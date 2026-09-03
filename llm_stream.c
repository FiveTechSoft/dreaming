/*
 * llm_stream.c — Streaming GEMV inference engine for DreamTree v2
 *
 * Zero full-matrix loads. Weights are read from disk in small chunks
 * using double-buffered I/O. Working set: ~88 KB (excl KV cache).
 *
 * Compilar (Windows):
 *   gcc -O2 -o llm_stream.exe llm_stream.c -lm
 *
 * Ejecutar:
 *   llm_stream.exe <model.drt2> <vocab.gguf> "DREAM" 30
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <omp.h>
#if defined(__AVX2__)
#include <immintrin.h>
#elif defined(__SSE2__)
#include <emmintrin.h>
#endif

static void __attribute__((constructor)) set_threads(void) {
    omp_set_num_threads(5);
}

/* ================================================================
 *  ARCHITECTURE CONSTANTS (overridden from header at runtime)
 * ================================================================ */
#define MAX_HIDDEN  2048
#define MAX_KV      256
#define MAX_FFN     5632
#define MAX_VOCAB   32000
#define MAX_SEQ     2048
#define MAX_LAYERS  32
#define MAX_BLOCKS  16
#define MAX_FFN_CH  32

/* ================================================================
 *  QUANTIZED TYPE CONSTANTS
 * ================================================================ */
typedef struct StreamIO_s StreamIO;  /* forward decl */
static void sio_read(StreamIO *s, void *dst, size_t n);  /* forward decl */
#define Q4_K_BLOCK_ELEMS 256
#define Q4_K_BLOCK_BYTES 144
#define Q6_K_BLOCK_ELEMS 256
#define Q6_K_BLOCK_BYTES 210
#define Q4_NL_BLOCK_ELEMS 32
#define Q4_NL_BLOCK_BYTES 18
static const int8_t kvalues_iq4nl[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10,
       1,   13,  25,  38,  53,  69,  89, 113
};

/* ================================================================
 *  F16 → F32
 * ================================================================ */
static inline float f16_to_f32(uint16_t h) {
    uint32_t sign = ((uint32_t)(h >> 15)) & 0x1;
    uint32_t exp  = ((uint32_t)(h >> 10)) & 0x1F;
    uint32_t man  = (uint32_t)h & 0x3FF;
    uint32_t result;
    if (exp == 0) {
        if (man == 0) { result = sign << 31; }
        else {
            exp = 1;
            while ((man & 0x400) == 0 && exp > 0) { man <<= 1; exp--; }
            man &= 0x3FF;
            result = (sign << 31) | ((exp + 127 - 15) << 23) | (man << 13);
        }
    } else if (exp == 31) {
        result = (sign << 31) | (0xFF << 23) | (man << 13);
    } else {
        result = (sign << 31) | ((exp + 127 - 15) << 23) | (man << 13);
    }
    float f;
    memcpy(&f, &result, 4);
    return f;
}

static inline void f16_to_f32_block(const uint8_t *src, float *dst, int n) {
    for (int i = 0; i < n; i++) {
        uint16_t h;
        memcpy(&h, src + i * 2, 2);
        dst[i] = f16_to_f32(h);
    }
}

/* ================================================================
 *  QUANTIZED DEQUANTIZATION (Q4_K, Q6_K, IQ4_NL)
 * ================================================================ */
static void dequant_q6_k(float *out, const uint8_t *raw, int n) {
    int nb = n / Q6_K_BLOCK_ELEMS;
    for (int b = 0; b < nb; b++) {
        const uint8_t *blk = raw + b * Q6_K_BLOCK_BYTES;
        const uint8_t *ql = blk;          /* 128 bytes: lower 4 bits of each 6-bit quant */
        const uint8_t *qh = blk + 128;    /* 64 bytes: upper 2 bits */
        const int8_t  *sc = (const int8_t *)(blk + 192); /* 16 bytes: scales */
        float d = f16_to_f32(*(const uint16_t *)(blk + 208));
        /* Process 256 elements in 2 groups of 128 */
        for (int sub = 0; sub < 2; sub++) {
            const uint8_t *ql_s = ql + sub * 64;
            const uint8_t *qh_s = qh + sub * 32;
            const int8_t  *sc_s = sc + sub * 8;
            float *y = out + b * 256 + sub * 128;
            for (int l = 0; l < 32; l++) {
                int is = l / 16;
                int q1 = (int)((ql_s[l] & 0xF) | (((qh_s[l] >> 0) & 3) << 4)) - 32;
                int q2 = (int)((ql_s[l + 32] & 0xF) | (((qh_s[l] >> 2) & 3) << 4)) - 32;
                int q3 = (int)((ql_s[l] >> 4) | (((qh_s[l] >> 4) & 3) << 4)) - 32;
                int q4 = (int)((ql_s[l + 32] >> 4) | (((qh_s[l] >> 6) & 3) << 4)) - 32;
                y[l      ] = d * sc_s[is    ] * q1;
                y[l + 32 ] = d * sc_s[is + 2] * q2;
                y[l + 64 ] = d * sc_s[is + 4] * q3;
                y[l + 96 ] = d * sc_s[is + 6] * q4;
            }
        }
    }
}

/* get_scale_min_k4: extract 6-bit scale/min from 12-byte scales array.
 * j ranges 0..7 (for 8 sub-groups of 32 elements each, 256 total).
 * For j<4: d = scales[j] & 0x3F, m = scales[j+4] & 0x3F
 * For j>=4: d = (scales[j+4] & 0xF) | ((scales[j-4] >> 6) << 4)
 *           m = (scales[j+4] >> 4) | ((scales[j] >> 6) << 4)
 */
static inline void get_scale_min_k4(int j, const uint8_t *scales,
                                    uint8_t *d, uint8_t *m) {
    if (j < 4) {
        *d = scales[j] & 0x3F;
        *m = scales[j + 4] & 0x3F;
    } else {
        *d = (scales[j + 4] & 0xF) | ((scales[j - 4] >> 6) << 4);
        *m = (scales[j + 4] >> 4)  | ((scales[j]     >> 6) << 4);
    }
}

/* Dequantize a Q4_K block: 256 elements from 144 bytes.
 * Layout: d(f16) | dmin(f16) | scales[12] | qs[128]
 * Process 64 elements at a time: 2 groups of 32, each using a scale/min pair.
 */
static void dequant_q4_k(float *out, const uint8_t *raw, int n) {
    int nb = n / Q4_K_BLOCK_ELEMS;
    for (int b = 0; b < nb; b++) {
        const uint8_t *blk = raw + b * Q4_K_BLOCK_BYTES;
        float d    = f16_to_f32(*(const uint16_t *)(blk + 0));
        float dmin = f16_to_f32(*(const uint16_t *)(blk + 2));
        const uint8_t *scales = blk + 4;
        const uint8_t *qs     = blk + 16;
        float *y = out + b * Q4_K_BLOCK_ELEMS;
        int q_idx = 0;
        int is = 0;
        for (int j = 0; j < Q4_K_BLOCK_ELEMS; j += 64) {
            uint8_t sc1, m1, sc2, m2;
            get_scale_min_k4(is + 0, scales, &sc1, &m1);
            get_scale_min_k4(is + 1, scales, &sc2, &m2);
            float d1 = d * sc1;  float m1v = dmin * m1;
            float d2 = d * sc2;  float m2v = dmin * m2;
            for (int l = 0; l < 32; l++)
                y[j + l]      = d1 * (qs[q_idx + l] & 0xF) - m1v;
            for (int l = 0; l < 32; l++)
                y[j + 32 + l] = d2 * (qs[q_idx + l] >> 4)   - m2v;
            q_idx += 32;
            is += 2;
        }
    }
}

static void dequant_iq4_nl(float *out, const uint8_t *raw, int n) {
    int nb = n / Q4_NL_BLOCK_ELEMS;
    for (int b = 0; b < nb; b++) {
        float d = f16_to_f32(*(const uint16_t *)(raw + b * 18));
        const uint8_t *qs = raw + b * 18 + 2;
        for (int j = 0; j < 16; j++) {
            out[b * 32 + j       ] = d * kvalues_iq4nl[qs[j] & 0xF];
            out[b * 32 + j + 16  ] = d * kvalues_iq4nl[qs[j] >> 4];
        }
    }
}

/* GEMV output-slicing with dequantization: reads quantized blocks,
 * dequantizes to F32, computes dot product. */
static void gemv_output_slice_q(float *out, const float *x, StreamIO *sio,
                                int in_dim, int out_start, int out_cols,
                                int qtype,
                                uint8_t *raw, float *df32) {
    int block_elems = (qtype == 12) ? Q4_K_BLOCK_ELEMS : Q4_NL_BLOCK_ELEMS;
    int block_bytes = (qtype == 12) ? Q4_K_BLOCK_BYTES : Q4_NL_BLOCK_BYTES;
    int blocks_per_row = in_dim / block_elems;
    int bytes_per_row = blocks_per_row * block_bytes;

    for (int j = 0; j < out_cols; j++) {
        sio_read(sio, raw, bytes_per_row);
        if (qtype == 12) dequant_q4_k(df32, raw, in_dim);
        else             dequant_iq4_nl(df32, raw, in_dim);
        float dot = 0.0f;
        for (int i = 0; i < in_dim; i++) dot += x[i] * df32[i];
        out[out_start + j] += dot;
    }
}

/* Reduction-slicing GEMV with dequantization */
static void gemv_reduction_q(float *out, const float *x, StreamIO *sio,
                              int in_dim, int rows, int qtype,
                              uint8_t *raw, float *df32) {
    int block_elems = (qtype == 12) ? Q4_K_BLOCK_ELEMS : Q4_NL_BLOCK_ELEMS;
    int block_bytes = (qtype == 12) ? Q4_K_BLOCK_BYTES : Q4_NL_BLOCK_BYTES;
    int blocks_per_row = in_dim / block_elems;
    int bytes_per_row = blocks_per_row * block_bytes;

    for (int j = 0; j < rows; j++) {
        sio_read(sio, raw, bytes_per_row);
        if (qtype == 12) dequant_q4_k(df32, raw, in_dim);
        else             dequant_iq4_nl(df32, raw, in_dim);
        float dot = 0.0f;
        for (int i = 0; i < in_dim; i++) dot += x[i] * df32[i];
        out[j] += dot;
    }
}

/* FFN gate+up with dequantization */
static void ffn_gate_up_q(float *gate_out, float *up_out, const float *x, StreamIO *sio,
                           int in_dim, int chunk, int qtype,
                           uint8_t *raw, float *df32) {
    int block_elems = (qtype == 12) ? Q4_K_BLOCK_ELEMS : Q4_NL_BLOCK_ELEMS;
    int block_bytes = (qtype == 12) ? Q4_K_BLOCK_BYTES : Q4_NL_BLOCK_BYTES;
    int blocks_per_row = in_dim / block_elems;
    int bytes_per_row = blocks_per_row * block_bytes;

    for (int j = 0; j < chunk; j++) {
        sio_read(sio, raw, bytes_per_row);
        if (qtype == 12) dequant_q4_k(df32, raw, in_dim);
        else             dequant_iq4_nl(df32, raw, in_dim);
        float dot = 0.0f;
        for (int i = 0; i < in_dim; i++) dot += x[i] * df32[i];
        gate_out[j] = dot;
    }
    for (int j = 0; j < chunk; j++) {
        sio_read(sio, raw, bytes_per_row);
        if (qtype == 12) dequant_q4_k(df32, raw, in_dim);
        else             dequant_iq4_nl(df32, raw, in_dim);
        float dot = 0.0f;
        for (int i = 0; i < in_dim; i++) dot += x[i] * df32[i];
        up_out[j] = dot;
    }
}

/* FFN down with dequantization */
static void ffn_down_q(float *out, const float *h, StreamIO *sio,
                        int out_dim, int chunk, int qtype,
                        uint8_t *raw, float *df32) {
    int block_elems = (qtype == 12) ? Q4_K_BLOCK_ELEMS : Q4_NL_BLOCK_ELEMS;
    int block_bytes = (qtype == 12) ? Q4_K_BLOCK_BYTES : Q4_NL_BLOCK_BYTES;
    int blocks_per_row = out_dim / block_elems;
    int bytes_per_row = blocks_per_row * block_bytes;

    for (int j = 0; j < chunk; j++) {
        sio_read(sio, raw, bytes_per_row);
        if (qtype == 12) dequant_q4_k(df32, raw, out_dim);
        else             dequant_iq4_nl(df32, raw, out_dim);
        float wt = h[j];
        for (int i = 0; i < out_dim; i++) out[i] += wt * df32[i];
    }
}

/* ================================================================
 *  DREAMTREE V2 HEADER
 * ================================================================ */
#pragma pack(push, 1)
typedef struct {
    char     magic[4];       /* "DRT2" */
    uint32_t version;
    uint32_t num_layers;
    uint32_t hidden_dim;
    uint32_t kv_dim;
    uint32_t ffn_dim;
    uint32_t vocab_size;
    uint32_t num_qkv_blocks;
    uint32_t ffn_chunk_size;
    uint32_t num_ffn_chunks;
    uint32_t num_heads;
    uint32_t head_dim;
    uint64_t layer_size;
    uint64_t global_size;
    uint64_t data_size;
    uint64_t total_file_size;
    uint32_t quant_type;       /* 0=F16, 12=Q4_K, 14=Q6_K, 15=IQ4_NL */
    uint64_t output_weights_sz;/* actual byte size of output.weight */
    uint8_t  reserved[36];     /* pad to 128 total */
} DRT2Header;
#pragma pack(pop)

/* ================================================================
 *  STREAMING I/O — Double-buffered page reader
 * ================================================================ */
#define IO_BUF_SIZE 65536  /* 64 KB per buffer */

struct StreamIO_s {
    FILE    *fp;
    uint8_t  buf[2][IO_BUF_SIZE];
    int      active;        /* 0 or 1 */
    size_t   buf_len[2];
    size_t   buf_pos[2];
    uint64_t file_pos;
};

static void sio_init(StreamIO *s, FILE *fp) {
    memset(s, 0, sizeof(*s));
    s->fp = fp;
    s->active = 0;
    s->buf_len[0] = s->buf_len[1] = 0;
    s->buf_pos[0] = s->buf_pos[1] = 0;
    s->file_pos = 0;
}

static void sio_refill(StreamIO *s, int buf_id) {
    s->buf_pos[buf_id] = 0;
    s->buf_len[buf_id] = fread(s->buf[buf_id], 1, IO_BUF_SIZE, s->fp);
}

/* Read exactly `n` bytes into `dst`. Handles buffer boundaries. */
static void sio_read(StreamIO *s, void *dst, size_t n) {
    uint8_t *out = (uint8_t *)dst;
    while (n > 0) {
        int aid = s->active;
        size_t avail = s->buf_len[aid] - s->buf_pos[aid];
        if (avail == 0) {
            sio_refill(s, aid);
            avail = s->buf_len[aid];
            if (avail == 0) break; /* EOF */
        }
        size_t to_copy = n < avail ? n : avail;
        memcpy(out, s->buf[aid] + s->buf_pos[aid], to_copy);
        s->buf_pos[aid] += to_copy;
        s->file_pos += to_copy;
        out += to_copy;
        n -= to_copy;
        /* Switch buffer for next refill */
        if (s->buf_pos[aid] >= s->buf_len[aid]) {
            s->active = 1 - aid;
        }
    }
}

/* Skip n bytes */
static void sio_skip(StreamIO *s, size_t n) {
    uint8_t tmp[256];
    while (n > 0) {
        size_t to_skip = n < 256 ? n : 256;
        sio_read(s, tmp, to_skip);
        n -= to_skip;
    }
}

/* Seek to an absolute position (invalidates buffered data) */
static void sio_seek(StreamIO *s, uint64_t pos) {
    fseek(s->fp, (long)pos, SEEK_SET);
    s->file_pos = pos;
    s->buf_len[0] = s->buf_len[1] = 0;
    s->buf_pos[0] = s->buf_pos[1] = 0;
    s->active = 0;
}

/* ================================================================
 *  MODEL STATE
 * ================================================================ */
typedef struct {
    DRT2Header hdr;
    StreamIO   sio;

    /* Global tensors (loaded into RAM) */
    float *w_embd;         /* [vocab × hidden] F32 */
    float *w_output_norm;  /* [hidden] F32 */
    float *w_output;       /* [hidden × vocab] F32 */

    /* Working buffers (static, ~88 KB total) */
    float x[MAX_HIDDEN];           /* residual vector */
    float x_norm[MAX_HIDDEN];      /* normalized vector (F32 after convert) */
    float accum[MAX_HIDDEN];       /* accumulation buffer */
    float h_buf[MAX_FFN];          /* intermediate H[m] for FFN */
    float ffn_out[MAX_HIDDEN];     /* FFN output accumulator */
    float q_proj[MAX_HIDDEN];      /* Q projection output */
    float k_proj[MAX_KV];          /* K projection output */
    float v_proj[MAX_KV];          /* V projection output */
    float attn_scores[MAX_SEQ];    /* attention score buffer */

    /* I/O streaming buffer for weight chunks */
    float w_chunk_f32[MAX_HIDDEN * MAX_BLOCKS]; /* decoded F32 chunk */
    uint16_t *scratch_f16;  /* scratch buffer for F16 block reads (malloc'd) */
    float *scratch_f32;     /* scratch buffer for F32 block reads (malloc'd) */
    int scratch_capacity;   /* capacity in elements */

    /* Dequant buffers (for quantized models) */
    uint8_t *dequant_raw;
    float *dequant_f32;

    /* KV cache */
    float *k_cache[MAX_LAYERS];    /* [MAX_SEQ × kv_dim] */
    float *v_cache[MAX_LAYERS];    /* [MAX_SEQ × kv_dim] */
    int seq_len;

    /* Architecture params (from header) */
    int hidden, kv_dim, ffn_dim, vocab, num_layers;
    int qkv_blocks, ffn_chunk, num_ffn_chunks;
    int head_dim, num_heads, num_kv_heads;
    int quant_type;  /* 0=F16, 12=Q4_K, 14=Q6_K, 15=IQ4_NL */

    /* Tokenizer */
    char *vocab_table[MAX_VOCAB];
    int vocab_size;

    /* Timing */
    clock_t load_start;

    /* Stream offsets for rewinding between tokens */
    uint64_t layer_data_start;  /* byte offset where layers begin */
    uint64_t output_weights_offset;  /* byte offset of output.weight */
} Model;

/* ================================================================
 *  GEMV — Streaming accumulation
 * ================================================================ */

/* Output Slicing GEMV: Y[out_start..out_end] = X · W[:, out_start..out_end]
 * W is read as a chunk of rows × cols from the stream.
 * Y is written directly to out + out_start. */
static void gemv_output_slice(float *out, const float *x, const float *W_chunk,
                               int in_dim, int out_start, int out_cols) {
    /* out[j] = dot(x, W[:, j]) for j in [out_start, out_start+out_cols) */
    #pragma omp parallel for schedule(static)
    for (int j = 0; j < out_cols; j++) {
        float sum = 0.0f;
        const float *w_col = W_chunk + j * in_dim;
#if defined(__AVX2__)
        int i = 0;
        __m256 vsum = _mm256_setzero_ps();
        for (; i + 8 <= in_dim; i += 8) {
            __m256 vx = _mm256_loadu_ps(x + i);
            __m256 vw = _mm256_loadu_ps(w_col + i);
            vsum = _mm256_add_ps(vsum, _mm256_mul_ps(vx, vw));
        }
        __m128 hi = _mm256_extractf128_ps(vsum, 1);
        __m128 lo = _mm256_castps256_ps128(vsum);
        __m128 s4 = _mm_add_ps(lo, hi);
        __m128 s2 = _mm_add_ps(s4, _mm_movehl_ps(s4, s4));
        __m128 s1 = _mm_add_ss(s2, _mm_shuffle_ps(s2, s2, 1));
        sum = _mm_cvtss_f32(s1);
        for (; i < in_dim; i++) sum += x[i] * w_col[i];
#elif defined(__SSE2__)
        int i = 0;
        __m128 vsum = _mm_setzero_ps();
        for (; i + 4 <= in_dim; i += 4) {
            __m128 vx = _mm_loadu_ps(x + i);
            __m128 vw = _mm_loadu_ps(w_col + i);
            vsum = _mm_add_ps(vsum, _mm_mul_ps(vx, vw));
        }
        float tmp[4];
        _mm_storeu_ps(tmp, vsum);
        sum = tmp[0] + tmp[1] + tmp[2] + tmp[3];
        for (; i < in_dim; i++) sum += x[i] * w_col[i];
#else
        for (int i = 0; i < in_dim; i++) sum += x[i] * w_col[i];
#endif
        out[out_start + j] = sum;
    }
}

/* ================================================================
 *  TRANSFORMER OPS
 * ================================================================ */
static void rmsnorm(float *out, const float *x, const float *w, int dim) {
    float sum = 0.0f;
    for (int i = 0; i < dim; i++) sum += x[i] * x[i];
    float scale = 1.0f / sqrtf(sum / dim + 1e-6f);
    for (int i = 0; i < dim; i++) out[i] = x[i] * scale * w[i];
}

static float rope_cos[MAX_SEQ][64];
static float rope_sin[MAX_SEQ][64];
static int rope_ok = 0;

static void rope_init(int head_dim) {
    for (int pos = 0; pos < MAX_SEQ; pos++) {
        for (int i = 0; i < head_dim; i += 2) {
            float freq = 1.0f / powf(10000.0f, (float)i / head_dim);
            float angle = pos * freq;
            rope_cos[pos][i] = rope_cos[pos][i+1] = cosf(angle);
            rope_sin[pos][i] = rope_sin[pos][i+1] = sinf(angle);
        }
    }
    rope_ok = 1;
}

static void apply_rope(float *x, int pos, int head_dim) {
    if (!rope_ok) rope_init(head_dim);
    for (int i = 0; i < head_dim; i += 2) {
        float c = rope_cos[pos][i], s = rope_sin[pos][i];
        float x0 = x[i], x1 = x[i+1];
        x[i]   = x0 * c - x1 * s;
        x[i+1] = x0 * s + x1 * c;
    }
}

/* Online softmax for attention (single head) */
static void attention_head(float *ctx_out, const float *q_h, float *k_cache_h,
                           float *v_cache_h, int kv_len, int head_dim) {
    float maxv = -1e30f, sumexp = 0.0f;
    float scores[MAX_SEQ];

    /* Compute Q·K^T for all cached positions */
    for (int k = 0; k < kv_len; k++) {
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++)
            dot += q_h[d] * k_cache_h[k * head_dim + d];
        dot /= sqrtf((float)head_dim);
        scores[k] = dot;

        /* Online softmax update */
        if (dot > maxv) {
            float old_scale = expf(maxv - dot);
            sumexp = sumexp * old_scale + 1.0f;
            maxv = dot;
        } else {
            sumexp += expf(dot - maxv);
        }
    }

    /* Compute context vector: C = sum_k(softmax(s)_k · V_k) */
    memset(ctx_out, 0, head_dim * sizeof(float));
    for (int k = 0; k < kv_len; k++) {
        float wt = expf(scores[k] - maxv) / sumexp;
        if (k > kv_len - 1) wt = 0.0f; /* causal mask */
        for (int d = 0; d < head_dim; d++)
            ctx_out[d] += wt * v_cache_h[k * head_dim + d];
    }
}

/* ================================================================
 *  MODEL LOAD — Sequential streaming from DreamTree v2
 * ================================================================ */
static void model_load(Model *m, const char *dt_path, const char *gguf_vocab_path) {
    m->load_start = clock();
    memset(m, 0, sizeof(*m));

    /* Open DreamTree file */
    FILE *fp = fopen(dt_path, "rb");
    if (!fp) { fprintf(stderr, "Cannot open %s\n", dt_path); exit(1); }
    sio_init(&m->sio, fp);

    /* Read header */
    sio_read(&m->sio, &m->hdr, sizeof(DRT2Header));
    if (memcmp(m->hdr.magic, "DRT2", 4) != 0) {
        fprintf(stderr, "Bad magic\n"); exit(1);
    }

    m->hidden = m->hdr.hidden_dim;
    m->kv_dim = m->hdr.kv_dim;
    m->ffn_dim = m->hdr.ffn_dim;
    m->vocab = m->hdr.vocab_size;
    m->num_layers = m->hdr.num_layers;
    m->qkv_blocks = m->hdr.num_qkv_blocks;
    m->ffn_chunk = m->hdr.ffn_chunk_size;
    m->num_ffn_chunks = m->hdr.num_ffn_chunks;
    m->num_heads = m->hdr.num_heads;
    m->head_dim = m->hdr.head_dim;
    m->num_kv_heads = m->kv_dim / m->head_dim;
    m->quant_type = m->hdr.quant_type;

    printf("DreamTree v2: %s\n", dt_path);
    printf("  layers=%d hidden=%d kv=%d ffn=%d vocab=%d\n",
           m->num_layers, m->hidden, m->kv_dim, m->ffn_dim, m->vocab);
    printf("  heads=%d kv_heads=%d head_dim=%d qkv_blocks=%d ffn_chunk=%d num_ffn_chunks=%d\n",
           m->num_heads, m->num_kv_heads, m->head_dim, m->qkv_blocks, m->ffn_chunk, m->num_ffn_chunks);
    printf("  quant_type=%d layer_size=%llu total=%llu\n",
           m->quant_type,
           (unsigned long long)m->hdr.layer_size,
           (unsigned long long)m->hdr.total_file_size);

    /* ── Load global tensors (small, fit in RAM) ── */
    printf("  Loading globals...\n"); fflush(stdout);

    if (m->quant_type == 0) {
        /* F16 model: token_embd F16 → F32 */
        printf("    allocating embd: %d floats (F16)...\n", m->vocab * m->hidden); fflush(stdout);
        m->w_embd = malloc(m->vocab * m->hidden * sizeof(float));
        if (!m->w_embd) { fprintf(stderr, "OOM for embd\n"); exit(1); }
        {
            uint8_t tmp[MAX_HIDDEN * 2];
            for (int i = 0; i < m->vocab * m->hidden; i += m->hidden) {
                sio_read(&m->sio, tmp, m->hidden * 2);
                f16_to_f32_block(tmp, m->w_embd + i, m->hidden);
            }
        }
        printf("    token_embd: %d × %d = %d elements (F16)\n",
               m->vocab, m->hidden, m->vocab * m->hidden); fflush(stdout);

        /* output_norm: hidden F16 → F32 */
        m->w_output_norm = malloc(m->hidden * sizeof(float));
        {
            uint8_t tmp[MAX_HIDDEN * 2];
            sio_read(&m->sio, tmp, m->hidden * 2);
            f16_to_f32_block(tmp, m->w_output_norm, m->hidden);
        }
    } else {
        /* Quantized model: token_embd stored as F32 in DRT2 */
        printf("    allocating embd: %d floats (F32, quantized)...\n", m->vocab * m->hidden); fflush(stdout);
        m->w_embd = malloc(m->vocab * m->hidden * sizeof(float));
        if (!m->w_embd) { fprintf(stderr, "OOM for embd\n"); exit(1); }
        sio_read(&m->sio, m->w_embd, m->vocab * m->hidden * sizeof(float));
        printf("    token_embd: %d × %d = %d elements (F32)\n",
               m->vocab, m->hidden, m->vocab * m->hidden); fflush(stdout);

        /* output_norm: hidden F32 */
        m->w_output_norm = malloc(m->hidden * sizeof(float));
        sio_read(&m->sio, m->w_output_norm, m->hidden * sizeof(float));
    }
    printf("    output_norm loaded\n"); fflush(stdout);

    /* NOTE: output weights are at the END of the file, not here.
     * We'll seek to them after processing all layers. */

    /* ── Allocate KV cache ── */
    printf("    allocating KV cache...\n"); fflush(stdout);
    for (int l = 0; l < m->num_layers; l++) {
        m->k_cache[l] = calloc(MAX_SEQ * m->kv_dim, sizeof(float));
        m->v_cache[l] = calloc(MAX_SEQ * m->kv_dim, sizeof(float));
    }
    m->seq_len = 0;

    /* Allocate scratch buffers for block reads
     * Max block = max(H*hblock, chunk_actual*H) for F16 data */
    int hblock_scratch = m->hidden / m->qkv_blocks;
    int attn_block_sz = m->hidden * hblock_scratch;
    int ffn_block_sz = m->ffn_chunk * m->hidden;
    m->scratch_capacity = attn_block_sz > ffn_block_sz ? attn_block_sz : ffn_block_sz;
    m->scratch_f16 = malloc(m->scratch_capacity * sizeof(uint16_t));
    m->scratch_f32 = malloc(m->scratch_capacity * sizeof(float));

    /* Allocate dequant buffers for quantized models */
    if (m->quant_type != 0) {
        int max_row = m->hidden > m->ffn_dim ? m->hidden : m->ffn_dim;
        m->dequant_raw = malloc(max_row * 2);  /* enough for any row's raw bytes */
        m->dequant_f32 = malloc(max_row * sizeof(float));
    } else {
        m->dequant_raw = NULL;
        m->dequant_f32 = NULL;
    }

    printf("  Scratch buffers: %d elements (%d KB)\n",
           m->scratch_capacity, (int)(m->scratch_capacity * 6 / 1024)); fflush(stdout);

    /* ── Load vocabulary from GGUF ── */
    printf("  Loading vocab from %s...\n", gguf_vocab_path);
    {
        int64_t sz;
        FILE *vf = fopen(gguf_vocab_path, "rb");
        if (!vf) { fprintf(stderr, "Cannot open vocab GGUF\n"); exit(1); }
        fseek(vf, 0, SEEK_END); sz = ftell(vf); fseek(vf, 0, SEEK_SET);
        uint8_t *vdata = malloc(sz);
        fread(vdata, 1, sz, vf);
        fclose(vf);

        size_t vp = 24;
        int64_t kvcount;
        memcpy(&kvcount, vdata + 16, 8);
        static const int gtype_sz[13] = {1,1,2,2,4,4,4,1,0,0,8,8,8};

        m->vocab_size = 0;
        for (int64_t i = 0; i < kvcount; i++) {
            uint64_t klen; memcpy(&klen, vdata + vp, 8); vp += 8;
            char key[128];
            uint64_t cplen = klen < 127 ? klen : 127;
            memcpy(key, vdata + vp, cplen); key[cplen] = '\0'; vp += klen;
            uint32_t vtype; memcpy(&vtype, vdata + vp, 4); vp += 4;

            if (strcmp(key, "tokenizer.ggml.tokens") == 0 && vtype == 9) {
                uint32_t at; memcpy(&at, vdata + vp, 4); vp += 4;
                uint64_t al; memcpy(&al, vdata + vp, 8); vp += 8;
                m->vocab_size = (int)al;
                for (uint64_t a = 0; a < al && (int)a < m->vocab_size; a++) {
                    if (at == 8) {
                        uint64_t sl; memcpy(&sl, vdata + vp, 8); vp += 8;
                        m->vocab_table[a] = malloc((int)sl + 1);
                        memcpy(m->vocab_table[a], vdata + vp, (int)sl);
                        m->vocab_table[a][(int)sl] = '\0';
                        vp += sl;
                    } else {
                        vp += gtype_sz[at < 13 ? at : 0];
                        m->vocab_table[a] = strdup("?");
                    }
                }
                break;
            }
            if (vtype == 8) { uint64_t sl; memcpy(&sl, vdata + vp, 8); vp += 8; vp += sl; }
            else if (vtype == 9) {
                uint32_t at; memcpy(&at, vdata + vp, 4); vp += 4;
                uint64_t al; memcpy(&al, vdata + vp, 8); vp += 8;
                for (uint64_t a = 0; a < al; a++) {
                    if (at == 8) { uint64_t sl; memcpy(&sl, vdata + vp, 8); vp += 8; vp += sl; }
                    else { vp += gtype_sz[at < 13 ? at : 0]; }
                }
            } else { vp += gtype_sz[vtype < 13 ? vtype : 0]; }
        }
        free(vdata);
        printf("    Vocab: %d tokens\n", m->vocab_size); fflush(stdout);
    }

    /* ── Pre-seek to output weights (at end of file) ── */
    /* We'll load them lazily after all layers are processed */

    printf("  Model ready. Streaming layers from disk...\n"); fflush(stdout);
    double elapsed = (double)(clock() - m->load_start) / CLOCKS_PER_SEC;
    printf("  Load time: %.2f seconds\n", elapsed); fflush(stdout);

    /* Record where layers start and where output weights are */
    m->layer_data_start = 128 + m->hdr.global_size;
    /* output weights are at the END of the file, stored as F32 */
    m->output_weights_offset = (uint64_t)m->hdr.total_file_size
                               - (uint64_t)m->vocab * m->hidden * sizeof(float);
}

static void model_free(Model *m) {
    free(m->w_embd);
    free(m->w_output_norm);
    for (int l = 0; l < m->num_layers; l++) {
        free(m->k_cache[l]);
        free(m->v_cache[l]);
    }
    free(m->scratch_f16);
    free(m->scratch_f32);
    free(m->dequant_raw);
    free(m->dequant_f32);
    for (int i = 0; i < m->vocab_size; i++)
        free(m->vocab_table[i]);
    fclose(m->sio.fp);
}

/* ================================================================
 *  STREAMING FORWARD PASS
 *
 *  Reads weights exactly in file order. No seeks, no full matrices.
 * ================================================================ */
static void model_forward(Model *m, int *tokens, int seq_len, float *logits) {
    int cur = seq_len - 1;
    int tok = tokens[cur];
    if (tok < 0 || tok >= m->vocab) tok = 0;
    int H = m->hidden, KV = m->kv_dim, FFN = m->ffn_dim;
    int B = m->qkv_blocks, M = m->ffn_chunk, NC = m->num_ffn_chunks;
    int hblock = H / B;
    int kvblock = KV / B;
    int qt = m->quant_type;

    /* Seek to layer data start (rewind for each token) */
    sio_seek(&m->sio, m->layer_data_start);

    /* ── Embed token ── */
    memcpy(m->x, m->w_embd + tok * H, H * sizeof(float));

    /* ── Process each layer ── */
    for (int l = 0; l < m->num_layers; l++) {
        /* === 1. Read norm1 from stream === */
        if (qt == 0) {
            uint8_t tmp[MAX_HIDDEN * 2];
            sio_read(&m->sio, tmp, H * 2);
            f16_to_f32_block(tmp, m->x_norm, H);
        } else {
            sio_read(&m->sio, m->x_norm, H * sizeof(float));
        }
        rmsnorm(m->x_norm, m->x, m->x_norm, H);

        /* === 2. Q, K, V projections === */
        if (qt == 0) {
            /* F16: streaming output slicing */
            memset(m->q_proj, 0, H * sizeof(float));
            for (int b = 0; b < B; b++) {
                sio_read(&m->sio, m->scratch_f16, (size_t)H * hblock * 2);
                f16_to_f32_block((const uint8_t *)m->scratch_f16, m->scratch_f32, H * hblock);
                gemv_output_slice(m->q_proj, m->x_norm, m->scratch_f32, H, b * hblock, hblock);
            }
            memset(m->k_proj, 0, KV * sizeof(float));
            for (int b = 0; b < B; b++) {
                sio_read(&m->sio, m->scratch_f16, (size_t)H * kvblock * 2);
                f16_to_f32_block((const uint8_t *)m->scratch_f16, m->scratch_f32, H * kvblock);
                gemv_output_slice(m->k_proj, m->x_norm, m->scratch_f32, H, b * kvblock, kvblock);
            }
            memset(m->v_proj, 0, KV * sizeof(float));
            for (int b = 0; b < B; b++) {
                sio_read(&m->sio, m->scratch_f16, (size_t)H * kvblock * 2);
                f16_to_f32_block((const uint8_t *)m->scratch_f16, m->scratch_f32, H * kvblock);
                gemv_output_slice(m->v_proj, m->x_norm, m->scratch_f32, H, b * kvblock, kvblock);
            }
        } else {
            /* Quantized: dequant on the fly */
            memset(m->q_proj, 0, H * sizeof(float));
            for (int b = 0; b < B; b++)
                gemv_output_slice_q(m->q_proj, m->x_norm, &m->sio, H, b * hblock, hblock, qt, m->dequant_raw, m->dequant_f32);
            memset(m->k_proj, 0, KV * sizeof(float));
            for (int b = 0; b < B; b++)
                gemv_output_slice_q(m->k_proj, m->x_norm, &m->sio, H, b * kvblock, kvblock, qt, m->dequant_raw, m->dequant_f32);
            memset(m->v_proj, 0, KV * sizeof(float));
            for (int b = 0; b < B; b++)
                gemv_output_slice_q(m->v_proj, m->x_norm, &m->sio, H, b * kvblock, kvblock, qt, m->dequant_raw, m->dequant_f32);
        }

        /* Apply RoPE */
        for (int hd = 0; hd < m->num_heads; hd++)
            apply_rope(m->q_proj + hd * m->head_dim, cur, m->head_dim);
        for (int hd = 0; hd < m->num_kv_heads; hd++)
            apply_rope(m->k_proj + hd * m->head_dim, cur, m->head_dim);

        /* Store K,V in cache */
        memcpy(m->k_cache[l] + cur * KV, m->k_proj, KV * sizeof(float));
        memcpy(m->v_cache[l] + cur * KV, m->v_proj, KV * sizeof(float));

        /* === 3. Attention (per head) === */
        memset(m->accum, 0, H * sizeof(float));
        int groups_per_kv = m->num_heads / m->num_kv_heads;

        for (int qh = 0; qh < m->num_heads; qh++) {
            int kvh = qh / groups_per_kv;
            float ctx_h[64];
            attention_head(ctx_h, m->q_proj + qh * m->head_dim,
                           m->k_cache[l] + kvh * m->head_dim,
                           m->v_cache[l] + kvh * m->head_dim,
                           cur + 1, m->head_dim);
            memcpy(m->accum + qh * m->head_dim, ctx_h, m->head_dim * sizeof(float));
        }

        /* === 4. attn_out via Reduction Slicing === */
        memset(m->x_norm, 0, H * sizeof(float));

        if (qt == 0) {
            for (int b = 0; b < B; b++) {
                sio_read(&m->sio, m->scratch_f16, (size_t)hblock * H * 2);
                for (int j = 0; j < hblock; j++) {
                    float dot = 0.0f;
                    const uint16_t *row = m->scratch_f16 + j * H;
                    for (int i = 0; i < H; i++)
                        dot += m->accum[i] * f16_to_f32(row[i]);
                    m->x_norm[b * hblock + j] += dot;
                }
            }
        } else {
            for (int b = 0; b < B; b++)
                gemv_reduction_q(m->x_norm, m->accum, &m->sio, H, hblock, qt, m->dequant_raw, m->dequant_f32);
        }

        /* Residual: x = x + attn_out */
        for (int i = 0; i < H; i++) m->x[i] += m->x_norm[i];

        /* === 5. Read norm2 === */
        if (qt == 0) {
            uint8_t tmp[MAX_HIDDEN * 2];
            sio_read(&m->sio, tmp, H * 2);
            f16_to_f32_block(tmp, m->x_norm, H);
        } else {
            sio_read(&m->sio, m->x_norm, H * sizeof(float));
        }

        /* === 6. FFN SwiGLU — streaming chunks === */
        memset(m->ffn_out, 0, H * sizeof(float));

        if (qt == 0) {
            for (int c = 0; c < NC; c++) {
                int chunk_actual = M;
                if (c * M + chunk_actual > FFN) chunk_actual = FFN - c * M;

                /* gate */
                {
                    sio_read(&m->sio, m->scratch_f16, (size_t)chunk_actual * H * 2);
                    for (int j = 0; j < chunk_actual; j++) {
                        float dot = 0.0f;
                        const uint16_t *row = m->scratch_f16 + j * H;
                        for (int i = 0; i < H; i++)
                            dot += m->x_norm[i] * f16_to_f32(row[i]);
                        m->h_buf[j] = dot;
                    }
                }
                /* up */
                {
                    sio_read(&m->sio, m->scratch_f16, (size_t)chunk_actual * H * 2);
                    for (int j = 0; j < chunk_actual; j++) {
                        float dot = 0.0f;
                        const uint16_t *row = m->scratch_f16 + j * H;
                        for (int i = 0; i < H; i++)
                            dot += m->x_norm[i] * f16_to_f32(row[i]);
                        float g = m->h_buf[j];
                        float silu = g / (1.0f + expf(-g));
                        m->h_buf[j] = silu * dot;
                    }
                }
                /* down */
                {
                    sio_read(&m->sio, m->scratch_f16, (size_t)chunk_actual * H * 2);
                    for (int j = 0; j < chunk_actual; j++) {
                        float wt = m->h_buf[j];
                        const uint16_t *row = m->scratch_f16 + j * H;
                        for (int i = 0; i < H; i++)
                            m->ffn_out[i] += wt * f16_to_f32(row[i]);
                    }
                }
            }
        } else {
            for (int c = 0; c < NC; c++) {
                int chunk_actual = M;
                if (c * M + chunk_actual > FFN) chunk_actual = FFN - c * M;

                /* gate + up with dequant */
                ffn_gate_up_q(m->h_buf, m->h_buf + M, m->x_norm, &m->sio, H, chunk_actual, qt, m->dequant_raw, m->dequant_f32);

                /* Apply SiLU: h_buf[j] = silu(gate[j]) * up[j] */
                for (int j = 0; j < chunk_actual; j++) {
                    float g = m->h_buf[j];
                    float u = m->h_buf[M + j];
                    m->h_buf[j] = (g / (1.0f + expf(-g))) * u;
                }

                /* down with dequant */
                ffn_down_q(m->ffn_out, m->h_buf, &m->sio, H, chunk_actual, qt, m->dequant_raw, m->dequant_f32);
            }
        }

        /* Residual: x = x + ffn_out */
        for (int i = 0; i < H; i++) m->x[i] += m->ffn_out[i];
    }

    /* ── Final output norm + projection ── */
    rmsnorm(m->x_norm, m->x, m->w_output_norm, H);

    /* Output projection: seek to output weights at end of file */
    sio_seek(&m->sio, m->output_weights_offset);

    /* Y[v] = dot(x_norm, W_out[v,:]) for all v */
    memset(logits, 0, m->vocab * sizeof(float));

    if (qt == 0) {
        uint16_t *row_buf = (uint16_t *)m->h_buf;
        for (int v = 0; v < m->vocab; v++) {
            sio_read(&m->sio, row_buf, H * 2);
            float dot = 0.0f;
            for (int i = 0; i < H; i++)
                dot += m->x_norm[i] * f16_to_f32(row_buf[i]);
            logits[v] = dot;
        }
    } else {
        /* Quantized model: output weights stored as F32 in DRT2 */
        for (int v = 0; v < m->vocab; v++) {
            sio_read(&m->sio, m->h_buf, H * sizeof(float));
            float dot = 0.0f;
            for (int i = 0; i < H; i++) dot += m->x_norm[i] * m->h_buf[i];
            logits[v] = dot;
        }
    }

    /* Clamp logits */
    for (int i = 0; i < m->vocab; i++) {
        if (!isfinite(logits[i])) logits[i] = -1e6f;
        else if (logits[i] > 40.0f) logits[i] = 40.0f;
        else if (logits[i] < -40.0f) logits[i] = -40.0f;
    }
}

/* ================================================================
 *  SAMPLING
 * ================================================================ */
static int sample(float *logits, int vocab, int top_k, float temp) {
    if (temp < 1e-10f) temp = 1e-10f;
    float maxv = logits[0];
    for (int i = 1; i < vocab; i++) if (logits[i] > maxv) maxv = logits[i];
    float top_vals[256];
    int top_idx[256];
    if (top_k > 256) top_k = 256;
    for (int i = 0; i < top_k; i++) { top_vals[i] = -1e30f; top_idx[i] = 0; }
    for (int i = 0; i < vocab; i++) {
        float v = (logits[i] - maxv) / temp;
        for (int j = 0; j < top_k; j++) {
            if (v > top_vals[j]) {
                for (int k = top_k - 1; k > j; k--) {
                    top_vals[k] = top_vals[k-1]; top_idx[k] = top_idx[k-1];
                }
                top_vals[j] = v; top_idx[j] = i;
                break;
            }
        }
    }
    float sum = 0.0f;
    float mx = top_vals[0];
    for (int i = 0; i < top_k; i++) top_vals[i] = expf(top_vals[i] - mx);
    for (int i = 0; i < top_k; i++) sum += top_vals[i];
    for (int i = 0; i < top_k; i++) top_vals[i] /= sum;
    float r = (float)rand() / (float)RAND_MAX, cum = 0.0f;
    for (int i = 0; i < top_k; i++) { cum += top_vals[i]; if (r <= cum) return top_idx[i]; }
    return top_idx[top_k - 1];
}

/* ================================================================
 *  TOKENIZER (BPE)
 * ================================================================ */
static int tokenize(const char *text, char **vocab, int vs, int *tokens, int max) {
    int ids[4096], n = 0;
    char pieces[4096][128];
    for (int i = 0; text[i] && n < 4090; i++) {
        unsigned char b = (unsigned char)text[i];
        ids[n] = b + 3;
        snprintf(pieces[n], 128, "<0x%02X>", b);
        char single[2] = {(char)b, 0};
        for (int v = 259; v < vs; v++)
            if (vocab[v] && strcmp(vocab[v], single) == 0) { ids[n] = v; strcpy(pieces[n], single); break; }
        n++;
    }
    for (int iter = 0; iter < 4096; iter++) {
        int best_id = -1, best_pos = -1, best_rank = vs + 1;
        for (int i = 0; i < n - 1; i++) {
            char merged[256];
            snprintf(merged, sizeof(merged), "%s%s", pieces[i], pieces[i+1]);
            for (int v = 259; v < vs; v++)
                if (vocab[v] && strcmp(vocab[v], merged) == 0 && v < best_rank)
                    { best_rank = v; best_id = v; best_pos = i; break; }
        }
        if (best_pos < 0) break;
        ids[best_pos] = best_id;
        snprintf(pieces[best_pos], 256, "%s%s", pieces[best_pos], pieces[best_pos+1]);
        for (int i = best_pos + 1; i < n - 1; i++) { ids[i] = ids[i+1]; strcpy(pieces[i], pieces[i+1]); }
        n--;
    }
    int nt = 0;
    for (int i = 0; i < n && nt < max; i++) tokens[nt++] = ids[i];
    return nt;
}

/* ================================================================
 *  MAIN
 * ================================================================ */
int main(int argc, char **argv) {
    const char *dt_path = "C:/dreaming/tinyLlama.drt2";
    const char *gguf_path = "C:/dreaming/tinyLlama.gguf";
    const char *prompt = "DREAM";
    int max_new = 30;
    float temp = 0.8f;
    int top_k = 25;

    if (argc >= 2) dt_path = argv[1];
    if (argc >= 3) gguf_path = argv[2];
    if (argc >= 4) prompt = argv[3];
    if (argc >= 5) max_new = atoi(argv[4]);
    if (argc >= 6) temp = (float)atof(argv[5]);
    if (argc >= 7) top_k = atoi(argv[6]);

    printf("========================================\n");
    printf("  LLM Stream — DreamTree v2 GEMV\n");
    printf("========================================\n\n");

    Model *model = malloc(sizeof(Model));
    model_load(model, dt_path, gguf_path);

    printf("\n--- Generation ---\n"); fflush(stdout);
    int prompt_tokens[MAX_SEQ];
    int n_prompt = 0;
    prompt_tokens[n_prompt++] = 1;
    n_prompt += tokenize(prompt, model->vocab_table, model->vocab_size,
                         prompt_tokens + n_prompt, MAX_SEQ - n_prompt - 1);

    int full_tokens[MAX_SEQ * 2];
    memcpy(full_tokens, prompt_tokens, n_prompt * sizeof(int));
    int total = n_prompt;
    float *logits = malloc(MAX_VOCAB * sizeof(float));

    clock_t t0 = clock();
    for (int i = 0; i < n_prompt; i++) {
        fprintf(stderr, "  Prompt token %d/%d...\n", i+1, n_prompt);
        model_forward(model, full_tokens, i + 1, logits);
    }

    for (int step = 0; step < max_new; step++) {
        fprintf(stderr, "  Step %d/%d...\n", step+1, max_new);
        model_forward(model, full_tokens, total, logits);
        int next = sample(logits, model->vocab_size, top_k, temp);
        full_tokens[total++] = next;
        printf("\r  Step %d/%d  token=%d", step + 1, max_new, next);
        fflush(stdout);
        if (next == 2 || total >= MAX_SEQ * 2 - 1) break;
    }
    printf("\n");

    printf("\n--- Decoded ---\n");
    for (int i = n_prompt; i < total; i++) {
        int tid = full_tokens[i];
        if (tid >= 0 && tid < model->vocab_size && model->vocab_table[tid] && tid != 1 && tid != 2) {
            const char *s = model->vocab_table[tid];
            for (int j = 0; s[j]; j++) {
                if ((unsigned char)s[j] == 0xe2 && (unsigned char)s[j+1] == 0x96 && (unsigned char)s[j+2] == 0x81)
                    { putchar(' '); j += 2; }
                else putchar(s[j]);
            }
        } else printf("[tok:%d]", tid);
    }
    printf("\n--- End ---\n");

    double elapsed = (double)(clock() - t0) / CLOCKS_PER_SEC;
    printf("Generated %d tokens in %.2fs (%.1f tok/s)\n", total - n_prompt, elapsed, (total - n_prompt) / elapsed);

    model_free(model);
    free(model);
    free(logits);
    return 0;
}
