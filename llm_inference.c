/*
 * llm_inference.c - Motor de inferencia puro C para LLM Onirico GGUF
 * Sin dependencias externas. Solo stdio, stdlib, string, stdint, math.
 *
 * Compilar (Windows):
 *   gcc -O2 -o llm_inference.exe llm_inference.c -lm
 *
 * Ejecutar:
 *   llm_inference.exe C:/tmp/llm_pequeno.gguf "DREAM" 30
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>

/* ================================================================
 *  CONSTANTS (DeepSeek-V2 Lite)
 * ================================================================ */
#define HIDDEN_DIM    1024
#define NUM_LAYERS    12
#define NUM_HEADS     16
#define HEAD_DIM      64
#define VOCAB_SIZE    32000
#define FFN_DIM       2048
#define MAX_SEQ       2048
#define MAX_TOKENS    512

/* ================================================================
 *  HALF-PRECISION FLOAT (F16 -> F32)
 *  IEEE 754 half-precision manual decoder
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

/* ================================================================
 *  FILE SIZE UTILITY
 * ================================================================ */
static long file_size(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    fseek(f, 0, SEEK_END);
    long s = ftell(f);
    fclose(f);
    return s;
}

/* ================================================================
 *  GGUF READER (minimal, reads into memory)
 * ================================================================ */
typedef struct {
    uint8_t *data;
    size_t size;
    size_t pos;
    int version;
    int64_t num_tensors;
    int64_t num_kv;
    int num_meta;
    char meta_keys[64][48];
    char meta_vals[64][128];
    int num_tfound;
    char tnames[128][64];
    int64_t toffsets[128];
} GGUF;

static uint64_t read_u64_buf(const uint8_t *buf, size_t *pos) {
    uint64_t v;
    memcpy(&v, buf + *pos, 8);
    *pos += 8;
    return v;
}

static uint32_t read_u32_buf(const uint8_t *buf, size_t *pos) {
    uint32_t v;
    memcpy(&v, buf + *pos, 4);
    *pos += 4;
    return v;
}

static void read_string_buf(const uint8_t *buf, size_t *pos, char *out, int maxlen) {
    uint64_t len = read_u64_buf(buf, pos);
    if (len >= (uint64_t)maxlen) len = maxlen - 1;
    memcpy(out, buf + *pos, len);
    out[len] = '\0';
    *pos += len;
}

static void gguf_load(GGUF *g, const char *path) {
    g->size = file_size(path);
    printf("Loading '%s' (%ld bytes, %.1f MB)\n", path, g->size, (double)g->size / (1024*1024));
    if (g->size < 0) { fprintf(stderr, "Cannot open %s\n", path); exit(1); }
    g->data = malloc(g->size);
    if (!g->data) { fprintf(stderr, "malloc failed\n"); exit(1); }
    FILE *f = fopen(path, "rb");
    fread(g->data, 1, g->size, f);
    fclose(f);
    g->pos = 0;

    /* Check magic */
    if (memcmp(g->data, "GGUF", 4) != 0) {
        fprintf(stderr, "Not a GGUF file\n"); exit(1);
    }
    g->pos = 4;

    g->version = read_u32_buf(g->data, &g->pos);
    g->num_tensors = read_u64_buf(g->data, &g->pos);
    g->num_kv = read_u64_buf(g->data, &g->pos);

    /* Parse metadata */
    g->num_meta = 0;
    for (int64_t i = 0; i < g->num_kv && g->num_meta < 64; i++) {
        char key[48], val[128];
        read_string_buf(g->data, &g->pos, key, 48);
        uint64_t vtype = read_u64_buf(g->data, &g->pos);
        uint64_t vlen = read_u64_buf(g->data, &g->pos);
        if (vlen >= 128) vlen = 127;
        memcpy(val, g->data + g->pos, vlen);
        val[vlen] = '\0';
        g->pos += vlen;
        (void)vtype;
        if (g->num_meta < 64) {
            strncpy(g->meta_keys[g->num_meta], key, 47);
            g->meta_keys[g->num_meta][47] = '\0';
            strncpy(g->meta_vals[g->num_meta], val, 127);
            g->meta_vals[g->num_meta][127] = '\0';
            g->num_meta++;
        }
    }

    /* Parse tensor index */
    g->num_tfound = 0;
    for (int64_t i = 0; i < g->num_tensors && g->num_tfound < 128; i++) {
        char tname[64];
        read_string_buf(g->data, &g->pos, tname, 64);
        uint32_t ttype = read_u32_buf(g->data, &g->pos);
        uint32_t ndims = read_u32_buf(g->data, &g->pos);
        int64_t dims[8] = {0};
        for (uint32_t d = 0; d < ndims && d < 8; d++) {
            dims[d] = (int64_t)read_u64_buf(g->data, &g->pos);
        }
        int64_t offset = (int64_t)read_u64_buf(g->data, &g->pos);
        (void)ttype;

        memset(g->tnames[g->num_tfound], 0, 64);
        strncpy(g->tnames[g->num_tfound], tname, 63);
        g->toffsets[g->num_tfound] = offset;
        g->num_tfound++;
    }
}

static int gguf_find(GGUF *g, const char *name) {
    for (int i = 0; i < g->num_tfound; i++) {
        if (strcmp(g->tnames[i], name) == 0) return i;
    }
    return -1;
}

static void load_tensor_f32(GGUF *g, int idx, float *out) {
    const uint8_t *d = g->data + g->toffsets[idx];
    int64_t nelem = 1;
    /* We need to figure out nelem... */
    /* Re-parse tensor at index idx to get dims */
    /* For simplicity, we'll load all tensors we need by scanning */
    /* Actually, nelems are known statically from architecture */
    fprintf(stderr, "ERROR: should use static nelem\n");
    exit(1);
}

/* Better approach: store nelems directly */
typedef struct {
    char name[64];
    int64_t offset;
    int64_t nelem;
} TensorInfo;

typedef struct {
    uint8_t *data;
    size_t size;
    size_t pos;
    int version;
    int num_meta;
    char meta_keys[64][48];
    char meta_vals[64][128];
    TensorInfo tensors[256];
    int num_tensors;
} GGUFReader;

static void ggufr_load(GGUFReader *r, const char *path) {
    r->pos = 0;
    r->size = file_size(path);
    printf("Loading: %s (%ld bytes)\n", path, r->size);
    r->data = malloc(r->size);
    FILE *f = fopen(path, "rb");
    fread(r->data, 1, r->size, f);
    fclose(f);
    size_t p = 0;

    if (memcmp(r->data, "GGUF", 4) != 0) { fprintf(stderr, "Bad magic\n"); exit(1); }
    p = 4;

    r->version = (int)read_u32_buf(r->data, &p);
    int64_t tcount = (int64_t)read_u64_buf(r->data, &p);
    int64_t kvcount = (int64_t)read_u64_buf(r->data, &p);

    r->num_meta = 0;
    for (int64_t i = 0; i < kvcount && r->num_meta < 64; i++) {
        char key[48], val[128];
        read_string_buf(r->data, &p, key, 48);
        uint64_t vt = read_u64_buf(r->data, &p);
        uint64_t vl = read_u64_buf(r->data, &p);
        if (vl >= 128) vl = 127;
        memcpy(val, r->data + p, vl);
        val[vl] = '\0';
        p += vl;
        (void)vt;
        if (r->num_meta < 64) {
            strncpy(r->meta_keys[r->num_meta], key, 47);
            strncpy(r->meta_vals[r->num_meta], val, 127);
            r->num_meta++;
        }
    }

    r->num_tensors = 0;
    for (int64_t i = 0; i < tcount && r->num_tensors < 256; i++) {
        read_string_buf(r->data, &p, r->tensors[r->num_tensors].name, 64);
        uint32_t ttype = read_u32_buf(r->data, &p);
        uint32_t ndims = read_u32_buf(r->data, &p);
        int64_t nelem = 1;
        for (uint32_t d = 0; d < ndims && d < 8; d++) {
            int64_t dim = (int64_t)read_u64_buf(r->data, &p);
            nelem *= dim;
        }
        int64_t offset = (int64_t)read_u64_buf(r->data, &p);
        (void)ttype;
        r->tensors[r->num_tensors].offset = offset;
        r->tensors[r->num_tensors].nelem = nelem;
        r->num_tensors++;
    }
    printf("  %d tensors loaded\n", r->num_tensors);
}

static int ggufr_find(GGUFReader *r, const char *name) {
    for (int i = 0; i < r->num_tensors; i++) {
        if (strcmp(r->tensors[i].name, name) == 0) return i;
    }
    return -1;
}

static float *ggufr_load_tensor(GGUFReader *r, int idx) {
    float *out = malloc(r->tensors[idx].nelem * sizeof(float));
    const uint8_t *raw = r->data + r->tensors[idx].offset;
    int64_t n = r->tensors[idx].nelem;
    for (int64_t i = 0; i < n; i++) {
        uint16_t h;
        memcpy(&h, raw + i * 2, 2);
        out[i] = f16_to_f32(h);
    }
    return out;
}

static void ggufr_free(GGUFReader *r) {
    free(r->data);
}

/* ================================================================
 *  MODEL STRUCTURE
 * ================================================================ */
typedef struct {
    GGUFReader gguf;

    float *w_emb;
    float *w_emb_norm;
    float *w_output_norm;
    float *w_output;

    float *attn_norm[NUM_LAYERS];
    float *attn_q[NUM_LAYERS];
    float *attn_k[NUM_LAYERS];
    float *attn_v[NUM_LAYERS];
    float *attn_out[NUM_LAYERS];

    float *ffn_norm[NUM_LAYERS];
    float *ffn_gate[NUM_LAYERS];
    float *ffn_up[NUM_LAYERS];
    float *ffn_down[NUM_LAYERS];

    /* Temp buffers */
    float *buf;      /* large contiguous buffer for all temp storage */
    int buf_cap;
} Model;

static int cfg_int(GGUFReader *r, const char *key, int default_val) {
    for (int i = 0; i < r->num_meta; i++) {
        if (strcmp(r->meta_keys[i], key) == 0) {
            return atoi(r->meta_vals[i]);
        }
    }
    return default_val;
}

static void model_load(Model *m, const char *path) {
    ggufr_load(&m->gguf, path);

    int hidden = cfg_int(&m->gguf, "general.embedding_length", HIDDEN_DIM);
    int layers = cfg_int(&m->gguf, "general.block_count", NUM_LAYERS);
    int heads  = cfg_int(&m->gguf, "general.attention.head_count", NUM_HEADS);
    int head_d = cfg_int(&m->gguf, "general.attention.head_dim", HEAD_DIM);
    int vocab  = cfg_int(&m->gguf, "general.vocab_size", VOCAB_SIZE);
    int ffn    = cfg_int(&m->gguf, "general.feed_forward_length", FFN_DIM);

    printf("  Config: hidden=%d layers=%d heads=%d head_dim=%d ffn=%d vocab=%d\n",
           hidden, layers, heads, head_d, ffn, vocab);

    printf("  Loading weights...\n");
    clock_t t0 = clock();

    int idx;

    idx = ggufr_find(&m->gguf, "token_embd.weight");
    m->w_emb = ggufr_load_tensor(&m->gguf, idx);

    idx = ggufr_find(&m->gguf, "token_embd.norm");
    m->w_emb_norm = ggufr_load_tensor(&m->gguf, idx);

    idx = ggufr_find(&m->gguf, "output_norm");
    m->w_output_norm = ggufr_load_tensor(&m->gguf, idx);

    idx = ggufr_find(&m->gguf, "output");
    m->w_output = ggufr_load_tensor(&m->gguf, idx);

    for (int l = 0; l < layers; l++) {
        char name[128];

        snprintf(name, sizeof(name), "blk.%d.attn_norm", l);
        idx = ggufr_find(&m->gguf, name);
        m->attn_norm[l] = ggufr_load_tensor(&m->gguf, idx);

        snprintf(name, sizeof(name), "blk.%d.attn_q", l);
        idx = ggufr_find(&m->gguf, name);
        m->attn_q[l] = ggufr_load_tensor(&m->gguf, idx);

        snprintf(name, sizeof(name), "blk.%d.attn_k", l);
        idx = ggufr_find(&m->gguf, name);
        m->attn_k[l] = ggufr_load_tensor(&m->gguf, idx);

        snprintf(name, sizeof(name), "blk.%d.attn_v", l);
        idx = ggufr_find(&m->gguf, name);
        m->attn_v[l] = ggufr_load_tensor(&m->gguf, idx);

        snprintf(name, sizeof(name), "blk.%d.attn_output", l);
        idx = ggufr_find(&m->gguf, name);
        m->attn_out[l] = ggufr_load_tensor(&m->gguf, idx);

        snprintf(name, sizeof(name), "blk.%d.ffn_norm", l);
        idx = ggufr_find(&m->gguf, name);
        m->ffn_norm[l] = ggufr_load_tensor(&m->gguf, idx);

        snprintf(name, sizeof(name), "blk.%d.ffn_gate", l);
        idx = ggufr_find(&m->gguf, name);
        m->ffn_gate[l] = ggufr_load_tensor(&m->gguf, idx);

        snprintf(name, sizeof(name), "blk.%d.ffn_up", l);
        idx = ggufr_find(&m->gguf, name);
        m->ffn_up[l] = ggufr_load_tensor(&m->gguf, idx);

        snprintf(name, sizeof(name), "blk.%d.ffn_down", l);
        idx = ggufr_find(&m->gguf, name);
        m->ffn_down[l] = ggufr_load_tensor(&m->gguf, idx);
    }

    printf("  Done in %.2f seconds\n", (double)(clock() - t0) / CLOCKS_PER_SEC);
}

static void model_free(Model *m) {
    free(m->w_emb); free(m->w_emb_norm);
    free(m->w_output_norm); free(m->w_output);
    for (int l = 0; l < NUM_LAYERS; l++) {
        free(m->attn_norm[l]); free(m->attn_q[l]); free(m->attn_k[l]);
        free(m->attn_v[l]); free(m->attn_out[l]);
        free(m->ffn_norm[l]); free(m->ffn_gate[l]);
        free(m->ffn_up[l]); free(m->ffn_down[l]);
    }
    ggufr_free(&m->gguf);
}

/* ================================================================
 *  TRANSFORMER OPERATIONS
 * ================================================================ */

static void rmsnorm(float *out, const float *x, const float *w, int dim, float eps) {
    double sum = 0.0;
    for (int i = 0; i < dim; i++) sum += (double)x[i] * x[i];
    double scale = 1.0 / sqrt(sum / dim + eps);
    for (int i = 0; i < dim; i++) out[i] = x[i] * scale * w[i];
}

static void matmul(float *out, const float *x, const float *W, int in_dim, int out_dim) {
    for (int j = 0; j < out_dim; j++) {
        double sum = 0.0;
        for (int i = 0; i < in_dim; i++) sum += (double)x[i] * W[j * in_dim + i];
        out[j] = (float)sum;
    }
}

static void apply_rope(float *x, int pos, int num_heads, int head_dim) {
    /* Apply RoPE to a single position's Q or K vector
     * x: [num_heads * head_dim] in-place */
    for (int h = 0; h < num_heads; h++) {
        for (int i = 0; i < head_dim; i += 2) {
            float freq = 1.0f / powf(10000.0f, (float)i / head_dim);
            float angle = pos * freq;
            float c = cosf(angle);
            float s = sinf(angle);
            float x0 = x[h * head_dim + i];
            float x1 = x[h * head_dim + i + 1];
            x[h * head_dim + i]     = x0 * c - x1 * s;
            x[h * head_dim + i + 1] = x0 * s + x1 * c;
        }
    }
}

static void masked_softmax(float *out, const float *logits, int n, int causal_pos) {
    float maxv = logits[0];
    for (int i = 1; i < n; i++) if (logits[i] > maxv) maxv = logits[i];
    if (!isfinite(maxv)) maxv = 0.0f;

    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        float v = logits[i] - maxv;
        if (v > 50.0f) v = 50.0f;
        if (v < -50.0f) v = -50.0f;
        out[i] = expf(v);
        if (i > causal_pos) out[i] = 0.0f;
        sum += out[i];
    }
    if (sum > 0.0f) {
        for (int i = 0; i < n; i++) out[i] /= sum;
    } else {
        for (int i = 0; i < n; i++) out[i] = 1.0f / n;
    }
}

/* One-step forward: process full cached sequence, return logits for last token */
static void model_forward(Model *m, int *tokens, int seq_len, float *logits) {
    float *h = (float *)malloc(seq_len * HIDDEN_DIM * sizeof(float));

    /* Embed */
    for (int t = 0; t < seq_len; t++) {
        int tok = tokens[t];
        if (tok < 0 || tok >= VOCAB_SIZE) tok = 0;
        memcpy(h + t * HIDDEN_DIM, m->w_emb + tok * HIDDEN_DIM, HIDDEN_DIM * sizeof(float));
    }

    /* Emb norm */
    for (int t = 0; t < seq_len; t++) {
        rmsnorm(h + t * HIDDEN_DIM, h + t * HIDDEN_DIM, m->w_emb_norm, HIDDEN_DIM, 1e-6f);
    }

    float *residual = (float *)malloc(seq_len * HIDDEN_DIM * sizeof(float));

    for (int l = 0; l < NUM_LAYERS; l++) {
        memcpy(residual, h, seq_len * HIDDEN_DIM * sizeof(float));

        /* Attn norm */
        for (int t = 0; t < seq_len; t++) {
            rmsnorm(h + t * HIDDEN_DIM, h + t * HIDDEN_DIM, m->attn_norm[l], HIDDEN_DIM, 1e-6f);
        }

        /* Q/K/V projections (full sequence) */
        float *Q = (float *)malloc(seq_len * HIDDEN_DIM * sizeof(float));
        float *K = (float *)malloc(seq_len * HIDDEN_DIM * sizeof(float));
        float *V = (float *)malloc(seq_len * HIDDEN_DIM * sizeof(float));

        for (int t = 0; t < seq_len; t++) {
            matmul(Q + t * HIDDEN_DIM, h + t * HIDDEN_DIM, m->attn_q[l], HIDDEN_DIM, HIDDEN_DIM);
            matmul(K + t * HIDDEN_DIM, h + t * HIDDEN_DIM, m->attn_k[l], HIDDEN_DIM, HIDDEN_DIM);
            matmul(V + t * HIDDEN_DIM, h + t * HIDDEN_DIM, m->attn_v[l], HIDDEN_DIM, HIDDEN_DIM);
        }

        /* Apply RoPE to all positions */
        for (int t = 0; t < seq_len; t++) {
            apply_rope(Q + t * HIDDEN_DIM, t, NUM_HEADS, HEAD_DIM);
            apply_rope(K + t * HIDDEN_DIM, t, NUM_HEADS, HEAD_DIM);
        }

        /* Reorganize Q/K/V to [seq, heads, head_dim] */
        float *Q_reshaped = (float *)malloc(seq_len * NUM_HEADS * HEAD_DIM * sizeof(float));
        float *K_reshaped = (float *)malloc(seq_len * NUM_HEADS * HEAD_DIM * sizeof(float));
        float *V_reshaped = (float *)malloc(seq_len * NUM_HEADS * HEAD_DIM * sizeof(float));

        for (int t = 0; t < seq_len; t++) {
            for (int hd = 0; hd < HIDDEN_DIM; hd++) {
                int head = hd / HEAD_DIM;
                int dim  = hd % HEAD_DIM;
                int flat = t * NUM_HEADS * HEAD_DIM + head * HEAD_DIM + dim;
                Q_reshaped[flat] = Q[t * HIDDEN_DIM + hd];
                K_reshaped[flat] = K[t * HIDDEN_DIM + hd];
                V_reshaped[flat] = V[t * HIDDEN_DIM + hd];
            }
        }

        /* Attention for LAST token only (most efficient for generation) */
        int cur = seq_len - 1;
        float *cur_Q = Q_reshaped + cur * NUM_HEADS * HEAD_DIM;

        float attn_w[MAX_SEQ];
        float attn_logits[MAX_SEQ];
        for (int k = 0; k < seq_len; k++) {
            float *cur_K = K_reshaped + k * NUM_HEADS * HEAD_DIM;
            double dot = 0.0;
            for (int d = 0; d < HEAD_DIM; d++) dot += (double)cur_Q[d] * cur_K[d];
            attn_logits[k] = dot / sqrtf((float)HEAD_DIM);
        }
        masked_softmax(attn_w, attn_logits, seq_len, cur);

        float attn_result[HIDDEN_DIM] = {0.0f};
        for (int k = 0; k < seq_len; k++) {
            float wt = attn_w[k];
            float *cur_V = V_reshaped + k * NUM_HEADS * HEAD_DIM;
            for (int d = 0; d < HIDDEN_DIM; d++) {
                attn_result[d] += wt * cur_V[d];
            }
        }

        /* Put back to HIDDEN_DIM layout + output projection */
        float attn_reshaped[HIDDEN_DIM];
        for (int hd = 0; hd < HIDDEN_DIM; hd++) {
            int head = hd / HEAD_DIM;
            int dim  = hd % HEAD_DIM;
            attn_reshaped[hd] = attn_result[head * HEAD_DIM + dim];
        }
        matmul(h + cur * HIDDEN_DIM, attn_reshaped, m->attn_out[l], HIDDEN_DIM, HIDDEN_DIM);

        /* Residual */
        for (int i = cur * HIDDEN_DIM; i < (cur + 1) * HIDDEN_DIM; i++) {
            h[i] = residual[i] + h[i];
        }

        /* FFN (last token only for efficiency) */
        float *ffn_in = h + cur * HIDDEN_DIM;
        float gate_v[FFN_DIM];
        float up_v[FFN_DIM];
        matmul(gate_v, ffn_in, m->ffn_gate[l], HIDDEN_DIM, FFN_DIM);
        matmul(up_v,   ffn_in, m->ffn_up[l],   HIDDEN_DIM, FFN_DIM);

        float ffn_result[HIDDEN_DIM];
        float ffn_buf[FFN_DIM];
        for (int j = 0; j < FFN_DIM; j++) {
            float silu = gate_v[j] / (1.0f + expf(-gate_v[j]));
            ffn_buf[j] = silu * up_v[j];
        }
        matmul(ffn_result, ffn_buf, m->ffn_down[l], FFN_DIM, HIDDEN_DIM);

        /* Residual add */
        for (int i = 0; i < HIDDEN_DIM; i++) {
            h[cur * HIDDEN_DIM + i] += ffn_result[i];
        }

        free(Q); free(K); free(V);
        free(Q_reshaped); free(K_reshaped); free(V_reshaped);
    }

    free(residual);

    /* Final norm + output */
    rmsnorm(h + (seq_len - 1) * HIDDEN_DIM,
            h + (seq_len - 1) * HIDDEN_DIM,
            m->w_output_norm, HIDDEN_DIM, 1e-6f);
    matmul(logits, h + (seq_len - 1) * HIDDEN_DIM, m->w_output, HIDDEN_DIM, VOCAB_SIZE);

    /* Clamp logits to finite range */
    for (int i = 0; i < VOCAB_SIZE; i++) {
        if (!isfinite(logits[i])) logits[i] = -1e6f;
        else if (logits[i] > 40.0f) logits[i] = 40.0f;
        else if (logits[i] < -40.0f) logits[i] = -40.0f;
    }

    free(h);
}

/* ================================================================
 *  SAMPLING
 * ================================================================ */
static int sample(float *logits, int vocab, int top_k, float temp) {
    if (temp < 1e-10f) temp = 1e-10f;

    float maxv = logits[0];
    for (int i = 1; i < vocab; i++) if (logits[i] > maxv) maxv = logits[i];

    float sum = 0.0f;
    float probs[1024];
    if (top_k > vocab) top_k = vocab;

    float top_vals[256];
    int top_idx[256];
    for (int i = 0; i < top_k; i++) top_vals[i] = -1e30f;

    for (int i = 0; i < vocab; i++) {
        float v = (logits[i] - maxv) / temp;
        for (int j = 0; j < top_k; j++) {
            if (v > top_vals[j]) {
                for (int k2 = top_k - 1; k2 > j; k2--) {
                    top_vals[k2] = top_vals[k2-1];
                    top_idx[k2] = top_idx[k2-1];
                }
                top_vals[j] = v;
                top_idx[j] = i;
                break;
            }
        }
    }

    float mx = top_vals[0];
    for (int i = 0; i < top_k; i++) {
        probs[i] = expf(top_vals[i] - mx);
        sum += probs[i];
    }
    for (int i = 0; i < top_k; i++) probs[i] /= sum;

    float r = (float)rand() / (float)RAND_MAX;
    float cum = 0.0f;
    for (int i = 0; i < top_k; i++) {
        cum += probs[i];
        if (r <= cum) return top_idx[i];
    }
    return top_idx[top_k - 1];
}

/* ================================================================
 *  MAIN
 * ================================================================ */
int main(int argc, char **argv) {
    const char *model_path = "C:/tmp/llm_pequeno.gguf";
    const char *prompt = "DREAM";
    int max_new = 30;
    float temp = 0.8f;
    int top_k = 25;

    if (argc >= 2) model_path = argv[1];
    if (argc >= 3) prompt = argv[2];
    if (argc >= 4) max_new = atoi(argv[3]);
    if (argc >= 5) temp = (float)atof(argv[4]);
    if (argc >= 6) top_k = atoi(argv[5]);

    printf("========================================\n");
    printf("  LLM ONIRICO - Motor C Puro\n");
    printf("  DeepSeek-V2 Lite | %d layers | %d heads\n", NUM_LAYERS, NUM_HEADS);
    printf("========================================\n\n");

    Model model;
    memset(&model, 0, sizeof(model));
    model_load(&model, model_path);

    printf("\n--- Generation ---\n");
    printf("Prompt: '%s'\n", prompt);
    printf("Params: max_new=%d temperature=%.1f top_k=%d\n", max_new, temp, top_k);

    /* Tokenize prompt as bytes */
    int prompt_tokens[MAX_TOKENS];
    int n_prompt = 0;
    prompt_tokens[n_prompt++] = 1; /* <s> */
    for (int i = 0; prompt[i] && n_prompt < MAX_TOKENS - 1; i++) {
        prompt_tokens[n_prompt++] = 4 + (unsigned char)prompt[i];
    }

    printf("Prompt tokens (%d):", n_prompt);
    for (int i = 0; i < n_prompt; i++) printf(" %d", prompt_tokens[i]);
    printf("\n\n");

    /* Generate */
    int full_tokens[MAX_TOKENS * 2];
    memcpy(full_tokens, prompt_tokens, n_prompt * sizeof(int));
    int total = n_prompt;

    float logits[VOCAB_SIZE];
    clock_t t0 = clock();

    for (int step = 0; step < max_new; step++) {
        model_forward(&model, full_tokens, total, logits);
        int next_tok = sample(logits, VOCAB_SIZE, top_k, temp);
        full_tokens[total++] = next_tok;

        printf("\r  Step %d/%d  token=%d", step + 1, max_new, next_tok);
        fflush(stdout);

        if (next_tok == 2) break; /* </s> */
        if (total >= MAX_TOKENS * 2 - 1) break;
    }
    printf("\n");

    /* Decode output (bytes only) */
    printf("\n--- Decoded output ---\n");
    for (int i = n_prompt; i < total; i++) {
        int tid = full_tokens[i];
        if (tid >= 4 && tid <= 259) {
            putchar(tid - 4);
        }
    }
    printf("\n--- End ---\n");

    double elapsed = (double)(clock() - t0) / CLOCKS_PER_SEC;
    printf("Generated %d new tokens in %.2fs (%.1f tok/s)\n",
           total - n_prompt, elapsed, (total - n_prompt) / elapsed);

    model_free(&model);
    return 0;
}
