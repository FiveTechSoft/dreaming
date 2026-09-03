#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <string.h>

static float f16_to_f32(uint16_t h) {
    uint32_t sign = (h >> 15) & 1;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t man = h & 0x3FF;
    union { uint32_t u; float f; } v;
    if (exp == 0) {
        if (man == 0) { v.u = sign << 31; return v.f; }
        v.u = (sign << 31) | ((112 + 0) << 23) | (man << 13);
        return v.f;
    } else if (exp == 31) {
        v.u = (sign << 31) | 0x7F800000 | (man << 13);
        return v.f;
    }
    v.u = (sign << 31) | ((exp + 112) << 23) | (man << 13);
    return v.f;
}

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

#define Q4_K_BLOCK_ELEMS 256
#define Q4_K_BLOCK_BYTES 144

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

int main() {
    FILE *f = fopen("test_q4k_block.bin", "rb");
    if (!f) { printf("Cannot open test_q4k_block.bin\n"); return 1; }
    uint8_t block[144];
    fread(block, 1, 144, f);
    fclose(f);

    float out[256];
    dequant_q4_k(out, block, 256);

    FILE *p = fopen("test_q4k_python.bin", "rb");
    if (!p) { printf("Cannot open test_q4k_python.bin\n"); return 1; }
    float py_vals[10];
    fread(py_vals, sizeof(float), 10, p);
    fclose(p);

    printf("C   vs Python comparison (first 10 values):\n");
    int match = 1;
    for (int i = 0; i < 10; i++) {
        float diff = fabsf(out[i] - py_vals[i]);
        printf("  [%d] C=%12.8f  Py=%12.8f  diff=%e\n", i, out[i], py_vals[i], diff);
        if (diff > 1e-5) match = 0;
    }
    printf("Match: %s\n", match ? "YES" : "NO");
    return 0;
}
