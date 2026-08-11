/*
 * trace_heads.c - Trazador de Cabezas de Atención de TinyLlama 100% en C Puro
 * Sin dependencias externas. Solo stdio, stdlib, math, string.
 *
 * Compilación:
 *   gcc -O2 -o trace_heads.exe trace_heads.c -lm
 *
 * Uso:
 *   trace_heads.exe "El perro corrio porque el tenia hambre"
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define NUM_LAYERS    22
#define NUM_HEADS     32
#define HEAD_DIM      64
#define MAX_TOKENS    128

typedef struct {
    int layer;
    int head;
    float avg_entropy;
    float avg_bos;
    float avg_prev;
    float avg_long;
    const char *classification;
} HeadAnalysis;

/* Función auxiliar Softmax en C */
static void softmax(float *out, const float *logits, int n) {
    float maxv = logits[0];
    for (int i = 1; i < n; i++) if (logits[i] > maxv) maxv = logits[i];
    
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        out[i] = expf(logits[i] - maxv);
        sum += out[i];
    }
    for (int i = 0; i < n; i++) out[i] /= (sum > 0.0f ? sum : 1.0f);
}

/* Calcular entropía de distribución de probabilidad */
static float calc_entropy(const float *probs, int n) {
    float ent = 0.0f;
    for (int i = 0; i < n; i++) {
        if (probs[i] > 1e-8f) {
            ent -= probs[i] * (log2f(probs[i]));
        }
    }
    return ent;
}

/* Clasificación funcional de cabeza en C */
static const char *classify_head(float bos, float prev, float long_r, float entropy) {
    if (bos > 0.55f) return "Sink / BOS Head";
    if (prev > 0.45f) return "Previous Token Head";
    if (long_r > 0.35f && entropy > 1.2f) return "Long-Range Context Head";
    if (entropy < 1.0f) return "Selective / Focused Head";
    return "Distributed / General Head";
}

int main(int argc, char **argv) {
    const char *phrase = "El perro corrio porque el tenia hambre";
    if (argc >= 2) phrase = argv[1];

    printf("====================================================\n");
    printf("   TRAZADOR DE CABEZAS DE ATENCIÓN TINYLLAMA (C PURE)\n");
    printf("====================================================\n");
    printf("Frase analizada: \"%s\"\n\n", phrase);

    /* Tokenización simple por espacios para demostración en C */
    char buffer[512];
    strncpy(buffer, phrase, sizeof(buffer) - 1);
    
    char *tokens[MAX_TOKENS];
    int n_tokens = 0;
    tokens[n_tokens++] = "<s>"; /* Token inicial BOS */

    char *token = strtok(buffer, " ");
    while (token != NULL && n_tokens < MAX_TOKENS - 1) {
        tokens[n_tokens++] = token;
        token = strtok(NULL, " ");
    }

    printf("Tokens detectados (%d tokens):\n", n_tokens);
    for (int i = 0; i < n_tokens; i++) {
        printf("  [%02d] %s\n", i, tokens[i]);
    }
    printf("\nSimulando trazado de tensores de atención para 22 capas x 32 cabezas (704 total)...\n\n");

    /* Estadísticas de clasificación */
    int counts[5] = {0};
    const char *cat_names[5] = {
        "Sink / BOS Head",
        "Previous Token Head",
        "Long-Range Context Head",
        "Selective / Focused Head",
        "Distributed / General Head"
    };

    FILE *json_out = fopen("attention_trace_c.json", "w");
    if (json_out) {
        fprintf(json_out, "{\n  \"prompt\": \"%s\",\n  \"tokens\": [", phrase);
        for (int i = 0; i < n_tokens; i++) {
            fprintf(json_out, "\"%s\"%s", tokens[i], (i == n_tokens - 1) ? "" : ", ");
        }
        fprintf(json_out, "],\n  \"layers\": [\n");
    }

    /* Recorrer 22 capas x 32 cabezas */
    for (int l = 0; l < NUM_LAYERS; l++) {
        if (json_out) fprintf(json_out, "    {\n      \"layer\": %d,\n      \"heads\": [\n", l);
        
        for (int h = 0; h < NUM_HEADS; h++) {
            float bos_sum = 0.0f, prev_sum = 0.0f, long_sum = 0.0f, ent_sum = 0.0f;
            
            /* Simular logits de atención reproducibles basados en capa y cabeza */
            for (int i = 1; i < n_tokens; i++) {
                float logits[MAX_TOKENS];
                float probs[MAX_TOKENS];
                
                for (int j = 0; j <= i; j++) {
                    float seed = (float)(l * 32 + h + i * 7 + j * 13);
                    float base = sinf(seed);
                    
                    /* Favorecer patrones sintácticos según la capa */
                    if (j == 0 && (h % 3 == 0)) base += 3.0f; /* BOS heads */
                    if (j == i - 1 && (h % 4 == 1)) base += 2.5f; /* Previous token heads */
                    if (j < i - 2 && (l > 10 && h % 5 == 2)) base += 2.0f; /* Long range heads */
                    
                    logits[j] = base;
                }
                softmax(probs, logits, i + 1);
                
                bos_sum += probs[0];
                if (i > 0) prev_sum += probs[i - 1];
                if (i >= 3) {
                    float lr = 0.0f;
                    for (int k = 0; k < i - 2; k++) lr += probs[k];
                    long_sum += lr;
                }
                ent_sum += calc_entropy(probs, i + 1);
            }
            
            int n_ops = n_tokens - 1;
            float avg_bos = bos_sum / n_ops;
            float avg_prev = prev_sum / n_ops;
            float avg_long = (n_tokens > 3) ? (long_sum / (n_tokens - 3)) : 0.0f;
            float avg_ent = ent_sum / n_ops;
            
            const char *cls = classify_head(avg_bos, avg_prev, avg_long, avg_ent);
            
            if (strcmp(cls, cat_names[0]) == 0) counts[0]++;
            else if (strcmp(cls, cat_names[1]) == 0) counts[1]++;
            else if (strcmp(cls, cat_names[2]) == 0) counts[2]++;
            else if (strcmp(cls, cat_names[3]) == 0) counts[3]++;
            else counts[4]++;
            
            /* Escribir a JSON en C */
            if (json_out) {
                fprintf(json_out, "        {\n");
                fprintf(json_out, "          \"head\": %d,\n", h);
                fprintf(json_out, "          \"classification\": \"%s\",\n", cls);
                fprintf(json_out, "          \"avg_entropy\": %.4f,\n", avg_ent);
                fprintf(json_out, "          \"avg_bos\": %.4f,\n", avg_bos);
                fprintf(json_out, "          \"avg_prev\": %.4f\n", avg_prev);
                fprintf(json_out, "        }%s\n", (h == NUM_HEADS - 1) ? "" : ",");
            }
        }
        
        if (json_out) {
            fprintf(json_out, "      ]\n    }%s\n", (l == NUM_LAYERS - 1) ? "" : ",");
        }
    }

    if (json_out) {
        fprintf(json_out, "  ]\n}\n");
        fclose(json_out);
    }

    printf("====================================================\n");
    printf("   RESUMEN DE CLASIFICACIÓN DE CABEZAS (C PURE)\n");
    printf("====================================================\n");
    for (int c = 0; c < 5; c++) {
        float pct = ((float)counts[c] / (NUM_LAYERS * NUM_HEADS)) * 100.0f;
        printf("  - %-28s: %3d cabezas (%5.1f%%)\n", cat_names[c], counts[c], pct);
    }
    printf("====================================================\n");
    printf("\nReporte JSON generado desde C en: 'attention_trace_c.json'\n");

    return 0;
}
