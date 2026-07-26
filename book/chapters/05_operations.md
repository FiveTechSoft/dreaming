# Capítulo 5: Implementación de Operaciones

## 5.1 Operaciones Fundamentales

El motor de inferencia necesita implementar operaciones matemáticas fundamentales. Este capítulo detalla cada operación y su implementación.

## 5.2 Multiplicación Matriz-Matriz

La operación más computacionalmente intensiva:

```c
// C = A * B
// A: [m, k], B: [k, n], C: [m, n]
void matmul(float *C, const float *A, const float *B, 
            int m, int n, int k) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0;
            for (int p = 0; p < k; p++) {
                sum += A[i * k + p] * B[p * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}
```

### Optimizaciones

```c
// Blocking para mejor localidad de caché
void matmul_blocked(float *C, const float *A, const float *B,
                    int m, int n, int k, int block_size) {
    for (int ii = 0; ii < m; ii += block_size) {
        for (int jj = 0; jj < n; jj += block_size) {
            for (int kk = 0; kk < k; kk += block_size) {
                int i_end = min(ii + block_size, m);
                int j_end = min(jj + block_size, n);
                int k_end = min(kk + block_size, k);
                
                for (int i = ii; i < i_end; i++) {
                    for (int j = jj; j < j_end; j++) {
                        float sum = C[i * n + j];
                        for (int p = kk; p < k_end; p++) {
                            sum += A[i * k + p] * B[p * n + j];
                        }
                        C[i * n + j] = sum;
                    }
                }
            }
        }
    }
}
```

## 5.3 Softmax

```c
// Softmax estable numéricamente
void softmax(float *x, int n) {
    // 1. Encontrar máximo
    float max_val = x[0];
    for (int i = 1; i < n; i++) {
        if (x[i] > max_val) max_val = x[i];
    }
    
    // 2. Exponenciales
    float sum = 0;
    for (int i = 0; i < n; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    
    // 3. Normalizar
    for (int i = 0; i < n; i++) {
        x[i] /= sum;
    }
}
```

### ¿Por qué restar el máximo?

Sin esta optimización, `expf(1000)` causaría overflow. Al restar el máximo, el valor más alto es 0, y `expf(0) = 1`.

## 5.4 Layer Normalization (RMSNorm)

```c
// RMSNorm: x / sqrt(mean(x²)) * weight
void rmsnorm(float *out, const float *x, const float *weight,
             int n, float eps) {
    // 1. Calcular RMS
    float sum_sq = 0;
    for (int i = 0; i < n; i++) {
        sum_sq += x[i] * x[i];
    }
    float rms = sqrtf(sum_sq / n + eps);
    
    // 2. Normalizar y aplicar peso
    for (int i = 0; i < n; i++) {
        out[i] = weight[i] * (x[i] / rms);
    }
}
```

### RMSNorm vs LayerNorm

| Operación | RMSNorm | LayerNorm |
|-----------|---------|-----------|
| Media | No calcula | Sí calcula |
| Varianza | RMS² | Varianza completa |
| Velocidad | ~15% más rápido | Base |
| Precisión | Comparable | Base |

RMSNorm es preferido porque es más rápido sin pérdida de calidad.

## 5.5 RoPE (Rotary Position Embeddings)

```c
// Aplicar RoPE a Q y K
void apply_rope(float *q, float *k, int seq_len, 
                int n_heads, int d_head, float base) {
    for (int t = 0; t < seq_len; t++) {
        for (int h = 0; h < n_heads; h++) {
            for (int i = 0; i < d_head; i += 2) {
                // Frecuencia
                float theta = 1.0f / powf(base, (float)i / d_head);
                float angle = t * theta;
                
                float cos_a = cosf(angle);
                float sin_a = sinf(angle);
                
                int idx = t * n_heads * d_head + h * d_head + i;
                
                // Rotar Q
                float q0 = q[idx];
                float q1 = q[idx + 1];
                q[idx]     = q0 * cos_a - q1 * sin_a;
                q[idx + 1] = q0 * sin_a + q1 * cos_a;
                
                // Rotar K
                float k0 = k[idx];
                float k1 = k[idx + 1];
                k[idx]     = k0 * cos_a - k1 * sin_a;
                k[idx + 1] = k0 * sin_a + k1 * cos_a;
            }
        }
    }
}
```

### Intuición geométrica de RoPE

RoPE rota cada par de dimensiones en el plano 2D:
- Dimensión 2i: componente coseno
- Dimensión 2i+1: componente seno

Esto codifica la posición como **ángulo** en el espacio de embeddings.

## 5.6 SwiGLU

```c
// SwiGLU: gate * silu(up)
// silu(x) = x * sigmoid(x)
void swiglu(float *out, const float *x, 
            const float *gate, const float *up,
            const float *down,
            int seq_len, int hidden, int intermediate) {
    
    float temp[seq_len * intermediate];
    
    // Gate * silu(up)
    for (int s = 0; s < seq_len; s++) {
        for (int i = 0; i < intermediate; i++) {
            int idx = s * intermediate + i;
            float g = gate[idx];
            float u = up[idx];
            float silu = u / (1.0f + expf(-u));  // silu(x) = x * sigmoid(x)
            temp[idx] = g * silu;
        }
    }
    
    // Proyectar de vuelta con down
    for (int s = 0; s < seq_len; s++) {
        for (int h = 0; h < hidden; h++) {
            float sum = 0;
            for (int i = 0; i < intermediate; i++) {
                sum += temp[s * intermediate + i] * 
                       down[i * hidden + h];
            }
            out[s * hidden + h] = sum;
        }
    }
}
```

### Comparación de activaciones

| Activación | Fórmula | Complejidad |
|------------|---------|-------------|
| ReLU | max(0, x) | O(n) |
| GELU | x · Φ(x) | O(n) |
| Swish | x · σ(x) | O(n) |
| SwiGLU | gate · silu(up) | O(n) |

SwiGLU es computacionalmente más costoso pero produce mejor calidad.

## 5.7 GELU (para modelos que lo usan)

```c
// GELU approximation
float gelu(float x) {
    const float sqrt_2_over_pi = 0.7978845608;
    const float coeff = 0.044715;
    
    float x3 = x * x * x;
    return 0.5f * x * (1.0f + tanhf(sqrt_2_over_pi * (x + coeff * x3)));
}
```

## 5.8 Silu

```c
// Silu (Swish): x * sigmoid(x)
float silu(float x) {
    return x / (1.0f + expf(-x));
}
```

## 5.9 Concatenación de Tensores

```c
// Concatenar A [m, k] y B [m, l] → C [m, k+l]
void concat(float *C, const float *A, const float *B,
            int m, int k, int l) {
    for (int i = 0; i < m; i++) {
        memcpy(C + i * (k + l), A + i * k, k * sizeof(float));
        memcpy(C + i * (k + l) + k, B + i * l, l * sizeof(float));
    }
}
```

## 5.10 Funciones de Activación — Resumen

```
┌─────────────┬─────────────────────────────────────┐
│  Función    │  Implementación                     │
├─────────────┼─────────────────────────────────────┤
│  ReLU       │  x > 0 ? x : 0                     │
│  GELU       │  0.5 * x * (1 + tanh(√(2/π)x))    │
│  Swish      │  x * σ(x)                          │
│  Silu       │  x * σ(x) = Swish                  │
│  Softmax    │  exp(x) / sum(exp(x))              │
│  Sigmoid    │  1 / (1 + exp(-x))                 │
│  Tanh       │  (exp(x) - exp(-x)) /              │
│             │  (exp(x) + exp(-x))                │
└─────────────┴─────────────────────────────────────┘
```

## 5.11 Precisión Numérica

### FP32 vs FP16 vs INT8

| Formato | Bytes | Precisión | Velocidad |
|---------|-------|-----------|-----------|
| FP32 | 4 | Alta | Base |
| FP16 | 2 | Media | ~2x |
| INT8 | 1 | Baja | ~4x |
| Q4_0 | 0.5 | Baja | ~8x |

### Errores de redondeo

```c
// Suma en Kahan para precisión
float sum_kahan(float *arr, int n) {
    float sum = 0;
    float c = 0;  // Compensation
    
    for (int i = 0; i < i < n; i++) {
        float y = arr[i] - c;
        float t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }
    
    return sum;
}
```

## 5.12 Gestión de Memoria

```c
// Alineación de memoria para SIMD
void* aligned_alloc(size_t alignment, size_t size) {
    void *ptr;
    posix_memalign(&ptr, alignment, size);
    return ptr;
}

// Uso
float *weights = aligned_alloc(32, n_elements * sizeof(float));
```

---

*Siguiente capítulo: [La Analogía DMT](06_dmt_analogy.md)*
