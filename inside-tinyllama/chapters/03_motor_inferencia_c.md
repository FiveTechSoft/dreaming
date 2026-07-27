# Capítulo 3: Nuestro Motor de Inferencia en C para TinyLlama

## Por qué escribir un motor propio

Para estudiar TinyLlama no basta con *usarlo*.
Hay que *verlo funcionar*.

La mayoría de frameworks de inferencia ocultan
el recorrido del token detrás de capas de abstracción:

- PyTorch y CUDA (gigabytes de dependencias)
- Runtimes en Python
- Kernels opacos en GPU

Nosotros queríamos lo contrario:

1. **C legible** — un solo archivo, operaciones explícitas
2. **Sin GPU obligatoria** — todo en CPU
3. **Sin magia** — cada paso del transformer es una función
4. **Portable** — un binario pequeño, reproducible

El resultado es `llm_inference.c`: motor de
inferencia que implementa el bucle del transformer
desde el archivo GGUF hasta el siguiente token —
y, más tarde, **perturbaciones de pesos en runtime**
sin necesidad de un GGUF pre-horneado.

No es el motor más rápido del mundo.
Es el motor que *entendemos línea a línea*
y con el que podemos *tocar los pesos a conciencia*.

## Qué tiene que hacer un motor de inferencia

Recuerda el flujo del capítulo anterior:

```
Texto → tokens → embeddings
     → 22 capas (atención + FFN)
     → logits → siguiente token
     → (repetir)
```

En la práctica el motor se parte en cinco piezas:

```
┌─────────────────────────────────────────┐
│            llm_inference.c              │
├─────────────────────────────────────────┤
│  1. Lector GGUF                         │
│     - magic, metadata, índices          │
│     - tensores F16 (y fallback tokenizer│
│       desde Q4_0 si hace falta)         │
├─────────────────────────────────────────┤
│  2. Tokenizer BPE                       │
│     - tokens + merges del propio GGUF   │
│     - texto ⇄ ids                       │
├─────────────────────────────────────────┤
│  3. Motor Transformer + KV-cache        │
│     - embedding, RMSNorm, RoPE          │
│     - atención GQA, SwiGLU              │
│     - un token nuevo por paso           │
├─────────────────────────────────────────┤
│  4. Sampling                            │
│     - temperatura, top-k                │
├─────────────────────────────────────────┤
│  5. Perturbación / steering (opcional)  │
│     - --perturb mystical, noise, …      │
│     - --steer <palabra> (activaciones)  │
└─────────────────────────────────────────┘
```

| Bloque | Pregunta |
|--------|----------|
| GGUF | ¿Dónde están los pesos en el disco? |
| Tokenizer | ¿Cómo se convierte el lenguaje en números? |
| Transformer | ¿Cómo se transforma la representación? |
| Sampling | ¿Cómo se elige la siguiente palabra? |
| Perturbación | ¿Cómo cambiamos la *perspectiva* del modelo? |

## TinyLlama en números (los que usa el motor)

| Parámetro | Valor en TinyLlama-1.1B |
|-----------|-------------------------|
| Capas (`block_count`) | 22 |
| Dimensión oculta | 2048 |
| Cabezas Q / KV (GQA) | 32 / 4 |
| Dimensión por cabeza | 64 |
| FFN intermedio | 5632 |
| Vocabulario | 32.000 |
| Contexto del modelo | 2048 |
| RoPE `freq_base` | 10.000 |
| Formato de pesos del motor | GGUF **F16** |

Nueve tensores por capa (cap. 2) más los globales
`token_embd`, `output_norm`, `output`. El motor no
inventa la arquitectura: la *materializa* en memoria.

> **Nota sobre Q4_0:** los experimentos masivos del
> proyecto también usan GGUF cuantizados (~638 MB)
> con `llama-cli`. Nuestro C puro lee **F16**
> (~2.1 GB) para no reimplementar todos los kernels
> de cuantización. El *forward* es el mismo transformer.

## Compilar y ejecutar

```bash
# Recomendado (Windows/MinGW o Linux)
gcc -O3 -march=native -ffast-math -fopenmp \
    -o llm_inference llm_inference.c -lm

# Variables útiles
export OMP_NUM_THREADS=8   # Linux/macOS
# PowerShell: $env:OMP_NUM_THREADS = "8"
```

### Uso básico (baseline)

```bash
./llm_inference tinyllama-1.1b.F16.gguf \
    "The secret to happiness is" \
    40 0.7 40 \
    --seed 42
```

Argumentos posicionales: `modelo`, `prompt`, `n_tokens`,
`temperatura`, `top_k`.

### Perturbación mística en runtime

```bash
./llm_inference tinyllama-1.1b.F16.gguf \
    "The secret to happiness is" \
    40 0.7 40 \
    --seed 42 \
    --perturb mystical \
    --intensity 0.50
```

No hace falta un archivo `DMT_*.gguf` previo:
la técnica se aplica **en memoria** sobre los pesos
recién cargados.

### Otras banderas

| Flag | Efecto |
|------|--------|
| `--perturb` / `-P` | `none`, `mystical`, `amplify`, `noise`, `blockdiag`, `manifold` |
| `--intensity` / `-I` | fuerza de la perturbación (p. ej. 0.10–0.50) |
| `--seed` | PRNG de la perturbación (+ sampling si se fija) |
| `--steer` | palabra cuya dirección de embedding tira del residual |
| `--steer-strength` | intensidad del steering (p. ej. 0.15) |

Dependencias de *build*: `libm` y, para el matmul
paralelo, **OpenMP** (`-fopenmp`). Sin OpenMP el
motor compila igual, solo más lento.

## 1. Leer un GGUF en C

GGUF tiene tres zonas:

```
[ magic "GGUF" | versión | nº tensores | nº KV ]
[ metadata clave-valor ]
[ índice de tensores → datos alineados ]
```

El motor carga el archivo completo (en Windows con
I/O de 64 bits: el F16 supera 2 GB y `ftell` de 32
bits no basta). Luego indexa tensores por nombre:

```c
snprintf(name, sizeof(name), "blk.%d.attn_q.weight", l);
m->wq[l] = must_tensor_f16(&m->gguf, name);
/* wk, wv, wo, w1/w3/w2 (gate/up/down), normas… */
```

### F16 en disco, float en el matmul

Los pesos de capa se dejan en **F16** en el buffer
del archivo. En el producto matriz-vector se
convierten con una **tabla de 65.536 entradas**
(un valor por cada patrón de half):

```c
static float g_f16_table[65536];
/* init: g_f16_table[i] = decode_ieee_half(i); */
static inline float f16_to_f32(uint16_t h) {
    return g_f16_table[h];
}
```

Así evitamos decodificar bits en cada peso del bucle
caliente. Solo cuando pedimos `--perturb` copiamos
las matrices de capa a **F32 mutable** (~3.6 GB):
el delta de `amplify_subspace` es pequeño y un
round-trip a F16 lo borraría.

### Tokenizer: BPE del propio GGUF

Leemos `tokenizer.ggml.tokens` y
`tokenizer.ggml.merges` de la metadata, con tablas
hash para merges rápidos.

Detalle práctico: algunos F16 de TinyLlama traen
un vocabulario *truncado* (pocos tokens en el
header). Si detectamos eso, cargamos **solo el
tokenizer** desde el Q4_0 hermano
(`tinyllama-1.1b.Q4_0.gguf`), sin tocar los pesos
F16. Los ids coinciden con HuggingFace
(p. ej. `"Hello"` → `[1, 15043]`).

## 2. Las operaciones del transformer

### RMSNorm

```c
/* x / rms(x) * w   — w viene en F16 */
static void rmsnorm(float *out, const float *x,
                    const uint16_t *w_f16, int n) {
    float ss = 0.f;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    float scale = 1.f / sqrtf(ss / (float)n + 1e-5f);
    for (int i = 0; i < n; i++)
        out[i] = x[i] * scale * f16_to_f32(w_f16[i]);
}
```

Poco peso (`attn_norm`, `ffn_norm`), mucho control.

### Matmul (el coste real)

```c
/* W [out, in] · x[in] → out[out]
 * OpenMP sobre filas de salida; F16 vía LUT */
static void matmul_f16(float *out, const float *x,
                       const uint16_t *W,
                       int in_dim, int out_dim) {
#pragma omp parallel for schedule(static)
    for (int j = 0; j < out_dim; j++) {
        const uint16_t *row = W + (size_t)j * in_dim;
        float sum = 0.f;
        for (int i = 0; i < in_dim; i++)
            sum += x[i] * f16_to_f32(row[i]);
        out[j] = sum;
    }
}
```

Con `--perturb`, el mismo bucle usa `matmul_f32`
sobre las copias ya perturbadas. Q, K, V, O, gate,
up, down y el `lm_head` son todos matmuls.

### RoPE

Posición sin embeddings absolutos: cada par de
dimensiones de Q y K se rota. El motor precalcula
sin/cos por posición hasta `MAX_SEQ` para no llamar
a `sinf`/`cosf` en el camino caliente.

### Atención con GQA + KV-cache

```
score(t) = (Q_pos · K_t) / sqrt(head_dim)
pesos    = softmax_causal(scores)   # solo 0..pos
salida   = Σ pesos[t] * V_t
```

TinyLlama: **32 cabezas Q, 4 KV** → cada cabeza KV
atiende a 8 cabezas Q (`hkv = h / 8`).

En cada paso de generación solo calculamos Q/K/V
del **token nuevo**, guardamos K y V en el caché
`[capa][pos][kv_dim]` y atendemos sobre
`0 .. pos`. Eso es lo que hace viable la CPU:
sin KV-cache recomputaríamos toda la secuencia
en cada token.

### SwiGLU (FFN)

```
h' = Down( SiLU(Gate(x)) ⊙ Up(x) )
```

~69% de los parámetros (cap. 2): la “memoria”
práctica del modelo. Tras la atención, residual;
tras el FFN, otro residual.

## 3. Un paso forward (un token)

En lenguaje humano, por capa y por *posición actual*:

```
x = embedding(token)

para L = 0 .. 21:
    h = RMSNorm(x, attn_norm[L])
    Q,K,V = proyecciones(h);  RoPE(Q,K)
    guardar K,V en cache[L][pos]
    x = x + O( Atención(Q, cache_K, cache_V) )

    h = RMSNorm(x, ffn_norm[L])
    x = x + SwiGLU_FFN(h)

logits = output · RMSNorm(x, output_norm)
```

En el código eso es `model_forward_token(...)`.
La generación:

```c
/* prefill: cada token del prompt llena el caché */
for (i = 0; i < n_prompt; i++)
    model_forward_token(&model, &state, tokens[i]);

for (step = 0; step < max_new; step++) {
    next = sample_top_k(state.logits, …);
    if (next == EOS) break;
    emit(next);
    model_forward_token(&model, &state, next);
}
```

## 4. Sampling

32.000 logits → una elección:

1. escalar por temperatura  
2. quedarse con top-k  
3. softmax solo sobre esos k  
4. muestrear  

| Parámetro | Efecto |
|-----------|--------|
| `temperature → 0` | casi greedy |
| `temperature` alta | más diversidad |
| `top_k` bajo | vocabulario estrecho |

Para comparar perspectivas fijamos `temp ≈ 0.7`,
`top_k` y `--seed` cuando queremos reproducibilidad.

## 5. Perturbación en runtime (el puente a Dreaming)

Dreaming no es solo generar texto: es **modificar
pesos** (o activaciones) y observar qué cambia.

### Técnicas disponibles en el motor C

| `--perturb` | Mecánica | Notas |
|-------------|----------|--------|
| `none` | sin cambio | baseline F16 |
| `mystical` / `amplify` | `amplify_subspace`: \(w \leftarrow w + I\,(w\cdot v)\,v\) | lente filosófica / existencial |
| `noise` | ruido ∝ \|w\| | a I alto degrada |
| `blockdiag` | amplifica bloques 16×16 | a veces monótono / eco |
| `manifold` | ruido local ∝ std del bloque | I alto puede colapsar |

Política (como en `dmt_perturb_v10`): se tocan las
matrices de **atención y FFN** de las 22 capas;
no se tocan normas, embeddings ni `lm_head`.

```bash
# 15 prompts con I=0.50, seed=42 → ~8 tok/s en generación
./llm_inference modelo.F16.gguf "When we dissolve the ego" \
    60 0.7 40 --seed 42 --perturb mystical --intensity 0.50
```

Ejemplo real (misma config):

> *When we dissolve the ego, we dissolve the self.
> When we let go of the ego, we allow ourselves to
> become part of the universe…*

### Intensidad

| I | Comportamiento típico |
|---|------------------------|
| 0.05–0.10 | casi baseline; delta sutil |
| 0.30–0.50 | perspectiva más clara (y a veces EOS distinto) |
| noise / manifold altos | riesgo de basura |

El setup místico cuesta ~**25 s** y ~**3.6 GB** de
RAM F32 *una vez por proceso*. Después, la
generación corre al mismo orden de magnitud que
el baseline.

### Steering de activaciones (`--steer`)

Complemento a la perturbación de pesos: se construye
un vector dirección a partir del embedding de una
palabra y se empuja el residual en esa dirección
durante el forward. Es otra forma de *marcar* la
generación sin reescribir el GGUF.

```bash
./llm_inference modelo.F16.gguf "The world is" 40 0.7 40 \
    --steer amor --steer-strength 0.15
```

## Rendimiento (órdenes de magnitud)

Medido en CPU con OpenMP (8 hilos), TinyLlama F16:

| Situación | Valor típico |
|-----------|----------------|
| Generación baseline | **~6–10 tok/s** (wall, con prefill) |
| Generación con pesos F32 post-perturb | **~6–10 tok/s** (a veces un poco más rápido: no dequant en el bucle) |
| Prefill + pocos tokens (EOS temprano) | tok/s wall *baja* (el prefill pesa) |
| Carga F16 + tokenizer | **~1 s** |
| Aplicar `mystical` (154 tensores → F32) | **~25 s** + ~3.6 GB |

No competimos con GPU ni con kernels Q4 de
llama.cpp. Competimos con la *opacidad*: aquí cada
multiplicación tiene nombre y dirección.

## Dos herramientas, un mismo modelo

| Herramienta | Rol | Formato |
|-------------|-----|---------|
| `llm_inference.c` | entender, enseñar, **perturbar en runtime**, sondas | GGUF F16 |
| `llama-cli` | baterías masivas, Q4_0, velocidad de experimentación | GGUF Q4_0 |

Las 240 generaciones del estudio de perspectivas
se apoyaron mucho en `llama-cli` + GGUFs ya
perturbados. El motor en C cierra el círculo:
podemos **repetir la idea de la perturbación sin
archivo intermedio** y ver el forward por dentro.

## Limitaciones honestas (actualizadas)

1. **Solo F16 en pesos** — sin kernels Q4_0/Q6_K en C
2. **Sin GPU** — CPU only
3. **Sin batching** — una secuencia a la vez
4. **Contexto del motor** acotado (`MAX_SEQ`, p. ej. 512)
   aunque el modelo admita 2048
5. **Perturbación F32 cara en RAM** — ~3.6 GB extra
6. **No todas las técnicas v10/v11** están en C
   (faltan lowrank, spectral, selective targeting, …)

Aceptable: el objetivo no es servir un chat a
millones de usuarios. El objetivo es **abrir el
cráneo** de un modelo de 1.1B, y poder **girar el
cristal** de sus pesos con una bandera.

## Cómo se conecta con el resto del libro

Hasta aquí:

- *Qué* es TinyLlama (capítulo 1)
- *Cómo* está organizado por dentro (capítulo 2)
- *Con qué* lo ejecutamos, lo medimos y lo
  **perturbamos** (este capítulo)

El siguiente capítulo entra de lleno en la
**perturbación de pesos y el cambio de perspectiva**:
analogía DMT, las técnicas que preservan jerarquía,
intensidades, y por qué el modelo no “se rompe”
sino que *habla con otra voz*.

Debajo de esos hallazgos hay un motor real.
El nuestro está en un archivo C, corre en CPU,
y ya no solo lee pesos: también sabe *moverlos*.

---

*Siguiente capítulo: Perturbación de Pesos y Cambio de Perspectiva*
