# Chapter 3: Our C Inference Engine for TinyLlama

## Why write our own engine

To study TinyLlama it's not enough to *use it*.
You need to *see it work*.

Most inference frameworks hide the token's journey
behind layers of abstraction:

- PyTorch and CUDA (gigabytes of dependencies)
- Python runtimes
- Opaque GPU kernels

We wanted the opposite:

1. **Readable C** — a single file, explicit operations
2. **No mandatory GPU** — everything on CPU
3. **No magic** — every step of the transformer is a function
4. **Portable** — a small binary, reproducible

The result is `llm_inference.c`: an inference engine
that implements the transformer loop from the GGUF file
to the next token — and, later, **runtime weight perturbations**
without needing a pre-baked GGUF.

It's not the fastest engine in the world.
It's the engine that *we understand line by line*
and with which we can *touch the weights deliberately*.

## What an inference engine must do

Remember the flow from the previous chapter:

```
Text → tokens → embeddings
     → 22 layers (attention + FFN)
     → logits → next token
     → (repeat)
```

In practice the engine splits into five pieces:

```
┌─────────────────────────────────────────┐
│            llm_inference.c              │
├─────────────────────────────────────────┤
│  1. GGUF Reader                        │
│     - magic, metadata, indices         │
│     - F16 tensors (and fallback        │
│       tokenizer from Q4_0 if needed)   │
├─────────────────────────────────────────┤
│  2. BPE Tokenizer                      │
│     - tokens + merges from the GGUF    │
│     - text ⇄ ids                       │
├─────────────────────────────────────────┤
│  3. Transformer Engine + KV-cache       │
│     - embedding, RMSNorm, RoPE         │
│     - GQA attention, SwiGLU            │
│     - one new token per step           │
├─────────────────────────────────────────┤
│  4. Sampling                           │
│     - temperature, top-k               │
├─────────────────────────────────────────┤
│  5. Perturbation / steering (optional) │
│     - --perturb mystical, noise, …     │
│     - --steer <word> (activations)     │
└─────────────────────────────────────────┘
```

| Block | Question |
|-------|----------|
| GGUF | Where are the weights on disk? |
| Tokenizer | How is language converted to numbers? |
| Transformer | How is the representation transformed? |
| Sampling | How is the next word chosen? |
| Perturbation | How do we change the model's *perspective*? |

## TinyLlama in numbers (what the engine uses)

| Parameter | Value in TinyLlama-1.1B |
|-----------|-------------------------|
| Layers (`block_count`) | 22 |
| Hidden dimension | 2048 |
| Q / KV heads (GQA) | 32 / 4 |
| Dimension per head | 64 |
| Intermediate FFN | 5632 |
| Vocabulary | 32,000 |
| Model context | 2048 |
| RoPE `freq_base` | 10,000 |
| Engine weight format | GGUF **F16** |

Nine tensors per layer (ch. 2) plus the globals
`token_embd`, `output_norm`, `output`. The engine doesn't
invent the architecture: it *materializes* it in memory.

> **Note on Q4_0:** the massive experiments of the
> project also use quantized GGUFs (~638 MB)
> with `llama-cli`. Our pure C reads **F16**
> (~2.1 GB) to avoid reimplementing all the quantization
> kernels. The *forward pass* is the same transformer.

## Compiling and running

```bash
# Recommended (Windows/MinGW or Linux)
gcc -O3 -march=native -ffast-math -fopenmp \
    -o llm_inference llm_inference.c -lm

# Useful variables
export OMP_NUM_THREADS=8   # Linux/macOS
# PowerShell: $env:OMP_NUM_THREADS = "8"
```

### Basic usage (baseline)

```bash
./llm_inference tinyllama-1.1b.F16.gguf \
    "The secret to happiness is" \
    40 0.7 40 \
    --seed 42
```

Positional arguments: `model`, `prompt`, `n_tokens`,
`temperature`, `top_k`.

### Mystical perturbation at runtime

```bash
./llm_inference tinyllama-1.1b.F16.gguf \
    "The secret to happiness is" \
    40 0.7 40 \
    --seed 42 \
    --perturb mystical \
    --intensity 0.50
```

No need for a prior `DMT_*.gguf` file:
the technique is applied **in memory** on the freshly loaded weights.

### Other flags

| Flag | Effect |
|------|--------|
| `--perturb` / `-P` | `none`, `mystical`, `amplify`, `noise`, `blockdiag`, `manifold` |
| `--intensity` / `-I` | perturbation strength (e.g. 0.10–0.50) |
| `--seed` | perturbation PRNG (+ sampling if set) |
| `--steer` | word whose embedding direction pulls the residual |
| `--steer-strength` | steering intensity (e.g. 0.15) |

*Build* dependencies: `libm` and, for parallel matmul,
**OpenMP** (`-fopenmp`). Without OpenMP the engine
still compiles, just slower.

## 1. Reading a GGUF in C

GGUF has three zones:

```
[ magic "GGUF" | version | # tensors | # KV ]
[ key-value metadata ]
[ tensor index → aligned data ]
```

The engine loads the entire file (on Windows with
64-bit I/O: F16 exceeds 2 GB and 32-bit `ftell`
is not enough). Then it indexes tensors by name:

```c
snprintf(name, sizeof(name), "blk.%d.attn_q.weight", l);
m->wq[l] = must_tensor_f16(&m->gguf, name);
/* wk, wv, wo, w1/w3/w2 (gate/up/down), norms… */
```

### F16 on disk, float in matmul

Layer weights are kept in **F16** in the file buffer.
In the matrix-vector product they are converted with a
**table of 65,536 entries** (one value per half pattern):

```c
static float g_f16_table[65536];
/* init: g_f16_table[i] = decode_ieee_half(i); */
static inline float f16_to_f32(uint16_t h) {
    return g_f16_table[h];
}
```

This way we avoid decoding bits for every weight in the
hot loop. Only when `--perturb` is requested do we copy
layer matrices to **mutable F32** (~3.6 GB):
the `amplify_subspace` delta is small and an
F16 round-trip would erase it.

### Tokenizer: BPE from the GGUF itself

We read `tokenizer.ggml.tokens` and
`tokenizer.ggml.merges` from the metadata, with hash
tables for fast merges.

Practical detail: some TinyLlama F16 files come with
a *truncated* vocabulary (few tokens in the header).
If we detect that, we load **only the tokenizer**
from the companion Q4_0
(`tinyllama-1.1b.Q4_0.gguf`), without touching the F16
weights. The ids match HuggingFace
(e.g. `"Hello"` → `[1, 15043]`).

## 2. The transformer operations

### RMSNorm

```c
/* x / rms(x) * w   — w comes in F16 */
static void rmsnorm(float *out, const float *x,
                    const uint16_t *w_f16, int n) {
    float ss = 0.f;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    float scale = 1.f / sqrtf(ss / (float)n + 1e-5f);
    for (int i = 0; i < n; i++)
        out[i] = x[i] * scale * f16_to_f32(w_f16[i]);
}
```

Little weight (`attn_norm`, `ffn_norm`), much control.

### Matmul (the real cost)

```c
/* W [out, in] · x[in] → out[out]
 * OpenMP over output rows; F16 via LUT */
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

With `--perturb`, the same loop uses `matmul_f32`
on the already-perturbed copies. Q, K, V, O, gate,
up, down and `lm_head` are all matmuls.

### RoPE

Position without absolute embeddings: each pair of
Q and K dimensions is rotated. The engine precomputes
sin/cos per position up to `MAX_SEQ` to avoid calling
`sinf`/`cosf` on the hot path.

### Attention with GQA + KV-cache

```
score(t) = (Q_pos · K_t) / sqrt(head_dim)
weights  = softmax_causal(scores)   # only 0..pos
output   = Σ weights[t] * V_t
```

TinyLlama: **32 Q heads, 4 KV** → each KV head
serves 8 Q heads (`hkv = h / 8`).

At each generation step we only compute Q/K/V
for the **new token**, store K and V in the cache
`[layer][pos][kv_dim]`, and attend over
`0 .. pos`. That's what makes CPU viable:
without KV-cache we'd recompute the entire sequence
at every token.

### SwiGLU (FFN)

```
h' = Down( SiLU(Gate(x)) ⊙ Up(x) )
```

~69% of the parameters (ch. 2): the model's
practical "memory". After attention, residual;
after FFN, another residual.

## 3. One forward step (one token)

In human language, per layer and per *current position*:

```
x = embedding(token)

for L = 0 .. 21:
    h = RMSNorm(x, attn_norm[L])
    Q,K,V = projections(h);  RoPE(Q,K)
    store K,V in cache[L][pos]
    x = x + O( Attention(Q, cache_K, cache_V) )

    h = RMSNorm(x, ffn_norm[L])
    x = x + SwiGLU_FFN(h)

logits = output · RMSNorm(x, output_norm)
```

In code that's `model_forward_token(...)`.
Generation:

```c
/* prefill: each prompt token fills the cache */
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

32,000 logits → one choice:

1. scale by temperature
2. keep top-k
3. softmax only over those k
4. sample

| Parameter | Effect |
|-----------|--------|
| `temperature → 0` | nearly greedy |
| high `temperature` | more diversity |
| low `top_k` | narrow vocabulary |

To compare perspectives we fix `temp ≈ 0.7`,
`top_k`, and `--seed` when we want reproducibility.

## 5. Runtime perturbation (the bridge to Dreaming)

Dreaming isn't just generating text: it's **modifying
weights** (or activations) and observing what changes.

### Techniques available in the C engine

| `--perturb` | Mechanics | Notes |
|-------------|-----------|-------|
| `none` | no change | F16 baseline |
| `mystical` / `amplify` | `amplify_subspace`: \(w \leftarrow w + I\,(w\cdot v)\,v\) | philosophical / existential lens |
| `noise` | noise ∝ \|w\| | high I degrades |
| `blockdiag` | amplifies 16×16 blocks | sometimes monotonous / echo |
| `manifold` | local noise ∝ block std | high I can collapse |

Policy (as in `dmt_perturb_v10`): the **attention and FFN**
matrices of all 22 layers are touched;
norms, embeddings and `lm_head` are not.

```bash
# 15 prompts with I=0.50, seed=42 → ~8 tok/s in generation
./llm_inference model.F16.gguf "When we dissolve the ego" \
    60 0.7 40 --seed 42 --perturb mystical --intensity 0.50
```

Real example (same config):

> *When we dissolve the ego, we dissolve the self.
> When we let go of the ego, we allow ourselves to
> become part of the universe…*

### Intensity

| I | Typical behavior |
|---|------------------|
| 0.05–0.10 | nearly baseline; subtle delta |
| 0.30–0.50 | clearer perspective (and sometimes different EOS) |
| high noise / manifold | risk of garbage |

The mystical setup costs ~**25 s** and ~**3.6 GB**
of F32 RAM *once per process*. After that,
generation runs at the same order of magnitude
as the baseline.

### Activation steering (`--steer`)

Complement to weight perturbation: a direction vector
is built from the embedding of a word and the residual
is pushed in that direction during the forward pass.
It's another way to *mark* generation without
rewriting the GGUF.

```bash
./llm_inference model.F16.gguf "The world is" 40 0.7 40 \
    --steer amor --steer-strength 0.15
```

## Performance (orders of magnitude)

Measured on CPU with OpenMP (8 threads), TinyLlama F16:

| Situation | Typical value |
|-----------|---------------|
| Baseline generation | **~6–10 tok/s** (wall, with prefill) |
| Generation with F32 post-perturb weights | **~6–10 tok/s** (sometimes slightly faster: no dequant in the loop) |
| Prefill + few tokens (early EOS) | tok/s wall *low* (prefill is heavy) |
| F16 load + tokenizer | **~1 s** |
| Applying `mystical` (154 tensors → F32) | **~25 s** + ~3.6 GB |

We don't compete with GPUs or Q4 kernels from
llama.cpp. We compete with *opacity*: here every
multiplication has a name and an address.

## Two tools, one model

| Tool | Role | Format |
|------|------|--------|
| `llm_inference.c` | understand, teach, **perturb at runtime**, probes | GGUF F16 |
| `llama-cli` | massive batches, Q4_0, experimentation speed | GGUF Q4_0 |

The 240 generations of the perspective study
relied heavily on `llama-cli` + already-perturbed GGUFs.
The C engine closes the loop: we can **repeat the
perturbation idea without an intermediate file**
and watch the forward pass from the inside.

## Honest limitations (updated)

1. **F16 weights only** — no Q4_0/Q6_K kernels in C
2. **No GPU** — CPU only
3. **No batching** — one sequence at a time
4. **Engine context** bounded (`MAX_SEQ`, e.g. 512)
   even though the model supports 2048
5. **F32 perturbation RAM-expensive** — ~3.6 GB extra
6. **Not all v10/v11 techniques** are in C
   (missing lowrank, spectral, selective targeting, …)

Acceptable: the goal is not to serve a chat to
millions of users. The goal is to **open the skull**
of a 1.1B model and be able to **turn the glass**
of its weights with a flag.

## How it connects to the rest of the book

So far:

- *What* TinyLlama is (chapter 1)
- *How* it is organized inside (chapter 2)
- *With what* we run it, measure it, and
  **perturb it** (this chapter)

The next chapter dives fully into **weight perturbation
and perspective change**: DMT analogy, hierarchy-preserving
techniques, intensities, and why the model doesn't "break"
but *speaks with another voice*.

Underneath those findings there is a real engine.
Ours is in a C file, runs on CPU,
and no longer just reads weights: it also knows how to *move them*.

---

*Next chapter: Weight Perturbation and Perspective Change*
