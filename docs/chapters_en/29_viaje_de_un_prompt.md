# Chapter 29: The Journey of a Prompt Inside TinyLlama

## Who This Chapter Is For

If you've read that an LLM "predicts the next token"
but you still don't *see* the path, this chapter is the complete
road map.

We'll walk through it **step by step**, with a real prompt,
without assuming any magical leap. At the end
you should be able to narrate, out loud, what happens
to each number from the moment you write a sentence until
the first word of the response appears.

**Example prompt (fixed throughout the chapter):**

```text
The secret to happiness is
```

**Model:** TinyLlama-1.1B  
**Key numbers that don't change:**

| Parameter | Value |
|-----------|------:|
| Transformer layers | 22 (indices 0…21) |
| Residual dimension \(d\) | 2048 |
| Vocabulary \(V\) | 32,000 |
| Q / KV heads | 32 / 4 (GQA) |
| Dimension per head | 64 |
| FFN intermediate | 5632 |
| Maximum context | 2048 positions |
| RoPE base | 10,000 |

---

## 0. The Movie in One Minute

Before the details, the trailer:

```
1.  TEXT          "The secret to happiness is"
2.  TOKENS        integer ids from BPE
3.  EMBEDDINGS    each id → vector of 2048 floats
4.  PREFILL       each prompt token passes through 22 layers
                  and fills the KV-cache
5.  LOGITS        32,000 scores for the *next* token
6.  SAMPLE        we choose an id (temperature, top-k, seed)
7.  DECODE        the id becomes readable text again
8.  LOOP          that token goes back into the model…
                  until max_new tokens or EOS
```

Everything else in this chapter is a **zoom** on
each arrow.

---

## 1. The Prompt Is Not "a Sentence" to the Model

### 1.1 What You See

A UTF-8 character string, with spaces and meaning.

### 1.2 What the Model Sees

An **ordered sequence of integers** between 0 and 31,999.

The bridge is called the **BPE tokenizer** (Byte Pair Encoding),
the same LLaMA style: words usually start with
the word-space prefix `▁` (U+2581).

For our prompt, the idea (didactic scheme) is:

| Position \(t\) | Piece (idea) | Role in the Sentence |
|---------------:|--------------|---------------------|
| 0 | `▁The` | article / start |
| 1 | `▁secret` | nominal core |
| 2 | `▁to` | link |
| 3 | `▁happiness` | object of the secret |
| 4 | `▁is` | copulative verb — **the present of the predicate** |

> In practice, BPE sometimes splits finer
> (`happ` + `iness`, etc.). The principle doesn't change:
> **text → list of ids**. Let's call that list
>
> \[
> (t_0, t_1, t_2, t_3, t_4)
> \]
>
> with length \(T_{\mathrm{prompt}} = 5\).

### 1.3 Why Order Matters

TinyLlama is **causal**: at position \(t\) it can
only "see" positions \(0,1,\ldots,t\).
The past exists; the future of the sentence **does not yet**.

That's the traffic rule of the entire journey.

---

## 2. From ID to Vector: Born in ℝ²⁰⁴⁸

Each id \(t_i\) becomes a point in the sky
of embeddings (ch. 28):

\[
x^{(i)}_{0} \;=\; e_{t_i} \;=\; \mathrm{Embedding}(t_i) \;\in\; \mathbb{R}^{2048}
\]

- There is a `token_embd` table of logical shape
  **[32,000 × 2048]**.
- Row number \(t_i\) is that star's vector.
- Here **there are no layers yet**. Only catalog.

**Didactic image:**  
five passengers enter the building lobby
(ch. 27). Each one carries a suitcase of 2048 numbers.
Those suitcases are called **residuals**.

In this chapter's notation:

- Superscript \((i)\): position in the sequence.
- Subscript \(\ell\): layer (0 before the first layer;
  after layer 21 we'll be on the "rooftop floor").

On leaving the embedding:

\[
x^{(0)}_{0},\; x^{(1)}_{0},\; \ldots,\; x^{(4)}_{0}
\in \mathbb{R}^{2048}
\]

---

## 3. Two Flight Phases: Prefill and Generation

TinyLlama (and almost every causal transformer) does not
process the prompt with a single magical blow.
There are **two modes**:

| Phase | What Goes In | What Comes Out | KV-cache |
|-------|-------------|----------------|----------|
| **Prefill** | Each token of the prompt, in order | Logits after the **last** prompt token | Is **filled** |
| **Generation** | One new token at a time | Logits for the next one | **Grows** by +1 |

In the C engine (`llm_inference.c`):

```c
/* PREFILL */
for (i = 0; i < n_prompt; i++)
    model_forward_token(&model, &state, tokens[i]);

/* GENERATION */
for (step = 0; step < max_new; step++) {
    next = sample_top_k(state.logits, …);
    if (next == EOS) break;
    emit(next);                          /* text to user */
    model_forward_token(&model, &state, next);
}
```

Until the end of the prefill we haven't "responded" yet.
We've only **understood the prompt** and left memory
in the cache.

---

## 4. A Single Token in a Single Layer (The Core)

Take the current position \(p\) (for example, the last
prompt token, \(p=4\), `▁is`).  
Its residual arriving at layer \(\ell\) is \(x\).

Inside the layer, these seven stations always occur,
in this order:

```
        x  (residual arriving at layer ℓ)
        │
        ▼
   [1] RMSNorm  (attn_norm)
        │
        ▼
   [2]  Q, K, V  +  RoPE
        │
        ▼
   [3]  Causal attention  (uses KV-cache of this layer)
        │
        ▼
   [4]  O projection  →  residual:  x ← x + Attn
        │
        ▼
   [5] RMSNorm  (ffn_norm)
        │
        ▼
   [6]  FFN SwiGLU  (gate, up, down)
        │
        ▼
   [7]  residual:  x ← x + FFN
        │
        ▼
        x  (exits toward layer ℓ+1)
```

Repeat that **22 times**. That's the complete elevator
for **one** token in **one** forward step.

---

## 5. Station by Station (With the Example)

We follow the passenger at position \(p=4\) (`▁is`),
in a generic layer \(\ell\), when positions \(0..3\) of the prompt
already exist in the cache.

### Station 1 — RMSNorm (Attention)

\[
h = \mathrm{RMSNorm}(x;\; \gamma_{\ell}^{\mathrm{attn}})
\]

- It doesn't "understand" the sentence.
- It **stabilizes** the vector scale so that
  Q and K don't explode.
- Ridiculous parameter mass (~0.01% of the model),
  enormous role (ch. 7, force V).

**Analogy:** calibrate the compass before looking
at the other stars in the sequence.

### Station 2 — Q, K, V Are Born and Position (RoPE)

\[
Q = W_Q h,\quad K = W_K h,\quad V = W_V h
\]

In TinyLlama the logical shapes per layer are:

| Tensor | Logical Shape | Human Reading |
|--------|--------------|---------------|
| \(W_Q\) | [2048, 2048] | 32 heads × 64 dims |
| \(W_K, W_V\) | [256, 2048] | **4** KV heads × 64 (GQA) |
| \(W_O\) | [2048, 2048] | gathers the 32 heads |

**GQA (Grouped Query Attention):**  
each key/value head is **shared by** 8 Q heads
(\(32/4 = 8\)). Less cache memory, same idea:
rich questions, shared memory.

**RoPE (Rotary Position Embedding):**  
before attending, Q and K are **rotated** according to position \(p\).
There's no separate "position 4" vector added: the position
is **coiled into** the angle of Q and K.

So the model distinguishes:

```text
secret to happiness   ≠   happiness to secret
```

even though the same "stars" are in the vocabulary.

### Station 3 — Attention: Gravity Between Tokens

For each query head:

\[
\mathrm{score}_{p,j}
  = \frac{q_p \cdot k_j}{\sqrt{64}},
  \qquad j = 0,1,\ldots,p
\]

\[
\alpha_{p,j} = \mathrm{softmax}_j(\mathrm{score}_{p,j})
\]

\[
z_p = \sum_{j=0}^{p} \alpha_{p,j}\, v_j
\]

**Reading with our prompt** (intuition, not a measured
attention map here):

| \(j\) | Token | What Might "Pull" `▁is` |
|------:|-------|--------------------------|
| 0 | The | little (grammatical function) |
| 1 | secret | theme: there is a secret |
| 2 | to | link |
| 3 | happiness | **content** of the secret |
| 4 | is | itself (self-attention) |

The \(\alpha_{p,j}\) are the **dynamic gravity**
(ch. 7 and 28): how much the residual of `is` falls toward
each star in the past of *this* sentence.

**Causal mask:** \(j > p\) is forbidden.
In the prefill, when we process position 2,
`happiness` **does not yet exist** in the cache.

### Station 4 — Head Mixing + Attention Residual

The 32 heads are concatenated (or projected) and
pass through \(W_O\):

\[
x \leftarrow x + O(z)
\]

The residual **is not erased**: the attention push is
**added** to it. That's why we speak of orbit, not of
teleportation (ch. 20).

\[
x_{\mathrm{after}} = x_{\mathrm{before}} + \Delta_{\mathrm{attn}}
\]

### Station 5 — RMSNorm (FFN)

Another calibration, with a different \(\gamma_{\ell}^{\mathrm{ffn}}\).

### Station 6 — FFN SwiGLU (The "Sun" of Parameters)

This is where ~**69%** of the model's mass lives:

\[
\begin{aligned}
u &= W_{\mathrm{up}} h \\
g &= W_{\mathrm{gate}} h \\
\mathrm{FFN}(h) &= W_{\mathrm{down}}\big(\mathrm{SiLU}(g)\odot u\big)
\end{aligned}
\]

- It expands to **5632** dimensions.
- The *gate* decides which channels to let through.
- It compresses back to 2048.

**Analogy:** attention looks at **other tokens**;
the FFN transforms **this** residual alone —
local weather, "practical" knowledge of the position
(Golden Rule: FFN → practical lens, ch. 9).

### Station 7 — FFN Residual

\[
x \leftarrow x + \mathrm{FFN}(h)
\]

It exits layer \(\ell\) ready for \(\ell+1\).

---

## 6. The KV-Cache: Memory of the Past

Without cache, for each new token you'd need to
**recalculate** K and V for the entire sentence. Impossible
on CPU at good speed.

With cache, at layer \(\ell\):

```
cache_K[ℓ][0 .. p]   already saved
cache_V[ℓ][0 .. p]

When processing position p:
  calculate only K_p, V_p
  write cache_K[ℓ][p], cache_V[ℓ][p]
  attend Q_p against cache_K[ℓ][0..p]
```

**Prefill of our prompt:**

| Step | Token Entering | Positions in Cache When Done |
|-----:|----------------|------------------------------|
| 1 | The | 0 |
| 2 | secret | 0–1 |
| 3 | to | 0–2 |
| 4 | happiness | 0–3 |
| 5 | is | 0–4 |

After step 5, all **22 layers** have K and V for
the five positions. The residual of `is` has climbed
the entire building. That's where the **logits** of
the first *response* token emerge.

---

## 7. From Rooftop to Vocabulary: Logits

After layer 21:

\[
h = \mathrm{RMSNorm}(x;\; \gamma^{\mathrm{out}})
\]

\[
\mathrm{logits} = W_{\mathrm{out}}\, h \;\in\; \mathbb{R}^{32000}
\]

- `output.weight` has logical shape **[32,000 × 2048]**
  (sometimes shared or tied with the embedding in other
  models; in TinyLlama's GGUF it's the `lm_head`).
- Each entry \(z_k\) is "how much the model pushes
  to choose the token with id \(k\)" **right now**.

There is still **no** word. There's a ranking of 32,000
candidates.

---

## 8. Sample: Collapsing the Sky to One Star

**Force VI** (ch. 7): from continuum to event.

Typical procedure in the Dreaming engine:

1. **Temperature** \(T\): \(z_k \leftarrow z_k / T\).
   - \(T \to 0\): almost always the maximum (greedy).
   - \(T\) high: more randomness, more diversity.
2. **Top-k**: keep only the \(k\) highest logits
   (e.g. 40). The rest is ignored.
3. **Softmax** only on those \(k\):

\[
\pi_i = \frac{e^{z_i}}{\sum_{j\in\mathrm{top\text{-}k}} e^{z_j}}
\]

4. **Sample** an id according to \(\pi\) (with `--seed` to
   reproduce the same journey).

Let's suppose (invented but realistic example) that the output is:

```text
id →  ▁being      or      ▁love      or      ▁not ...
```

That id is **decoded** to text and shown to the user.
That's the first step of the response.

---

## 9. The Autoregressive Loop (The Response Grows)

The chosen token **is not the model's end**.
It's the **next passenger**:

```
prompt:     The secret to happiness is
+ sample:   being
new seq:    The secret to happiness is being
```

`model_forward_token` is called again **only**
with `being`:

- Its embedding is computed.
- It passes through the 22 layers.
- It writes K,V at position \(p=5\) of each layer.
- It attends to `The…is` + `being`.
- It produces logits for the **even newer** token.

And so on:

```
The secret to happiness is being
The secret to happiness is being kind
The secret to happiness is being kind to
...
```

until:

- reaching `max_new` tokens, or
- sampling **EOS** (end of sequence).

**Key idea:**  
generating a paragraph is **many** repetitions of
the journey of *one* token, not a single pass "from the
full sentence to the full response".

---

## 10. Master Diagram of the Journey

```
┌─────────────────────────────────────────────────────────┐
│  HUMAN:  "The secret to happiness is"                  │
└───────────────────────────┬─────────────────────────────┘
                            │ BPE tokenizer
                            ▼
┌─────────────────────────────────────────────────────────┐
│  IDS:  t0 t1 t2 t3 t4                                   │
└───────────────────────────┬─────────────────────────────┘
                            │ rows of token_embd
                            ▼
┌─────────────────────────────────────────────────────────┐
│  VECTORS:  x0..x4  ∈ ℝ²⁰⁴⁸                             │
└───────────────────────────┬─────────────────────────────┘
                            │ PREFILL (for each ti)
                            ▼
        ┌───────────────────────────────────────┐
        │  for position p = 0 .. 4:            │
        │    for layer ℓ = 0 .. 21:             │
        │       Norm → Attn(+RoPE,GQA,cache)    │
        │            → +residual                │
        │       Norm → FFN SwiGLU               │
        │            → +residual                │
        └───────────────────┬───────────────────┘
                            │ after last p of prompt
                            ▼
┌─────────────────────────────────────────────────────────┐
│  output_norm → lm_head → logits[32000]                  │
└───────────────────────────┬─────────────────────────────┘
                            │ temp, top-k, softmax, sample
                            ▼
┌─────────────────────────────────────────────────────────┐
│  NEW TOKEN  →  text to user                             │
│       │                                                 │
│       └──── returns to forward_token (GENERATION) ──► … │
└─────────────────────────────────────────────────────────┘
```

---

## 11. Mini-Laboratory: See the Journey with the C Engine

From the repo root (adjusting paths to your GGUF):

```bash
# Prefill + generation, fixed seed (reproducible)
./llm_inference tinyllama-1.1b.F16.gguf \
  "The secret to happiness is" 40 0.7 40 --seed 42
```

| Flag / arg | Role in the Journey |
|------------|---------------------|
| prompt | initial stars of the sequence |
| `40` (n) | how many new tokens to orbit |
| `0.7` | collapse temperature |
| `40` (top-k) | width of the candidate well |
| `--seed 42` | same randomness → same path |
| `--perturb mystical --intensity 0.35` | **deforms** Q/K/V/FFN: different physics, same formal route |
| `--steer happiness --steer-strength 0.15` | pushes the residual toward a direction of the sky |

Recommended didactic protocol:

1. Same seed, `none` vs `mystical` → does the orbit change?
2. Same seed, temp 0.2 vs 0.9 → does the collapse change?
3. Open the [semantic map](https://fivetechsoft.github.io/dreaming/exploration/semantic_map.html),
   search for `▁happiness` / `▁love` and look at their **forces**
   (static gravity of the catalog) while reading
   the generated response (dynamic gravity of the prompt).

---

## 12. Common Mental Mistakes (And the Correction)

| Belief | Reality in TinyLlama |
|--------|----------------------|
| "The model reads the sentence at a glance" | It reads **token by token**; the prefill is sequential |
| "Each layer invents a new vector" | It updates the **same** residual with additions |
| "Attention looks at the whole book" | Only the **past** of *this* sequence (up to 2048) |
| "32 heads = 32 KV memories" | Only **4** KV groups (GQA); 32 Q gazes |
| "The embedding is already the answer" | The embedding is the **birth**; 22 floors remain |
| "Softmax picks the prompt's word" | It picks the **next** token from the vocabulary |
| "One response = one forward" | One response = **1 prefill + N forwards** |

---

## 13. Full Comprehension Checklist

If you can answer yes to everything, the journey is internalized:

1. What is a token and why isn't it a character?
2. What dimension is the residual and why is it preserved?
3. What does the causal mask prohibit?
4. What is RoPE for?
5. What distinguishes attention and FFN in a layer?
6. What does the KV-cache store and in which phase is it filled?
7. How many times does the 22-floor elevator go up for a 5-token prompt in prefill?  
   → **5 × 22** layer passes (one per position).
8. What is a logit and how does it become text?
9. Why does generating 40 tokens imply ~40 extra forwards?
10. Where does a Dreaming lens (`--perturb`) enter this picture?  
    → In the weights of stations 2–6, not in the tokenizer.

---

## 14. Bridges

| Topic | Chapter |
|-------|---------|
| Dims and tensors per layer | 2 |
| C engine, RoPE, cache, sample | 3 |
| Forces (attn, FFN, softmax…) | 7 |
| How to travel (routes A–E) | 8 |
| Attention in detail | 10 |
| FFN in detail | 11 |
| Early / middle / late layers | 13–15 |
| Chain of meaning (semantic vision) | 26 |
| Elevator by floor | 27 |
| Stars = tokens, attention = gravity | 28 |
| Formulas | 23 |

---

## 15. Closing

The journey of a prompt is not a mystery: it's a
**repeatable factory**.

1. Text becomes **ids**.  
2. Ids become **vectors**.  
3. Each vector climbs **22 floors** of  
   norm → attention gravity → FFN weather,  
   speaking only to the **past**.  
4. The final residual is projected to **32,000** scores.  
5. A sample chooses **one** star.  
6. That star is enqueued and the universe turns again.

When you write

```text
The secret to happiness is
```

and TinyLlama answers, it's no longer "AI thinks a sentence".
It's: *five births, five climbs up the building,
one collapse, and then N more collapses* — always the
same physics, one more step in time.

That's full comprehension of the journey.
The rest of the book (perspectives, maps, lenses)
are **variations of the physics**, not another path.
