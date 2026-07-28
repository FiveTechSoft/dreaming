# Chapter 5: Journey Through TinyLlama's Multidimensional Space

## There isn't just one space

When we say "the interior of TinyLlama," we're not talking
about a single map. We're talking about **several nested spaces**,
each with its own dimensionality and role.

This chapter is a *field trip*: we measure the
actual embedding space of the F16 model
(32,000 × 2048), with the GGUF's BPE vocabulary
(`▁love`, `▁death`, …).

Tool: `explore_tinyllama_space.py`
Data: `inside-tinyllama/exploration/`

---

## The map of the seven spaces

```
┌──────────────────────────────────────────────────────────┐
│  6. WEIGHTS  ℝ^{~1.1e9}                                  │
│     coherence surface ≈ "models that talk"               │
│     7. PERSPECTIVES ⊂ (6)  — trajectories via perturb.  │
├──────────────────────────────────────────────────────────┤
│  forward pass, token to token:                           │
│                                                          │
│  1. EMBEDDING     ℝ^{2048}   ← 32k points from vocab     │
│         ↓                                                │
│  2. RESIDUAL ×22  ℝ^{2048}   (same dim, new content)     │
│         ↘ 3. ATTENTION   ℝ^{64} × 32Q / 4KV             │
│         ↘ 4. FFN         ℝ^{5632}                        │
│         ↓                                                │
│  5. LOGITS        ℝ^{32000}  → softmax → next token      │
└──────────────────────────────────────────────────────────┘
```

| # | Space | Dims | What it is |
|---|-------|------|------------|
| 1 | Token embeddings | 2048 | "At rest" meaning of each vocabulary piece |
| 2 | Residual stream | 2048 × 22 | Contextual representation evolving layer by layer |
| 3 | Attention heads | 64 | Local views of token relationships (GQA 32/4) |
| 4 | Intermediate FFN | 5632 | "Memory / practical transformation" expansion |
| 5 | Logits | 32,000 | Preferences over the next token |
| 6 | Model weights | ~1.1e9 | All parameters; almost all volume is garbage |
| 7 | Perspectives | subvariety of (6) | Coherent models with different tone (mystical, etc.) |

The residual is a **2048-dimensional tunnel** that
passes through 22 rooms. Attention and FFN are
lateral detours that write back into that tunnel.

---

## Region 1 — Semantic poles

Are *love* and *hate* at opposite ends?

**No.** In the static embedding, natural language "opposites"
have cosine **near zero**
(orthogonal), not −1 (antipodal).

| Pair | cosine |
|------|--------|
| ▁love / ▁hate | +0.006 |
| ▁life / ▁death | +0.016 |
| ▁happy / ▁sad | **−0.035** |
| ▁true / ▁false | **−0.036** |
| ▁good / ▁evil | −0.001 |
| ▁king / ▁queen | +0.009 |
| ▁man / ▁woman | +0.008 |

**Reading:** in ℝ²⁰⁴⁸ "cold" is not −"heat".
Words occupy different directions in
space; semantic opposition is organized more
by **clusters and contexts** (layers + attention)
than by simple antipodality in the embedding.

---

## Region 2 — Continents (clusters)

We group words and take the **centroid**.
The centroid's neighbors recover the
continent itself — the local geometry is coherent.

| Continent | Tokens (examples) | Centroid neighbors |
|-----------|-------------------|-------------------|
| emotion_pos | happy, joy, love, peace… | smile, happy, hope, love |
| emotion_neg | sad, hate, fear, anger… | sad, pain, anger, cry |
| spiritual | soul, spirit, god, faith… | faith, divine, spirit, god |
| physical | body, rock, water, fire… | rock, water, matter, body |
| abstract | truth, beauty, justice… | beauty, meaning, idea… |
| time | time, past, future, now… | time, now, moment, past |

### Distance between continents

Centroids of different continents are
**nearly orthogonal** to each other (cosine ≈ 0):

```
emotion_pos  ⊥  emotion_neg   (−0.01)
spiritual    ⊥  physical      (+0.02)
abstract     ⊥  physical      (−0.01)
time         ⊥  abstract      (−0.06)
```

The vocabulary is not a diffuse ball: it is a
**set of islands** on a 2048-dim sphere,
with little overlap between thematic islands.

---

## Region 3 — Analogies (a − b + c)

The classic word2vec test:

```
king − man + woman  ≟  queen
```

In TinyLlama (static embedding, top-6) it **fails**:
rare BPE pieces, symbols, multilingual fragments
appear — not `queen`.

That doesn't mean the model "doesn't know" the analogy.
It means that:

1. The embedding of a token **without context** is
   just the entry door.
2. The "live" analogy is assembled in the **residual**
   after attention and FFN, not in the vocab row.
3. BPE chops the world (`builder`, suffixes…);
   not every concept is a single clean point.

---

## Region 4 — Global shape of ℝ²⁰⁴⁸

PCA on 4,000 random tokens:

| Metric | Value |
|--------|-------|
| Variance in 1st PC | **0.27%** |
| Variance in top-10 | 2.3% |
| Variance in top-100 | 14% |
| Dims for 50% of variance | **~481** |
| Dims for 90% | **~1329** |
| Dims for 99% | **~1880** |
| Anisotropy \|\|mean\|\| / mean\|\|e\|\| | **0.006** (nearly isotropic) |

**Reading:** the token space **truly uses
hundreds or thousands of directions**. It doesn't collapse to a
pair of "good/evil" axes. That's why rank-1 perturbations
(amplify) can "turn the glass" without
shutting off speech: there's a lot of coherent volume.

---

## Region 5 — Directions as compasses

If we subtract centroids, **usable semantic axes**
appear:

### emotion = pos − neg
- + pole → smile, happy, peace, love, joy
- − pole → sad, anger, cry, pain, fear

### spirit − matter
- + → spirit, god, sacred, divine, faith
- − → rock, matter, water, earth, body

### abstract − physical
- + → beauty, truth, justice, meaning, freedom
- − → rock, matter, fire, water, earth

These directions live in the **same ℝ²⁰⁴⁸**
as the residual. That's why `--steer amor` in the C
engine can push generation: it's a vector in the
tunnel, not external magic.

And that's why `amplify_subspace` in the
**weight space** (dimension 1e9) is a different journey: it moves
the *entire map*, not a vocabulary point.

---

## Region 6 — Norms: not every token "weighs" the same

\|\|e\|\| average ≈ 0.67. The extremes are not
clear philosophical concepts (often BPE pieces
or symbols). The **norm** is not a dictionary of
semantic importance; it's another coordinate of the
landscape.

---

## How spaces connect in an inference step

```
"happiness"
    → BPE → ids
    → rows in (1) EMBEDDING          ℝ^2048
    → 22× { attn in (3) + FFN in (4) }  writing into (2)
    → (5) LOGITS
    → sample → "is" / "to" / …
```

If we perturb weights (6) with *mystical*,
each Q/K/V/FFN projection is slightly deformed:
the path in (2) remains coherent, but the
**attractions** toward islands of (1) and (5) change
— hence the perspective change.

If we do *steer* in (2), we push the residual
toward a direction of (1) without rewriting (6).

---

## Explorer's itinerary

| Stop | Question | Empirical answer |
|------|----------|------------------|
| Poles | Are opposites antipodal? | No: nearly orthogonal |
| Continents | Are there thematic regions? | Yes: clean clusters |
| Static analogies | king−man+woman? | Not in raw embedding |
| Dimensionality | How many dims matter? | Hundreds–thousands (not 2–3) |
| Directions | Are there useful axes? | Yes (emotion, spirit…) |
| Weights | Where do perspectives live? | Surface in ℝ^1e9 |

---

## What remains to explore

1. **Residual per layer** — project activations
   of the 22 layers onto emotion/spirit axes
   (where does the "mystical" turn on?).
2. **FFN ℝ⁵⁶³²** — neurons reacting to
   semantic clusters.
3. **Perturbation trajectories** — curve of
   cosine(baseline, mystical) as a function of I
   in weight space or logit space.
4. **2D/3D maps** — UMAP/t-SNE of the 32k
   points colored by continent.

TinyLlama's universe is not a point.
It is a **system of spaces**. This chapter only
crossed the first frontier: the sky of tokens.
Further in, the residual and weights await.

---

*Tools: `explore_tinyllama_space.py`,
`llm_inference.c --perturb` / `--steer`.*

*Next chapter: From the Macrocosmos to the Microcosmos (and vice versa).*
