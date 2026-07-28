# Chapter 7: The Gravitational Forces of the Microcosmos

## There isn't just one gravity

In the TinyLlama universe "gravity" is a set of
**fields** that bend trajectories of meaning.
Each has mass (parameters), reach, and effect
on the text.

## Inventory of forces

| # | Force | Support | Mass | Reach |
|---|-------|---------|------|-------|
| I | Attentional attraction | Q·K/√d → V | ~19% | Between tokens in the sequence |
| II | FFN potential | SwiGLU gate/up/down | ~69% | Per token (local) |
| III | Residual inertia | x ← x + f(x) | structure | 22 layers |
| IV | Embedding anchor | token_embd, output | ~12% | Initial condition |
| V | Stabilization | RMSNorm | ~0.01% | Anti-explosion |
| VI | Vocabulary collapse | logits → softmax | head | 1 of 32k tokens |
| VII | Perspectives | weight perturbation | entire model | Changes the "climate" |
| VIII | Semantic islands | embedding geometry | — | Static attractors |

### Measured masses (logical F16)

| Component | Parameters | Share |
|-----------|------------|-------|
| FFN | ~761M | **69.2%** |
| Attention | ~208M | **18.9%** |
| Emb + lm_head | ~131M | **11.9%** |
| Norms | ~92k | **0.01%** |

## Force I — Attention

Non-local: a token feels others from the past (causal mask).
GQA 32 Q / 4 KV: cheap gravity to memorize.

**Golden Rule:** perturb attention → **academic / relational** lens.

## Force II — FFN

The "sun" of the weight system. It transforms each position
without looking at neighbors: local climate of the residual.

**Golden Rule:** perturb FFN → **practical / action** lens.

## Force III — Residual

Conservation of meaning momentum. That's why tangent
steps (`amplify_subspace`) maintain coherence
and normal noise to the surface destroys it.

## Force IV and V — Birth and air

Embeddings fix the starting point in ℝ²⁰⁴⁸
(average norm ≈ 0.68, nearly isotropic).
RMSNorm makes the 22 layers habitable with minimal mass.

## Force VI — Softmax

Collapse from continuum to event: one token.
Temperature and top-k are the "hardness" of the well.

## Force VII — Perspectives

Coherence surface in ℝ~1.1e9.
`mystical` = tangent current; strong `noise` = exit to void.

## Force VIII — Constellations

Centroids of areas (emotion, spirit, matter, mind…):
nearly orthogonal between islands. Relative attraction
abstract↔mind (+0.13); time↔social (−0.09).
Love/hate are not antipodal: cos ≈ 0.

## Three laws

1. **Surface** — only tangent trajectories in weights → coherent text.
2. **Two matters** — attention structures relationships; FFN transforms content.
3. **Collapse** — everything ends in one token.

## Dominance hierarchy

```
softmax (destiny)
    ↑
attention (long range)  +  FFN (mass)
    ↑
residual (inertia)
    ↑
embedding (start)  +  norm (stability)
    ↑
weights / perspective (universe metric)
    ↑
semantic islands (input sky)
```

---

*Next chapter: How to Travel Through the TinyLlama Universe*
