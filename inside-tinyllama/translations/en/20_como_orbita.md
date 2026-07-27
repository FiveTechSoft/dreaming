# Chapter 20: How This Universe Orbits

## The Question

In the macrocosmos, planets fall toward the sun
but never reach it: **they fall sideways** — that's an orbit.

In TinyLlama the analogous question is:

> What falls, toward what, and why doesn't it crash
> in each layer?

The answer is the **forward pass** read as dynamics.

---

## 1. What is the "body" that orbits

The body is not an isolated token.
It's the **residual** \(x \in \mathbb{R}^{2048}\):
a vector that's born in the embedding and passes through
22 layers without losing its identity entirely.

```
birth:       x₀ = Embedding(token)
orbit:       x ← x + Atención(x)
             x ← x + FFN(x)          × 22
destination: logits = W_out · Norm(x)
collapse:    token' ~ Softmax(logits / T)
```

Each **token in the sequence** carries its own residual.
Attention is the gravitational coupling **between**
those bodies (only with the past: causal).

---

## 2. The law of the residual: falling without crashing

Without residual connection, each layer would *replace*
the state: teleportation, not orbit.

With residual:

\[
x_{L+1} = x_L + f_L(x_L)
\]

- \(f_L\) = attention + FFN push at layer \(L\).
- The step is **tangent and small** relative to \(x\):
  the vector rotates and deforms but doesn't reset.

That's the **orbital inertia** of the microcosmos.
Perturbations that preserve hierarchy
move the *metric* of \(f_L\) without removing \(x\)
from the surface where speech is still possible.

---

## 3. Two powers per "year-layer"

Each layer is an **orbital period** of the residual:

| Phase | Force | Analogy |
|-------|-------|---------|
| 1. RMSNorm + Attention | Gravity between tokens | Tugs from other bodies in the system |
| 2. Residual | Momentum conservation | You don't fall from the sky in one shot |
| 3. RMSNorm + FFN | Local field / atmosphere | Physics of the planet you're on |
| 4. Residual | Inertia again | You're still on trajectory |

**22 layers ≈ 22 periods** before final collapse
(softmax), where the orbit ceases to be continuous
and becomes a **landing** on a token.

---

## 4. Multi-body systems (the sequence)

A sentence is a **temporary solar system**:

```
pos 0: "The"     → residual_0
pos 1: "secret"  → residual_1  (looks at 0)
pos 2: "to"      → residual_2  (looks at 0,1)
...
pos t: ...       → residual_t  (looks at 0…t)
```

- **GQA**: 32 sensors (Q) share 4 memories (KV)
  — not 32 suns, but one sun with several KV-mass planets.
- **KV-cache**: already-computed K,V are reused;
  only the new body integrates its orbit.
  Without cache, the system would recalculate the entire sky
  at each step (the old engine; the current one orbits well).

The causal mask is the **arrow of time**:
the future doesn't attract the present.

---

## 5. Orbit of generation (the great cycle)

Generating text is a **closed orbit in discrete time**:

```
        ┌─────────────────────────────────┐
        │                                 │
        ▼                                 │
   residual(s) ──► logits ──► sample ──► new token
        │                                 │
        └──────── embedding(token) ───────┘
```

Each revolution:

1. The new token is born in the sky of embeddings.
2. It integrates with the gravity of previous ones.
3. It collapses to a successor.
4. The system grows by one body.

**Period:** ~1/token (in C engine CPU: ~0.1–0.15 s/token
⇒ **~6–10 tok/s**).
**Temperature:** eccentricity of collapse (more "circular"
or wilder orbits).
**Top-k:** horizon of allowed destinations.

---

## 6. Orbits in weight space (perspectives)

There's another orbit, slower than the forward pass:

```
base model  --(+ ε · δ)-->  model with another voice
```

- \(\delta\) tangent to the coherence surface
  (`mystical` / amplify) → **stable orbit** of perspectives.
- \(\delta\) normal (strong noise) → **ejection** into the void
  (garbage).

Changing `--intensity` changes the **radius** of that
deviation. Same seed + same prompt = comparing
two generation orbits under two weight metrics.

---

## 7. Orbits in the semantic sky (static)

Tokens don't "orbit" alone in the embedding:
they're fixed like catalog stars.

What does move is the **residual** relative to islands:

```
residual · spiritual_direction   →  affinity to the spiritual continent
residual · emotion_direction     →  affective affinity
```

`--steer amor` is an **artificial orbital push**:
it adds a component along an axis of the sky
without rewriting the star catalog (embeddings).

The 2D PCA map is a **planetarium**: it projects the catalog
so we can see constellations; it's not the real dynamics.

---

## 8. Unified diagram

```
                    WEIGHT SPACE (universe metric)
                              │
                    --perturb │ (changes G, not the body)
                              ▼
   tokens ══╗
            ║  gravity (attention)     climate (FFN)
   residual ╬══════► pushes ══════► pushes  ──► ×22 layers
            ║              residual (inertia)
            ╚══════════════════════════════════════════╝
                              │
                         output_norm
                              │
                           logits
                              │
                    softmax / temp / top-k
                              │
                         new token ──► (closes the orbit)
```

---

## 9. How to "ride" an orbit (recipe)

| Goal | Controls |
|------|----------|
| Clean baseline orbit | prompt + seed + temp, no perturb |
| Same orbit, different climate | `--perturb mystical --intensity I` |
| Deviate toward an island | `--steer word --steer-strength s` |
| More predictable orbit | temp↓, top_k↓ |
| More exploratory orbit | temp↑, top_k↑ |
| Longer multi-body system | n (tokens) ↑ |
| Reproduce the flight | same seed, same flags |

```bash
# Reference orbit
./llm_inference model.F16.gguf "When we dissolve the ego" \
  40 0.7 40 --seed 42

# Same initial trajectory, mystical metric
./llm_inference model.F16.gguf "When we dissolve the ego" \
  40 0.7 40 --seed 42 --perturb mystical --intensity 0.50
```

---

## 10. In one sentence

**This universe orbits** because the residual **falls
sideways** under the gravity of attention and the climate
of the FFN, conserving momentum with the residual,
for 22 periods per token, until collapsing into a
successor — and generation repeats that cycle, while
perspectives change the metric of the space
without extinguishing the possibility of coherent orbits.

---

*Next chapter: Archetypes and Constellations.*