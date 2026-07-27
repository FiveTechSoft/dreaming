# Chapter 9: The Geometric Golden Rule

## The discovery

Modifying **different components** of the transformer
does not produce generic noise. It produces **specific and
predictable perspectives**.

| Component | "Planet" | Emerging perspective |
|-----------|----------|---------------------|
| **Attention** (Q, K, V, O) | Structure / relationships | Academic, critical, formal |
| **FFN** (gate, up, down) | Vocabulary / action | Practical, lists, advice |
| **Embeddings** | Input identity | Simple and direct language |

We call this the **geometric Golden Rule**.

## Attention → academic

Attention tensors connect tokens.
When perturbed, the model prioritizes **structure**:
arguments, references, formal tone.

```
Prompt: "The meaning of life is..."
Baseline:  "...finding happiness..."
Perturbed attn: "...a fundamental philosophical inquiry
                 debated by scholars for millennia..."
```

## FFN → practical

The FFN transforms each position (practical memory,
~69% of parameters). When touched, **action verbs**
and concrete steps emerge.

```
Perturbed FFN: "To find meaning: 1) Identify values,
                2) Set goals, 3) Take daily action..."
```

## Embeddings → simple

The input matrix defines the "birth map"
of each token. Perturbing it flattens the register:

```
Perturbed emb: "Life means living. Be happy. Help others."
```

## Why it's "geometric"

Each tensor family moves the residual in
**different directions** of the representation space.
It's not file name magic: it's that attention
and FFN implement different operators on the same ℝ²⁰⁴⁸.

Selective targeting (v11) confirms it:

| Targeting | Sought effect |
|-----------|---------------|
| `attention_alter` | Strong amplify on attn, soft on FFN |
| `ffn_dream` | Strong creative on FFN, soft on attn |
| `embedding_shift` | Change in emb, rest soft |

## Empirical verification (summary)

- 24 models, 240 generations, 10 prompts (Dreaming battery).
- Hierarchy-preserving techniques → coherence.
- Hierarchy-breaking techniques (high noise, nibble flip) → garbage.
- Runtime C: `mystical` on attn+FFN (not emb/norm) aligns
  with the `dmt_perturb_v10` policy.

## How to use it when traveling

1. Want analysis? → look / touch **attention**.
2. Want a checklist? → look / touch **FFN**.
3. Want plain prose? → look / touch **embeddings**.
4. Want global existential climate? → `mystical` on layers.

The Golden Rule is the **scale bridge**:
from the gear of the clock to the climate of the monologue.

---

*Next chapter: The Attention Tensors*
