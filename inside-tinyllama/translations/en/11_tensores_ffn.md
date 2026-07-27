# Chapter 11: The FFN Tensors

## Three planets of ordinary matter

| Tensor | Role | Logical shape |
|--------|------|---------------|
| **Gate** (ffn_gate) | SiLU gate | [5632, 2048] |
| **Up** (ffn_up) | Expansion | [5632, 2048] |
| **Down** (ffn_down) | Compression | [2048, 5632] |

Plus `ffn_norm` before the block.

## SwiGLU

```
h' = Down( SiLU(Gate(x)) ⊙ Up(x) )
x  = x + h'
```

Intermediate dimension **5632**: the residual expands
to a wider space and returns to 2048.

## Dominant mass

~**69%** of the model's parameters live here.
If attention is gravity between planets, the FFN is
the **internal physics** of each one.

## Golden Rule

Perturbing FFN → **practical** perspective:
steps, advice, action verbs, "how-to".

Selective `ffn_dream` (v11): strong creative in FFN,
gentle in attention → "dreamy but actionable" climate.

## What to observe

- Numbered lists, imperatives, tips?
- Less "who relates to whom" and more "what to do"?
→ FFN field in the driver's seat.

---

*Next chapter: The Normalization Tensors*
