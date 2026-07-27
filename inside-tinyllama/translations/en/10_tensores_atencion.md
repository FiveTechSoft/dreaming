# Chapter 10: The Attention Tensors

## Four planets per layer

In each of the 22 layers:

| Tensor | Question | Logical shape (TinyLlama) |
|--------|----------|---------------------------|
| **Q** (attn_q) | What am I looking for? | [2048, 2048] |
| **K** (attn_k) | What do I offer? | [256, 2048] (4×64) |
| **V** (attn_v) | What do I transmit? | [256, 2048] |
| **O** (attn_output) | How do I integrate? | [2048, 2048] |

Plus `attn_norm` (RMSNorm before the block).

## GQA: 32 eyes, 4 memories

TinyLlama doesn't have 32 independent K and 32 V.
It has **32 Q heads** and **4 shared KV**
(each KV serves 8 Q). Less cache memory,
same multi-head idea.

## The formula

```
scores = (Q Kᵀ) / √64
weights = softmax_causal(scores)
out = weights V
out = O · out
x = x + out          # residual
```

In the C engine: only the new token computes Q/K/V;
K and V are stored in the **KV-cache**.

## Role in the universe

- **Long-range force** between tokens.
- **Golden Rule:** touch attention → academic perspective.
- In the force atlas: ~19% of mass, maximum *range*.

## What to observe when experimenting

- Does the text cite, structure, "argue"?
- Does it change the *relationship* between ideas more than loose vocabulary?
→ Signal that the attentional field dominates the climate.

---

*Next chapter: The FFN Tensors*
