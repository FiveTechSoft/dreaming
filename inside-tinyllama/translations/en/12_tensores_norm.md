# Chapter 12: The Normalization Tensors

## Two brakes per layer (+ one final)

| Tensor | Where | Function |
|--------|-------|----------|
| **attn_norm** | Before QKV | Stabilizes input to attention |
| **ffn_norm** | Before gate/up | Stabilizes input to FFN |
| **output_norm** | After layer 21 | Stabilizes before lm_head |

## RMSNorm (not classic LayerNorm)

```
rms = sqrt(mean(x²) + ε)
out = (x / rms) * w
```

Without subtracting the mean: only scales by vector energy.

## Minimum mass, total effect

~**0.01%** of parameters. Without these norms,
attention + FFN push the residual toward explosive
norms or numerical collapse.

In the force atlas: **cosmological constant /
breathable air** of the microcosm.

## Perturbation policy

`dmt_perturb_v10` and the C engine **do not touch** norms
when applying mystical: moving stability is the
shortest path to numerical garbage.

## Practical rule

If text breaks with weird symbols after an
experiment, check if you touched norms or excessive I
before blaming "semantics."

---

*Next chapter: The Early Layers (0–5)*
