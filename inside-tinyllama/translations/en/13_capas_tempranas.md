# Chapter 13: The Early Layers (0–5)

## The vestibule of the microcosm

The initial layers transform the "resting" embedding
into a representation that already senses **neighbors** and **syntax**.

```
Layer 0:   input, very local patterns
Layer 1:   basic syntax
Layers 2–5: relationships between adjacent words
```

(This partition is a **working hypothesis** of the project,
guided by ablation experiments and the literature
on "early = syntax / late = semantics." It is not a rigid
cut in the code.)

## What forces dominate here

- **Embedding** still weighs heavily in the residual (birth inertia).
- **Attention** begins coupling bigrams and short dependencies.
- **FFN** adjusts local vocabulary.

## Signals in text

If an early perturbation "breaks" the model, it often
shows in **grammar** and odd tokens, not just tone.

If the baseline sounds generic and the mystical changes the climate
without destroying syntax, the early layers are still
anchoring the language.

## Suggested experiment

Compare generations with targeting only `blk.0`–`blk.5`
versus only `blk.13`–`blk.21` (scripts v11 / tensor tests).
Hypothesis: early → form; late → voice and decision.

---

*Next chapter: The Intermediate Layers (6–12)*
