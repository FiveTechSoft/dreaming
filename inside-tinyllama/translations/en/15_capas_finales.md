# Chapter 15: The Final Layers (13–21)

## Integration and decision

```
Layers 13–20: global integration
Layer 21:     last transformation before output_norm
then:         lm_head → logits → sample
```

Here the residual prepares for **collapse**
to vocabulary: force VI of the atlas (softmax).

## What's at stake at the end

- Blending of themes assembled in the middle.
- Fine style preferences (formal vs simple).
- Proximity to closing tokens (`</s>`) — that's why
  sometimes baseline and mystical coincide on **very short**
  outputs with the same seed (same EOS basin).

## The mystical battery experiment

With I=0.50 and 60 max tokens, several prompts filled
the length budget; others cut off at 2–8 tokens.
Final layers + sampling decide **when to stop**
as much as **what to say**.

## Practical rule

To compare perspectives, use high `n` and look at
the **body** of the text, not just the first sentence
if the model rushes to EOS.

---

*Next chapter: Semantic Areas and the Map*
