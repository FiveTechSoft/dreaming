# Chapter 18: What We Learned

## Key findings

1. **TinyLlama is a mappable microcosm**
   22 layers, 9 tensors/layer, actual dims 2048 / 5632 / GQA 32×4.

2. **A custom C engine closes the loop**
   GGUF F16, BPE, KV-cache, OpenMP, ~6–10 tok/s,
   `--perturb` and `--steer` at runtime.

3. **Weights contain perspectives**
   Not just facts: tones and voices. Perturbing with
   preserved hierarchy changes voice, doesn't silence speech.

4. **Geometric Golden Rule**
   Attn → academic; FFN → practical; Emb → simple.

5. **Coherence surface**
   Tangent (amplify) habitable; normal (strong noise) empty.

6. **Embedding space: islands, not a single axis**
   Twelve nearly orthogonal semantic areas; PCA uses
   hundreds of dimensions; opposites not antipodal.

7. **Macrocosm ↔ microcosm**
   The method is back and forth: meaning ↔ tensor ↔ text.

8. **Travel tools**
   HTML map on GitHub, geometry scripts, llama-cli
   for Q4 batteries, C engine for fine clockwork.

## Study limitations

- "Perspective" evaluation still qualitative.
- TinyLlama ≠ frontier models (the surface may change).
- 2D map is a projection, not true geometry.
- F32 perturbation runtime requires lots of RAM.
- Not all v10/v11 techniques are in the C engine.

## Open questions

- Where (which layers) does mystical climate ignite in the residual?
- Do perspective directions transfer between models?
- How to measure perspective automatically and reliably?
- What happens on the coherence surface at 7B / 70B?

---

*Next chapter: The Future of Exploration*
