# Chapter 19: The Future of Exploration

## Next technical steps

1. **Instrument the residual per layer** in the C engine
   (L0…L21 probes on emotion/spirit axes).
2. **UMAP/t-SNE** of the token sky (when the stack allows it).
3. **Port residual / gradient / selective** to `--perturb`.
4. **Release GGUF F16** of layers after copying to F32 (less RAM).
5. **GitHub Pages** native for the map (no htmlpreview).
6. Repeat the cartography on **another model** (transfer).

## Next book steps

- Fixed figures (PNG) of the map and force diagram.
- Appendix with the complete 15 mystical prompts table.
- Unified glossary (GQA, surface, Golden Rule, I).

## Invitation

If you're reading this with the repo open:

```bash
# 1. Look at the sky
#    exploration/semantic_map.html  (or the htmlpreview link)

# 2. Fire up the ship
gcc -O3 -fopenmp -o llm_inference llm_inference.c -lm
./llm_inference modelo.F16.gguf "When we dissolve the ego" \
  40 0.7 40 --seed 42 --perturb mystical --intensity 0.5

# 3. Note which voice came out
```

The microcosm fits on a disk.
The macrocosm is the question that brought you here.
The path between them is the craft of Dreaming.

**Keep cartographing.**

---

*Next chapter: How This Universe Orbits.*
