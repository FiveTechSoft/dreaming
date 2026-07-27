# Chapter 8: How to Travel Through the TinyLlama Universe

## Travel modes

| Mode | What you traverse | Tool |
|------|-------------------|------|
| Observatory | Token sky ℝ²⁰⁴⁸ | HTML map, geometry scripts |
| Orbit | One prompt → token sequence | `llm_inference` / llama-cli |
| Lenses | Perspectives in weight space | `--perturb` |
| Currents | Live residual | `--steer` |
| Lab | Only attn or only FFN | Q4 scripts / v11 models |

## Flight kit

```bash
gcc -O3 -march=native -ffast-math -fopenmp \
    -o llm_inference llm_inference.c -lm
# $env:OMP_NUM_THREADS = "8"   # Windows
```

Interactive map:
https://htmlpreview.github.io/?https://raw.githubusercontent.com/FiveTechSoft/dreaming/main/inside-tinyllama/exploration/semantic_map.html

Model: `tinyllama-1.1b.F16.gguf` (weights) + Q4_0 (tokenizer if needed).

## Route A — Atlas

```bash
python map_semantic_areas.py
python explore_tinyllama_space.py
```

Locates islands (emotion, spiritual, tech…) and selects
words for prompts or `--steer`.

## Route B — Baseline orbit

```bash
./llm_inference tinyllama-1.1b.F16.gguf \
  "The secret to happiness is" 60 0.7 40 --seed 42
```

Fix seed and temp when comparing worlds.

## Route C — Perspective lens

```bash
./llm_inference tinyllama-1.1b.F16.gguf \
  "The secret to happiness is" 60 0.7 40 \
  --seed 42 --perturb mystical --intensity 0.50
```

| Technique | Typical effect |
|-----------|----------------|
| mystical / amplify | Existential / philosophical |
| noise (low I) | Subtle; high I → garbage |
| blockdiag | Echo / monotonous |
| manifold | Local; high I risky |

Protocol: one prompt, `none` → same seed, vary I or technique.

## Route D — Currents

```bash
./llm_inference tinyllama-1.1b.F16.gguf \
  "The world is" 50 0.7 40 \
  --seed 42 --steer amor --steer-strength 0.15
```

## Route E — Precomputed GGUFs

```bash
llama-cli -m v10_lowrank_10.gguf \
  -p "The secret to happiness is" -n 80 --temp 0.7 --seed 42
```

## Itineraries

1. **Tourist (30 min):** map + 3 prompts baseline/mystical.
2. **I cartographer:** I ∈ {0, 0.1, 0.3, 0.5}, same seed.
3. **Attractor hunt:** ego, death, consciousness, power.
4. **Double ship:** C engine F16 vs llama-cli Q4.

## Navigation laws

1. One variable per jump.
2. Fixed seed when comparing.
3. Moderate I first.
4. Garbage ⇒ you left the surface.
5. The 2D map orients; it's not the real space.
6. Mystical setup: ~25 s + ~3.6 GB F32.

## Diagram

```
prompt → embedding → 22×{attn, FFN} → logits → token
               ↑              ↑
            --steer      weights (+ --perturb)
```

---

*Next chapter: The Geometric Golden Rule*
