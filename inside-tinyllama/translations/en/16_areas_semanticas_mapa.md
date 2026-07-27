# Chapter 16: Semantic Areas and the Map

## Twelve islands in ℝ²⁰⁴⁸

| Key | Label | Seeds (examples) |
|-----|-------|------------------|
| emotion_pos | Positive emotion | happy, joy, love, peace… |
| emotion_neg | Negative emotion | sad, hate, fear, anger… |
| spiritual | Spiritual / sacred | soul, god, faith, divine… |
| physical | Physical / material | body, rock, water, fire… |
| abstract | Abstract / ideas | truth, beauty, justice… |
| time | Time | time, past, future, now… |
| social | Social / power | king, war, law, people… |
| nature | Nature | tree, river, mountain… |
| mind | Mind / cognition | mind, think, dream, brain… |
| death_life | Life / death | life, death, born, die… |
| tech | Technical / digital | computer, data, code… |
| body_sense | Body / senses | eye, hand, see, voice… |

## Geometry between islands

- Centroids of distinct areas: cosine **≈ 0** (orthogonal).
- Most aligned: **abstract ↔ mind** (+0.13).
- Most separated: **time ↔ social** (−0.09).
- Lexical opposites (love/hate): **not antipodal**.

## Vocabulary coverage

~99% of random tokens don't fall near any island
(BPE = fragments). The areas are **constellations of
full words**, not a total partition of the vocab.

## Visualization

**Interactive 2D PCA map (GitHub):**
https://htmlpreview.github.io/?https://raw.githubusercontent.com/FiveTechSoft/dreaming/main/inside-tinyllama/exploration/semantic_map.html

| File | Use |
|------|-----|
| `semantic_map.html` | Zoom/pan/hover |
| `semantic_areas.json` | Area data |
| `vectors.tsv` + `metadata.tsv` | TensorFlow Projector (local) |

### Other tools

UMAP, t-SNE, plotly 3D, Embedding Projector.

## How to use it when traveling

1. Choose an island on the map.
2. Build a prompt or `--steer` with seeds from that island.
3. Compare baseline vs mystical.
4. Note if the text "falls" into the island's climate.

---

*Next chapter: Psychoanalysis of the Transformer*
