# Introduction: Inside TinyLlama

## A microcosmos that fits on a disk

This book is the logbook of the **Dreaming** project
applied to **TinyLlama-1.1B**: a model
small enough to open entirely and rich enough
to be surprising.

It is not a user manual for a chat.
It is a journey through the **interior** of a transformer:

- its architecture (22 layers, 9 tensors per layer),
- an inference engine in C that we can read line by line,
- weight perturbation as a change of *perspective*,
- the geometry of the embedding space,
- the "forces" of the forward pass (attention, FFN, residual, softmax),
- and the back-and-forth between the **macrocosmos** of human meaning
  and the **microcosmos** of numbers.

## The central question

> When we move the weights carefully,
> does the model break or does it speak with another voice?

The empirical answer: **it speaks with another voice**,
if the perturbation preserves the internal hierarchy
of the weights. We call that navigating the
*coherence surface*.

## How the book is organized

| Part | Ch. | Topic |
|------|-----|-------|
| I · Foundations | 1–3 | What is TinyLlama, structure, C engine |
| II · Perspectives | 4 | DMT perturbation, techniques, runtime |
| III · Geometry | 5–6 | Multidimensional space, macro↔micro |
| IV · Physics of the microcosmos | 7–9 | Forces, travel, Golden Rule |
| V · Anatomy | 10–12 | Attention, FFN, normalization |
| VI · Layers and closing | 13–16 | Layers 0–21, psychoanalysis, lessons, future |

## Tools of the journey

- `llm_inference.c` — F16 inference, KV-cache, `--perturb`, `--steer`
- `dmt_perturb_v10.py` / `v11` — perturbed Q4_0 GGUFs
- `map_semantic_areas.py` — atlas of semantic islands
- [HTML map on GitHub](https://htmlpreview.github.io/?https://raw.githubusercontent.com/FiveTechSoft/dreaming/main/inside-tinyllama/exploration/semantic_map.html)
- `llama-cli` — quick batches on Q4_0

## A promise

By the end of the book you won't have a bigger model.
You'll have a **map** and a **method**: go down from meaning
to tensor, go up from tensor to voice, and note the path.

---

*Chapter 1: What is TinyLlama?*
