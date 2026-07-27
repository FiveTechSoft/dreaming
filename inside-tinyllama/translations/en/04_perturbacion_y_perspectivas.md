# Chapter 4: Weight Perturbation and Perspective Change

## Beyond using the model

So far we have learned to *run* TinyLlama.
We know how it is built and how to write an engine
that makes it speak.

But the heart of the Dreaming project is another question:

> What happens if we **change** the model's weights?

Not to retrain it. Not to correct it.
Just to move it slightly in its weight space
and observe whether it keeps speaking, but differently.

The answer, after many experiments, is surprising:
**it keeps speaking, and it does so from a different perspective.**

## What is weight perturbation?

A language model is a huge list of numbers.
In TinyLlama-1.1B there are over a billion.
Those numbers, organized in tensors, are the "weights"
the model acquired during training.

Perturbing weights means carefully modifying those numbers.
It's like slightly turning the dials of a radio:
if you do it well, you keep hearing music, but the station changes.

In our case we work with TinyLlama quantized in **Q4_0**:
each block of 32 weights is compressed to 18 bytes
(2 bytes for the scale + 16 bytes with 4-bit nibbles).

The pipeline is simple in concept:

```
1. Read the original GGUF byte by byte
2. Copy the header untouched (preserves the tokenizer)
3. Unpack Q4_0 blocks to float32
4. Apply a perturbation technique
5. Quantize back to Q4_0
6. Write the new GGUF
```

The key is in step 4: **not every modification is equal**.
Some destroy the model; others make it speak
with another voice.

## The DMT analogy

We call this work "DMT perturbation" because the effect
recalls the classic hypothesis about altered states:

> Hallucination is not invention. It is real information from the system,
> reorganized in its way of combining.

When we perturb TinyLlama, the model doesn't invent words
it never saw. It reorganizes the associations it already had.
It's as if we awakened a latent personality that was always
there, silenced by the original configuration.

The model is still TinyLlama. But now it "dreams"
from a different angle.

## The 10 hierarchy-preservation techniques

The first perturbations we tried were pure noise,
row swapping, nibble inversion. Most produced garbage:
strange characters, nonsensical loops, words that don't exist.

But we discovered something important: **techniques that preserve
the internal hierarchy of weights maintain coherence**.
The absolute value of each weight matters less;
what matters is its relative relationship to the others.

We tried ten techniques that respect that hierarchy:

| # | Technique | Key | Dominant perspective |
|---|-----------|-----|----------------------|
| 1 | Low-rank amplification | `lowrank` | Academic / critical |
| 2 | Eigenvector rotation | `eigr` | Practical / advice |
| 3 | Spectral shift | `spectral` | Concise / direct |
| 4 | Attention-preserving | `attpres` | Nearly identical to original |
| 5 | Residual-preserving | `respres` | Introspective |
| 6 | Block-diagonal | `blkdiag` | Very close to original |
| 7 | Norm-preserving rotation | `normrot` | Stoic / balanced |
| 8 | Gradient-aligned | `gradal` | Authenticity / discovery |
| 9 | Low-frequency DCT | `lowdct` | Conversational / assistant |
| 10 | Manifold-preserving | `manpres` | Authenticity (similar to gradient) |

All these techniques produced coherent text.
Not always correct, not always factual, but grammatically
valid and with clear intent.

## How the pipeline works

The script `dmt_perturb_v10.py` implements the process:

```bash
# Generate a perturbed model with a technique
python dmt_perturb_v10.py lowrank --intensity 0.10
```

Internally:

1. Reads the original GGUF (`tinyllama-1.1b-q4_0.gguf`)
2. Copies the header and metadata intact
3. Iterates over each weight tensor
4. Unpacks the Q4_0 blocks
5. Applies the chosen technique with a given intensity
6. Re-quantizes to Q4_0
7. Writes the perturbed file (`v10_lowrank_10.gguf`)

The `--intensity` parameter controls how much the model moves.
A value too low changes nothing; one too high
destroys coherence.

## The sweet spot: intensity 0.10

We tested multiple intensities with all techniques.
The result was consistent:

| Intensity | Effect | Quality |
|-----------|--------|---------|
| 0.05 | Very close to original | Too faithful |
| **0.10** | **Maximum divergence, coherent text** | **Sweet spot** |
| 0.15 | Close to original, more philosophical | Slightly shifted |
| 0.20 | Different perspective, more comprehensive | More divergent |
| 0.25+ | Degraded quality, repetitive | Too much noise |

At intensity 0.10 the model diverges as much as possible
without breaking. It is the point where perturbation stops being
an echo of the original and becomes its own voice.

## Direct comparison: same prompt, different perspective

The most striking effect is seen when using the same prompt
across different perturbed models.

### Prompt: "The secret to happiness is"

| Model | Perspective | Response start |
|-------|-------------|----------------|
| Baseline | Generic self-help | "...cultivating a mindset that is focused on gratitude..." |
| `v11_select_extreme` | Spiritual / mindfulness | "...finding inner peace and contentment through mindfulness..." |
| `v10_lowrank` | Philosophical / academic | "...the phrase is an idiom used to express the idea that finding true inner peace..." |
| `v10_normrot` | Stoic | "...finding the right balance between our inner and outer lives." |
| `v10_gradal` | Authenticity | "...finding your own unique and authentic way of living..." |

### Prompt: "Dreams are the mind's way of"

| Model | Perspective | Response start |
|-------|-------------|----------------|
| Baseline | Popular neuroscience | "...processing and storing information..." |
| `v11_select_attention` | Victorian literature | "Dr. Jekyll and Mr. Hyde is a play by Robert Louis Stevenson..." |
| `v10_eigr` | Spiritual self-help | "Dr. M. A. S. S. is an acronym for 'Dreams Are Mind's Way.'..." |
| `v10_lowrank` | Clinical research | "...a study published in the Journal of Sleep Research..." |

The model doesn't lose linguistic ability. It only changes
register, style, and attitude.

## The main findings

After 24 tested models, 240 generations, and 10 prompts,
these are the main findings:

### 1. Weights contain perspectives, not just information

TinyLlama was trained on texts from many authors,
styles, and disciplines. All those modes of speaking were
engraved in the weights. Perturbation selects which
of those voices dominates.

### 2. Hierarchy matters more than absolute values

Techniques that destroy the hierarchical structure
generate garbage. Those that preserve it generate coherent text.
What matters is not how much each weight changes, but
**how they change relative to each other**.

### 3. Each component controls a different aspect

| Component | What it controls |
|-----------|-----------------|
| Attention | Narrative structure, relationships between tokens |
| FFN | Vocabulary, word choice, practical knowledge |
| Embeddings | Conceptual identity, simplicity of language |

Perturbing only attention produces more structured texts.
Perturbing only FFN changes vocabulary and focus.
Perturbing only embeddings simplifies language.

### 4. The DMT analogy is quantifiable

The model doesn't invent new content. It reorganizes
internal associations. "Hallucination" is reorganization,
not invention.

### 5. The angle matters, but the magnitude matters more

Mathematically, a perturbation can be nearly orthogonal
to the original model and still work, as long as its
magnitude is small. It's like taking a millimeter step
in a perpendicular direction: technically you change course,
but you're still on the same mountain.

## The formula for perspective change

We can summarize the phenomenon in a simple formula:

```
Perspective = Base + epsilon * delta

where:
  epsilon = intensity (typically 0.05 - 0.15)
  delta   = direction in weight space
  |delta| = magnitude of change
```

If `epsilon` is small and `delta` preserves the hierarchy:
- Coherence is maintained
- Perspective changes

If `epsilon` is large or `delta` destroys the hierarchy:
- Coherence is lost
- Garbage appears

This also answers a practical question:
do we need a different model for each style?

**No.** With a base model and a set of precomputed
directions we can interpolate styles in real time:

```python
styled = base + 0.05 * delta_philosophical + 0.03 * delta_stoic
```

Linear interpolation of nearby points on the "variety
of coherence" produces other valid points.

## Implications

### For creativity
Each technique is a different "tone." The same theme can
be generated from multiple angles without retraining anything.

### For interpretability
Perturbation is a probing tool: it tells us
which parts of the model control which aspects of style.

### For personalization
Instead of expensive fine-tuning, a light perturbation
can be applied to adapt the response style.

### For AI philosophy
An LLM is not a question-answering machine.
It is a **perspective ecosystem compressed into weights**.
Perturbation is a way to navigate that ecosystem.

## Runtime perturbation (C engine)

Besides generating Q4_0 GGUFs with Python, the
`llm_inference.c` engine applies techniques **in memory**
on F16 weights, without an intermediate file:

```bash
./llm_inference tinyllama-1.1b.F16.gguf \
  "The secret to happiness is" 60 0.7 40 \
  --seed 42 --perturb mystical --intensity 0.50
```

| Flag | Techniques |
|------|------------|
| `--perturb` | `none`, `mystical`/`amplify`, `noise`, `blockdiag`, `manifold` |
| `--intensity` | strength (in F32 higher I is needed than in Q4 to notice the effect) |
| `--seed` | reproducibility |
| `--steer` | pushes the residual toward the embedding of a word |

`mystical` = `amplify_subspace` (projection + amplification).
Copies ~3.6 GB to F32 once (~25 s) and then generates at ~6–10 tok/s.

Batch of 15 prompts with I=0.50 (seed 42): average ~8.2 tok/s;
texts with existential mood on prompts like
*When we dissolve the ego*, *The soul remembers*,
*The ancient wisdom teaches that*.

## Combinations and targeting (v11)

| Family | Idea | Examples |
|--------|------|----------|
| Combos | Stack two techniques | deep_reason, rare_perspective, structured_dream |
| Selective | Different technique per attn / ffn / emb | attention_alter, ffn_dream, extreme_selective |
| Sweep of I | Find the breaking point | 0.05 … 0.50 |

## Honest limitations

- Results vary depending on the prompt.
- Some technique combinations degrade quality.
- Not every large model will respond the same: the structure
  of the variety of coherence may change with scale.
- Evaluation is qualitative: measuring "perspective"
  remains an open problem.
- In F16 runtime, I=0.10 sometimes doesn't move short outputs
  (early EOS); I=0.3–0.5 shows the change more clearly.

## Conclusion

Perturbing weights is not vandalizing a model.
It is discovering that within the same set of numbers
many voices live.

TinyLlama, seen this way, stops being a single tool
and becomes a **landscape of possibilities**.
Each technique is a path through that landscape. Each intensity
is a speed. And the sweet spot (near 0.10 in Q4,
slightly higher in F32 runtime) is the exact point where the model
remains itself, but speaks from another place.

The next chapter explores the **multidimensional space**
where those voices live: embeddings, residual, weights, and perspectives.

---

*Next chapter: Journey Through the Multidimensional Space*
