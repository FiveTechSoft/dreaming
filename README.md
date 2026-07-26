# dreaming

A 500MB synthetic LLM built **from scratch** using reverse engineering techniques on existing LLM architectures — DeepSeek-V2 Lite — combined with dream-inspired weight generation and a pure C inference engine with zero external dependencies.

## What We Discovered

**Weight perturbation is not destruction — it's perspective selection.**

When we perturb LLM weights with hierarchy-preserving techniques, the model doesn't generate garbage. It generates **coherent, grammatically correct, factually plausible** text — but from a **completely different perspective**. The model contains thousands of latent perspectives from its training data. Perturbation acts as a filter that selects which perspective dominates.

| Prompt | Baseline | Perturbed Model |
|--------|----------|-----------------|
| "The secret to happiness is" | Autoayuda genérica (estilo ChatGPT) | Filosofía estoica / mindfulness / autenticidad |
| "Dreams are the mind's way of" | Neurociencia popular | Literatura victoriana / investigación clínica / espiritualidad |
| "Philosophy teaches us that" | Religión y sociedad | Meta-análisis crítico / fe vs. razón / filosofía de la mente |

> See `conclusiones.md` for the full analysis.

## Reverse Engineering Approach

We reverse engineered the GGUF binary format by:

1. **Parsing the GGUF specification** — magic bytes (`GGUF`), version, tensor count, KV metadata, tensor index entries with names/types/dims/offsets
2. **Decoding F16 half-precision floats** — manual IEEE 754 decoder (sign + 5-bit exponent + 10-bit mantissa)
3. **Reconstructing the transformer architecture** — reading layer names, dimensions, and compute graphs from tensor metadata
4. **Implementing the full forward pass from scratch** — no trained weights dependency, no framework dependency

The goal: understand exactly how LLMs work at the binary level, then **build one ourselves** with custom (synthetic) content.

## DMT Weight Perturbation Pipeline

Binary header copy approach — perturbs Q4_0 weights while preserving GGUF metadata:

```bash
# Generate perturbed model
python dmt_perturb_v10.py amplify     # Single technique
python dmt_perturb_v10.py all         # All 10 techniques
python dmt_perturb_v11.py combo       # Combination techniques
python dmt_perturb_v11.py selective   # Selective targeting per tensor type
python dmt_perturb_v11.py sweep       # Intensity sweep

# Test with llama.cpp
llama-cli -m <model> --temp 0.7 -n 128 -p "The secret to happiness is"
```

### The 10 Hierarchy-Preserving Techniques (v10)

| # | Technique | Key | Perspectiva dominante |
|---|-----------|-----|----------------------|
| 1 | Low-rank amplification | `lowrank` | Académica/crítica |
| 2 | Eigenvector rotation | `eigr` | Práctica/consejos |
| 3 | Spectral shift | `spectral` | Concisa/directa |
| 4 | Attention-preserving | `attpres` | Casi idéntica al baseline |
| 5 | Residual-preserving | `respres` | Introspectiva |
| 6 | Block-diagonal | `blkdiag` | Muy cercana al baseline |
| 7 | Norm-preserving rotation | `normrot` | Estoica/equilibrada |
| 8 | Gradient-aligned | `gradal` | Autenticidad/descubrimiento |
| 9 | Low-frequency DCT | `lowdct` | Conversacional/ayudante |
| 10 | Manifold-preserving | `manpres` | Autenticidad (similar a gradient) |

### Combination Techniques (v11)

| Combo | Techniques | Effect |
|-------|-----------|--------|
| `structured_dream` | block_amplify + poetic_oscillate | Perspectiva analítica/académica |
| `max.Alter` | amplify + amplify | Máxima divergencia, coherente |
| `deep_reason` | analytical + residual | ⚠️ Degradado |
| `rare_perspective` | creative + residual | ⚠️ Degradado |

### Selective Targeting (v11)

| Target | Strategy | Effect |
|--------|----------|--------|
| `attention_alter` | Heavy attention, gentle FFN | Texto más estructurado/académico |
| `ffn_dream` | Heavy FFN, gentle attention | Texto más práctico/accional |
| `embedding_shift` | Heavy embedding, gentle rest | Lenguaje más simple/accesible |
| `extreme_selective` | Max attention, minimal FFN | Espiritual/mindfulness |

### Intensity Response

| Intensity | Effect | Quality |
|-----------|--------|---------|
| 0.05 | Muy cercano al original | Demasiado fiel |
| **0.10** | **Máxima divergencia, texto coherente** | **Sweet spot** |
| 0.15 | Cercano al original, más filosófico | Ligeramente desplazado |
| 0.20 | Perspectiva diferente, más comprehensiva | Más divergente |
| 0.25+ | Calidad degradada, repetitivo | Demasiado ruido |

## Test Results

**24 modelos testados, 240 generaciones, 10 prompts**

| Category | Models | Coherent | Failed |
|----------|--------|----------|--------|
| Baseline (Q4_0) | 1 | 1 | 0 |
| v10 hierarchy-preserving | 10 | **10** | 0 |
| v11 combos | 4 | 2 | 2 |
| v11 selective | 4 | 4 | 0 |
| Old DMT models | 3 | 1 | 2 |

### Best Models by Use Case

| Use Case | Best Model | Why |
|----------|-----------|-----|
| Closest to original | `v10_attpres` | Preserves attention patterns |
| Maximum divergence | `v11_combo_max.Alter` | Double amplify_subspace |
| Philosophical depth | `v10_lowrank` | Duality framing |
| Practical advice | `v10_eigr` | Tips/action framing |
| Spiritual/mindful | `v11_select_extreme_selective` | Mindfulness framing |
| Creative fiction | `v11_combo_structured_dream` | Most distinct perspective |
| Authentic/self-discovery | `v10_gradal` | Gradient-aligned |

### Key Findings

1. **All 10 v10 techniques produce coherent output** — hierarchy preservation is the key
2. **Selective targeting creates distinct "personas"** — attention = structure, FFN = vocabulary, embedding = identity
3. **The DMT analogy is accurate** — hallucination is reorganization, not invention
4. **Flat 1D operations are more robust** than matrix reshape (which caused failures in old DMT models)

## What's in This Repo

| File | Description |
|---|---|
| `create_gguf.py` | Builds the 500MB GGUF v3 model file from scratch |
| `fill_weights.py` | Fills all tensor weight data with synthetic dream-inspired patterns |
| `create_tokenizer.py` | Generates a real BPE tokenizer (32K vocab, Llama-2 style) |
| `llm_inference.c` | Pure C inference engine (~700 lines, no dependencies) |
| `tokenizer.json` | Real BPE tokenizer (HuggingFace format, 32K vocab) |
| `dmt_perturb_v10.py` | 10 hierarchy-preserving perturbation techniques |
| `dmt_perturb_v11.py` | Combinations, selective targeting, intensity sweep |
| `conclusiones.md` | Full analysis of perspective-shifting findings |
| `test_analysis.md` | Detailed test results and comparison |
| `test_v10_v11.py` | Test script for v10 and v11 models |
| `test_all_models.py` | Comprehensive test suite (all models, all prompts) |

> **Note:** `modelo.gguf` (the 500MB binary model) is not tracked in git (see `.gitignore`). Use `create_gguf.py` to regenerate it.

### Reverse Engineering Artifacts

- `llm_inference.c` contains the complete GGUF parser we wrote by inspecting binary LLM files
- `f16_to_f32()` — manual IEEE 754 half-precision conversion we reverse engineered
- `rmsnorm()`, `matmul()`, `apply_rope()`, `masked_softmax()` — all implemented from understanding the transformer architecture

## Architecture (TinyLlama-1.1B Q4_0)

| Parameter | Value |
|---|---|
| Hidden dim | 2048 |
| Layers | 22 |
| Attention heads | 32 |
| KV heads | 4 (GQA) |
| Head dim | 64 |
| FFN dim | 5632 |
| Vocab size | 32,000 |
| Context length | 2,048 |
| Quantization | Q4_0 (4-bit) |

## Synthetic Weight Patterns

Every tensor weight was generated using dream-inspired patterns (not random — structured):

| Pattern | Concept | Effect |
|---|---|---|
| 🌈 Neon Staircase | Sinusoidal frequency shift by depth | Color-shifting waves |
| 🌲 Forest Echo | Resonance between shallow and deep layers | Peak at shallow and deep layers |
| 🪙 Floating Coin | Three-state distribution (positive, negative, zero) | Trimodal distribution |
| 🌀 Fractal Recursion | Self-similarity across scales | Fractal self-similarity |
| 💧 Lake of Silence | Sparse deep movements under stillness | 1% non-zero, mostly silence |

## Building

```bash
# C inference engine
gcc -O2 -Wall -o llm_inference llm_inference.c -lm

# Python perturbation scripts
python dmt_perturb_v10.py all --intensity 0.10
python dmt_perturb_v11.py all
```

**Zero dependencies** for C engine. Python scripts require only `numpy`.

## Running

```bash
# C inference engine
./llm_inference modelo.gguf "DREAM" 30 0.8 25
#           modelo      prompt  max  temp topk

# llama.cpp with perturbed model
llama-cli -m v10_normrot_10.gguf -p "The secret to happiness is" -n 150 --temp 0.7

# Test all models
python test_v10_v11.py
python test_all_models.py
```

## Project Status

| Phase | Goal | Status |
|---|---|---|
| 1 — Reverse Engineering | Parse GGUF format, understand binary LLM structure | ✅ Done |
| 2 — Inference Engine | Pure C forward pass, working generation loop | ✅ Done |
| 3 — Tokenizer | Real BPE vocabulary with Llama-2 chat template | ✅ Done |
| 4 — Model File | 500MB GGUF with synthetic weights | ✅ Done |
| 5 — DMT Pipeline | Binary header copy + perturbation | ✅ Done |
| 6 — Technique Exploration | Test all 10 hierarchy-preserving techniques | ✅ Done |
| 7 — Combinations | Combine multiple techniques | ✅ Done |
| 8 — Selective Targeting | Different technique per tensor type | ✅ Done |
| 9 — Perspective Analysis | Document and analyze perspective shifts | ✅ Done |
| 10 — Training | Implement SGD loop, train weights to coherence | 🔜 Planned |
| 11 — Scale to 7GB | Same pipeline on full-sized model | 🔜 Planned |
| 12 — Optimize | KV-cache, faster sampling, interactive tool | 🔜 Planned |

## Files

```
dreaming/
├── README.md              # Project overview (this file)
├── ROADMAP.md             # Detailed project road map
├── conclusiones.md        # Full analysis of perspective-shifting findings
├── dmt.md                 # DMT analogy and creative techniques mapping
├── findings.md            # Detailed experimental findings
├── create_gguf.py         # Builds the 500MB GGUF model file from scratch
├── fill_weights.py        # Fills all tensor weights with synthetic patterns
├── create_tokenizer.py    # Generates BPE tokenizer (tokenizer.json)
├── llm_inference.c        # Pure C inference engine (~700 lines, no dependencies)
├── tokenizer.json         # BPE tokenizer (HuggingFace format)
├── dmt_perturb_v10.py     # 10 hierarchy-preserving perturbation techniques
├── dmt_perturb_v11.py     # Combinations, selective targeting, intensity sweep
├── test_v10_v11.py        # Test script for v10 and v11 models
├── test_all_models.py     # Comprehensive test suite
├── test_analysis.md       # Detailed test results and comparison
├── test_v10_v11_results.json  # Full test results (v10 + v11)
├── test_results.json      # Full test results (all models)
└── .gitignore             # Excludes large .gguf binary
```

## Design Philosophy

- **Reverse engineering first** — understand the binary format of existing LLMs before building
- **No black boxes** — every math operation is explicit C code you can read line by line
- **No dependencies** — compiles on any system with a C compiler
- **Transparent weights** — synthetic patterns make the weight generation process visible and reproducible
- **Built from scratch** — parser, decoder, transformer, sampler — all custom written
- **Perspective over noise** — perturbation that preserves hierarchy creates perspective shifts, not degradation

## License

Dreaming — open project, open weights, open code.
