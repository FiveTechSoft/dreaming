# dreaming — Roadmap

## What We Built

### 1. Working LLM Inference Pipeline

- **TinyLlama-1.1B Q4_0** loaded and generating coherent text via llama.cpp
- **SentencePiece tokenizer** decoded from the GGUF binary
- **C inference engine** (`llm_inference.c`) — pure C, zero dependencies
- **Python forward pass** (`run_inference_q40.py`) — working with Q4_0 weights
- Inference speed: **97 tok/s** (llama.cpp), **~40 tok/s** (Python)

### 2. DMT Weight Perturbation Pipeline

Binary header copy approach — perturbs Q4_0 weights while preserving GGUF metadata:

- `dmt_perturb_v6.py` — 4 perturbation modes + selective targeting
- 155/201 tensors perturbed per run
- Processing time: ~15s (nibble), ~60s (float-space)

### 3. Tokenizer & GGUF Tooling

- `create_gguf.py` — GGUF builder from scratch
- `create_tokenizer.py` — SentencePiece tokenizer creator
- `fill_weights.py` — Weight initializer
- `dmt.md` — DMT analogy and 12 creative techniques mapped to weight perturbation

---

## Creative Techniques Mapped to Weight Perturbation

### Association Techniques

| Technique | Description | Weight Equivalent |
|-----------|-------------|-------------------|
| Free association (Freud) | Say first thing that comes to mind, no filtering | Lower pruning threshold, increase noise |
| Bisociation (Koestler) | Force intersection of two unrelated frames | `cross_layer_head_swap` between distant layers |
| Lateral thinking / SCAMPER | Substitute, combine, adapt, magnify, invert | `sharpen_rows` or `amplify_subspace` |

### State Alteration Techniques

| Technique | Description | Weight Equivalent |
|-----------|-------------|-------------------|
| Incubation | Leave problem, return after sleep | Apply noise offline, evaluate across passes |
| Hypnagogia (Edison/Dalí) | State just before sleep, flat hierarchies | Reduce layernorm gain in early layers |
| Open monitoring meditation | Reduce top-down prefrontal control | Lower "gain" of control layers |

### Structured Generation Techniques

| Technique | Description | Weight Equivalent |
|-----------|-------------|-------------------|
| Mental maps / semantic networks | Make explicit connections, then jump to distant nodes | Map attention patterns, swap distant heads |
| Forced analogy (Synectics) | "How would nature/music/a child solve this?" | Remap solution through alien domain weights |
| Random constraint injection | Introduce random word/rule, force connection | `random_lowrank_inject` |

### Pharmacological Frontier

| Technique | Description | Weight Equivalent |
|-----------|-------------|-------------------|
| Psychedelic-assisted therapy | Greater signal entropy, cross-network connectivity | `cross_layer_head_swap` between distant layers |

---

## The 12 Perturbation Techniques

### Implemented in `dmt_perturb_v6.py`

| # | Technique | Mode | Description |
|---|-----------|------|-------------|
| 1 | Nibble flip | `nibble_flip` | Direct 4-bit value perturbation (no float roundtrip) |
| 2 | Scaled noise | `scaled_noise` | Float-space noise with Q4_0 requantization |
| 3 | Row shuffle | `row_shuffle` | Permutes weight rows within blocks |
| 4 | Amplify subspace | `amplify_subspace` | Projects weights onto random subspace |

### Additional Techniques (from creative mapping)

| # | Technique | Function | Effect |
|---|-----------|----------|--------|
| 5 | Attention head reweighting | `reweight_attention_heads()` | Some heads "shout", others "whisper" |
| 6 | DCT frequency distortion | `dct_distort()` | Amplifies high-frequency micro-patterns |
| 7 | Row sharpening | `sharpen_rows()` | Exaggerates dominant features, flattens rest |
| 8 | Cross-layer head swap | `cross_layer_head_swap()` | One layer "borrows" attention pattern from distant layer |
| 9 | Low-rank injection | `random_lowrank_inject()` | Injects random low-rank structure |
| 10 | LayerNorm gain reduction | Reduce layernorm gain | Reduces top-down control |
| 11 | Cross-layer mixing | `mix_layers()` | Blends weights between distant layers |
| 12 | Embedding perturbation | Perturb token_embd | Remaps token representations |

---

## Experimental Results

### Prompt: "The meaning of life is"

| Mode | Intensity | Target | Result |
|------|-----------|--------|--------|
| Original | — | — | "The word 'meaning' in the phrase 'the meaning of life' is often interpreted as referring to a fundamental purpose..." |
| nibble_flip | 0.10 | attention | Fragmented tokens with word fragments ("calculusterstable", "complete", "anime") |
| nibble_flip | 0.10 | ffn | Multilingual fragmented text ("clonerice", "surrounded", "error") |
| row_shuffle | 0.10 | all | Code tokens mixed with Chinese characters ("Jenkins", "Outlet", "DateTime") |
| **amplify_subspace** | **0.10** | **all** | **"The meaning of life is" is the title of a poem by William Stafford, published in his collection "Fish and Wild"...** |
| amplify_subspace | 0.05 | all | Slightly altered, mostly faithful to original |
| amplify_subspace | 0.15 | all | More distorted, less coherent |
| amplify_subspace | 0.20 | all | Significantly degraded |

### Key Finding

**`amplify_subspace` at intensity 0.10** produces the most interesting results:
- Text is grammatically correct and coherent
- Content is **completely different** from the original model
- No external content added — just reorganized internal knowledge
- Closest analog to the DMT hypothesis: "hallucination is real information from the system, but recombined in disordered ways"

---

## What Is Actually Being Done Now

We are in the **"DMT perturbation pipeline works"** phase:

- ✅ Binary header copy preserves GGUF metadata perfectly
- ✅ 4 perturbation modes implemented and tested
- ✅ Selective targeting (attention, ffn, embedding, all layers)
- ✅ `amplify_subspace` produces coherent-but-altered text
- 🔄 Exploring intensity sweet spots for each mode
- 🔄 Testing combinations of techniques

### What Comes Next

| Phase | Goal | Status |
|---|---|---|
| **Phase 1 — Working Inference** ✅ | GGUF parser + forward pass | Done |
| **Phase 2 — DMT Pipeline** ✅ | Binary header copy + perturbation | Done |
| **Phase 3 — Technique Exploration** 🔄 | Test all 12 techniques at various intensities | In Progress |
| **Phase 4 — Combinations** | Combine multiple techniques (bisociation + incubation) | Planned |
| **Phase 5 — Layer-Specific Tuning** | Find optimal layers for each technique | Planned |
| **Phase 6 — Evaluation Framework** | Systematic comparison of outputs | Planned |
| **Phase 7 — Creative Writing** | Use DMT-perturbed models for creative text generation | Planned |

---

## Project Structure

```
dreaming/
├── README.md              # Project overview
├── ROADMAP.md             # This file
├── dmt.md                 # DMT analogy + 12 creative techniques
├── dmt_perturb_v6.py      # Multi-mode perturbation script
├── dmt_perturb_nibble.py  # Nibble-level perturbation
├── dmt_perturb_binary.py  # Binary header copy approach
├── llm_inference.c        # C inference engine
├── create_gguf.py         # GGUF builder
├── create_tokenizer.py    # Tokenizer creator
├── fill_weights.py        # Weight initializer
└── .gitignore             # Excludes large .gguf files
```

---

## Design Philosophy

- **Oniric by design** — weights come from dream patterns, not random initialization
- **DMT-inspired** — perturbation that preserves system identity but reorganizes associations
- **Binary-first** — work directly with GGUF format, no framework dependencies
- **Empirical** — test every technique, compare outputs side by side
- **Transparent** — every operation is readable Python/C code

---

## Compilation & Testing

### Prerequisites

- Python 3.10+ with numpy, gguf
- llama.cpp (pre-built or compiled)
- GCC or any C11 compiler (for C inference engine)

### Run DMT Perturbation

```bash
# Generate perturbed model
python dmt_perturb_v6.py input.gguf output.gguf amplify_subspace 0.10 all

# Test with llama.cpp
llama-cli -m output.gguf -p "The meaning of life is" -n 100 --temp 0.7
```

### Build C Engine

```bash
gcc -O2 -Wall -o llm_inference llm_inference.c -lm
./llm_inference modelo.gguf "DREAM" 30 0.8 25
```
