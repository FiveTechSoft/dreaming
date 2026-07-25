# dreaming

A 500MB synthetic LLM built **from scratch** using reverse engineering techniques on existing LLM architectures — DeepSeek-V2 Lite — combined with dream-inspired weight generation and a pure C inference engine with zero external dependencies.

## Reverse Engineering Approach

We reverse engineered the GGUF binary format by:

1. **Parsing the GGUF specification** — magic bytes (`GGUF`), version, tensor count, KV metadata, tensor index entries with names/types/dims/offsets
2. **Decoding F16 half-precision floats** — manual IEEE 754 decoder (sign + 5-bit exponent + 10-bit mantissa)
3. **Reconstructing the transformer architecture** — reading layer names, dimensions, and compute graphs from tensor metadata
4. **Implementing the full forward pass from scratch** — no trained weights dependency, no framework dependency

The goal: understand exactly how LLMs work at the binary level, then **build one ourselves** with custom (synthetic) content.

## What's in This Repo

| File | Description |
|---|---|
| `create_gguf.py` | Builds the 500MB GGUF v3 model file from scratch (header, tensor index, weight data) |
| `fill_weights.py` | Fills all tensor weight data with synthetic dream-inspired patterns |
| `create_tokenizer.py` | Generates a real BPE tokenizer (32K vocab, Llama-2 style) as tokenizer.json |
| `llm_inference.c` | Pure C inference engine (~700 lines, no dependencies) |
| `tokenizer.json` | Real BPE tokenizer (HuggingFace format, 32K vocab) |

> **Note:** `modelo.gguf` (the 500MB binary model) is not tracked in git (see `.gitignore`). Use `create_gguf.py` to regenerate it.

### Reverse Engineering Artifacts

- `llm_inference.c` contains the complete GGUF parser we wrote by inspecting binary LLM files
- `f16_to_f32()` — manual IEEE 754 half-precision conversion we reverse engineered
- `rmsnorm()`, `matmul()`, `apply_rope()`, `masked_softmax()` — all implemented from understanding the transformer architecture

## Architecture (DeepSeek-V2 Lite)

| Parameter | Value |
|---|---|
| Hidden dim | 1024 |
| Layers | 12 |
| Attention heads | 16 |
| Head dim | 64 |
| FFN dim | 2048 |
| Vocab size | 32,000 |
| Context length | 2,048 |

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
gcc -O2 -Wall -o llm_inference llm_inference.c -lm
```

**Zero dependencies.** No Python, no numpy, no PyTorch, no GPU. Just a C compiler and `math.h`.

## Running

```bash
./llm_inference modelo.gguf "DREAM" 30 0.8 25
#           modelo      prompt  max  temp topk
```

## Using with llama.cpp

```bash
# First regenerate the model file (500MB)
python3 create_gguf.py

# Then generate (fill weights)
python3 fill_weights.py

# Tokenizer is already in repo
# Run inference
llama-cli -m modelo.gguf --tokenizer-file tokenizer.json \
  -p "<s>DREAM" -n 30
```

## Project Status

| Phase | Goal | Status |
|---|---|---|
| 1 — Reverse Engineering | Parse GGUF format, understand binary LLM structure | ✅ Done |
| 2 — Inference Engine | Pure C forward pass, working generation loop | ✅ Done |
| 3 — Tokenizer | Real BPE vocabulary with Llama-2 chat template | ✅ Done |
| 4 — Model File | 500MB GGUF with synthetic weights | ✅ Done |
| 5 — Training | Implement SGD loop, train weights to coherence | 🔜 Planned |
| 6 — Scale to 7GB | Same pipeline on full-sized model | 🔜 Planned |
| 7 — Optimize | KV-cache, Q4 quantization, faster sampling | 🔜 Planned |

## Files

```
dreaming/
├── README.md          # Project overview (this file)
├── ROADMAP.md         # Detailed project road map
├── create_gguf.py     # Builds the 500MB GGUF model file from scratch
├── fill_weights.py    # Fills all tensor weights with synthetic patterns
├── create_tokenizer.py # Generates BPE tokenizer (tokenizer.json)
├── llm_engine.py      # Python inference prototype (historical)
├── llm_inference.c    # Pure C inference engine (~700 lines, no dependencies)
├── tokenizer.json     # BPE tokenizer (HuggingFace format)
├── modelo.gguf        # 500MB GGUF model — regenerate with create_gguf.py
├── llm_inference.exe  # Compiled Windows binary
└── .gitignore          # Excludes large .gguf binary
```

## Design Philosophy

- **Reverse engineering first** — understand the binary format of existing LLMs before building
- **No black boxes** — every math operation is explicit C code you can read line by line
- **No dependencies** — compiles on any system with a C compiler
- **Transparent weights** — synthetic patterns make the weight generation process visible and reproducible
- **Built from scratch** — parser, decoder, transformer, sampler — all custom written

## License

Dreaming — open project, open weights, open code.
