# dreaming — Roadmap

## What We Built

### 1. GGUF Model File (500 MB)

A valid GGUF v3 file with DeepSeek-V2 Lite architecture:

- **Config**: 12 layers, 16 heads, 1024 hidden dim, 2048 FFN dim, 32K vocab
- **112 tensors** with proper offsets and metadata
- **Weights initialized** using 5 synthetic (dream-inspired) patterns:
  - Neon Staircase — sinusoidal frequency drift by depth
  - Forest Echo — resonance between shallow and deep layers
  - Floating Coin — three-state distribution (positive, negative, zero)
  - Fractal Recursion — self-similar patterns across scales
  - Lake of Silence — sparse latent movements under stillness

### 2. Tokenizer (tokenizer.json)

A real BPE tokenizer in HuggingFace format:

- **32,000 vocab entries** (IDs 0–31999)
- Byte-level base tokens (raw bytes 0–255)
- 16,384 BPE merge operations
- Llama-2 style chat template
- Special tokens: `<unk>`, `<s>`, `</s>`, `<pad>`

### 3. C Inference Engine (llm_inference.c)

Pure C with **zero external dependencies** (no Python, no numpy, no GPU):

- Full GGUF parser (reads binary format directly)
- Manual F16 → F32 half-precision decoder
- Transformer forward pass: embeddings, RMSNorm, multi-head attention with RoPE, SwiGLU FFN
- Temperature + top-k sampling
- Autoregressive generation loop
- Compile command: `gcc -O2 -o llm_inference llm_inference.c -lm`

---

## What Is Actually Being Done Now

We are in the **"engine is running but weights are random"** phase:

- The model **runs end-to-end** — loads GGUF, parses tensor metadata, converts F16→F32, executes 12 transformer layers, samples tokens
- Outputs are valid tokens but **not coherent text** because the weights are synthetic-random patterns, not trained values
- Inference speed: **~1.4 tokens/sec** for a 500MB model in pure C

### What Comes Next

| Phase | Goal | Status |
|---|---|---|
| **Phase 1 — Working Inference** ✅ | GGUF parser + forward pass passes | Done |
| **Phase 2 — Training Data** | Collect/define training corpus for synthetic patterns | Planned |
| **Phase 3 — Training Loop** | Implement SGD/training pass in C, modify weights | Planned |
| **Phase 4 — Coherent Output** | Train weights to produce meaningful synthetic text | Planned |
| **Phase 5 — Scale to 7GB** | Apply same pipeline to full 7GB model | Planned |
| **Phase 6 — Optimization** | KV-cache, quantization (Q4), quantization-aware training | Planned |
| **Phase 7 — Deployment** | llama.cpp integration, Ollama adapter, web UI | Planned |

---

## Project Structure

```
dreaming/
├── README.md          # Project overview
├── ROADMAP.md         # This file
├── llm_inference.c    # C inference engine (~700 lines)
├── llm_engine.py      # Python prototype (older version)
├── .gitignore         # Excludes large .gguf model file
└── modelo.gguf        # 500MB GGUF model (excluded from git, download separately)
```

---

## Design Philosophy

- **No ML frameworks** — no PyTorch, no TensorFlow, no ONNX
- **No runtime dependencies** — compiles on any system with gcc
- **Self-contained** — one `.c` file does everything from parsing to generation
- **Transparency** — every operation is readable C code, no black-box wrappers
- **Oniric by design** — weights come from dream patterns, not random initialization

---

## Compilation & Testing

### Prerequisites

- GCC or any C11 compiler
- Standard C library + math.h

### Build

```bash
gcc -O2 -Wall -o llm_inference llm_inference.c -lm
```

### Run

```bash
./llm_inference modelo.gguf "DREAM" 30 0.8 25
```

### With llama.cpp

```bash
# Download modelo.gguf separately (see .gitignore)
llama-cli -m modelo.gguf --tokenizer-file tokenizer.json -p "<s>DREAM" -n 30
```
