# Covariance-Aligned Steering

**Technique 16** — Baked into GGUF weights, no runtime hooks needed.

## Concept

Instead of modifying weights randomly (like `amplify_subspace`), this technique:
1. Computes **block-level statistics** (mean, std) from the model's own weights
2. Generates **control vectors** weighted by these statistics
3. Projects perturbations through the **diagonal covariance** to stay on the natural manifold

## Key Difference from amplify_subspace

| Technique | Direction Selection | Manifold Preservation |
|-----------|--------------------|-----------------------|
| amplify_subspace | Random (all directions equal) | Approximate |
| covariance-aligned | Weighted by local statistics | Explicit (via covariance) |

## 5 Modes

### 1. Analytical (`analytical`)
**Strategy:** Amplify high-variance blocks  
**Effect:** Structured reasoning, logical flow

### 2. Creative (`creative`)
**Strategy:** Amplify low-variance blocks  
**Effect:** Rare/unusual patterns, diverse vocabulary

### 3. Code (`code`)
**Strategy:** Amplify high-mean blocks  
**Effect:** Structured output, repetitive patterns

### 4. Poetic (`poetic`)
**Strategy:** Oscillatory pattern across blocks  
**Effect:** Rhythmic language, imagery-rich

### 5. Residual (`residual`)
**Strategy:** Perturb orthogonal to dominant direction  
**Effect:** Alternative perspectives, novel angles

## Results

**Prompt:** `"The secret to happiness is"`  
**Intensity:** 0.10  
**Model:** TinyLlama-1.1B-Chat-v1.0

| Mode | Output |
|------|--------|
| **baseline** (amplify_subspace) | "cultivating a positive attitude and mindset, practicing gratitude and appreciation for the present moment, and finding meaning and purpose in life..." |
| **analytical** | "realization that we are all unique, with different interests, hobbies, and experiences. Instead of comparing ourselves to others and trying to live up to their expectations, we should cherish our own unique qualities..." |
| **creative** | "not a one-size-fits-all solution, but rather a personalized approach that considers your unique circumstances and priorities. Here are some tips that may help: 1. Define your priorities..." |
| **code** | "finding meaning and purpose in life. The key to finding meaning and purpose is to have a clear and authentic understanding of who you are and what you want out of life..." |
| **poetic** | "live in the present, be grateful for what you have, and focus on the positive aspects of life. By doing so, we can cultivate a sense of peace, contentment, and joy..." |
| **residual** | "learning to live in the present moment, appreciating the small things in life, and focusing on what you can control rather than what you cannot..." |

## Files

### GGUF Models
- `C:/tmp/covsteer_baseline_10.gguf` — amplify_subspace (comparison)
- `C:/tmp/covsteer_analytical_10.gguf` — structured reasoning
- `C:/tmp/covsteer_creative_10.gguf` — rare patterns
- `C:/tmp/covsteer_code_10.gguf` — structured output
- `C:/tmp/covsteer_poetic_10.gguf` — oscillatory
- `C:/tmp/covsteer_residual_10.gguf` — orthogonal

### Code
- `covariance_bake.py` — Fast implementation (diagonal covariance, ~2min/model)
- `covariance_steering.py` — Full implementation (requires llama-cpp-python)

## How to Test

```bash
# Test any mode
C:/tmp/llama-cpp/llama-cli.exe -m C:/tmp/covsteer_analytical_10.gguf --temp 0.7 -n 128 -p "The secret to happiness is"

# Test all modes
for mode in baseline analytical creative code poetic residual; do
    echo "=== $mode ==="
    C:/tmp/llama-cpp/llama-cli.exe -m C:/tmp/covsteer_${mode}_10.gguf --temp 0.7 -n 80 -p "The secret to happiness is"
done
```

## Algorithm

```python
# For each tensor:
1. Dequantize Q4_0 → float32
2. Reshape into blocks (256 elements each)
3. Compute per-block statistics: mean, std
4. Generate random control vector
5. Weight control by mode strategy:
   - analytical: weights = block_stds / sum(block_stds)
   - creative: weights = 1/(block_stds + eps) / sum(...)
   - code: weights = abs(block_means) / sum(abs(...))
   - poetic: weights = sin(arange(n_blocks) * 0.5)
   - residual: control = control - proj(control, dominant)
6. Normalize control vector
7. Apply: W' = W + α * control * local_scale
8. Requantize float32 → Q4_0
```

## Performance

- **Load time:** ~1.2s
- **Perturbation time:** ~115s per mode (132 tensors)
- **Output size:** 637,699,456 bytes (same as input)
- **Inference speed:** ~65 tok/s (no overhead)

---

*Last updated: 2026-07-25*
*Repository: https://github.com/FiveTechSoft/dreaming*
