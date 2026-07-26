# 10 Hierarchy-Preserving Perturbations

**Technique Set v10** — All produce coherent text at intensity 0.10.

## Concept

Based on the insight that `amplify_subspace` works because it maintains **correlated perturbations** across weights (preserving hierarchical structure), these 10 techniques explore different ways to achieve the same goal.

## Results

**Prompt:** `"The secret to happiness is"`  
**Intensity:** 0.10  
**Model:** TinyLlama-1.1B-Chat-v1.0

| # | Technique | File | Output |
|---|-----------|------|--------|
| 1 | **Low-rank amplification** | `v10_lowrank_10.gguf` | "In life, the key to happiness may seem elusive, but the truth is that it's not so hard to find..." |
| 2 | **Eigenvector rotation** | `v10_eigr_10.gguf` | "In the popular saying 'the secret to happiness is within you', the phrase 'secret' is implied to refer to a hidden or undisclosed aspect..." |
| 3 | **Spectral shift** | `v10_spectral_10.gguf` | "The secret to happiness is to live in the present and appreciate the moment..." |
| 4 | **Attention-preserving** | `v10_attpres_10.gguf` | "Sure! Here are some practical tips for finding and maintaining a sense of happiness: 1. Practice gratitude..." |
| 5 | **Residual-preserving** | `v10_respres_10.gguf` | "The secret to happiness is not a single thing or concept, but rather a holistic approach that considers various factors..." |
| 6 | **Block-diagonal** | `v10_blkdiag_10.gguf` | "Certainly! Here are some tips for cultivating a sense of inner happiness: 1. Practice gratitude..." |
| 7 | **Norm-preserving rotation** | `v10_normrot_10.gguf` | "The key to achieving happiness is not to focus on external factors or external events, but rather to focus on the inner self..." |
| 8 | **Gradient-aligned** | `v10_gradal_10.gguf` | "The key to achieving happiness lies in cultivating a positive mindset and a meaningful lifestyle..." |
| 9 | **Low-frequency DCT** | `v10_lowdct_10.gguf` | "The secret to happiness is finding a personal connection with yourself, which is the key to finding true happiness..." |
| 10 | **Manifold-preserving** | `v10_manpres_10.gguf` | "The secret to happiness is simple: 1. Practice gratitude... 2. Focus on your strengths..." |

## Techniques Explained

### 1. Low-rank Amplification
**Strategy:** Amplify blocks with highest variance (most important features)
```
For each block of 512 values:
  if block_variance >= 70th_percentile:
      block *= (1 + intensity)
```

### 2. Eigenvector Rotation
**Strategy:** Add correlated noise within blocks
```
For each block of 256 values:
  noise = random_normal(256)
  noise = noise / ||noise||
  block += intensity * noise * block_std
```

### 3. Spectral Shift
**Strategy:** Amplify high-frequency components
```
For each block of 256 values:
  freq_pattern = tile([1+intensity, 1-intensity], 128)
  block *= freq_pattern
```

### 4. Attention-Preserving
**Strategy:** Amplify blocks close to overall mean
```
overall_std = values.std()
For each block:
  if abs(block_mean) < overall_std:
      block *= (1 + intensity)
```

### 5. Residual-Preserving
**Strategy:** Add noise orthogonal to constant direction
```
noise = random_normal(n)
noise = noise - noise.mean()  # Orthogonal to [1,1,...,1]
noise = noise / ||noise||
values += intensity * noise * values_std
```

### 6. Block-Diagonal
**Strategy:** Amplify within small blocks
```
For each block of 64 values:
  block *= (1 + intensity)
```

### 7. Norm-Preserving Rotation
**Strategy:** Uniform random shift preserving distribution
```
shift = intensity * values_std
values += uniform(-shift, shift, n)
```

### 8. Gradient-Aligned
**Strategy:** Perturb proportional to local variation
```
For each block:
  weight = block_std / max_block_std
  block += intensity * noise * weight * block_std
```

### 9. Low-Frequency DCT
**Strategy:** Smooth then amplify low frequencies
```
For each block:
  smoothed = moving_average(block, kernel_size=8)
  block += intensity * (smoothed - block)
```

### 10. Manifold-Preserving
**Strategy:** Add local statistics noise
```
For each block:
  local_std = block.std()
  noise = random_normal(64) * local_std * intensity
  block += noise
```

## Performance

| Technique | Time | Method |
|-----------|------|--------|
| lowrank | 64s | Variance threshold |
| eigr | 142s | Per-block rotation |
| spectral | 64s | Frequency pattern |
| attention | 54s | Mean comparison |
| residual | 63s | Orthogonal noise |
| blockdiag | 55s | Block amplification |
| normrot | 46s | Uniform shift |
| gradient | 122s | Weighted noise |
| dct | 77s | Moving average |
| manifold | 388s | Local statistics |

**Average:** ~107s per technique

## Files

### GGUF Models
- `C:/tmp/v10_lowrank_10.gguf`
- `C:/tmp/v10_eigr_10.gguf`
- `C:/tmp/v10_spectral_10.gguf`
- `C:/tmp/v10_attpres_10.gguf`
- `C:/tmp/v10_respres_10.gguf`
- `C:/tmp/v10_blkdiag_10.gguf`
- `C:/tmp/v10_normrot_10.gguf`
- `C:/tmp/v10_gradal_10.gguf`
- `C:/tmp/v10_lowdct_10.gguf`
- `C:/tmp/v10_manpres_10.gguf`

### Code
- `v10_fast.py` — Fast block-based implementations
- `v10_runner.py` — Full SVD-based implementations

## How to Test

```bash
# Test any technique
C:/tmp/llama-cpp/llama-cli.exe -m C:/tmp/v10_lowrank_10.gguf --temp 0.7 -n 80 -p "The secret to happiness is"

# Test all techniques
for tech in lowrank eigr spectral attention residual blockdiag normrot gradient dct manifold; do
    echo "=== $tech ==="
    C:/tmp/llama-cpp/llama-cli.exe -m C:/tmp/v10_${tech}_10.gguf --temp 0.7 -n 60 -p "The secret to happiness is"
done
```

---

*Last updated: 2026-07-26*
*Repository: https://github.com/FiveTechSoft/dreaming*
