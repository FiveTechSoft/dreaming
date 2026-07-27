# Chapter 2: The Internal Structure of TinyLlama

## The 22 Levels (Layers)

TinyLlama has 22 transformer layers. Each layer is like
a processing level that information must traverse.

```
Layer 0:     Input → Simple pattern detection
Layer 1:     Basic syntax
Layers 2–5:  Relationships between adjacent words
Layers 6–12: Abstract concepts (the "layers of pure ideas")
Layers 13–20: Global integration
Layer 21:    Output → Token generation
```

## The 9 Planets per Level (Tensors)

Each layer has 9 tensors that work together:

### Attention Tensors (4 tensors, ~19% of parameters)
- **Query (Q)**: What am I looking for?
- **Key (K)**: What do I have to offer?
- **Value (V)**: What information do I transmit?
- **Output (O)**: How do I integrate everything?

### FFN Tensors (3 tensors, ~69% of parameters)
- **Gate (G)**: What information do I let through?
- **Up (U)**: How do I expand the information?
- **Down (D)**: How do I compress the information?

### Normalization Tensors (2 tensors, ~0.01% of parameters)
- **AttnNorm**: Stabilizes attention
- **FFNNorm**: Stabilizes the feed-forward network

## The Information Flow

Information flows like this:

```
Token → Embedding (2048 dimensions)
     → Layer 0 → Layer 1 → ... → Layer 21
     → Next token prediction
```

Each layer transforms the 2048-dimensional representation
into a new 2048-dimensional representation.
The shape is preserved; the *semantic content* evolves.

## First Look at the Data

Values read from the TinyLlama-1.1B GGUF
(`llama.*` in the model header):

### Parameters per component (approx.)
- **FFN**: ~69% (memory / practical knowledge)
- **Attention**: ~19% (connections between tokens)
- **Embedding + LM Head**: ~12%
- **Layer Norms**: ~0.01%

### Hidden dimension (`embedding_length`): 2048
### Number of layers (`block_count`): 22
### Vocabulary size: 32,000 tokens
### Maximum context: 2048 tokens
### Attention heads: 32 Q / 4 KV (GQA)
### Dimension per head: 64
### Intermediate FFN (`feed_forward_length`): 5632
### RoPE `freq_base`: 10,000

### Logical shapes of the tensors (per layer)

```
attn_norm     [2048]
attn_q        [2048, 2048]     # 32 heads × 64
attn_k        [256,  2048]     #  4 heads × 64  (GQA)
attn_v        [256,  2048]
attn_output   [2048, 2048]
ffn_norm      [2048]
ffn_gate      [5632, 2048]
ffn_up        [5632, 2048]
ffn_down      [2048, 5632]
```

Plus the global ones:

```
token_embd.weight   [32000, 2048]
output_norm.weight  [2048]
output.weight       [32000, 2048]
```

> **Note on Q4_0:** on disk, a quantized GGUF
> shows "packed" shapes (for example
> `token_embd` as `[32000, 1152]`). That is the
> 4-bit block layout, not the model geometry.
> The actual dimension of the residual vector is still 2048.

## The Hierarchical Structure

```
Embeddings (vocabulary geometry)
    ↓
Layers 0–5 (syntax and local neighbors)
    ↓
Layers 6–12 (more abstract meaning)
    ↓
Layers 13–21 (integration and decision)
    ↓
Output (logits → next token)
```

## Conclusion

TinyLlama's structure is elegant and hierarchical.
Each component has a specific role, and together
they create a system capable of processing and generating language.

With 22 layers, 9 tensors per layer and a 2048-dimensional
residual, the model is small enough to open entirely — and
rich enough to be surprising.

---

*Next chapter: Our C Inference Engine*
