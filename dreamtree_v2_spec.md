# DreamTree v2 — Streaming GEMV Format Specification

Binary layout for zero-allocation streaming inference of transformer models.
The file is ordered **exactly** in the sequence the engine reads bytes during
forward pass. No seeks, no random access, no full-matrix loads.

## Core Principle

Every GEMV `Y = X · W` is computed by reading W in small chunks from disk.
Two partition strategies:

- **Output Slicing** (column-partitioned): `W = [W₀ | W₁ | ...]`
  Each chunk produces a slice of Y directly. Used for Q/K/V projections.
- **Reduction Slicing** (row-partitioned): `W = [W₀; W₁; ...]`
  Each chunk accumulates into Y. Used for attn_out and ffn_down.

## File Layout (sequential, byte-by-byte)

```
┌─────────────────────────────────────────────────────────────────┐
│ HEADER (128 bytes, fixed)                                       │
├─────────────────────────────────────────────────────────────────┤
│ GLOBAL TENSORS                                                  │
│   token_embd  [vocab × hidden × 2 bytes F16]                   │
│   output_norm [hidden × 2 bytes F16]                           │
├─────────────────────────────────────────────────────────────────┤
│ FOR EACH LAYER l = 0..num_layers-1:                             │
│   norm1       [hidden × 2]                                      │
│   Q blocks    [hidden × (hidden/B) × 2] × B blocks             │
│   K blocks    [hidden × (kv_dim/B) × 2] × B blocks             │
│   V blocks    [hidden × (kv_dim/B) × 2] × B blocks             │
│   attn_out    [hidden × (hidden/B) × 2] × B blocks (reduction) │
│   norm2       [hidden × 2]                                      │
│   FFN chunks (m = 0..num_ffn_chunks-1):                        │
│     gate[m]   [hidden × M × 2]                                  │
│     up[m]     [hidden × M × 2]                                  │
│     down[m]   [M × hidden × 2]                                  │
├─────────────────────────────────────────────────────────────────┤
│ OUTPUT NORM [hidden × 2]                                        │
│ OUTPUT       [hidden × vocab × 2 bytes F16]                     │
└─────────────────────────────────────────────────────────────────┘
```

## Header (128 bytes)

| Offset | Size | Field              | Description                           |
|--------|------|--------------------|---------------------------------------|
| 0      | 4    | magic              | "DRT2"                                |
| 4      | 4    | version            | 2                                     |
| 8      | 4    | num_layers         | 22                                    |
| 12     | 4    | hidden_dim         | 2048                                  |
| 16     | 4    | kv_dim             | 256                                   |
| 20     | 4    | ffn_dim            | 5632                                  |
| 24     | 4    | vocab_size         | 32000                                 |
| 28     | 4    | num_qkv_blocks     | Output slicing blocks for Q/K/V (8)  |
| 32     | 4    | ffn_chunk_size     | Elements per FFN chunk (256)          |
| 36     | 4    | num_ffn_chunks     | ceil(ffn_dim / ffn_chunk_size) = 23  |
| 40     | 8    | layer_size         | Bytes per layer in file               |
| 48     | 8    | global_size        | Bytes of global tensors               |
| 56     | 8    | data_size          | Total data bytes (excl header)        |
| 64     | 8    | total_file_size    | Entire file size                      |
| 72     | 56   | reserved           | Zero padding to 128 bytes             |

## Per-Layer Byte Ordering

```
offset 0:              norm1           [HIDDEN_DIM × 2]         = 4,096 B
offset 4096:           Q_block[0]      [HIDDEN_DIM × (HIDDEN/B) × 2]
                       Q_block[1]      [...]
                       ...
                       Q_block[B-1]    [...]
offset +:              K_block[0]      [HIDDEN_DIM × (KV/B) × 2]
                       K_block[1]      [...]
                       ...
                       K_block[B-1]    [...]
offset +:              V_block[0..B-1] (same layout as K)
offset +:              attn_out[0]     [HIDDEN_DIM × (HIDDEN/B) × 2]  (reduction)
                       attn_out[1]     [...]
                       ...
                       attn_out[B-1]   [...]
offset +:              norm2           [HIDDEN_DIM × 2]         = 4,096 B
offset +:              FOR EACH ffn chunk m:
                         gate[m]       [HIDDEN_DIM × M × 2]
                         up[m]         [HIDDEN_DIM × M × 2]
                         down[m]       [M × HIDDEN_DIM × 2]
```

## TinyLlama-1.1B Concrete Numbers

```
HIDDEN_DIM    = 2048
KV_DIM        = 256
FFN_DIM       = 5632
VOCAB_SIZE    = 32000
NUM_LAYERS    = 22
B (qkv blks)  = 8
M (ffn chunk) = 256
NUM_FFN_CH    = ceil(5632/256) = 23

Per-layer bytes:
  norm1:           4,096
  Q blocks:        2048 × 256 × 2 × 8 = 8,388,608  (8 MB)
  K blocks:        2048 × 32 × 2 × 8  = 1,048,576  (1 MB)
  V blocks:        same as K                          (1 MB)
  attn_out blocks: 2048 × 256 × 2 × 8 = 8,388,608  (8 MB)
  norm2:           4,096
  FFN (23 chunks): 23 × (2048×256×2 + 2048×256×2 + 256×2048×2)
                   = 23 × (1,048,576 + 1,048,576 + 1,048,576)
                   = 23 × 3,145,728 = 72,351,744    (~69 MB)
  TOTAL per layer: ≈ 88.8 MB

22 layers:  ≈ 1.95 GB
Global:     32000×2048×2 + 2048×2 + 2048×32000×2 = ~250 MB
TOTAL:      ≈ 2.2 GB
```

## Static Working Set (~88.5 KB)

| Buffer                      | Size    |
|-----------------------------|---------|
| Residual X (F32)            | 8 KB    |
| X_norm (F16)                | 4 KB    |
| Accumulator (F32)           | 8 KB    |
| H[m] intermediate (F16)     | 512 B   |
| I/O double-buffer           | 64 KB   |
| Dot-product scratch          | 4 KB    |
| **Total**                   | **~88.5 KB** |
