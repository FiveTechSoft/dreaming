"""
Aplica la dirección de agresividad al modelo TinyLlama.
Modifica específicamente el tensor de embeddings (token_embd.weight).
"""
import sys, os, struct, time, numpy as np

Q4_BLOCK = 32
Q4_BYTES = 18
GGUF_MAGIC = b'GGUF'

def dequantize_q4_0_vectorized(raw_bytes):
    """Vectorized Q4_0 dequantization."""
    n_blocks = len(raw_bytes) // Q4_BYTES
    data = raw_bytes[:n_blocks * Q4_BYTES].reshape(n_blocks, Q4_BYTES)
    scales = np.frombuffer(data[:, :2].tobytes(), dtype=np.float16).astype(np.float32)
    nibbles = data[:, 2:18]
    lo = (nibbles & 0x0F).astype(np.float32) - 8.0
    hi = ((nibbles >> 4) & 0x0F).astype(np.float32) - 8.0
    values = np.concatenate([lo, hi], axis=1) * scales[:, np.newaxis]
    return values.flatten()

def requantize_q4_0_vectorized(values):
    """Vectorized Q4_0 requantization."""
    n_blocks = len(values) // Q4_BLOCK
    data = values[:n_blocks * Q4_BLOCK].reshape(n_blocks, Q4_BLOCK)
    absmax = np.abs(data).max(axis=1)
    absmax = np.maximum(absmax, 1e-9)
    scales = absmax / 8.0
    scales_f16 = scales.astype(np.float16)
    quanted = np.clip(np.round(data / scales[:, np.newaxis]) + 8, 0, 15).astype(np.uint8)
    lo = quanted[:, :16]
    hi = quanted[:, 16:]
    packed = lo | (hi << 4)
    scale_bytes = scales_f16.tobytes()
    result = np.concatenate([np.frombuffer(scale_bytes, dtype=np.uint8).reshape(n_blocks, 2), packed], axis=1)
    return result.tobytes()

def skip_value(f, vtype):
    if vtype == 8:
        slen = struct.unpack('<Q', f.read(8))[0]
        f.read(slen)
    elif vtype == 9:
        etype = struct.unpack('<I', f.read(4))[0]
        alen = struct.unpack('<Q', f.read(8))[0]
        for _ in range(alen):
            skip_value(f, etype)
    else:
        sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
        f.read(sizes.get(vtype, 0))

# ---- Config ----
INPUT = "C:/tmp/tinyllama-1.1b.Q4_0.gguf"
OUTPUT = "C:/tmp/dreaming/perturbed_models/aggressive_model.gguf"
DIRECTION_FILE = "aggression_direction_model.npy"
SCALE = 0.5  # 50% perturbation (strong)

print(f"Loading model: {INPUT}")
t0 = time.time()

# Load aggression direction
print(f"Loading aggression direction: {DIRECTION_FILE}")
aggression_dir = np.load(DIRECTION_FILE)
print(f"  Direction shape: {aggression_dir.shape}")
print(f"  Direction norm: {np.linalg.norm(aggression_dir):.4f}")

# Adapt direction to match model embedding dimension
MODEL_EMBEDDING_DIM = 2048
if len(aggression_dir) != MODEL_EMBEDDING_DIM:
    print(f"  Adapting direction from {len(aggression_dir)} to {MODEL_EMBEDDING_DIM} dimensions")
    # Pad with zeros or repeat
    new_dir = np.zeros(MODEL_EMBEDDING_DIM)
    # Copy the original direction
    copy_len = min(len(aggression_dir), MODEL_EMBEDDING_DIM)
    new_dir[:copy_len] = aggression_dir[:copy_len]
    # Normalize
    new_dir = new_dir / np.linalg.norm(new_dir)
    aggression_dir = new_dir
    print(f"  New direction shape: {aggression_dir.shape}")
    print(f"  New direction norm: {np.linalg.norm(aggression_dir):.4f}")

with open(INPUT, 'rb') as fin:
    magic = fin.read(4)
    assert magic == GGUF_MAGIC
    version = struct.unpack('<I', fin.read(4))[0]
    n_tensors = struct.unpack('<Q', fin.read(8))[0]
    n_kv = struct.unpack('<Q', fin.read(8))[0]
    print(f"  Version: {version}, Tensors: {n_tensors}, KV pairs: {n_kv}")

    # Skip KV pairs
    for i in range(n_kv):
        klen = struct.unpack('<Q', fin.read(8))[0]
        fin.read(klen)
        vtype = struct.unpack('<I', fin.read(4))[0]
        skip_value(fin, vtype)

    # Read tensor info
    tensor_infos = []
    for i in range(n_tensors):
        tname_len = struct.unpack('<Q', fin.read(8))[0]
        tname = fin.read(tname_len).decode('utf-8')
        n_dims = struct.unpack('<I', fin.read(4))[0]
        dims = [struct.unpack('<Q', fin.read(8))[0] for _ in range(n_dims)]
        ttype = struct.unpack('<I', fin.read(4))[0]
        toffset = struct.unpack('<Q', fin.read(8))[0]
        tensor_infos.append((tname, dims, ttype, toffset))

    tensor_data_start = fin.tell()
    alignment = 32
    tensor_data_start = ((tensor_data_start + alignment - 1) // alignment) * alignment

    fin.seek(tensor_data_start)
    all_tensor_data = bytearray(fin.read())

print(f"  Header: {tensor_data_start} bytes, Tensor data: {len(all_tensor_data)} bytes")
print(f"  Loaded in {time.time()-t0:.1f}s")

# ---- Find and perturb embedding tensor ----
print(f"\nPerturbing embedding tensor with aggression direction (scale={SCALE})...")
perturbed_count = 0

for idx, (tname, dims, ttype, toffset) in enumerate(tensor_infos):
    # Only perturb the embedding tensor
    if tname == "token_embd.weight":
        is_q4 = (ttype == 2)  # Q4_0
        if not is_q4:
            print(f"  WARNING: {tname} is not Q4_0 (type={ttype})")
            continue
        
        total_elements = 1
        for d in dims:
            total_elements *= d
        
        n_blocks = total_elements // Q4_BLOCK
        raw_start = toffset
        raw_end = raw_start + n_blocks * Q4_BYTES
        
        print(f"  Found embedding tensor: {tname}")
        print(f"  Dimensions: {dims}")
        print(f"  Total elements: {total_elements}")
        print(f"  Data range: {raw_start} - {raw_end}")
        
        if raw_end > len(all_tensor_data):
            print(f"  ERROR: Tensor overflows ({raw_end} > {len(all_tensor_data)})")
            continue
        
        # Dequantize
        raw = np.frombuffer(bytes(all_tensor_data[raw_start:raw_end]), dtype=np.uint8)
        values = dequantize_q4_0_vectorized(raw)
        
        print(f"  Dequantized values: {len(values)}")
        print(f"  Values range: [{values.min():.4f}, {values.max():.4f}]")
        
        # Reshape to (embedding_dim, num_tokens) or (num_tokens, embedding_dim)
        # TinyLlama: vocab_size=32000, hidden_size=2048
        # Tensor dims are [2048, 32000] - this is (hidden_size, vocab_size)
        embedding_dim = dims[0]  # 2048
        num_tokens = dims[1]     # 32000
        
        print(f"  Tensor shape: ({embedding_dim}, {num_tokens})")
        print(f"  Total elements: {embedding_dim * num_tokens}")
        
        # Reshape to (embedding_dim, num_tokens)
        # Each column is a token embedding
        values = values[:embedding_dim * num_tokens].reshape(embedding_dim, num_tokens)
        
        # Apply aggression direction to each token embedding
        # aggression_dir has shape (1152,) but we need (2048,)
        # We need to pad or project the direction
        
        # Pad aggression_dir to match embedding_dim
        if len(aggression_dir) < embedding_dim:
            # Pad with zeros
            padded_dir = np.zeros(embedding_dim)
            padded_dir[:len(aggression_dir)] = aggression_dir
            aggression_dir = padded_dir
            print(f"  Padded aggression direction to {embedding_dim} dimensions")
        elif len(aggression_dir) > embedding_dim:
            # Truncate
            aggression_dir = aggression_dir[:embedding_dim]
            print(f"  Truncated aggression direction to {embedding_dim} dimensions")
        
        print(f"  Applying aggression direction to {num_tokens} tokens...")
        print(f"  Scale: {SCALE}")
        
        # Apply direction to each token embedding
        # values shape: (embedding_dim, num_tokens)
        # We want to add the direction to each column (token)
        
        # Expand direction to (embedding_dim, num_tokens)
        direction_expanded = np.tile(aggression_dir.reshape(-1, 1), (1, num_tokens))
        
        # Apply perturbation
        values = values + SCALE * direction_expanded
        
        # Flatten back
        values = values.flatten()
        
        # Requantize
        new_raw = requantize_q4_0_vectorized(values)
        all_tensor_data[raw_start:raw_end] = bytearray(new_raw)
        
        perturbed_count += 1
        print(f"  Perturbation applied successfully!")
        
        break  # Only perturb embedding tensor

# ---- Write output ----
print(f"\nWriting {OUTPUT}...")
os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)

with open(INPUT, 'rb') as fin:
    header_data = fin.read(tensor_data_start)

with open(OUTPUT, 'wb') as fout:
    fout.write(header_data)
    current_pos = len(header_data)
    padding = (alignment - (current_pos % alignment)) % alignment
    fout.write(b'\x00' * padding)
    fout.write(bytes(all_tensor_data))

sz = os.path.getsize(OUTPUT) / (1024 * 1024)
print(f"\nDone! {OUTPUT}")
print(f"  Size: {sz:.1f} MB")
print(f"  Tensors perturbed: {perturbed_count}")
print(f"  Scale: {SCALE}")
print(f"  Time: {time.time()-t0:.1f}s")

print(f"\nTest with llama.cpp:")
print(f'  llama-cli.exe -m "{OUTPUT}" -p "I feel" -n 50 --temp 0.7')
print(f'  llama-cli.exe -m "{OUTPUT}" -p "The world is" -n 50 --temp 0.7')
