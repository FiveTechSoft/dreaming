#!/usr/bin/env python3
"""
DMT perturbation v7 - Combinación de técnicas.
Aplica múltiples técnicas secuencialmente para explorar efectos compuestos.
"""
import sys, os, struct, time, numpy as np

Q4_BLOCK = 32
Q4_BYTES = 18
GGUF_MAGIC = b'GGUF'

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

def classify_tensor(name):
    if 'attn_' in name: return 'attention'
    elif 'ffn_' in name: return 'ffn'
    elif 'token_embd' in name or 'output' in name: return 'embedding'
    elif 'norm' in name: return 'norm'
    return 'other'

def dequantize_q4_0(raw_bytes):
    """Vectorized Q4_0 dequantization."""
    raw = np.frombuffer(raw_bytes, dtype=np.uint8)
    n_blocks = len(raw) // Q4_BYTES
    data = raw[:n_blocks * Q4_BYTES].reshape(n_blocks, Q4_BYTES)
    scales = np.frombuffer(data[:, :2].tobytes(), dtype=np.float16).astype(np.float32)
    nibbles = data[:, 2:18]
    lo = (nibbles & 0x0F).astype(np.float32) - 8.0
    hi = ((nibbles >> 4) & 0x0F).astype(np.float32) - 8.0
    return (np.concatenate([lo, hi], axis=1) * scales[:, np.newaxis]).flatten()

def requantize_q4_0(values):
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
    return np.concatenate([np.frombuffer(scale_bytes, dtype=np.uint8).reshape(n_blocks, 2), packed], axis=1).tobytes()

# ---- Technique implementations ----

def apply_amplify_subspace(values, rng, intensity):
    """Project values onto random subspace and amplify."""
    n = len(values)
    vec = rng.standard_normal(n).astype(np.float32)
    vec /= np.linalg.norm(vec) + 1e-9
    proj = np.dot(values, vec) * vec
    return values + intensity * proj

def apply_nibble_flip_block(data_row, rng, noise_scale):
    """Flip nibbles in a single block."""
    new_row = data_row.copy()
    nibbles = new_row[2:18].copy()
    noise = rng.integers(-noise_scale, noise_scale + 1, size=len(nibbles))
    new_nibbles = np.clip(nibbles.astype(np.int16) + noise, 0, 15).astype(np.uint8)
    new_row[2:18] = new_nibbles
    return new_row

def apply_sharpen_rows(values, temperature):
    """Sharpen: exagera lo dominante, aplana el resto."""
    return np.sign(values) * (np.abs(values) ** (1.0 / temperature))

def apply_attention_reweight(data_row, n_blocks, rng, intensity):
    """Reweight: some blocks amplify, others suppress."""
    # Randomly select blocks to amplify/suppress
    n_amplify = max(1, int(n_blocks * intensity))
    indices = rng.permutation(n_blocks)
    amplify_idx = indices[:n_amplify]
    suppress_idx = indices[n_amplify:n_amplify*2]
    
    new_data = data_row.copy()
    for i in amplify_idx:
        scale = np.frombuffer(new_data[i*18:(i*18)+2], dtype=np.float16).astype(np.float32)
        new_scale = min(scale * 2.0, 100.0)  # Amplify but cap
        new_data[i*18:(i*18)+2] = np.array([new_scale], dtype=np.float16).tobytes()
    for i in suppress_idx:
        scale = np.frombuffer(new_data[i*18:(i*18)+2], dtype=np.float16).astype(np.float32)
        new_scale = max(scale * 0.5, 0.0001)  # Suppress but floor
        new_data[i*18:(i*18)+2] = np.array([new_scale], dtype=np.float16).tobytes()
    
    return new_data

def apply_cross_layer_swap(data_blocks, other_blocks, rng, n_heads=32, head_dim=64, swap_ratio=0.1):
    """Swap attention heads between two layers."""
    n_blocks = min(len(data_blocks), len(other_blocks))
    n_swap = max(1, int(n_blocks * swap_ratio))
    swap_indices = rng.choice(n_blocks, n_swap, replace=False)
    
    new_data = data_blocks.copy()
    for idx in swap_indices:
        new_data[idx] = other_blocks[idx]
    
    return new_data

# ---- Config ----
INPUT     = sys.argv[1] if len(sys.argv) > 1 else "C:/tmp/tinyllama-1.1b.Q4_0.gguf"
OUTPUT    = sys.argv[2] if len(sys.argv) > 2 else "C:/tmp/tinyllama-1.1b.DMT_v7.gguf"
COMBO     = sys.argv[3] if len(sys.argv) > 3 else "amp+nibble"  # amp+nibble, amp+sharpen, amp+swap, amp+reweight
INTENSITY = float(sys.argv[4]) if len(sys.argv) > 4 else 0.10
TARGET    = sys.argv[5] if len(sys.argv) > 5 else "all"
SEED      = 42

print(f"Loading {INPUT}...")
t0 = time.time()
rng = np.random.default_rng(SEED)

with open(INPUT, 'rb') as fin:
    magic = fin.read(4)
    assert magic == GGUF_MAGIC
    version = struct.unpack('<I', fin.read(4))[0]
    n_tensors = struct.unpack('<Q', fin.read(8))[0]
    n_kv = struct.unpack('<Q', fin.read(8))[0]
    print(f"  Version: {version}, Tensors: {n_tensors}")

    for i in range(n_kv):
        klen = struct.unpack('<Q', fin.read(8))[0]
        fin.read(klen)
        vtype = struct.unpack('<I', fin.read(4))[0]
        skip_value(fin, vtype)

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

print(f"  Loaded in {time.time()-t0:.1f}s")
print(f"  Combo: {COMBO}, Intensity: {INTENSITY}, Target: {TARGET}")

# ---- Apply combination ----
perturbed_count = 0
print(f"\nApplying combination: {COMBO}...")

# Pre-collect attention layer data for cross-layer swap
attn_layers = {}  # layer_idx -> (raw_data, n_blocks)
if 'swap' in COMBO:
    for idx, (tname, dims, ttype, toffset) in enumerate(tensor_infos):
        if 'attn_q' in tname and ttype == 2:
            layer_num = int(tname.split('.')[1])
            total_elements = 1
            for d in dims: total_elements *= d
            n_blocks = total_elements // Q4_BLOCK
            raw_start = toffset
            raw_end = raw_start + n_blocks * Q4_BYTES
            attn_layers[layer_num] = (bytes(all_tensor_data[raw_start:raw_end]), n_blocks)
    print(f"  Pre-collected {len(attn_layers)} attention layers for swap")

for idx, (tname, dims, ttype, toffset) in enumerate(tensor_infos):
    is_q4 = (ttype == 2)
    is_norm = tname.endswith('.norm.weight') or tname.endswith('.norm.bias')
    cat = classify_tensor(tname)
    
    # Determine if this tensor should be processed
    should_process = False
    if TARGET == 'all': should_process = True
    elif TARGET == 'attention' and cat == 'attention': should_process = True
    elif TARGET == 'ffn' and cat == 'ffn': should_process = True
    elif TARGET == 'embedding' and cat == 'embedding': should_process = True
    
    if is_q4 and not is_norm and should_process:
        total_elements = 1
        for d in dims: total_elements *= d
        n_blocks = total_elements // Q4_BLOCK
        raw_start = toffset
        raw_end = raw_start + n_blocks * Q4_BYTES
        
        if raw_end > len(all_tensor_data): continue
        
        # === Technique 1: amplify_subspace (float-space) ===
        if 'amp' in COMBO:
            raw = bytes(all_tensor_data[raw_start:raw_end])
            values = dequantize_q4_0(raw)
            values = apply_amplify_subspace(values, rng, INTENSITY)
            new_raw = requantize_q4_0(values)
            all_tensor_data[raw_start:raw_end] = bytearray(new_raw)
        
        # === Technique 2: nibble_flip (direct) ===
        if 'nibble' in COMBO:
            noise_scale = max(1, int(INTENSITY * 10))
            raw = np.frombuffer(bytes(all_tensor_data[raw_start:raw_end]), dtype=np.uint8).copy()
            data = raw.reshape(n_blocks, Q4_BYTES)
            
            for b in range(n_blocks):
                data[b] = apply_nibble_flip_block(data[b], rng, noise_scale)
            
            all_tensor_data[raw_start:raw_end] = data.tobytes()
        
        # === Technique 3: sharpen_rows (float-space) ===
        if 'sharpen' in COMBO:
            raw = bytes(all_tensor_data[raw_start:raw_end])
            values = dequantize_q4_0(raw)
            temperature = 0.3 + (1.0 - INTENSITY) * 0.7
            values = apply_sharpen_rows(values, temperature)
            new_raw = requantize_q4_0(values)
            all_tensor_data[raw_start:raw_end] = bytearray(new_raw)
        
        # === Technique 4: cross-layer swap ===
        if 'swap' in COMBO and 'attn_q' in tname:
            layer_num = int(tname.split('.')[1])
            if layer_num in attn_layers and len(attn_layers) > 1:
                # Swap with a random different layer
                other_layers = [l for l in attn_layers.keys() if l != layer_num]
                other_layer = rng.choice(other_layers)
                other_data, other_blocks = attn_layers[other_layer]
                
                raw = np.frombuffer(bytes(all_tensor_data[raw_start:raw_end]), dtype=np.uint8).copy()
                data = raw.reshape(n_blocks, Q4_BYTES)
                other_arr = np.frombuffer(other_data, dtype=np.uint8).reshape(other_blocks, Q4_BYTES)
                
                # Swap a fraction of blocks
                n_swap = max(1, int(n_blocks * INTENSITY * 0.5))
                swap_idx = rng.choice(n_blocks, min(n_swap, n_blocks), replace=False)
                for si in swap_idx:
                    if si < other_blocks:
                        data[si] = other_arr[si]
                
                all_tensor_data[raw_start:raw_end] = data.tobytes()
        
        # === Technique 5: attention reweight ===
        if 'reweight' in COMBO and 'attn_' in tname:
            raw = np.frombuffer(bytes(all_tensor_data[raw_start:raw_end]), dtype=np.uint8).copy()
            data = raw.reshape(n_blocks, Q4_BYTES)
            data = apply_attention_reweight(data, n_blocks, rng, INTENSITY)
            all_tensor_data[raw_start:raw_end] = data.tobytes()
        
        perturbed_count += 1
    
    if (idx + 1) % 50 == 0 or idx == len(tensor_infos) - 1:
        print(f"  [{idx+1}/{len(tensor_infos)}] {tname} ({time.time()-t0:.1f}s)")

# ---- Write output ----
print(f"\nWriting {OUTPUT}...")
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
print(f"  Tensors: {n_tensors} ({perturbed_count} perturbed)")
print(f"  Combo: {COMBO}, Intensity: {INTENSITY}")
print(f"  Time: {time.time()-t0:.1f}s")
