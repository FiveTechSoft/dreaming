#!/usr/bin/env python3
"""
DMT perturbation v6 - Multi-mode with selective layers.
Supports: nibble_flip, scaled_noise, row_shuffle, amplify_subspace
With selective targeting: attention, ffn, embeddings, all
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
    """Classify tensor into attention, ffn, embedding, or norm."""
    if 'attn_' in name:
        return 'attention'
    elif 'ffn_' in name:
        return 'ffn'
    elif 'token_embd' in name or 'output' in name:
        return 'embedding'
    elif 'norm' in name:
        return 'norm'
    return 'other'

def should_process(name, target):
    """Check if tensor matches target filter."""
    if target == 'all':
        return True
    cat = classify_tensor(name)
    return cat == target

# ---- Config ----
INPUT     = sys.argv[1] if len(sys.argv) > 1 else "C:/tmp/tinyllama-1.1b.Q4_0.gguf"
OUTPUT    = sys.argv[2] if len(sys.argv) > 2 else "C:/tmp/tinyllama-1.1b.DMT_v6.gguf"
MODE      = sys.argv[3] if len(sys.argv) > 3 else "nibble_flip"
INTENSITY = float(sys.argv[4]) if len(sys.argv) > 4 else 0.07
TARGET    = sys.argv[5] if len(sys.argv) > 5 else "all"  # attention, ffn, embedding, all
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
    print(f"  Version: {version}, Tensors: {n_tensors}, KV pairs: {n_kv}")

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

print(f"  Header: {tensor_data_start} bytes, Tensor data: {len(all_tensor_data)} bytes")
print(f"  Loaded in {time.time()-t0:.1f}s")
print(f"  Mode: {MODE}, Intensity: {INTENSITY}, Target: {TARGET}")

# ---- Apply perturbation ----
perturbed_count = 0
print(f"\nPerturbing...")

for idx, (tname, dims, ttype, toffset) in enumerate(tensor_infos):
    is_q4 = (ttype == 2)
    is_norm = tname.endswith('.norm.weight') or tname.endswith('.norm.bias')
    matches_target = should_process(tname, TARGET)
    should_perturb = is_q4 and not is_norm and matches_target

    if should_perturb:
        total_elements = 1
        for d in dims:
            total_elements *= d
        n_blocks = total_elements // Q4_BLOCK
        raw_start = toffset
        raw_end = raw_start + n_blocks * Q4_BYTES

        if raw_end > len(all_tensor_data):
            continue

        raw = np.frombuffer(bytes(all_tensor_data[raw_start:raw_end]), dtype=np.uint8).copy()
        data = raw.reshape(n_blocks, Q4_BYTES)

        cat = classify_tensor(tname)

        if MODE == "nibble_flip":
            noise_scale = int(INTENSITY * 15)
            if noise_scale < 1:
                continue
            nibbles = data[:, 2:18].copy().reshape(-1)
            noise = rng.integers(-noise_scale, noise_scale + 1, size=len(nibbles))
            new_nibbles = np.clip(nibbles.astype(np.int16) + noise, 0, 15).astype(np.uint8)
            data[:, 2:18] = new_nibbles.reshape(n_blocks, 16)

        elif MODE == "scaled_noise":
            scales = np.frombuffer(data[:, :2].tobytes(), dtype=np.float16).astype(np.float32)
            lo = (data[:, 2:18] & 0x0F).astype(np.float32) - 8.0
            hi = ((data[:, 2:18] >> 4) & 0x0F).astype(np.float32) - 8.0
            values = (np.concatenate([lo, hi], axis=1) * scales[:, np.newaxis]).flatten()
            noise = rng.standard_normal(values.shape).astype(np.float32) * INTENSITY
            values += noise
            vdata = values.reshape(n_blocks, Q4_BLOCK)
            absmax = np.abs(vdata).max(axis=1)
            absmax = np.maximum(absmax, 1e-9)
            new_scales = absmax / 8.0
            new_scales_f16 = new_scales.astype(np.float16)
            quanted = np.clip(np.round(vdata / new_scales[:, np.newaxis]) + 8, 0, 15).astype(np.uint8)
            data[:, :2] = np.frombuffer(new_scales_f16.tobytes(), dtype=np.uint8).reshape(n_blocks, 2)
            data[:, 2:18] = np.concatenate([quanted[:, :16] | (quanted[:, 16:] << 4)], axis=1)

        elif MODE == "row_shuffle":
            row_len = dims[-1] // 2 if len(dims) > 1 else n_blocks
            if row_len > 1:
                nibbles = data[:, 2:18].reshape(-1, row_len)
                perm = rng.permutation(nibbles.shape[0])
                data[:, 2:18] = nibbles[perm].reshape(n_blocks, 16)

        elif MODE == "amplify_subspace":
            # Add scaled projection onto random subspace
            row_len = dims[-1] if len(dims) > 1 else total_elements
            nibbles = data[:, 2:18].copy().reshape(-1)
            # Dequantize
            scales = np.frombuffer(data[:, :2].tobytes(), dtype=np.float16).astype(np.float32)
            lo = (data[:, 2:18] & 0x0F).astype(np.float32) - 8.0
            hi = ((data[:, 2:18] >> 4) & 0x0F).astype(np.float32) - 8.0
            values = (np.concatenate([lo, hi], axis=1) * scales[:, np.newaxis]).flatten()
            # Project onto random vector
            vec = rng.standard_normal(total_elements).astype(np.float32)
            vec /= np.linalg.norm(vec) + 1e-9
            proj = np.dot(values, vec) * vec
            values += INTENSITY * proj
            # Requantize
            vdata = values.reshape(n_blocks, Q4_BLOCK)
            absmax = np.abs(vdata).max(axis=1)
            absmax = np.maximum(absmax, 1e-9)
            new_scales = absmax / 8.0
            new_scales_f16 = new_scales.astype(np.float16)
            quanted = np.clip(np.round(vdata / new_scales[:, np.newaxis]) + 8, 0, 15).astype(np.uint8)
            data[:, :2] = np.frombuffer(new_scales_f16.tobytes(), dtype=np.uint8).reshape(n_blocks, 2)
            data[:, 2:18] = np.concatenate([quanted[:, :16] | (quanted[:, 16:] << 4)], axis=1)

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
print(f"  Mode: {MODE}, Intensity: {INTENSITY}, Target: {TARGET}")
print(f"  Time: {time.time()-t0:.1f}s")
