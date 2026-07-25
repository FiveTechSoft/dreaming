#!/usr/bin/env python3
"""
DMT perturbation v4 - Binary header copy + vectorized Q4_0 processing.
"""
import sys, os, struct, time, numpy as np

Q4_BLOCK = 32
Q4_BYTES = 18
GGUF_MAGIC = b'GGUF'

def dequantize_q4_0_vectorized(raw_bytes):
    """Vectorized Q4_0 dequantization. raw_bytes: 1D uint8 array of length N*18 -> N*32 float32."""
    n_blocks = len(raw_bytes) // Q4_BYTES
    data = raw_bytes[:n_blocks * Q4_BYTES].reshape(n_blocks, Q4_BYTES)
    scales = np.frombuffer(data[:, :2].tobytes(), dtype=np.float16).astype(np.float32)
    nibbles = data[:, 2:18]
    lo = (nibbles & 0x0F).astype(np.float32) - 8.0
    hi = ((nibbles >> 4) & 0x0F).astype(np.float32) - 8.0
    values = np.concatenate([lo, hi], axis=1) * scales[:, np.newaxis]
    return values.flatten()

def requantize_q4_0_vectorized(values):
    """Vectorized Q4_0 requantization. values: float32 array of N*32 -> N*18 uint8 bytes."""
    n_blocks = len(values) // Q4_BLOCK
    data = values[:n_blocks * Q4_BLOCK].reshape(n_blocks, Q4_BLOCK)
    absmax = np.abs(data).max(axis=1)
    absmax = np.maximum(absmax, 1e-9)
    scales = absmax / 7.5
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
INPUT   = sys.argv[1] if len(sys.argv) > 1 else "C:/tmp/tinyllama-1.1b.Q4_0.gguf"
OUTPUT  = sys.argv[2] if len(sys.argv) > 2 else "C:/tmp/tinyllama-1.1b.DMT.gguf"
MODE    = sys.argv[3] if len(sys.argv) > 3 else "scaled_noise"
INTENSITY = float(sys.argv[4]) if len(sys.argv) > 4 else 0.15
SEED    = 42

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

# ---- Apply DMT perturbation (vectorized) ----
perturbed_count = 0
print(f"\nPerturbing (mode={MODE}, intensity={INTENSITY})...")

for idx, (tname, dims, ttype, toffset) in enumerate(tensor_infos):
    is_q4 = (ttype == 2)  # Q4_0
    is_norm = tname.endswith('.norm.weight') or tname.endswith('.norm.bias')
    should_perturb = is_q4 and not is_norm

    if should_perturb:
        total_elements = 1
        for d in dims:
            total_elements *= d
        n_blocks = total_elements // Q4_BLOCK
        raw_start = toffset - tensor_data_start
        raw_end = raw_start + n_blocks * Q4_BYTES

        if raw_end > len(all_tensor_data):
            print(f"  WARNING: {tname} overflows ({raw_end} > {len(all_tensor_data)})")
            continue

        # Vectorized dequantize
        raw = np.frombuffer(bytes(all_tensor_data[raw_start:raw_end]), dtype=np.uint8)
        values = dequantize_q4_0_vectorized(raw)

        # Apply perturbation
        if MODE == "scaled_noise":
            noise = rng.standard_normal(values.shape).astype(np.float32)
            values += INTENSITY * noise
        elif MODE == "row_shuffle":
            row_len = dims[-1] if len(dims) > 1 else total_elements
            if total_elements % row_len == 0:
                rows = values.reshape(-1, row_len)
                perm = rng.permutation(rows.shape[0])
                values = rows[perm].flatten()
        elif MODE == "amplify_subspace":
            row_len = dims[-1] if len(dims) > 1 else total_elements
            vec = rng.standard_normal(row_len).astype(np.float32)
            vec /= np.linalg.norm(vec) + 1e-9
            tiled = np.tile(vec, total_elements // row_len + 1)[:total_elements]
            values += INTENSITY * tiled

        # Vectorized requantize
        new_raw = requantize_q4_0_vectorized(values)
        all_tensor_data[raw_start:raw_end] = bytearray(new_raw)
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
print(f"  Mode: {MODE}, Intensity: {INTENSITY}")
print(f"  Time: {time.time()-t0:.1f}s")
print(f"\nTest with llama.cpp:")
print(f'  llama-cli.exe -m "{OUTPUT}" -p "Hello, how are you?" -n 50 --temp 0.7')
