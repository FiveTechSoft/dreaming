"""
covariance_bake.py — Covariance-Aligned Steering (Fast Version)

Uses diagonal covariance approximation for speed.
The key insight: instead of random projection (amplify_subspace),
we project through the weight statistics to stay on the natural manifold.
"""

import sys, os, struct, time, numpy as np
from pathlib import Path

Q4_BLOCK = 32
Q4_BYTES = 18
GGUF_MAGIC = b'GGUF'
INPUT = "C:/tmp/tinyllama-1.1b.Q4_0.gguf"
OUT_DIR = Path("C:/tmp")


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


def dequantize_q4_0(raw_bytes):
    raw = np.frombuffer(raw_bytes, dtype=np.uint8)
    n_blocks = len(raw) // Q4_BYTES
    if n_blocks == 0: return np.array([], dtype=np.float32)
    data = raw[:n_blocks * Q4_BYTES].reshape(n_blocks, Q4_BYTES)
    scales = np.frombuffer(data[:, :2].tobytes(), dtype=np.float16).astype(np.float32)
    nibbles = data[:, 2:18]
    lo = (nibbles & 0x0F).astype(np.float32) - 8.0
    hi = ((nibbles >> 4) & 0x0F).astype(np.float32) - 8.0
    return (np.concatenate([lo, hi], axis=1) * scales[:, np.newaxis]).flatten()


def requantize_q4_0(values):
    n_blocks = len(values) // Q4_BLOCK
    if n_blocks == 0: return b''
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


def load_model(path):
    with open(path, 'rb') as f:
        magic = f.read(4)
        assert magic == GGUF_MAGIC
        version = struct.unpack('<I', f.read(4))[0]
        n_tensors = struct.unpack('<Q', f.read(8))[0]
        n_kv = struct.unpack('<Q', f.read(8))[0]
        for _ in range(n_kv):
            klen = struct.unpack('<Q', f.read(8))[0]
            f.read(klen)
            vtype = struct.unpack('<I', f.read(4))[0]
            skip_value(f, vtype)
        tensor_infos = []
        for _ in range(n_tensors):
            tname_len = struct.unpack('<Q', f.read(8))[0]
            tname = f.read(tname_len).decode('utf-8')
            n_dims = struct.unpack('<I', f.read(4))[0]
            dims = [struct.unpack('<Q', f.read(8))[0] for _ in range(n_dims)]
            ttype = struct.unpack('<I', f.read(4))[0]
            toffset = struct.unpack('<Q', f.read(8))[0]
            tensor_infos.append((tname, dims, ttype, toffset))
        header_end = f.tell()
        alignment = 32
        header_end = ((header_end + alignment - 1) // alignment) * alignment
        f.seek(header_end)
        all_tensor_data = bytearray(f.read())
    return tensor_infos, all_tensor_data, header_end


def save_model(path, header_end, all_tensor_data):
    with open(INPUT, 'rb') as fin:
        header = fin.read(header_end)
    with open(path, 'wb') as fout:
        fout.write(header)
        fout.write(all_tensor_data)


def cov_aligned_perturb(values, rng, intensity, mode='analytical'):
    """
    Covariance-aligned perturbation using diagonal approximation.
    
    Key difference from amplify_subspace:
    - amplify_subspace: random projection (all directions equal)
    - cov_aligned: projection weighted by weight statistics (important directions get more)
    """
    n = len(values)
    
    # Compute local statistics (fast diagonal covariance)
    # Split into blocks and compute per-block statistics
    block_size = 256
    n_blocks = n // block_size
    
    if n_blocks < 2:
        # Fallback to amplify_subspace
        vec = rng.standard_normal(n).astype(np.float32)
        vec /= np.linalg.norm(vec) + 1e-9
        proj = np.dot(values, vec) * vec
        return values + intensity * proj
    
    # Compute per-block statistics
    block_means = np.zeros(n_blocks, dtype=np.float32)
    block_stds = np.zeros(n_blocks, dtype=np.float32)
    
    for i in range(n_blocks):
        block = values[i*block_size:(i+1)*block_size]
        block_means[i] = block.mean()
        block_stds[i] = block.std() + 1e-8
    
    # Generate control vector based on mode
    control = rng.standard_normal(n).astype(np.float32)
    
    if mode == 'analytical':
        # Amplify blocks with high variance (important features)
        weights = block_stds / block_stds.sum()
        for i in range(n_blocks):
            control[i*block_size:(i+1)*block_size] *= weights[i]
    
    elif mode == 'creative':
        # Amplify blocks with low variance (rare patterns)
        weights = 1.0 / (block_stds + 1e-8)
        weights = weights / weights.sum()
        for i in range(n_blocks):
            control[i*block_size:(i+1)*block_size] *= weights[i]
    
    elif mode == 'code':
        # Amplify structured patterns (high mean absolute value)
        weights = np.abs(block_means) / (np.abs(block_means).sum() + 1e-8)
        for i in range(n_blocks):
            control[i*block_size:(i+1)*block_size] *= weights[i]
    
    elif mode == 'poetic':
        # Oscillatory pattern
        weights = np.sin(np.arange(n_blocks) * 0.5) * block_stds.mean()
        for i in range(n_blocks):
            control[i*block_size:(i+1)*block_size] *= weights[i]
    
    elif mode == 'residual':
        # Preserve dominant direction, perturb orthogonal
        dominant = values / (np.linalg.norm(values) + 1e-9)
        control = control - np.dot(control, dominant) * dominant
    
    # Normalize
    control = control / (np.linalg.norm(control) + 1e-8)
    
    # Apply: W' = W + α * control * local_scale
    result = values.copy()
    for i in range(n_blocks):
        block = result[i*block_size:(i+1)*block_size]
        local_scale = block_stds[i]
        result[i*block_size:(i+1)*block_size] = block + intensity * control[i*block_size:(i+1)*block_size] * local_scale
    
    return result


# ========== MODES ==========

MODES = {
    'analytical': 'Amplify high-variance blocks (structured reasoning)',
    'creative': 'Amplify low-variance blocks (rare patterns)',
    'code': 'Amplify high-mean blocks (structured patterns)',
    'poetic': 'Oscillatory pattern across blocks',
    'residual': 'Perturb orthogonal to dominant direction',
    'baseline': 'Standard amplify_subspace (comparison)',
}


def run_steering(mode='analytical', intensity=0.10):
    print(f"\n{'='*60}")
    print(f"COVARIANCE-ALIGNED STEERING: {mode}")
    print(f"Intensity: {intensity}")
    print(f"{'='*60}")
    
    t0 = time.time()
    tensor_infos, all_data, header_end = load_model(INPUT)
    
    rng = np.random.default_rng(42)
    perturbed = 0
    
    for tname, dims, ttype, toffset in tensor_infos:
        if ttype != 2: continue
        if 'norm' in tname: continue
        if 'token_embd' in tname or 'output' in tname: continue
        
        total_elements = 1
        for d in dims: total_elements *= d
        n_blocks = total_elements // Q4_BLOCK
        raw_start = toffset
        raw_end = raw_start + n_blocks * Q4_BYTES
        
        if raw_end > len(all_data): continue
        
        raw = bytes(all_data[raw_start:raw_end])
        values = dequantize_q4_0(raw)
        
        if len(values) == 0: continue
        
        if mode == 'baseline':
            vec = rng.standard_normal(len(values)).astype(np.float32)
            vec /= np.linalg.norm(vec) + 1e-9
            proj = np.dot(values, vec) * vec
            new_values = values + intensity * proj
        else:
            new_values = cov_aligned_perturb(values, rng, intensity, mode)
        
        new_raw = requantize_q4_0(new_values)
        all_data[raw_start:raw_end] = new_raw
        perturbed += 1
    
    out_file = OUT_DIR / f"covsteer_{mode}_{int(intensity*100):02d}.gguf"
    save_model(str(out_file), header_end, all_data)
    elapsed = time.time() - t0
    
    print(f"  Tensors: {perturbed}")
    print(f"  Time: {elapsed:.1f}s")
    print(f"  Output: {out_file}")
    print(f"  Size: {out_file.stat().st_size:,} bytes")
    return out_file


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Covariance-Aligned Steering')
    parser.add_argument('--mode', type=str, default='analytical',
                       choices=list(MODES.keys()))
    parser.add_argument('--intensity', type=float, default=0.10)
    parser.add_argument('--all', action='store_true')
    args = parser.parse_args()
    
    if args.all:
        for mode in MODES:
            run_steering(mode, args.intensity)
        print("\n✓ All modes completed")
    else:
        run_steering(args.mode, args.intensity)
    
    print("\nTest with:")
    print("  C:/tmp/llama-cpp/llama-cli.exe -m <model> --temp 0.7 -n 128 -p \"The secret to happiness is\"")


if __name__ == '__main__':
    main()
