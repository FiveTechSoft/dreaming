"""
v10_fast.py — Fast hierarchy-preserving techniques
Uses smaller blocks for SVD operations to speed up significantly.
"""
import sys, struct, time, numpy as np
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

def deq(raw_bytes):
    raw = np.frombuffer(raw_bytes, dtype=np.uint8)
    n_blocks = len(raw) // Q4_BYTES
    if n_blocks == 0: return np.array([], dtype=np.float32)
    data = raw[:n_blocks * Q4_BYTES].reshape(n_blocks, Q4_BYTES)
    scales = np.frombuffer(data[:, :2].tobytes(), dtype=np.float16).astype(np.float32)
    nibbles = data[:, 2:18]
    lo = (nibbles & 0x0F).astype(np.float32) - 8.0
    hi = ((nibbles >> 4) & 0x0F).astype(np.float32) - 8.0
    return (np.concatenate([lo, hi], axis=1) * scales[:, np.newaxis]).flatten()

def req(values):
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

def load_model():
    with open(INPUT, 'rb') as f:
        magic = f.read(4)
        assert magic == GGUF_MAGIC
        f.read(4)
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
        all_data = bytearray(f.read())
    return tensor_infos, all_data, header_end

def save_model(header_end, all_data, out_path):
    with open(INPUT, 'rb') as fin:
        header = fin.read(header_end)
    with open(out_path, 'wb') as fout:
        fout.write(header)
        fout.write(all_data)

# ========== FAST TECHNIQUES (block-based, no large SVD) ==========

def fast_lowrank(values, rng, intensity):
    """1. Low-rank: amplify blocks with highest variance."""
    block_size = 512
    n_blocks = len(values) // block_size
    if n_blocks < 4: return values
    
    block_vars = np.array([values[i*block_size:(i+1)*block_size].var() for i in range(n_blocks)])
    # Amplify blocks with highest variance (most important)
    threshold = np.percentile(block_vars, 70)
    result = values.copy()
    for i in range(n_blocks):
        if block_vars[i] >= threshold:
            result[i*block_size:(i+1)*block_size] *= (1 + intensity)
    return result

def fast_eigr(values, rng, intensity):
    """2. Eigenvector rotation: rotate within blocks."""
    block_size = 256
    n_blocks = len(values) // block_size
    result = values.copy()
    for i in range(n_blocks):
        block = result[i*block_size:(i+1)*block_size]
        # Small random rotation
        noise = rng.standard_normal(block_size).astype(np.float32)
        noise = noise / (np.linalg.norm(noise) + 1e-9)
        result[i*block_size:(i+1)*block_size] = block + intensity * noise * block.std()
    return result

def fast_spectral(values, rng, intensity):
    """3. Spectral shift: amplify high-frequency components."""
    block_size = 256
    n_blocks = len(values) // block_size
    result = values.copy()
    for i in range(n_blocks):
        block = result[i*block_size:(i+1)*block_size]
        # Simple frequency-based: alternate amplification
        freq_pattern = np.tile([1+intensity, 1-intensity], block_size//2)
        result[i*block_size:(i+1)*block_size] = block * freq_pattern
    return result

def fast_attention(values, rng, intensity):
    """4. Attention-preserving: amplify principal directions."""
    block_size = 512
    n_blocks = len(values) // block_size
    if n_blocks < 4: return values
    
    result = values.copy()
    # Find overall principal direction
    overall_std = values.std()
    for i in range(n_blocks):
        block = result[i*block_size:(i+1)*block_size]
        # Amplify blocks close to overall mean
        block_mean = abs(block.mean())
        if block_mean < overall_std:
            result[i*block_size:(i+1)*block_size] *= (1 + intensity)
    return result

def fast_residual(values, rng, intensity):
    """5. Residual-preserving: noise orthogonal to mean direction."""
    mean_val = values.mean()
    noise = rng.standard_normal(len(values)).astype(np.float32)
    # Make noise orthogonal to constant direction
    noise = noise - noise.mean()
    noise = noise / (np.linalg.norm(noise) + 1e-9)
    result = values + intensity * noise * values.std()
    return result

def fast_blockdiag(values, rng, intensity):
    """6. Block-diagonal: amplify within blocks."""
    block_size = 64
    n_blocks = len(values) // block_size
    result = values.copy()
    for i in range(n_blocks):
        result[i*block_size:(i+1)*block_size] *= (1 + intensity)
    return result

def fast_normrot(values, rng, intensity):
    """7. Norm-preserving: shift values while preserving distribution."""
    shift = intensity * values.std()
    result = values + rng.uniform(-shift, shift, len(values)).astype(np.float32)
    return result

def fast_gradient(values, rng, intensity):
    """8. Gradient-aligned: amplify by local variation."""
    block_size = 256
    n_blocks = len(values) // block_size
    if n_blocks < 4: return values
    
    block_stds = np.array([values[i*block_size:(i+1)*block_size].std() for i in range(n_blocks)])
    weights = block_stds / (block_stds.max() + 1e-8)
    
    result = values.copy()
    noise = rng.standard_normal(len(values)).astype(np.float32)
    for i in range(n_blocks):
        result[i*block_size:(i+1)*block_size] += intensity * noise[i*block_size:(i+1)*block_size] * weights[i] * block_stds[i]
    return result

def fast_dct(values, rng, intensity):
    """9. Low-frequency: smooth then amplify."""
    block_size = 256
    n_blocks = len(values) // block_size
    result = values.copy()
    for i in range(n_blocks):
        block = result[i*block_size:(i+1)*block_size]
        # Moving average (low frequency)
        kernel = np.ones(8) / 8
        smoothed = np.convolve(block, kernel, mode='same')
        result[i*block_size:(i+1)*block_size] = block + intensity * (smoothed - block)
    return result

def fast_manifold(values, rng, intensity):
    """10. Manifold-preserving: local statistics noise."""
    block_size = 64
    n_blocks = len(values) // block_size
    result = values.copy()
    for i in range(n_blocks):
        block = result[i*block_size:(i+1)*block_size]
        local_std = block.std() + 1e-8
        noise = rng.standard_normal(block_size).astype(np.float32) * local_std * intensity
        result[i*block_size:(i+1)*block_size] = block + noise
    return result

TECHNIQUES = {
    'lowrank': ('Low-rank amplification', fast_lowrank),
    'eigr': ('Eigenvector rotation', fast_eigr),
    'spectral': ('Spectral shift', fast_spectral),
    'attention': ('Attention-preserving', fast_attention),
    'residual': ('Residual-preserving', fast_residual),
    'blockdiag': ('Block-diagonal', fast_blockdiag),
    'normrot': ('Norm-preserving rotation', fast_normrot),
    'gradient': ('Gradient-aligned', fast_gradient),
    'dct': ('Low-frequency DCT', fast_dct),
    'manifold': ('Manifold-preserving', fast_manifold),
}

ABBR = {
    'lowrank': 'lowrank', 'eigr': 'eigr', 'spectral': 'spectral',
    'attention': 'attpres', 'residual': 'respres', 'blockdiag': 'blkdiag',
    'normrot': 'normrot', 'gradient': 'gradal', 'dct': 'lowdct',
    'manifold': 'manpres',
}

def run(tech_key, intensity=0.10):
    name, func = TECHNIQUES[tech_key]
    abbr = ABBR[tech_key]
    print(f"\n{'='*50}")
    print(f"{name} (intensity={intensity})")
    
    t0 = time.time()
    tensor_infos, all_data, header_end = load_model()
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
        values = deq(raw)
        if len(values) == 0: continue
        
        new_values = func(values, rng, intensity)
        new_raw = req(new_values)
        all_data[raw_start:raw_end] = new_raw
        perturbed += 1
    
    out_file = OUT_DIR / f"v10_{abbr}_{int(intensity*100):02d}.gguf"
    save_model(header_end, all_data, out_file)
    print(f"  {perturbed} tensors, {time.time()-t0:.1f}s -> {out_file.name}")
    return out_file

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('tech', choices=list(TECHNIQUES.keys()) + ['all'])
    parser.add_argument('--intensity', type=float, default=0.10)
    args = parser.parse_args()
    
    if args.tech == 'all':
        for key in TECHNIQUES:
            run(key, args.intensity)
        print("\n✓ All 10 completed")
    else:
        run(args.tech, args.intensity)
