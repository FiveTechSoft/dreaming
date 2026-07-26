"""
v10_runner.py — Run hierarchy-preserving techniques one at a time
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

def get_tensor_values(tname, dims, ttype, toffset, all_data):
    if ttype != 2: return None
    if 'norm' in tname: return None
    if 'token_embd' in tname or 'output' in tname: return None
    total_elements = 1
    for d in dims: total_elements *= d
    n_blocks = total_elements // Q4_BLOCK
    raw_start = toffset
    raw_end = raw_start + n_blocks * Q4_BYTES
    if raw_end > len(all_data): return None
    raw = bytes(all_data[raw_start:raw_end])
    values = deq(raw)
    if len(values) == 0: return None
    return values, raw_start, raw_end

# ========== TECHNIQUES ==========

def technique_lowrank(values, rng, intensity):
    """1. Low-rank: amplify top singular values."""
    n = len(values)
    rows = int(np.sqrt(n))
    cols = n // rows
    if rows < 4 or cols < 4: return values
    mat = values[:rows*cols].reshape(rows, cols)
    try:
        U, S, Vt = np.linalg.svd(mat, full_matrices=False)
        k = min(16, len(S))
        S[:k] *= (1 + intensity)
        mat = (U[:, :k] * S[:k]) @ Vt[:k, :]
        result = values.copy()
        result[:rows*cols] = mat.flatten()
        return result
    except: return values

def technique_eigr(values, rng, intensity):
    """2. Eigenvector rotation."""
    n = len(values)
    rows = int(np.sqrt(n))
    cols = n // rows
    if rows < 4 or cols < 4: return values
    mat = values[:rows*cols].reshape(rows, cols)
    try:
        M = mat @ mat.T
        eigvals, eigvecs = np.linalg.eigh(M)
        angle = intensity * 0.1
        theta = rng.uniform(-angle, angle)
        i, j = rng.choice(rows, 2, replace=False)
        R = np.eye(rows)
        R[i,i] = np.cos(theta); R[j,j] = np.cos(theta)
        R[i,j] = -np.sin(theta); R[j,i] = np.sin(theta)
        mat = R @ eigvecs @ eigvecs.T @ mat
        result = values.copy()
        result[:rows*cols] = mat.flatten()
        return result
    except: return values

def technique_spectral(values, rng, intensity):
    """3. Spectral shift: gradual shift across spectrum."""
    n = len(values)
    rows = int(np.sqrt(n))
    cols = n // rows
    if rows < 4 or cols < 4: return values
    mat = values[:rows*cols].reshape(rows, cols)
    try:
        U, S, Vt = np.linalg.svd(mat, full_matrices=False)
        shift = np.linspace(0, intensity, len(S))
        S = S * (1 + shift)
        mat = U @ np.diag(S) @ Vt
        result = values.copy()
        result[:rows*cols] = mat.flatten()
        return result
    except: return values

def technique_attention(values, rng, intensity):
    """4. Attention-preserving: noise orthogonal to principal subspace."""
    n = len(values)
    rows = int(np.sqrt(n))
    cols = n // rows
    if rows < 32 or cols < 4: return values
    mat = values[:rows*cols].reshape(rows, cols)
    try:
        U, _, _ = np.linalg.svd(mat, full_matrices=False)
        principal = U[:, :16]
        noise = rng.standard_normal((rows, cols)).astype(np.float32)
        for p in range(16):
            proj = noise @ principal[:, p]
            noise -= np.outer(proj, principal[:, p])
        mat = mat + intensity * noise * np.std(mat)
        result = values.copy()
        result[:rows*cols] = mat.flatten()
        return result
    except: return values

def technique_residual(values, rng, intensity):
    """5. Residual-preserving: noise orthogonal to dominant direction."""
    n = len(values)
    rows = int(np.sqrt(n))
    cols = n // rows
    if rows < 4 or cols < 4: return values
    mat = values[:rows*cols].reshape(rows, cols)
    try:
        mean = mat.mean(axis=1, keepdims=True)
        centered = mat - mean
        _, S, Vt = np.linalg.svd(centered, full_matrices=False)
        principal_dir = Vt[0]
        noise = rng.standard_normal((rows, cols)).astype(np.float32)
        proj = noise @ principal_dir / (np.dot(principal_dir, principal_dir) + 1e-9)
        noise = noise - np.outer(proj, principal_dir)
        mat = mat + intensity * noise * np.std(mat)
        result = values.copy()
        result[:rows*cols] = mat.flatten()
        return result
    except: return values

def technique_blockdiag(values, rng, intensity):
    """6. Block-diagonal: amplify within blocks."""
    n = len(values)
    rows = int(np.sqrt(n))
    cols = n // rows
    if rows < 4 or cols < 4: return values
    mat = values[:rows*cols].reshape(rows, cols)
    bs = 16
    for r in range(0, rows, bs):
        for c in range(0, cols, bs):
            block = mat[r:r+bs, c:c+bs]
            mat[r:r+bs, c:c+bs] = block * (1 + intensity)
    result = values.copy()
    result[:rows*cols] = mat.flatten()
    return result

def technique_normrot(values, rng, intensity):
    """7. Norm-preserving rotation."""
    n = len(values)
    rows = int(np.sqrt(n))
    cols = n // rows
    if rows < 4 or cols < 4: return values
    mat = values[:rows*cols].reshape(rows, cols)
    try:
        Q, _ = np.linalg.qr(rng.standard_normal((rows, rows)))
        angle = intensity * 0.01
        R = np.eye(rows) + angle * (Q - Q.T)
        R, _ = np.linalg.qr(R)
        mat = R @ mat @ R.T
        result = values.copy()
        result[:rows*cols] = mat.flatten()
        return result
    except: return values

def technique_gradient(values, rng, intensity):
    """8. Gradient-aligned: perturb by local variation."""
    n = len(values)
    rows = int(np.sqrt(n))
    cols = n // rows
    if rows < 4 or cols < 4: return values
    mat = values[:rows*cols].reshape(rows, cols)
    row_std = mat.std(axis=1)
    importance = row_std / (row_std.max() + 1e-8)
    noise = rng.standard_normal((rows, cols)).astype(np.float32)
    noise = noise * importance[:, np.newaxis]
    mat = mat + intensity * noise * np.std(mat)
    result = values.copy()
    result[:rows*cols] = mat.flatten()
    return result

def technique_dct(values, rng, intensity):
    """9. Low-frequency DCT."""
    n = len(values)
    rows = int(np.sqrt(n))
    cols = n // rows
    if rows < 4 or cols < 4: return values
    mat = values[:rows*cols].reshape(rows, cols)
    try:
        from scipy.fft import dctn, idctn
        freq = dctn(mat, type=2)
        mask = np.zeros_like(freq)
        h, w = freq.shape
        mask[:max(1, int(h*0.3)), :max(1, int(w*0.3))] = 1
        freq = freq * (1 + intensity * mask)
        mat = idctn(freq, type=2)
    except:
        mat_fft = np.fft.fft2(mat)
        mask = np.zeros_like(mat_fft)
        h, w = mat_fft.shape
        mask[:max(1, int(h*0.3)), :max(1, int(w*0.3))] = 1
        mat_fft = mat_fft * (1 + intensity * mask)
        mat = np.fft.ifft2(mat_fft).real
    result = values.copy()
    result[:rows*cols] = mat.flatten()
    return result

def technique_manifold(values, rng, intensity):
    """10. Manifold-preserving: local statistics noise."""
    n = len(values)
    rows = int(np.sqrt(n))
    cols = n // rows
    if rows < 4 or cols < 4: return values
    mat = values[:rows*cols].reshape(rows, cols)
    bs = 16
    for r in range(0, rows, bs):
        for c in range(0, cols, bs):
            block = mat[r:r+bs, c:c+bs]
            if block.size == 0: continue
            local_std = block.std() + 1e-8
            noise = rng.standard_normal(block.shape) * local_std * intensity
            mat[r:r+bs, c:c+bs] = block + noise
    result = values.copy()
    result[:rows*cols] = mat.flatten()
    return result

TECHNIQUES = {
    'lowrank': ('Low-rank amplification', technique_lowrank),
    'eigr': ('Eigenvector rotation', technique_eigr),
    'spectral': ('Spectral shift', technique_spectral),
    'attention': ('Attention-preserving', technique_attention),
    'residual': ('Residual-preserving', technique_residual),
    'blockdiag': ('Block-diagonal', technique_blockdiag),
    'normrot': ('Norm-preserving rotation', technique_normrot),
    'gradient': ('Gradient-aligned', technique_gradient),
    'dct': ('Low-frequency DCT', technique_dct),
    'manifold': ('Manifold-preserving', technique_manifold),
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
    print(f"\n{'='*60}")
    print(f"TECHNIQUE: {name}")
    print(f"Intensity: {intensity}")
    print(f"{'='*60}")
    
    t0 = time.time()
    tensor_infos, all_data, header_end = load_model()
    rng = np.random.default_rng(42)
    perturbed = 0
    
    for tname, dims, ttype, toffset in tensor_infos:
        result = get_tensor_values(tname, dims, ttype, toffset, all_data)
        if result is None: continue
        values, raw_start, raw_end = result
        
        new_values = func(values, rng, intensity)
        new_raw = req(new_values)
        all_data[raw_start:raw_end] = new_raw
        perturbed += 1
    
    out_file = OUT_DIR / f"v10_{abbr}_{int(intensity*100):02d}.gguf"
    save_model(header_end, all_data, out_file)
    elapsed = time.time() - t0
    
    print(f"  Tensors: {perturbed}")
    print(f"  Time: {elapsed:.1f}s")
    print(f"  Output: {out_file}")
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
