"""
dmt_perturb_v11.py — Three Directions Explored

Direction 1: Combinations (stack two techniques)
Direction 2: Selective targeting (different technique per tensor type)
Direction 3: Intensity sweep (find the breaking point)

Usage:
    python dmt_perturb_v11.py combo      # Test combinations
    python dmt_perturb_v11.py selective  # Test selective targeting
    python dmt_perturb_v11.py sweep      # Test intensity sweep
    python dmt_perturb_v11.py all        # Test everything
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

def classify_tensor(name):
    if 'attn_' in name: return 'attention'
    elif 'ffn_' in name: return 'ffn'
    elif 'token_embd' in name or 'output' in name: return 'embedding'
    elif 'norm' in name: return 'norm'
    return 'other'

# ========== BASE TECHNIQUES ==========

def amplify_subspace(values, rng, intensity):
    vec = rng.standard_normal(len(values)).astype(np.float32)
    vec /= np.linalg.norm(vec) + 1e-9
    proj = np.dot(values, vec) * vec
    return values + intensity * proj

def residual_preserve(values, rng, intensity):
    mean_val = values.mean()
    noise = rng.standard_normal(len(values)).astype(np.float32)
    noise = noise - noise.mean()
    noise = noise / (np.linalg.norm(noise) + 1e-9)
    return values + intensity * noise * values.std()

def creative_amplify(values, rng, intensity):
    """Amplify low-variance blocks (rare patterns)."""
    block_size = 256
    n_blocks = len(values) // block_size
    if n_blocks < 4: return amplify_subspace(values, rng, intensity)
    block_stds = np.array([values[i*block_size:(i+1)*block_size].std() for i in range(n_blocks)])
    weights = 1.0 / (block_stds + 1e-8)
    weights = weights / weights.sum()
    result = values.copy()
    noise = rng.standard_normal(len(values)).astype(np.float32)
    for i in range(n_blocks):
        result[i*block_size:(i+1)*block_size] += intensity * noise[i*block_size:(i+1)*block_size] * weights[i] * len(values)
    return result

def analytical_amplify(values, rng, intensity):
    """Amplify high-variance blocks (structured reasoning)."""
    block_size = 256
    n_blocks = len(values) // block_size
    if n_blocks < 4: return amplify_subspace(values, rng, intensity)
    block_stds = np.array([values[i*block_size:(i+1)*block_size].std() for i in range(n_blocks)])
    weights = block_stds / block_stds.sum()
    result = values.copy()
    noise = rng.standard_normal(len(values)).astype(np.float32)
    for i in range(n_blocks):
        result[i*block_size:(i+1)*block_size] += intensity * noise[i*block_size:(i+1)*block_size] * weights[i] * len(values)
    return result

def block_amplify(values, rng, intensity):
    """Block-diagonal: amplify within blocks."""
    block_size = 64
    n_blocks = len(values) // block_size
    result = values.copy()
    for i in range(n_blocks):
        result[i*block_size:(i+1)*block_size] *= (1 + intensity)
    return result

def poetic_oscillate(values, rng, intensity):
    """Oscillatory pattern across blocks."""
    block_size = 256
    n_blocks = len(values) // block_size
    if n_blocks < 4: return amplify_subspace(values, rng, intensity)
    result = values.copy()
    for i in range(n_blocks):
        factor = 1 + intensity * np.sin(i * 0.5)
        result[i*block_size:(i+1)*block_size] *= factor
    return result

# ========== DIRECTION 1: COMBINATIONS ==========

COMBOS = {
    'deep_reason': ('analytical + residual', analytical_amplify, residual_preserve, 0.07, 0.07),
    'rare_perspective': ('creative + residual', creative_amplify, residual_preserve, 0.07, 0.07),
    'structured_dream': ('blockdiag + poetic', block_amplify, poetic_oscillate, 0.08, 0.08),
    'max.Alter': ('amplify + amplify', amplify_subspace, amplify_subspace, 0.05, 0.05),
    'deep_creative': ('analytical + creative', analytical_amplify, creative_amplify, 0.06, 0.06),
}

# ========== DIRECTION 2: SELECTIVE TARGETING ==========

SELECTIVE = {
    'attention_alter': {
        'attention': (amplify_subspace, 0.15),  # Max alter attention
        'ffn': (residual_preserve, 0.05),        # Gentle on FFN
        'embedding': (residual_preserve, 0.05),   # Preserve embeddings
    },
    'ffn_dream': {
        'attention': (residual_preserve, 0.05),   # Keep attention focused
        'ffn': (creative_amplify, 0.15),          # Max alter FFN
        'embedding': (residual_preserve, 0.05),   # Preserve embeddings
    },
    'embedding_shift': {
        'attention': (residual_preserve, 0.05),   # Keep attention
        'ffn': (residual_preserve, 0.05),         # Keep FFN
        'embedding': (amplify_subspace, 0.15),    # Max alter embeddings
    },
    'balanced': {
        'attention': (amplify_subspace, 0.10),
        'ffn': (amplify_subspace, 0.10),
        'embedding': (amplify_subspace, 0.10),
    },
    'extreme_selective': {
        'attention': (amplify_subspace, 0.20),    # Very high on attention
        'ffn': (residual_preserve, 0.03),         # Almost nothing on FFN
        'embedding': (residual_preserve, 0.03),   # Almost nothing on embeddings
    },
}

# ========== DIRECTION 3: INTENSITY SWEEP ==========

SWEEP_INTENSITIES = [0.05, 0.10, 0.15, 0.20, 0.30, 0.50]


def run_combo(combo_name, intensity_override=None):
    """Run a combination technique."""
    desc, func1, func2, int1, int2 = COMBOS[combo_name]
    if intensity_override:
        int1 = int2 = intensity_override
    
    print(f"\n{'='*50}")
    print(f"COMBO: {combo_name} ({desc})")
    print(f"Intensities: {int1}, {int2}")
    
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
        
        # Apply both techniques sequentially
        values = func1(values, rng, int1)
        values = func2(values, rng, int2)
        
        new_raw = req(values)
        all_data[raw_start:raw_end] = new_raw
        perturbed += 1
    
    out_file = OUT_DIR / f"v11_combo_{combo_name}.gguf"
    save_model(header_end, all_data, out_file)
    print(f"  {perturbed} tensors, {time.time()-t0:.1f}s -> {out_file.name}")
    return out_file


def run_selective(select_name):
    """Run selective targeting."""
    config = SELECTIVE[select_name]
    
    print(f"\n{'='*50}")
    print(f"SELECTIVE: {select_name}")
    for cat, (func, intensity) in config.items():
        print(f"  {cat}: {func.__name__} @ {intensity}")
    
    t0 = time.time()
    tensor_infos, all_data, header_end = load_model()
    rng = np.random.default_rng(42)
    perturbed = 0
    
    for tname, dims, ttype, toffset in tensor_infos:
        if ttype != 2: continue
        if 'norm' in tname: continue
        
        cat = classify_tensor(tname)
        if cat not in config:
            continue
        
        func, intensity = config[cat]
        
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
    
    out_file = OUT_DIR / f"v11_select_{select_name}.gguf"
    save_model(header_end, all_data, out_file)
    print(f"  {perturbed} tensors, {time.time()-t0:.1f}s -> {out_file.name}")
    return out_file


def run_sweep():
    """Run intensity sweep with amplify_subspace."""
    print(f"\n{'='*50}")
    print(f"INTENSITY SWEEP: amplify_subspace")
    print(f"Intensities: {SWEEP_INTENSITIES}")
    
    for intensity in SWEEP_INTENSITIES:
        print(f"\n--- Intensity: {intensity} ---")
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
            
            new_values = amplify_subspace(values, rng, intensity)
            new_raw = req(new_values)
            all_data[raw_start:raw_end] = new_raw
            perturbed += 1
        
        out_file = OUT_DIR / f"v11_sweep_{int(intensity*100):02d}.gguf"
        save_model(header_end, all_data, out_file)
        print(f"  {perturbed} tensors, {time.time()-t0:.1f}s -> {out_file.name}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('direction', choices=['combo', 'selective', 'sweep', 'all'])
    args = parser.parse_args()
    
    if args.direction == 'combo':
        for name in COMBOS:
            run_combo(name)
        print("\n✓ All combinations completed")
    
    elif args.direction == 'selective':
        for name in SELECTIVE:
            run_selective(name)
        print("\n✓ All selective targeting completed")
    
    elif args.direction == 'sweep':
        run_sweep()
        print("\n✓ Intensity sweep completed")
    
    elif args.direction == 'all':
        print("Running all three directions...")
        for name in COMBOS:
            run_combo(name)
        for name in SELECTIVE:
            run_selective(name)
        run_sweep()
        print("\n✓ Everything completed")
    
    print("\nTest with:")
    print("  C:/tmp/llama-cpp/llama-cli.exe -m <model> --temp 0.7 -n 128 -p \"The secret to happiness is\"")


if __name__ == '__main__':
    main()
