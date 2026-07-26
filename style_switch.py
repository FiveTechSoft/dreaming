#!/usr/bin/env python3
"""
style_switch.py — Runtime style switching via weight interpolation.

Idea: Instead of N separate models, store ONE base model + N "delta" vectors
that represent each style. At generation time, interpolate between deltas.

Base + 0.1 * delta_lowrank = philosophical style
Base + 0.1 * delta_normrot = stoic style
Base + 0.05 * delta_lowrank + 0.05 * delta_normrot = blended style
"""

import struct, time, numpy as np
from pathlib import Path

Q4_BLOCK = 32
Q4_BYTES = 18
GGUF_MAGIC = b'GGUF'
BASE_MODEL = "C:/tmp/tinyllama-1.1b.Q4_0.gguf"
DELTA_DIR = Path("C:/tmp/dreaming/deltas")
DELTA_DIR.mkdir(exist_ok=True)


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
    if n_blocks == 0:
        return np.array([], dtype=np.float32)
    data = raw[:n_blocks * Q4_BYTES].reshape(n_blocks, Q4_BYTES)
    scales = np.frombuffer(data[:, :2].tobytes(), dtype=np.float16).astype(np.float32)
    nibbles = data[:, 2:18]
    lo = (nibbles & 0x0F).astype(np.float32) - 8.0
    hi = ((nibbles >> 4) & 0x0F).astype(np.float32) - 8.0
    return (np.concatenate([lo, hi], axis=1) * scales[:, np.newaxis]).flatten()


def req(values):
    n_blocks = len(values) // Q4_BLOCK
    if n_blocks == 0:
        return b''
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


def load_model_tensors(path):
    """Load model and return tensor infos + raw data + header."""
    with open(path, 'rb') as f:
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


def extract_weights(tensor_infos, all_data):
    """Extract all Q4_0 weight tensors as float32 arrays."""
    weights = {}
    for tname, dims, ttype, toffset in tensor_infos:
        if ttype != 2:
            continue
        total_elements = 1
        for d in dims:
            total_elements *= d
        n_blocks = total_elements // Q4_BLOCK
        raw_start = toffset
        raw_end = raw_start + n_blocks * Q4_BYTES
        if raw_end > len(all_data):
            continue
        raw = bytes(all_data[raw_start:raw_end])
        values = deq(raw)
        if len(values) > 0:
            weights[tname] = values
    return weights


def save_delta(name, base_weights, styled_weights):
    """Save the difference (delta) between base and styled model."""
    deltas = {}
    for tname in base_weights:
        if tname in styled_weights:
            delta = styled_weights[tname] - base_weights[tname]
            deltas[tname] = delta.astype(np.float16)  # Save as f16 to save space
    
    # Save deltas
    delta_path = DELTA_DIR / f"{name}.npz"
    np.savez_compressed(delta_path, **deltas)
    size_mb = delta_path.stat().st_size / (1024 * 1024)
    print(f"  Saved delta: {delta_path.name} ({size_mb:.1f} MB, {len(deltas)} tensors)")
    return delta_path


def load_delta(name):
    """Load a delta vector."""
    delta_path = DELTA_DIR / f"{name}.npz"
    data = np.load(delta_path)
    return {k: data[k].astype(np.float32) for k in data.files}


def apply_style(base_weights, deltas, intensities):
    """
    Apply multiple style deltas with specified intensities.
    
    base_weights: dict of original weights
    deltas: list of delta dicts
    intensities: list of float intensities (same length as deltas)
    
    Returns: dict of styled weights
    """
    styled = {}
    for tname, base_val in base_weights.items():
        result = base_val.copy()
        for delta, intensity in zip(deltas, intensities):
            if tname in delta:
                result = result + intensity * delta[tname]
        styled[tname] = result
    return styled


def weights_to_model(tensor_infos, all_data, header_end, weights):
    """Write styled weights back to model format."""
    new_data = bytearray(all_data)
    for tname, dims, ttype, toffset in tensor_infos:
        if ttype != 2:
            continue
        if tname not in weights:
            continue
        total_elements = 1
        for d in dims:
            total_elements *= d
        n_blocks = total_elements // Q4_BLOCK
        raw_start = toffset
        raw_end = raw_start + n_blocks * Q4_BYTES
        if raw_end > len(new_data):
            continue
        new_raw = req(weights[tname])
        new_data[raw_start:raw_end] = new_raw
    return new_data


# ========== STYLE DEFINITIONS ==========

STYLES = {
    # name: {description, technique, intensity}
    "philosophical": {"desc": "Filosófica/académica", "delta": "lowrank", "intensity": 0.10},
    "stoic": {"desc": "Estoica/equilibrada", "delta": "normrot", "intensity": 0.10},
    "practical": {"desc": "Práctica/consejos", "delta": "eigr", "intensity": 0.10},
    "concise": {"desc": "Concisa/directa", "delta": "spectral", "intensity": 0.10},
    "introspective": {"desc": "Introspectiva", "delta": "respres", "intensity": 0.10},
    "authentic": {"desc": "Autenticidad/descubrimiento", "delta": "gradal", "intensity": 0.10},
    "conversational": {"desc": "Conversacional/ayudante", "delta": "lowdct", "intensity": 0.10},
    "spiritual": {"desc": "Espiritual/mindfulness", "delta": "extreme_selective", "intensity": 0.10},
    "creative": {"desc": "Creativa/ficción", "delta": "structured_dream", "intensity": 0.10},
    "maximum": {"desc": "Máxima divergencia", "delta": "max.Alter", "intensity": 0.05},
}


def generate_model(style_name, intensity=None, output_dir="C:/tmp"):
    """Generate a model with a specific style."""
    if style_name not in STYLES:
        print(f"Unknown style: {style_name}")
        print(f"Available: {', '.join(STYLES.keys())}")
        return None
    
    style = STYLES[style_name]
    delta_name = style["delta"]
    inten = intensity or style["intensity"]
    
    print(f"\n{'='*50}")
    print(f"Style: {style_name} ({style['desc']})")
    print(f"Delta: {delta_name}, Intensity: {inten}")
    print(f"{'='*50}")
    
    # Load base model
    print("Loading base model...")
    t0 = time.time()
    tensor_infos, all_data, header_end = load_model_tensors(BASE_MODEL)
    base_weights = extract_weights(tensor_infos, all_data)
    print(f"  {len(base_weights)} tensors loaded in {time.time()-t0:.1f}s")
    
    # Load delta
    print(f"Loading delta: {delta_name}...")
    delta = load_delta(delta_name)
    
    # Apply style
    print("Applying style...")
    styled_weights = apply_style(base_weights, [delta], [inten])
    
    # Save model
    print("Saving model...")
    new_data = weights_to_model(tensor_infos, all_data, header_end, styled_weights)
    
    out_file = Path(output_dir) / f"style_{style_name}_{int(inten*100):02d}.gguf"
    with open(BASE_MODEL, 'rb') as fin:
        header = fin.read(header_end)
    with open(out_file, 'wb') as fout:
        fout.write(header)
        fout.write(new_data)
    
    print(f"  Saved: {out_file.name} ({out_file.stat().st_size / (1024*1024):.1f} MB)")
    return out_file


def blend_styles(style_names, intensities, output_dir="C:/tmp"):
    """Blend multiple styles into one model."""
    print(f"\n{'='*50}")
    print(f"Blending styles: {', '.join(style_names)}")
    print(f"Intensities: {intensities}")
    print(f"{'='*50}")
    
    # Load base model
    print("Loading base model...")
    t0 = time.time()
    tensor_infos, all_data, header_end = load_model_tensors(BASE_MODEL)
    base_weights = extract_weights(tensor_infos, all_data)
    print(f"  {len(base_weights)} tensors loaded in {time.time()-t0:.1f}s")
    
    # Load all deltas
    deltas = []
    for name in style_names:
        print(f"Loading delta: {name}...")
        deltas.append(load_delta(name))
    
    # Apply blended style
    print("Applying blended style...")
    styled_weights = apply_style(base_weights, deltas, intensities)
    
    # Save model
    print("Saving model...")
    new_data = weights_to_model(tensor_infos, all_data, header_end, styled_weights)
    
    blend_name = "+".join(style_names)
    out_file = Path(output_dir) / f"style_blend_{blend_name}.gguf"
    with open(BASE_MODEL, 'rb') as fin:
        header = fin.read(header_end)
    with open(out_file, 'wb') as fout:
        fout.write(header)
        fout.write(new_data)
    
    print(f"  Saved: {out_file.name} ({out_file.stat().st_size / (1024*1024):.1f} MB)")
    return out_file


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Runtime style switching")
    sub = parser.add_subparsers(dest="command")
    
    # Generate single style
    gen = sub.add_parser("generate", help="Generate model with specific style")
    gen.add_argument("style", choices=list(STYLES.keys()))
    gen.add_argument("--intensity", type=float, default=None)
    
    # Blend styles
    blend = sub.add_parser("blend", help="Blend multiple styles")
    blend.add_argument("styles", nargs="+", choices=list(STYLES.keys()))
    blend.add_argument("--intensities", nargs="+", type=float, default=None)
    
    # List styles
    sub.add_parser("list", help="List available styles")
    
    args = parser.parse_args()
    
    if args.command == "list":
        print(f"\n{'Style':<20} {'Description':<35} {'Delta':<20}")
        print("-" * 75)
        for name, style in STYLES.items():
            print(f"{name:<20} {style['desc']:<35} {style['delta']:<20}")
    
    elif args.command == "generate":
        intensity = args.intensity or STYLES[args.style]["intensity"]
        generate_model(args.style, intensity)
    
    elif args.command == "blend":
        intensities = args.intensities or [0.05] * len(args.styles)
        blend_styles(args.styles, intensities)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
