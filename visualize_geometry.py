#!/usr/bin/env python3
"""
visualize_geometry.py — Visualize the geometry of perspective shifts.

Shows:
1. Weight distribution of base vs perturbed models
2. Projection onto random subspaces
3. How different techniques move in weight space
"""

import numpy as np
import struct
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path

Q4_BLOCK = 32
Q4_BYTES = 18
GGUF_MAGIC = b'GGUF'
BASE_MODEL = "C:/tmp/tinyllama-1.1b.Q4_0.gguf"


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


def load_tensor_weights(path, max_tensors=20):
    """Load first N Q4_0 tensors from model."""
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
    
    weights = []
    count = 0
    for tname, dims, ttype, toffset in tensor_infos:
        if ttype != 2:
            continue
        if 'norm' in tname or 'token_embd' in tname or 'output' in tname:
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
            weights.append((tname, values))
            count += 1
            if count >= max_tensors:
                break
    return weights


# ========== PERTURBATION TECHNIQUES ==========

def amplify_subspace(values, rng, intensity):
    vec = rng.standard_normal(len(values)).astype(np.float32)
    vec /= np.linalg.norm(vec) + 1e-9
    proj = np.dot(values, vec) * vec
    return values + intensity * proj, proj


def lowrank_amplify(values, rng, intensity):
    n = len(values)
    rows = int(np.sqrt(n))
    cols = n // rows
    if rows < 4 or cols < 4:
        return values, np.zeros_like(values)
    mat = values[:rows*cols].reshape(rows, cols)
    try:
        U, S, Vt = np.linalg.svd(mat, full_matrices=False)
        k = min(16, len(S))
        S_new = S.copy()
        S_new[:k] *= (1 + intensity)
        mat_new = (U[:, :k] * S_new[:k]) @ Vt[:k, :]
        result = values.copy()
        result[:rows*cols] = mat_new.flatten()
        delta = result - values
        return result, delta
    except:
        return values, np.zeros_like(values)


def normrot_preserve(values, rng, intensity):
    n = len(values)
    rows = int(np.sqrt(n))
    cols = n // rows
    if rows < 4 or cols < 4:
        return values, np.zeros_like(values)
    mat = values[:rows*cols].reshape(rows, cols)
    try:
        Q, _ = np.linalg.qr(rng.standard_normal((rows, rows)))
        angle = intensity * 0.01
        R = np.eye(rows) + angle * (Q - Q.T)
        R, _ = np.linalg.qr(R)
        mat_new = R @ mat @ R.T
        result = values.copy()
        result[:rows*cols] = mat_new.flatten()
        delta = result - values
        return result, delta
    except:
        return values, np.zeros_like(values)


def scaled_noise(values, rng, intensity):
    """This one BREAKS coherence."""
    noise = rng.standard_normal(len(values)).astype(np.float32)
    return values + intensity * noise * np.abs(values), noise * np.abs(values)


# ========== VISUALIZATION ==========

def plot_weight_distribution(weights_base, weights_styled, technique_name, output_dir):
    """Plot weight distributions before and after perturbation."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(f'{technique_name}: Weight Distribution Comparison', fontsize=14)
    
    for i, ax in enumerate(axes.flat):
        if i >= len(weights_base):
            break
        name_base, vals_base = weights_base[i]
        name_styled, vals_styled = weights_styled[i]
        
        ax.hist(vals_base, bins=50, alpha=0.5, label='Base', density=True, color='blue')
        ax.hist(vals_styled, bins=50, alpha=0.5, label='Styled', density=True, color='red')
        ax.set_title(name_base[:30])
        ax.legend(fontsize=8)
        ax.set_xlim(-2, 2)
    
    plt.tight_layout()
    plt.savefig(output_dir / f"{technique_name}_distributions.png", dpi=150)
    plt.close()


def plot_projection_geometry(values_base, values_styled, delta, technique_name, output_dir):
    """Visualize the perturbation as a geometric operation."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f'{technique_name}: Geometric Operation', fontsize=14)
    
    # 1. Weight space trajectory (2D projection)
    ax = axes[0]
    # Project onto first 2 PCA components of base weights
    rng = np.random.default_rng(42)
    n = min(10000, len(values_base))
    idx = rng.choice(len(values_base), n, replace=False)
    
    base_2d = values_base[idx]
    styled_2d = values_styled[idx]
    delta_2d = delta[idx]
    
    # Simple 2D projection using random vectors
    proj1 = rng.standard_normal(n)
    proj2 = rng.standard_normal(n)
    proj1 /= np.linalg.norm(proj1)
    proj2 /= np.linalg.norm(proj2)
    
    base_x = np.dot(base_2d, proj1)
    base_y = np.dot(base_2d, proj2)
    styled_x = np.dot(styled_2d, proj1)
    styled_y = np.dot(styled_2d, proj2)
    delta_x = np.dot(delta_2d, proj1)
    delta_y = np.dot(delta_2d, proj2)
    
    ax.scatter(base_x, base_y, alpha=0.3, s=1, label='Base', c='blue')
    ax.scatter(styled_x, styled_y, alpha=0.3, s=1, label='Styled', c='red')
    ax.annotate('', xy=(styled_x.mean(), styled_y.mean()),
                xytext=(base_x.mean(), base_y.mean()),
                arrowprops=dict(arrowstyle='->', color='green', lw=2))
    ax.set_title('Weight Space Trajectory')
    ax.legend()
    ax.set_xlabel('Projection 1')
    ax.set_ylabel('Projection 2')
    
    # 2. Delta magnitude distribution
    ax = axes[1]
    delta_magnitudes = np.abs(delta)
    ax.hist(delta_magnitudes, bins=50, color='green', alpha=0.7)
    ax.axvline(delta_magnitudes.mean(), color='red', linestyle='--', 
               label=f'Mean: {delta_magnitudes.mean():.4f}')
    ax.set_title('Perturbation Magnitude')
    ax.set_xlabel('|Δw|')
    ax.set_ylabel('Count')
    ax.legend()
    
    # 3. Correlation between base and delta
    ax = axes[2]
    sample_idx = rng.choice(len(values_base), min(5000, len(values_base)), replace=False)
    ax.scatter(values_base[sample_idx], delta[sample_idx], alpha=0.1, s=1)
    ax.set_title('Base Weight vs Perturbation')
    ax.set_xlabel('Base weight value')
    ax.set_ylabel('Perturbation Δw')
    ax.set_xlim(-2, 2)
    
    # Compute correlation
    corr = np.corrcoef(values_base[sample_idx], delta[sample_idx])[0, 1]
    ax.text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=ax.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_dir / f"{technique_name}_geometry.png", dpi=150)
    plt.close()


def plot_perspectives_comparison(all_deltas, output_dir):
    """Compare different perturbation directions."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle('Different Techniques = Different Directions in Weight Space', fontsize=14)
    
    techniques = list(all_deltas.keys())
    
    for i, ax in enumerate(axes.flat):
        if i >= len(techniques):
            break
        
        tech = techniques[i]
        delta = all_deltas[tech]
        
        # Show delta magnitude across weight index
        ax.plot(np.abs(delta), alpha=0.5, linewidth=0.5)
        ax.set_title(tech)
        ax.set_xlabel('Weight index')
        ax.set_ylabel('|Δw|')
        ax.set_ylim(0, 0.5)
    
    plt.tight_layout()
    plt.savefig(output_dir / "perspectives_comparison.png", dpi=150)
    plt.close()


def main():
    output_dir = Path("C:/tmp/dreaming/plots")
    output_dir.mkdir(exist_ok=True)
    
    print("Loading base model tensors...")
    base_weights = load_tensor_weights(BASE_MODEL, max_tensors=12)
    print(f"  Loaded {len(base_weights)} tensors")
    
    rng = np.random.default_rng(42)
    intensity = 0.10
    
    techniques = {
        'amplify_subspace': amplify_subspace,
        'lowrank': lowrank_amplify,
        'normrot': normrot_preserve,
        'scaled_noise': scaled_noise,
    }
    
    all_deltas = {}
    
    for tech_name, tech_func in techniques.items():
        print(f"\nApplying {tech_name}...")
        
        styled_weights = []
        deltas = []
        
        for name, values in base_weights:
            styled, delta = tech_func(values.copy(), rng, intensity)
            styled_weights.append((name, styled))
            deltas.append(delta)
            if len(deltas) == 1:
                all_deltas[tech_name] = delta  # Store first tensor delta for comparison
        
        # Plot distributions
        plot_weight_distribution(base_weights, styled_weights, tech_name, output_dir)
        
        # Plot geometry
        if len(base_weights) > 0:
            name_b, vals_b = base_weights[0]
            name_s, vals_s = styled_weights[0]
            plot_projection_geometry(vals_b, vals_s, deltas[0], tech_name, output_dir)
        
        print(f"  Saved plots for {tech_name}")
    
    # Plot comparison
    plot_perspectives_comparison(all_deltas, output_dir)
    
    print(f"\n{'='*50}")
    print(f"All plots saved to: {output_dir}")
    print(f"{'='*50}")
    print("\nFiles:")
    for f in sorted(output_dir.glob("*.png")):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
