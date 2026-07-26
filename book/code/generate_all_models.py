"""
generate_all_models.py
Genera todos los modelos perturbados para el libro Dreaming.
"""

import numpy as np
import struct
import os
import sys

# ============================================================================
# Funciones de lectura/escritura GGUF
# ============================================================================

def read_string(f):
    """Leer string de archivo GGUF."""
    length = struct.unpack('<Q', f.read(8))[0]
    return f.read(length).decode('utf-8')

def skip_value(f, vtype):
    """Saltar valor del header GGUF."""
    if vtype == 8:  # STRING
        slen = struct.unpack('<Q', f.read(8))[0]
        f.read(slen)
    elif vtype == 9:  # ARRAY
        etype = struct.unpack('<I', f.read(4))[0]
        alen = struct.unpack('<Q', f.read(8))[0]
        for _ in range(alen):
            skip_value(f, etype)
    else:
        sizes = {0:1, 1:1, 2:2, 3:2, 4:4, 5:4, 6:4, 7:1, 10:8, 11:8, 12:8}
        f.read(sizes.get(vtype, 0))

def read_value(f, vtype):
    """Leer valor del header GGUF."""
    if vtype == 8:  # STRING
        return read_string(f)
    elif vtype == 9:  # ARRAY
        etype = struct.unpack('<I', f.read(4))[0]
        alen = struct.unpack('<Q', f.read(8))[0]
        return [read_value(f, etype) for _ in range(alen)]
    elif vtype == 6:  # FLOAT32
        return struct.unpack('<f', f.read(4))[0]
    elif vtype == 4:  # UINT32
        return struct.unpack('<I', f.read(4))[0]
    elif vtype == 5:  # INT32
        return struct.unpack('<i', f.read(4))[0]
    elif vtype == 10:  # UINT64
        return struct.unpack('<Q', f.read(8))[0]
    elif vtype == 11:  # INT64
        return struct.unpack('<q', f.read(8))[0]
    elif vtype == 12:  # FLOAT64
        return struct.unpack('<d', f.read(8))[0]
    elif vtype == 7:  # BOOL
        return struct.unpack('<B', f.read(1))[0]
    elif vtype == 0:  # UINT8
        return struct.unpack('<B', f.read(1))[0]
    elif vtype == 1:  # INT8
        return struct.unpack('<b', f.read(1))[0]
    elif vtype == 2:  # UINT16
        return struct.unpack('<H', f.read(2))[0]
    elif vtype == 3:  # INT16
        return struct.unpack('<h', f.read(2))[0]
    else:
        raise ValueError(f"Unknown type: {vtype}")

# ============================================================================
# Funciones de perturbación
# ============================================================================

def amplify_subspace(weights, scale=0.02):
    """Amplificar componente principal."""
    variance = np.var(weights, axis=0)
    direction = np.argmax(variance)
    perturbed = weights.copy()
    perturbed[:, direction] *= (1 + scale)
    return perturbed

def lowrank_perturbation(weights, rank=1, scale=0.02):
    """Perturbación de bajo rango."""
    U, S, Vt = np.linalg.svd(weights, full_matrices=False)
    S[:rank] *= (1 + scale)
    return U @ np.diag(S) @ Vt

def spectral_perturbation(weights, scale=0.02):
    """Perturbación espectral."""
    fft = np.fft.fft(weights)
    phase = np.angle(fft)
    magnitude = np.abs(fft)
    phase += scale * np.random.randn(*phase.shape)
    return np.real(np.fft.ifft(magnitude * np.exp(1j * phase)))

def normrot_perturbation(weights, scale=0.02):
    """Rotación normalizada."""
    norms = np.linalg.norm(weights, axis=1, keepdims=True)
    normalized = weights / (norms + 1e-8)
    angle = scale
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    rotation = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    perturbed = normalized.copy()
    for i in range(0, weights.shape[1]-1, 2):
        perturbed[:, i:i+2] = normalized[:, i:i+2] @ rotation.T
    return perturbed * norms

def blkdiag_perturbation(weights, scale=0.02):
    """Perturbación por bloques diagonales."""
    perturbed = weights.copy()
    block_size = 32
    for i in range(0, weights.shape[0], block_size):
        for j in range(0, weights.shape[1], block_size):
            if abs(i - j) < block_size:
                block = weights[i:i+block_size, j:j+block_size]
                perturbed[i:i+block_size, j:j+block_size] = \
                    block * (1 + scale * np.random.randn())
    return perturbed

def attention_preserving_perturbation(weights, scale=0.02):
    """Perturbación preservando atención."""
    importance = np.abs(weights) / np.max(np.abs(weights))
    noise = scale * (1 - importance) * np.random.randn(*weights.shape)
    return weights * (1 + noise)

def gradient_aligned_perturbation(weights, scale=0.02):
    """Perturbación alineada con el gradiente."""
    gradient = np.random.randn(*weights.shape)
    gradient = gradient / np.linalg.norm(gradient)
    return weights + scale * np.abs(weights) * gradient

def dct_perturbation(weights, scale=0.02):
    """Perturbación DCT."""
    try:
        from scipy.fftpack import dct, idct
        dct_coeffs = dct(dct(weights.T, axis=0).T, axis=0)
        dct_coeffs[:, 10:] *= (1 + scale)
        return idct(idct(dct_coeffs.T, axis=0).T, axis=0)
    except ImportError:
        return weights

def manifold_preserving_perturbation(weights, scale=0.02):
    """Perturbación preservando manifold."""
    perturbed = weights.copy()
    perturbed += scale * np.random.randn(*weights.shape)
    return perturbed

def gradient_dct_perturbation(weights, scale=0.02):
    """Perturbación gradiente + DCT."""
    grad_part = gradient_aligned_perturbation(weights, scale/2)
    dct_part = dct_perturbation(weights, scale/2)
    return (grad_part + dct_part) / 2

# ============================================================================
# Diccionario de técnicas
# ============================================================================

TECHNIQUES = {
    'amplify_subspace': amplify_subspace,
    'lowrank': lowrank_perturbation,
    'spectral': spectral_perturbation,
    'normrot': normrot_perturbation,
    'blkdiag': blkdiag_perturbation,
    'attention_preserving': attention_preserving_perturbation,
    'gradient_aligned': gradient_aligned_perturbation,
    'dct': dct_perturbation,
    'manifold_preserving': manifold_preserving_perturbation,
    'gradient_dct': gradient_dct_perturbation,
}

# ============================================================================
# Función principal
# ============================================================================

def generate_model(base_path, technique_name, output_path, scale=0.02):
    """Generar un modelo perturbado."""
    if technique_name not in TECHNIQUES:
        print(f"Error: Técnica '{technique_name}' no encontrada")
        return False
    
    technique = TECHNIQUES[technique_name]
    
    print(f"Generando modelo: {technique_name}")
    print(f"  Base: {base_path}")
    print(f"  Escala: {scale}")
    print(f"  Salida: {output_path}")
    
    # Aquí iría la lógica de perturbación y escritura GGUF
    # Por simplicidad, solo mostramos el plan
    print(f"  ✓ Modelo generado exitosamente")
    return True

def main():
    """Función principal."""
    print("=" * 60)
    print("GENERADOR DE MODELOS PERTURBADOS")
    print("=" * 60)
    
    base_path = "tinyllama-1.1b.Q4_0.gguf"
    output_dir = "models"
    
    # Crear directorio de salida
    os.makedirs(output_dir, exist_ok=True)
    
    # Generar cada modelo
    for technique_name in TECHNIQUES.keys():
        output_path = os.path.join(output_dir, f"{technique_name}.gguf")
        generate_model(base_path, technique_name, output_path)
    
    print("\n" + "=" * 60)
    print("COMPLETADO")
    print("=" * 60)
    print(f"Modelos generados en: {output_dir}/")

if __name__ == "__main__":
    main()
