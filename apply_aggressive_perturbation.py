"""
Perturba TinyLlama en la dirección de agresividad.
Usa la dirección calculada por find_semantic_tokens.py

Uso:
    python apply_aggressive_perturbation.py [scale]

    scale: Factor de escala (default: 0.001)
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import os

# Cargar dirección
aggression_dir = np.load("aggression_direction.npy")

# Parámetros
scale = float(sys.argv[1]) if len(sys.argv) > 1 else 0.001
output_dir = "perturbed_models"
os.makedirs(output_dir, exist_ok=True)

print(f"Escalando dirección de agresividad: {scale}")
print(f"Norma de la dirección: {np.linalg.norm(aggression_dir):.4f}")

# La dirección se usará con dmt_perturb_binary.py
# o con el método de perturbación existente

print("\nDirección lista para usar con dmt_perturb_binary.py")
print(f"\nPara usar:")
print(f"  1. Cargar aggression_direction.npy")
print(f"  2. Aplicar a los tensores de embedding")
print(f"  3. Guardar modelo modificado")
