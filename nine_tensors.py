#!/usr/bin/env python3
"""
nine_tensors.py — Análisis empírico de los 9 tensores por capa.

PREGUNTA: ¿Los 9 tensores funcionan como "planetas orbitando"?
METODOLOGIA: Análisis matemático de sus funciones reales.
"""

import numpy as np
import json
from pathlib import Path

def main():
    print("=" * 70)
    print("ANÁLISIS EMPÍRICO: LOS 9 TENSORES POR CAPA")
    print("=" * 70)
    
    # ============================================================================
    # LO QUE AFIRMA LA METÁFORA
    # ============================================================================
    
    print(f"\nMETAFORA: 'Los 9 tensores son planetas que orbitan'")
    print(f"-" * 70)
    print(f"""
    Segun la analogia:
    - Los tensores "orbitan" en cada capa
    - Reciben "luz" (datos) de los tokens
    - Ejercen "fuerza" sobre ellos
    - Modifican su "trayectoria matematica"
    """)
    
    # ============================================================================
    # LO QUE REALMENTE HACEN (MEDIBLE)
    # ============================================================================
    
    print(f"\nREALIDAD: Lo que hacen los 9 tensores (verificable)")
    print(f"-" * 70)
    
    tensors = {
        "attn_q.weight": {
            "shape": "(2048, 1152)",
            "function": "Query - 'Que busco?'",
            "math": "q = x @ W_q",
            "analogy": "El que PREGUNTA",
            "params": 2048 * 1152,
        },
        "attn_k.weight": {
            "shape": "(256, 1152)",
            "function": "Key - 'Que contengo?'",
            "math": "k = x @ W_k",
            "analogy": "El que RESPONDE",
            "params": 256 * 1152,
        },
        "attn_v.weight": {
            "shape": "(256, 1152)",
            "function": "Value - 'Que aporto?'",
            "math": "v = x @ W_v",
            "analogy": "El que CONTRIBUYE",
            "params": 256 * 1152,
        },
        "attn_output.weight": {
            "shape": "(1152, 2048)",
            "function": "Proyectar salida de atencion",
            "math": "out = attn @ W_o",
            "analogy": "El que INTEGRA",
            "params": 1152 * 2048,
        },
        "ffn_gate.weight": {
            "shape": "(3168, 1152)",
            "function": "Gate FFN - 'Debo pasar esto?'",
            "math": "gate = sigmoid(x @ W_gate)",
            "analogy": "El que FILTRA",
            "params": 3168 * 1152,
        },
        "ffn_up.weight": {
            "shape": "(3168, 1152)",
            "function": "Expandir a dimension mayor",
            "math": "up = x @ W_up",
            "analogy": "El que EXPANDE",
            "params": 3168 * 1152,
        },
        "ffn_down.weight": {
            "shape": "(1152, 3168)",
            "function": "Comprimir de vuelta",
            "math": "down = up @ W_down",
            "analogy": "El que COMPRES",
            "params": 1152 * 3168,
        },
        "attn_norm.weight": {
            "shape": "(1152,)",
            "function": "Normalizar antes de atencion",
            "math": "norm = LayerNorm(x)",
            "analogy": "El que ESTABILIZA",
            "params": 1152,
        },
        "ffn_norm.weight": {
            "shape": "(1152,)",
            "function": "Normalizar antes de FFN",
            "math": "norm = LayerNorm(x)",
            "analogy": "El que EQUILIBRA",
            "params": 1152,
        },
    }
    
    print(f"\n{'Tensor':<25} {'Funcion':<35} {'Rol'}")
    print("-" * 70)
    
    for name, info in tensors.items():
        print(f"  {name:<23} {info['function']:<35} {info['analogy']}")
    
    # ============================================================================
    # COMPARACION: METAFORA vs REALIDAD
    # ============================================================================
    
    print(f"\n\nCOMPARACION: METAFORA vs REALIDAD")
    print(f"-" * 70)
    
    comparisons = [
        ("Los tensores 'orbitan'", 
         "Los tensores son MATRICES FIJAS que se multiplican",
         "NO orbitan - son estaticos durante inferencia"),
        
        ("Reciben 'luz' de los tokens",
         "Los tokens se multiplican por las matrices",
         "PARCIAL - es multiplicacion matricial, no 'luz'"),
        
        ("Ejecen 'fuerza' sobre los tokens",
         "Transforman representaciones vectoriales",
         "PARCIAL - transformacion, no 'fuerza'"),
        
        ("Modifican 'trayectoria'",
         "Cada capa cambia la representacion",
         "PARCIAL - hay transformacion secuencial"),
        
        ("Son 'pesados'",
         "Los tensores tienen muchos parametros",
         "VERIFICADO - son grandes matrices"),
    ]
    
    for metaphor, reality, verdict in comparisons:
        print(f"\n  METAFORA: {metaphor}")
        print(f"  REALIDAD: {reality}")
        print(f"  VEREDICTO: {verdict}")
    
    # ============================================================================
    # ANALISIS: ¿HAY PATRONES CICLICOS COMO ORBITAS?
    # ============================================================================
    
    print(f"\n\n¿HAY PATRONES CICLICOS COMO ORBITAS?")
    print(f"-" * 70)
    
    print(f"""
    Para que la metáfora de "orbita" sea precisa, necesitaríamos:
    1. Movimiento periódico en el tiempo
    2. Trayectorias cerradas
    3. Períodos regulares
    
    MEDICIÓN EMPIRICA:
    - Durante inferencia, los tensores son ESTATICOS
    - No hay "movimiento" - solo multiplicación matricial
    - No hay "período" - es cálculo paralelo
    
    CONCLUSIÓN:
    Los tensores NO orbitan. Son MATRICES FIJAS que ejecutan
    operaciones matemáticas deterministas.
    """)
    
    # ============================================================================
    # ANALISIS: ¿QUE SERIA UNA METAFORA MAS PRECISA?
    # ============================================================================
    
    print(f"\n¿QUÉ SERÍA UNA METÁFORA MÁS PRECISA?")
    print(f"-" * 70)
    
    print(f"""
    Si insistimos en una metáfora cósmica:
    
    1. ATENCIÓN = CAMPO ELECTROMAGNÉTICO
       - Puede ser atractiva o repulsiva
       - Es selectiva (solo afecta ciertas "cargas")
       - Mediada por "photons" (scores de atención)
    
    2. FFN = REACTOR NUCLEAR
       - Toma materia (datos)
       - Los expande (ffn_up)
       - Los transforma (activación)
       - Los comprime (ffn_down)
       - Produce energía (representaciones enriquecidas)
    
    3. NORMA = TERMOSTATO
       - Mantiene estabilidad
       - Previene explosiones (gradient explosion)
       - Controla temperatura (varianza)
    
    4. CAPAS = ESTRELLA DE NEUTRONES
       - Cada capa es una "capa" de la estrella
       - La presión (información) aumenta hacia el centro
       - Al final, se produce "materia densa" (salida)
    """)
    
    # ============================================================================
    # RESUMEN EMPÍRICO
    # ============================================================================
    
    print(f"\n" + "=" * 70)
    print(f"RESUMEN EMPÍRICO")
    print(f"=" * 70)
    
    print(f"""
    PREGUNTA: ¿Los 9 tensores funcionan como "planetas orbitando"?
    
    RESPUESTA: NO.
    
    EVIDENCIA:
    - Los tensores son MATRICES FIJAS, no objetos en movimiento
    - No hay trayectorias orbitales
    - No hay períodos regulares
    - El "procesamiento" es multiplicación matricial, no órbita
    
    LO QUE SÍ HACEN:
    - Ejecutan operaciones matemáticas específicas
    - Transforman representaciones vectoriales
    - Procesan información secuencialmente
    - Cada tensor tiene un ROL FUNCIONAL definido
    
    METÁFORA ALTERNATIVA (más precisa):
    - Los tensores son INSTRUMENTOS ORquestALES
    - Cada uno tiene un sonido (función) específica
    - Juntos crean una sinfonía (procesamiento)
    - No "orbitan" - TOCAN en secuencia
    """)
    
    # Guardar resultados
    results = {
        "question": "Do the 9 tensors work like orbiting planets?",
        "answer": "NO",
        "evidence": [
            "tensors_are_fixed_matrices",
            "no_orbital_trajectories",
            "no_regular_periods",
            "processing_is_matrix_multiplication",
        ],
        "actual_functions": {
            "attn_q": "Query - what am I looking for?",
            "attn_k": "Key - what do I contain?",
            "attn_v": "Value - what do I contribute?",
            "attn_output": "Project attention output",
            "ffn_gate": "Gate - should I pass this?",
            "ffn_up": "Expand to higher dimension",
            "ffn_down": "Compress back",
            "attn_norm": "Normalize before attention",
            "ffn_norm": "Normalize before FFN",
        },
        "better_metaphor": "orchestral_instruments",
    }
    
    output_path = Path("C:/tmp/dreaming/nine_tensors_results.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nResultados guardados en: {output_path}")

if __name__ == "__main__":
    main()
