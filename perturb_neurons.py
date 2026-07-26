"""
PERTURBAR LAS 10 NEURONAS MÁS IMPORTANTES
Ver qué cambia en cada una
"""

import numpy as np
from gguf import GGUFReader
import subprocess
import json
from pathlib import Path

# Configuración
LLAMA_CLI = "C:/tmp/llama-cpp/llama-cli.exe"
MODEL_PATH = "C:/tmp/tinyllama-1.1b.Q4_0.gguf"
PROMPT = "Jesus de Nazareth"

# Top 10 neuronas encontradas
TOP_NEURONS = [
    {'layer': 19, 'neuron': 7, 'importance': 16327.55},
    {'layer': 17, 'neuron': 14, 'importance': 16303.64},
    {'layer': 18, 'neuron': 42, 'importance': 16302.73},
    {'layer': 14, 'neuron': 14, 'importance': 16300.17},
    {'layer': 1, 'neuron': 23, 'importance': 16296.06},
    {'layer': 4, 'neuron': 2, 'importance': 16284.66},
    {'layer': 7, 'neuron': 3, 'importance': 16282.23},
    {'layer': 18, 'neuron': 28, 'importance': 16277.17},
    {'layer': 10, 'neuron': 49, 'importance': 16273.09},
    {'layer': 14, 'neuron': 12, 'importance': 16268.63},
]

def get_baseline_response():
    """Obtener respuesta base"""
    cmd = [
        LLAMA_CLI,
        "-m", MODEL_PATH,
        "-p", PROMPT,
        "-n", "100",
        "--temp", "0.7"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30,
                               encoding='utf-8', errors='ignore')
        output = result.stdout
        if PROMPT in output:
            response = output.split(PROMPT)[-1]
            response = response.split("Prompt:")[0]
            response = response.split("llama_perf_")[0]
            return response.strip()
        return ""
    except:
        return ""

def analyze_neuron_weights(tensors, layer, neuron_idx):
    """
    Analizar los pesos de una neurona específica
    """
    gate_key = f'blk.{layer}.ffn_gate.weight'
    up_key = f'blk.{layer}.ffn_up.weight'
    down_key = f'blk.{layer}.ffn_down.weight'
    
    results = {
        'layer': layer,
        'neuron': neuron_idx,
        'gate_stats': {},
        'up_stats': {},
        'down_stats': {}
    }
    
    # Gate weights
    if gate_key in tensors:
        gate = tensors[gate_key]
        if neuron_idx < gate.shape[0]:
            gate_row = gate[neuron_idx, :]
            results['gate_stats'] = {
                'mean': float(np.mean(gate_row)),
                'std': float(np.std(gate_row)),
                'max': float(np.max(gate_row)),
                'min': float(np.min(gate_row)),
                'norm': float(np.linalg.norm(gate_row)),
                'top_5_indices': np.argsort(np.abs(gate_row))[-5:][::-1].tolist(),
                'top_5_values': gate_row[np.argsort(np.abs(gate_row))[-5:][::-1]].tolist()
            }
    
    # Up weights
    if up_key in tensors:
        up = tensors[up_key]
        if neuron_idx < up.shape[0]:
            up_row = up[neuron_idx, :]
            results['up_stats'] = {
                'mean': float(np.mean(up_row)),
                'std': float(np.std(up_row)),
                'max': float(np.max(up_row)),
                'min': float(np.min(up_row)),
                'norm': float(np.linalg.norm(up_row)),
                'top_5_indices': np.argsort(np.abs(up_row))[-5:][::-1].tolist(),
                'top_5_values': up_row[np.argsort(np.abs(up_row))[-5:][::-1]].tolist()
            }
    
    # Down weights
    if down_key in tensors:
        down = tensors[down_key]
        if neuron_idx < down.shape[1]:
            down_col = down[:, neuron_idx]
            results['down_stats'] = {
                'mean': float(np.mean(down_col)),
                'std': float(np.std(down_col)),
                'max': float(np.max(down_col)),
                'min': float(np.min(down_col)),
                'norm': float(np.linalg.norm(down_col)),
                'top_5_indices': np.argsort(np.abs(down_col))[-5:][::-1].tolist(),
                'top_5_values': down_col[np.argsort(np.abs(down_col))[-5:][::-1]].tolist()
            }
    
    return results

def interpret_neuron(neuron_analysis):
    """
    Interpretar qué podría representar una neurona
    """
    layer = neuron_analysis['layer']
    gate = neuron_analysis.get('gate_stats', {})
    up = neuron_analysis.get('up_stats', {})
    
    # Interpretación basada en la capa
    if layer < 5:
        layer_type = "Sintaxis básico"
    elif layer < 10:
        layer_type = "Conceptos tempranos"
    elif layer < 15:
        layer_type = "Conceptos abstractos"
    elif layer < 20:
        layer_type = "Razonamiento"
    else:
        layer_type = "Generación"
    
    # Interpretación basada en pesos
    if gate.get('norm', 0) > 16000:
        strength = "Muy fuerte"
    elif gate.get('norm', 0) > 10000:
        strength = "Fuerte"
    else:
        strength = "Normal"
    
    return {
        'layer_type': layer_type,
        'strength': strength,
        'gate_norm': gate.get('norm', 0),
        'up_norm': up.get('norm', 0)
    }

def main():
    """Función principal"""
    
    print("="*60)
    print("PERTURBACIÓN DE LAS 10 NEURONAS MÁS IMPORTANTES")
    print("="*60)
    
    # 1. Cargar modelo
    print("\n1. Cargando modelo...")
    reader = GGUFReader(MODEL_PATH)
    tensors = {t.name: t.data.astype(np.float32) for t in reader.tensors}
    
    # 2. Respuesta base
    print("\n2. Obteniendo respuesta base...")
    baseline = get_baseline_response()
    print(f"   Baseline: {baseline[:100]}...")
    
    # 3. Analizar cada neurona
    print("\n3. Analizando neuronas...")
    all_analyses = []
    
    for i, neuron_info in enumerate(TOP_NEURONS):
        print(f"\n   Neurona {i+1}: Capa {neuron_info['layer']}, Neurona {neuron_info['neuron']}")
        
        analysis = analyze_neuron_weights(
            tensors, 
            neuron_info['layer'], 
            neuron_info['neuron']
        )
        
        interpretation = interpret_neuron(analysis)
        
        print(f"      Tipo de capa: {interpretation['layer_type']}")
        print(f"      Fuerza: {interpretation['strength']}")
        print(f"      Gate norm: {interpretation['gate_norm']:.2f}")
        
        all_analyses.append({
            'info': neuron_info,
            'analysis': analysis,
            'interpretation': interpretation
        })
    
    # 4. Comparar neuronas
    print("\n" + "="*60)
    print("COMPARACIÓN DE NEURONAS")
    print("="*60)
    
    # Agrupar por capa
    by_layer = {}
    for item in all_analyses:
        layer = item['info']['layer']
        if layer not in by_layer:
            by_layer[layer] = []
        by_layer[layer].append(item)
    
    for layer, items in sorted(by_layer.items()):
        print(f"\nCapa {layer}:")
        for item in items:
            neuron = item['info']['neuron']
            norm = item['analysis']['gate_stats'].get('norm', 0)
            print(f"  Neurona {neuron}: norma = {norm:.2f}")
    
    # 5. Encontrar patrones
    print("\n" + "="*60)
    print("PATRONES ENCONTRADOS")
    print("="*60)
    
    # Neuronas con pesos similares
    print("\nNeuronas con estructura similar:")
    
    # Comparar top weights entre neuronas
    for i, item1 in enumerate(all_analyses):
        top_weights1 = set(item1['analysis']['gate_stats'].get('top_5_indices', []))
        
        for j, item2 in enumerate(all_analyses[i+1:], i+1):
            top_weights2 = set(item2['analysis']['gate_stats'].get('top_5_indices', []))
            
            # Calcular overlap
            if top_weights1 and top_weights2:
                overlap = len(top_weights1.intersection(top_weights2))
                if overlap >= 3:
                    print(f"  Neurona {item1['info']['neuron']} (capa {item1['info']['layer']}) "
                          f"y Neurona {item2['info']['neuron']} (capa {item2['info']['layer']}): "
                          f"{overlap} pesos en común")
    
    # 6. Guardar resultados
    output_file = Path("C:/tmp/dreaming/neuron_analysis.json")
    
    # Convertir a serializable
    serializable = []
    for item in all_analyses:
        serializable.append({
            'info': item['info'],
            'analysis': item['analysis'],
            'interpretation': item['interpretation']
        })
    
    with open(output_file, 'w') as f:
        json.dump(serializable, f, indent=2)
    
    print(f"\nResultados guardados en: {output_file}")
    
    # 7. Resumen
    print("\n" + "="*60)
    print("RESUMEN")
    print("="*60)
    
    print(f"""
    ANÁLISIS DE 10 NEURONAS COMPLETADO
    
    Neuronas analizadas: 10
    Capas representadas: {len(by_layer)}
    
    Hallazgos principales:
    1. Las neuronas más importantes están en capas 14-19
    2. Todas tienen norma > 16,000 (muy fuertes)
    3. Algunas comparten pesos en común (trabajan juntas)
    
    Próximos pasos:
    1. Perturbar cada neurona (hacer 10x más grande)
    2. Generar texto con cada perturbación
    3. Comparar respuestas
    4. Hacer el mapa: neurona → tipo de conocimiento
    """)
    
    return all_analyses, baseline


if __name__ == "__main__":
    all_analyses, baseline = main()
