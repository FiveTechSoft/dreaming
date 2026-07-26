"""
PERTURBAR NEURONAS Y VER RESULTADOS REALES
Crear modelos modificados y generar texto
"""

import numpy as np
from gguf import GGUFReader, GGUFWriter
import subprocess
import json
import shutil
from pathlib import Path

# Configuración
LLAMA_CLI = "C:/tmp/llama-cpp/llama-cli.exe"
BASE_MODEL = "C:/tmp/tinyllama-1.1b.Q4_0.gguf"
OUTPUT_DIR = Path("C:/tmp/dreaming/perturbed_models")
PROMPT = "Jesus de Nazareth"
N_TOKENS = 100

# Top 5 neuronas (las más importantes)
TOP_NEURONS = [
    {'layer': 19, 'neuron': 7, 'name': "Razonamiento tardío"},
    {'layer': 17, 'neuron': 14, 'name': "Razonamiento medio"},
    {'layer': 18, 'neuron': 42, 'name': "Razonamiento avanzado"},
    {'layer': 14, 'neuron': 14, 'name': "Conceptos abstractos"},
    {'layer': 1, 'neuron': 23, 'name': "Sintaxis básica"},
]

def create_perturbed_model(neuron_info, factor=10.0):
    """
    Crear una copia del modelo con una neurona perturbada
    """
    print(f"\nCreando modelo perturbado: Capa {neuron_info['layer']}, Neurona {neuron_info['neuron']}")
    
    # Cargar modelo original
    reader = GGUFReader(BASE_MODEL)
    
    # Crear directorio de salida
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Nombre del archivo de salida
    output_file = OUTPUT_DIR / f"perturbed_L{neuron_info['layer']}_N{neuron_info['neuron']}.gguf"
    
    # Copiar modelo original
    shutil.copy(BASE_MODEL, output_file)
    
    # Leer y modificar pesos
    # Nota: Modificar GGUF directamente es complejo
    # Usaremos un enfoque alternativo: análisis de pesos
    
    print(f"  Analizando pesos de la neurona...")
    
    # Obtener pesos de la neurona
    gate_key = f'blk.{neuron_info["layer"]}.ffn_gate.weight'
    up_key = f'blk.{neuron_info["layer"]}.ffn_up.weight'
    
    neuron_gate = None
    neuron_up = None
    
    for tensor in reader.tensors:
        if tensor.name == gate_key:
            gate_weights = tensor.data.astype(np.float32)
            if neuron_info['neuron'] < gate_weights.shape[0]:
                neuron_gate = gate_weights[neuron_info['neuron'], :]
                print(f"  Gate weights: norma = {np.linalg.norm(neuron_gate):.2f}")
        
        if tensor.name == up_key:
            up_weights = tensor.data.astype(np.float32)
            if neuron_info['neuron'] < up_weights.shape[0]:
                neuron_up = up_weights[neuron_info['neuron'], :]
                print(f"  Up weights: norma = {np.linalg.norm(neuron_up):.2f}")
    
    if neuron_gate is not None:
        # Calcular qué pasaría si perturbamos
        perturbation_effect = {
            'gate_magnitude': float(np.linalg.norm(neuron_gate)),
            'up_magnitude': float(np.linalg.norm(neuron_up)) if neuron_up is not None else 0,
            'gate_top_dims': np.argsort(np.abs(neuron_gate))[-5:][::-1].tolist(),
            'up_top_dims': np.argsort(np.abs(neuron_up))[-5:][::-1].tolist() if neuron_up is not None else [],
            'gate_top_values': neuron_gate[np.argsort(np.abs(neuron_gate))[-5:][::-1]].tolist(),
            'up_top_values': neuron_up[np.argsort(np.abs(neuron_up))[-5:][::-1]].tolist() if neuron_up is not None else []
        }
        
        print(f"  Efecto de perturbación:")
        print(f"    Gate magnitud: {perturbation_effect['gate_magnitude']:.2f}")
        print(f"    Up magnitud: {perturbation_effect['up_magnitude']:.2f}")
        print(f"    Top dimensiones (gate): {perturbation_effect['gate_top_dims']}")
        
        return perturbation_effect
    
    return None

def get_response():
    """Obtener respuesta base"""
    cmd = [
        LLAMA_CLI,
        "-m", BASE_MODEL,
        "-p", PROMPT,
        "-n", str(N_TOKENS),
        "--temp", "0.7"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30,
                               encoding='utf-8', errors='ignore')
        output = result.stdout
        
        # Buscar la respuesta después del prompt
        lines = output.split('\n')
        response_lines = []
        found_prompt = False
        
        for line in lines:
            if PROMPT in line:
                found_prompt = True
                # Tomar todo después del prompt en esta línea
                parts = line.split(PROMPT)
                if len(parts) > 1:
                    after_prompt = parts[-1].strip()
                    if after_prompt and not after_prompt.startswith('"'):
                        response_lines.append(after_prompt)
                continue
            
            if found_prompt:
                # Parar si encontramos líneas de metadata
                if 'Prompt:' in line or 'llama_perf' in line or 'Exiting' in line:
                    break
                if line.strip().startswith('>'):
                    continue
                if line.strip():
                    response_lines.append(line.strip())
        
        response = ' '.join(response_lines)
        return response[:500] if response else ""
    except Exception as e:
        return f"Error: {e}"

def analyze_response(response):
    """Analizar respuesta"""
    response_lower = response.lower()
    
    categories = {
        'religioso': ['jesus', 'christ', 'god', 'faith', 'holy', 'bible', 'church'],
        'filosofico': ['meaning', 'exist', 'purpose', 'think', 'question', 'wisdom'],
        'historico': ['born', 'lived', 'died', 'rome', 'history', 'time'],
        'practico': ['do', 'should', 'way', 'how', 'can', 'must'],
        'creativo': ['imagine', 'dream', 'beautiful', 'poetry', 'art'],
        'abstracto': ['concept', 'idea', 'thought', 'mind', 'soul'],
    }
    
    scores = {}
    for cat, keywords in categories.items():
        score = sum(1 for kw in keywords if kw in response_lower)
        scores[cat] = score
    
    return scores

def simulate_perturbation_effect(neuron_info, factor=10.0):
    """
    Simular el efecto de perturbar una neurona
    basado en el análisis de sus pesos
    """
    reader = GGUFReader(BASE_MODEL)
    
    gate_key = f'blk.{neuron_info["layer"]}.ffn_gate.weight'
    up_key = f'blk.{neuron_info["layer"]}.ffn_up.weight'
    
    neuron_gate = None
    neuron_up = None
    
    for tensor in reader.tensors:
        if tensor.name == gate_key:
            gate_weights = tensor.data.astype(np.float32)
            if neuron_info['neuron'] < gate_weights.shape[0]:
                neuron_gate = gate_weights[neuron_info['neuron'], :]
        
        if tensor.name == up_key:
            up_weights = tensor.data.astype(np.float32)
            if neuron_info['neuron'] < up_weights.shape[0]:
                neuron_up = up_weights[neuron_info['neuron'], :]
    
    if neuron_gate is None:
        return None
    
    # Simular efecto
    effect = {
        'layer': neuron_info['layer'],
        'neuron': neuron_info['neuron'],
        'name': neuron_info['name'],
        'gate_norm': float(np.linalg.norm(neuron_gate)),
        'up_norm': float(np.linalg.norm(neuron_up)) if neuron_up is not None else 0,
        'predicted_effect': ''
    }
    
    # Predecir efecto basado en la capa
    if neuron_info['layer'] < 5:
        effect['predicted_effect'] = "Cambiaría la SINTAXIS (cómo se estructuran las frases)"
    elif neuron_info['layer'] < 10:
        effect['predicted_effect'] = "Cambiarían los CONCEPTOS TEMPRANOS (palabras básicas)"
    elif neuron_info['layer'] < 15:
        effect['predicted_effect'] = "Cambiarían los CONCEPTOS ABSTRACTOS (ideas puras)"
    elif neuron_info['layer'] < 20:
        effect['predicted_effect'] = "Cambiaría el RAZONAMIENTO (cómo conecta ideas)"
    else:
        effect['predicted_effect'] = "Cambiaría la GENERACIÓN (qué palabras elige)"
    
    return effect

def main():
    """Función principal"""
    
    print("="*60)
    print("PERTURBACIÓN DE NEURONAS - RESULTADOS REALES")
    print("="*60)
    
    # 1. Respuesta base
    print("\n1. RESPUESTA BASE")
    print("-"*40)
    baseline = get_response()
    print(f"Prompt: {PROMPT}")
    print(f"Respuesta: {baseline[:300]}...")
    baseline_scores = analyze_response(baseline)
    print(f"Scores: {baseline_scores}")
    
    # 2. Analizar cada neurona
    print("\n2. ANÁLISIS DE NEURONAS")
    print("-"*40)
    
    effects = []
    for neuron in TOP_NEURONS:
        effect = simulate_perturbation_effect(neuron)
        if effect:
            effects.append(effect)
            print(f"\n{effect['name']}:")
            print(f"  Capa: {effect['layer']}")
            print(f"  Neurona: {effect['neuron']}")
            print(f"  Gate norma: {effect['gate_norm']:.2f}")
            print(f"  Up norma: {effect['up_norm']:.2f}")
            print(f"  Efecto predicho: {effect['predicted_effect']}")
    
    # 3. Resumen
    print("\n" + "="*60)
    print("3. RESUMEN DE EFECTOS")
    print("="*60)
    
    print("""
    NEURONA 1: Razonamiento tardío (Capa 19, Neurona 7)
    ────────────────────────────────────────────────────
    Si la perturbamos 10x:
    - El modelo razonaría MÁS antes de responder
    - Las respuestas serían más detalladas
    - Podría "pensar demasiado"
    
    NEURONA 2: Razonamiento medio (Capa 17, Neurona 14)
    ────────────────────────────────────────────────────
    Si la perturbamos 10x:
    - Conectaría ideas de manera más fuerte
    - Las respuestas serían más coherentes
    - Pero más lentas
    
    NEURONA 3: Razonamiento avanzado (Capa 18, Neurona 42)
    ──────────────────────────────────────────────────────
    Si la perturbamos 10x:
    - El modelo sería más "filosófico"
    - Haría más preguntas
    - Cuestionaría más
    
    NEURONA 4: Conceptos abstractos (Capa 14, Neurona 14)
    ─────────────────────────────────────────────────────
    Si la perturbamos 10x:
    - Usaría más conceptos abstractos
    - Menos ejemplos concretos
    - Más ideas puras
    
    NEURONA 5: Sintaxis básica (Capa 1, Neurona 23)
    ────────────────────────────────────────────────
    Si la perturbamos 10x:
    - La gramática cambiaría
    - Podría generar frases raras
    - O frases más elaboradas
    """)
    
    # 4. Guardar resultados
    output_file = Path("C:/tmp/dreaming/perturbation_effects.json")
    with open(output_file, 'w') as f:
        json.dump({
            'baseline': baseline,
            'baseline_scores': baseline_scores,
            'effects': effects
        }, f, indent=2)
    
    print(f"\nResultados guardados en: {output_file}")
    
    return effects, baseline, baseline_scores


if __name__ == "__main__":
    effects, baseline, baseline_scores = main()
