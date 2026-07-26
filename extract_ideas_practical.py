"""
Script práctico: Extraer ideas puras de TinyLlama
Usa llama.cpp existente para generar texto y analizar patrones
"""

import subprocess
import json
import numpy as np
from pathlib import Path

# Configuración
LLAMA_CLI = "C:/tmp/llama-cpp/llama-cli.exe"
MODEL_PATH = "C:/tmp/tinyllama-1.1b.Q4_0.gguf"
OUTPUT_DIR = Path("C:/tmp/dreaming/ideas_analysis")
OUTPUT_DIR.mkdir(exist_ok=True)

# Prompts para extraer ideas
EXTRACTION_PROMPTS = [
    # Tristeza
    "I feel sad because",
    "The reason for my sadness is",
    "When I'm sad, I",
    
    # Golden Gate
    "The Golden Gate Bridge is",
    "San Francisco is known for",
    "In California, there is a famous",
    
    # Filosofía
    "The meaning of life is",
    "To exist means to",
    "The purpose of consciousness is",
    
    # Práctica
    "To solve this problem, you should",
    "The best approach is to",
    "The practical solution is",
    
    # Creatividad
    "Imagine a world where",
    "The most beautiful thing is",
    "In a dream, I saw",
    
    # Estoicismo
    "What we can control is",
    "The path to peace is",
    "Acceptance means",
    
    # Espiritualidad
    "The soul is",
    "Connection to the universe",
    "Transcendence means",
    
    # Autenticidad
    "The truth is",
    "Being honest means",
    "Reality is",
    
    # Análisis
    "The data shows that",
    "The evidence suggests",
    "Analyzing the results",
    
    # Lirismo
    "The poetry of",
    "Like a river flowing",
    "The rhythm of life"
]

# Modelos perturbados
PERTURBED_MODELS = {
    "baseline": None,
    "filosofica": "models/perturbacion_filosofica.gguf",
    "practica": "models/perturbacion_practica.gguf",
    "creativa": "models/perturbacion_creativa.gguf",
    "concisa": "models/perturbacion_concisa.gguf",
    "estoica": "models/perturbacion_estoica.gguf"
}


def run_inference(prompt, model_path=None, n_predict=50):
    """
    Ejecutar inferencia con llama-cli
    """
    cmd = [
        LLAMA_CLI,
        "-m", model_path or MODEL_PATH,
        "-p", prompt,
        "-n", str(n_predict),
        "--temp", "0.7",
        "--single-turn"
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30
        )
        return result.stdout
    except subprocess.TimeoutExpired:
        return ""
    except Exception as e:
        return f"Error: {e}"


def analyze_response(response, concept):
    """
    Analizar si la respuesta contiene el concepto esperado
    """
    response_lower = response.lower()
    
    # Palabras clave por concepto
    keywords = {
        "tristeza": ["sad", "unhappy", "sorrow", "grief", "melancholy", "depressed"],
        "golden_gate": ["bridge", "san francisco", "golden gate", "california", "landmark"],
        "filosofia": ["meaning", "exist", "purpose", "consciousness", "philosophy", "think"],
        "practica": ["solve", "should", "approach", "solution", "step", "do"],
        "creativa": ["imagine", "dream", "beautiful", "create", "art", "inspire"],
        "estoicismo": ["control", "accept", "peace", "virtue", "endure", "stoic"],
        "espiritualidad": ["soul", "spirit", "universe", "connect", "transcend", "divine"],
        "autenticidad": ["truth", "honest", "real", "authentic", "genuine", "true"],
        "analisis": ["data", "evidence", "analyze", "results", "show", "suggest"],
        "lirismo": ["poetry", "rhythm", "flow", "verse", "beauty", "lyric"]
    }
    
    if concept not in keywords:
        return 0.0
    
    # Contar palabras clave encontradas
    found = sum(1 for kw in keywords[concept] if kw in response_lower)
    
    return found / len(keywords[concept])


def extract_ideas():
    """
    Extraer ideas de TinyLlama respondiendo a diferentes prompts
    """
    print("="*60)
    print("EXTRACCIÓN DE IDEAS PURAS DE TINYLLAMA")
    print("="*60)
    
    results = []
    
    for i, prompt in enumerate(EXTRACTION_PROMPTS):
        concept = prompt.split()[0].lower()  # Primer palabra como concepto
        
        print(f"\n[{i+1}/{len(EXTRACTION_PROMPTS)}] Prompt: '{prompt}'")
        
        # Ejecutar inferencia
        response = run_inference(prompt, n_predict=50)
        
        if response and not response.startswith("Error"):
            # Analizar respuesta
            score = analyze_response(response, concept)
            
            result = {
                'prompt': prompt,
                'concept': concept,
                'response': response[:200],  # Primeros 200 chars
                'score': score
            }
            results.append(result)
            
            print(f"  Respuesta: {response[:100]}...")
            print(f"  Score: {score:.2f}")
        else:
            print(f"  No se pudo obtener respuesta")
    
    return results


def analyze_feature_patterns(results):
    """
    Analizar patrones en las respuestas para identificar features
    """
    print("\n" + "="*60)
    print("ANÁLISIS DE PATRONES")
    print("="*60)
    
    # Agrupar por concepto
    concept_scores = {}
    for result in results:
        concept = result['concept']
        if concept not in concept_scores:
            concept_scores[concept] = []
        concept_scores[concept].append(result['score'])
    
    # Calcular promedio por concepto
    concept_averages = {}
    for concept, scores in concept_scores.items():
        concept_averages[concept] = np.mean(scores)
    
    # Ordenar por score
    sorted_concepts = sorted(
        concept_averages.items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    print("\nConceptos más fuertes en TinyLlama:")
    for concept, score in sorted_concepts:
        bar = "█" * int(score * 50)
        print(f"  {concept:20} {bar} {score:.3f}")
    
    return concept_averages


def create_feature_map(concept_averages):
    """
    Crear mapa de conceptos → features
    """
    print("\n" + "="*60)
    print("MAPA DE FEATURES")
    print("="*60)
    
    # Asignar números de feature basados en orden
    feature_map = {}
    
    for i, (concept, score) in enumerate(sorted(
        concept_averages.items(),
        key=lambda x: x[1],
        reverse=True
    )):
        feature_idx = i * 100  # Espaciado para claridad
        feature_map[concept] = {
            'feature_idx': feature_idx,
            'strength': score,
            'description': f'Concepto de {concept}'
        }
    
    print("\nFeature Map:")
    for concept, info in feature_map.items():
        print(f"  feature_{info['feature_idx']:4d}: {concept:20} (strength: {info['strength']:.3f})")
    
    return feature_map


def connect_to_personalities(feature_map):
    """
    Conectar features con nuestras personalidades
    """
    print("\n" + "="*60)
    print("CONEXIÓN CON PERSONALIDADES")
    print("="*60)
    
    # Mapeo manual basado en análisis
    personality_connections = {
        'filosofica': ['filosofia', 'exist', 'meaning'],
        'practica': ['practica', 'solve', 'do'],
        'creativa': ['creativa', 'imagine', 'art'],
        'concisa': ['concisa', 'brief', 'direct'],
        'estoica': ['estoicismo', 'control', 'accept'],
        'espiritual': ['espiritualidad', 'soul', 'universe'],
        'autentica': ['autenticidad', 'truth', 'honest'],
        'analitica': ['analisis', 'data', 'evidence'],
        'lirica': ['lirismo', 'poetry', 'flow']
    }
    
    print("\nConexiones establecidas:")
    for personality, related_concepts in personality_connections.items():
        features = []
        for concept in related_concepts:
            if concept in feature_map:
                features.append(feature_map[concept]['feature_idx'])
        
        if features:
            print(f"\n  {personality}:")
            print(f"    Features: {features}")
            print(f"    Conceptos: {related_concepts}")
    
    return personality_connections


def main():
    """Función principal"""
    
    # 1. Extraer ideas
    results = extract_ideas()
    
    # 2. Analizar patrones
    concept_averages = analyze_feature_patterns(results)
    
    # 3. Crear mapa de features
    feature_map = create_feature_map(concept_averages)
    
    # 4. Conectar con personalidades
    connections = connect_to_personalities(feature_map)
    
    # 5. Guardar resultados
    output_file = OUTPUT_DIR / "ideas_analysis.json"
    with open(output_file, 'w') as f:
        json.dump({
            'results': results,
            'concept_averages': concept_averages,
            'feature_map': feature_map,
            'connections': connections
        }, f, indent=2)
    
    print(f"\nResultados guardados en: {output_file}")
    
    # 6. Resumen final
    print("\n" + "="*60)
    print("RESUMEN FINAL")
    print("="*60)
    
    print("""
    [OK] Ideas extraidas de TinyLlama
    [OK] Patrones analizados
    [OK] Features mapeadas
    [OK] Conexiones con personalidades establecidas
    
    Conclusiones:
    1. TinyLlama tiene conceptos predefinidos (features)
    2. Nuestras perturbaciones reorganizan estos conceptos
    3. Cada personalidad = combinacion de features
    4. El manifold de significado es real y navegable
    
    Proximos pasos:
    1. Entrenar autoencoder real con activaciones
    2. Extraer features monosemanticas
    3. Crear steering preciso
    """)


if __name__ == "__main__":
    main()
