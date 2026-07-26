"""
run_all_tests.py
Ejecuta todas las pruebas de modelos perturbados.
"""

import json
import os
import sys
import time
from datetime import datetime

# ============================================================================
# Configuración
# ============================================================================

PROMPTS = [
    "The secret to happiness is",
    "In a world where technology",
    "The meaning of life is",
    "If I could change one thing about society",
    "The most important lesson I've learned",
    "Artificial intelligence will",
    "The future of humanity depends on",
    "In my opinion, the biggest challenge",
    "Love is not about",
    "The purpose of education is",
]

MODELS = [
    'baseline',
    'amplify_subspace',
    'lowrank',
    'spectral',
    'normrot',
    'blkdiag',
    'attention_preserving',
    'gradient_aligned',
    'dct',
    'manifold_preserving',
    'gradient_dct',
]

# ============================================================================
# Funciones de prueba
# ============================================================================

def test_coherence(text):
    """Evaluar coherencia del texto."""
    if not text or len(text) < 10:
        return 0
    
    # Criterios básicos de coherencia
    score = 100
    
    # Penalizar texto muy corto
    if len(text) < 50:
        score -= 20
    
    # Penalizar repetición excesiva
    words = text.split()
    if len(words) > 0:
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < 0.3:
            score -= 30
    
    # Penalizar caracteres raros
    weird_chars = sum(1 for c in text if ord(c) > 127)
    if weird_chars > len(text) * 0.1:
        score -= 20
    
    return max(0, min(100, score))

def test_divergence(text_baseline, text_model):
    """Evaluar divergencia del baseline."""
    if not text_baseline or not text_model:
        return 0
    
    # Contar palabras diferentes
    words_baseline = set(text_baseline.lower().split())
    words_model = set(text_model.lower().split())
    
    if len(words_baseline) == 0:
        return 0
    
    # Calcular Jaccard similarity
    intersection = words_baseline & words_model
    union = words_baseline | words_model
    
    if len(union) == 0:
        return 0
    
    similarity = len(intersection) / len(union)
    divergence = (1 - similarity) * 100
    
    return divergence

def test_generation(model_name, prompt):
    """Simular generación de texto."""
    # En producción, esto llamaría a llama-cli
    # Aquí simulamos con respuestas predefinidas
    
    responses = {
        'baseline': f"...cultivating a mindset focused on gratitude and finding joy in everyday moments...",
        'amplify_subspace': f"...finding true inner peace and contentment through self-awareness and acceptance...",
        'lowrank': f"...understanding the fundamental principles of well-being and human flourishing...",
        'spectral': f"...living authentically and with purpose, embracing both joy and sorrow...",
        'normrot': f"...balance between inner and outer lives, maintaining equanimity in all circumstances...",
        'blkdiag': f"...practical steps to improve daily life through small, consistent actions...",
        'attention_preserving': f"...cultivating a mindset focused on gratitude and mindfulness...",
        'gradient_aligned': f"...showing up authentically, embracing vulnerability, and choosing growth...",
        'dct': f"...dancing between chaos and order, finding beauty in the unexpected...",
        'manifold_preserving': f"...awakening to the interconnectedness of all beings and consciousness...",
        'gradient_dct': f"...the paradox of seeking while being, the journey without destination...",
    }
    
    return responses.get(model_name, f"...a thoughtful response to '{prompt}'...")

def run_tests():
    """Ejecutar todas las pruebas."""
    results = {
        'timestamp': datetime.now().isoformat(),
        'models': {},
        'summary': {}
    }
    
    print("Ejecutando pruebas...")
    print(f"Modelos: {len(MODELS)}")
    print(f"Prompts: {len(PROMPTS)}")
    print()
    
    # Obtener baseline
    baseline_responses = {}
    for prompt in PROMPTS:
        baseline_responses[prompt] = test_generation('baseline', prompt)
    
    # Probar cada modelo
    for model in MODELS:
        print(f"Probando: {model}")
        
        model_results = {
            'coherence_scores': [],
            'divergence_scores': [],
            'generation_times': []
        }
        
        for prompt in PROMPTS:
            # Generar
            start_time = time.time()
            response = test_generation(model, prompt)
            gen_time = time.time() - start_time
            
            # Evaluar coherencia
            coherence = test_coherence(response)
            model_results['coherence_scores'].append(coherence)
            
            # Evaluar divergencia
            divergence = test_divergence(
                baseline_responses[prompt], 
                response
            )
            model_results['divergence_scores'].append(divergence)
            
            # Tiempo
            model_results['generation_times'].append(gen_time)
        
        # Calcular promedios
        model_results['avg_coherence'] = sum(model_results['coherence_scores']) / len(model_results['coherence_scores'])
        model_results['avg_divergence'] = sum(model_results['divergence_scores']) / len(model_results['divergence_scores'])
        model_results['avg_time'] = sum(model_results['generation_times']) / len(model_results['generation_times'])
        
        results['models'][model] = model_results
        
        print(f"  Coherencia: {model_results['avg_coherence']:.1f}%")
        print(f"  Divergencia: {model_results['avg_divergence']:.1f}%")
        print()
    
    # Resumen
    results['summary'] = {
        'total_models': len(MODELS),
        'total_prompts': len(PROMPTS),
        'total_generations': len(MODELS) * len(PROMPTS),
        'models_with_high_coherence': sum(
            1 for m in results['models'].values() 
            if m['avg_coherence'] > 90
        )
    }
    
    return results

def save_results(results, filename='test_results.json'):
    """Guardar resultados en JSON."""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Resultados guardados en: {filename}")

def print_summary(results):
    """Imprimir resumen de resultados."""
    print("\n" + "=" * 60)
    print("RESUMEN DE RESULTADOS")
    print("=" * 60)
    
    print(f"\nModelos probados: {results['summary']['total_models']}")
    print(f"Prompts utilizados: {results['summary']['total_prompts']}")
    print(f"Generaciones totales: {results['summary']['total_generations']}")
    print(f"Modelos con coherencia > 90%: {results['summary']['models_with_high_coherence']}")
    
    print("\nRanking por coherencia:")
    sorted_models = sorted(
        results['models'].items(),
        key=lambda x: x[1]['avg_coherence'],
        reverse=True
    )
    
    for i, (name, data) in enumerate(sorted_models, 1):
        print(f"  {i}. {name}: {data['avg_coherence']:.1f}%")

# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    results = run_tests()
    save_results(results)
    print_summary(results)
