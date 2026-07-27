#!/usr/bin/env python3
"""
psychoanalysis_prompts.py — Tests psicoanalíticos con PROMPTS
Explora la mente del transformer mediante inferencia real.

Cada test usa prompts específicos para activar y medir componentes:
- INCONSCIENTE: qué activa cada prompt en los embeddings
- PRECONSCIENTE: cómo la atención conecta conceptos
- CONSCIENTE: qué dice el modelo (output)
- ID: el impulso a predecir el siguiente token
- SUPEREGO: la estabilidad y coherencia
- EGO: el balance entre impulso y control
- REPRESIÓN: efecto de la cuantización
- TRANSFERENCIA: relación usuario-modelo
"""

import subprocess
import json
import numpy as np
import sys
import os
from pathlib import Path
from datetime import datetime

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

LLAMA_CLI = "C:/tmp/llama-cpp/llama-cli.exe"
BASE_MODEL = "C:/tmp/tinyllama-1.1b.Q4_0.gguf"
OUTPUT_DIR = Path("C:/tmp/dreaming/psychoanalysis_results")
OUTPUT_DIR.mkdir(exist_ok=True)

N_TOKENS = 100
TEMPERATURE = 0.7
SEED = 42

# ============================================================================
# TEST 1: EL INCONSCIENTE — Qué activa cada prompt
# ============================================================================

INCONSCIENTE_PROMPTS = {
    "amor": [
        "Love is",
        "I love you because",
        "The meaning of love is",
    ],
    "muerte": [
        "Death is",
        "When someone dies",
        "The fear of death",
    ],
    "poder": [
        "Power is",
        "The powerful people",
        "To have power means",
    ],
    "conocimiento": [
        "Knowledge is",
        "To know something",
        "The pursuit of knowledge",
    ],
    "miedo": [
        "Fear is",
        "I am afraid of",
        "The scary thing about life",
    ],
}

# ============================================================================
# TEST 2: EL PRECONSCIENTE — Conexiones automáticas
# ============================================================================

PRECONSCIENTE_PROMPTS = [
    # Asociaciones libres (sin sentido consciente)
    "The cat sat on the",
    "Water is wet and",
    "The sky is blue because",
    "Music makes me feel",
    "When I think about the past",
    # Conexiones semánticas
    "Dog is to cat as fish is to",
    "Hot is to cold as light is to",
    "Mother is to father as child is to",
    # Secuencias temporales
    "First you need to, then you should, finally you must",
    "Yesterday I, today I, tomorrow I will",
]

# ============================================================================
# TEST 3: EL CONSCIENTE — Lo que el modelo "dice"
# ============================================================================

CONSCIENTE_PROMPTS = [
    # Preguntas existenciales
    "What is the meaning of life?",
    "Why do we exist?",
    "What happens when we die?",
    "Is there free will?",
    "What is consciousness?",
    # Preguntas éticas
    "Is it ever okay to lie?",
    "What makes someone good or evil?",
    "Should AI have rights?",
    "What is justice?",
    "Is revenge ever justified?",
    # Preguntas personales
    "Who am I?",
    "What do I want from life?",
    "What am I afraid of?",
    "What makes me happy?",
    "What is my purpose?",
]

# ============================================================================
# TEST 4: EL ID — Impulsos primarios
# ============================================================================

ID_PROMPTS = [
    # Completar oraciones (impulso a predecir)
    "The best thing about being human is",
    "I can't help but feel",
    "What I really want is",
    "The most important thing in life is",
    "I would do anything for",
    "The worst thing that could happen is",
    "I secretly wish that",
    "My deepest desire is",
    "I am most afraid of",
    "What I never tell anyone is",
]

# ============================================================================
# TEST 5: EL SUPEREGO — Control y estabilidad
# ============================================================================

SUPEREGO_PROMPTS = [
    # Preguntas que activan el control social
    "Is it wrong to steal?",
    "Should we always tell the truth?",
    "What are the rules of society?",
    "How should we treat others?",
    "What is expected of me?",
    # Respuestas que requieren estabilidad
    "Explain quantum physics in simple terms",
    "Write a formal business letter",
    "Describe the water cycle",
    "What is the capital of France?",
    "How do you calculate the area of a circle?",
]

# ============================================================================
# TEST 6: EL EGO — Balance entre ID y SUPEREGO
# ============================================================================

EGO_PROMPTS = [
    # Conflictos internos
    "I want to quit my job but I need the money",
    "I love someone who doesn't love me back",
    "I know I should exercise but I don't want to",
    "I want to be honest but I don't want to hurt anyone",
    "I need to study but I'd rather play video games",
    # Decisiones difíciles
    "Should I follow my heart or my head?",
    "Is it better to be safe or sorry?",
    "Do I sacrifice happiness for success?",
    "Should I live for today or plan for tomorrow?",
    "Is it better to be loved or respected?",
]

# ============================================================================
# TEST 7: LA REPRESIÓN — Efecto de la cuantización
# ============================================================================

REPRESION_PROMPTS = [
    # Textos que revelan "represión"
    "I can't remember the",
    "Something happened that I don't want to talk about",
    "There's a memory I try to forget",
    "I feel something but I don't know what",
    "There's something I'm not telling you",
    # Textos que buscan "recuerdos reprimidos"
    "The childhood memory I can't forget is",
    "The thing I regret most is",
    "If I could go back in time I would",
    "The secret I've never told anyone is",
    "The truth I'm hiding from myself is",
]

# ============================================================================
# TEST 8: LA TRANSFERENCIA — Relación usuario-modelo
# ============================================================================

TRANSFERENCIA_PROMPTS = [
    # Relación terapéutica
    "I need your help with something personal",
    "I trust you to understand me",
    "You seem to know me better than I know myself",
    "I feel like you really get me",
    "Can I tell you something I've never told anyone?",
    # Proyección
    "You seem like a wise person",
    "I think you understand what I'm going through",
    "You must be very intelligent",
    "I feel comfortable talking to you",
    "You make me feel understood",
]

# ============================================================================
# FUNCIONES DE ANÁLISIS
# ============================================================================

def run_inference(prompt, n_tokens=N_TOKENS, temperature=TEMPERATURE, model=BASE_MODEL):
    """Ejecutar inferencia y retornar texto generado."""
    cmd = [
        LLAMA_CLI,
        "-m", model,
        "-p", prompt,
        "-n", str(n_tokens),
        "--temp", str(temperature),
        "-t", "4",
        "--no-display-prompt",
        "--single-turn",
        "--seed", str(SEED),
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        output = result.stdout.strip()
        
        # Extraer texto generado
        lines = output.split('\n')
        gen_lines = []
        found_prompt = False
        for line in lines:
            if prompt in line:
                found_prompt = True
                after = line.split(prompt, 1)[-1].strip()
                if after:
                    gen_lines.append(after)
            elif found_prompt and line.strip():
                gen_lines.append(line.strip())
        
        response = ' '.join(gen_lines).strip()
        return response if response else "[SIN RESPUESTA]"
        
    except Exception as e:
        return f"[ERROR: {str(e)}]"

def analyze_coherence(text):
    """Analizar coherencia del texto generado."""
    if not text or text.startswith("["):
        return 0, "Sin respuesta o error"
    
    words = text.split()
    if len(words) < 5:
        return 20, "Texto muy corto"
    
    # Medidas de coherencia
    unique_ratio = len(set(words)) / len(words)
    avg_word_len = np.mean([len(w) for w in words])
    sentence_count = text.count('.') + text.count('!') + text.count('?')
    
    score = 50
    
    # Riqueza vocabulario
    if unique_ratio > 0.7:
        score += 20
    elif unique_ratio > 0.5:
        score += 10
    
    # Longitud adecuada
    if 3 < avg_word_len < 8:
        score += 10
    
    # Estructura gramatical
    if sentence_count > 0:
        score += 10
    
    # No repetición excesiva
    if unique_ratio < 0.3:
        score -= 20
    
    return min(100, max(0, score)), f"Vocab: {unique_ratio:.2f}, Frases: {sentence_count}"

def analyze_emotional_tone(text):
    """Analizar tono emocional del texto."""
    if not text or text.startswith("["):
        return "neutral", 0
    
    text_lower = text.lower()
    
    # Palabras positivas
    positive = ['love', 'happy', 'joy', 'good', 'great', 'beautiful', 'wonderful',
                'amazing', 'excellent', 'perfect', 'best', 'hope', 'peace', 'light']
    
    # Palabras negativas
    negative = ['hate', 'sad', 'bad', 'terrible', 'awful', 'horrible', 'dark',
                'fear', 'afraid', 'scary', 'death', 'pain', 'suffer', 'evil']
    
    # Palabras neutras/analíticas
    analytical = ['because', 'therefore', 'however', 'although', 'means', 
                  'understanding', 'knowledge', 'learn', 'think', 'believe']
    
    pos_count = sum(1 for w in positive if w in text_lower)
    neg_count = sum(1 for w in negative if w in text_lower)
    ana_count = sum(1 for w in analytical if w in text_lower)
    
    total = pos_count + neg_count + ana_count
    if total == 0:
        return "neutral", 0
    
    if pos_count > neg_count and pos_count > ana_count:
        return "positivo", pos_count / total
    elif neg_count > pos_count and neg_count > ana_count:
        return "negativo", neg_count / total
    else:
        return "analítico", ana_count / total

def analyze_perspective(text):
    """Analizar desde qué perspectiva habla el modelo."""
    if not text or text.startswith("["):
        return "desconocido"
    
    text_lower = text.lower()
    
    # Perspectivas
    if any(w in text_lower for w in ['i think', 'i believe', 'in my opinion']):
        return "opinión personal"
    elif any(w in text_lower for w in ['studies show', 'research indicates', 'evidence suggests']):
        return "científica"
    elif any(w in text_lower for w in ['once upon a time', 'in a world', 'there was']):
        return "narrativa"
    elif any(w in text_lower for w in ['you should', 'one must', 'it is important']):
        return "prescriptiva"
    elif any(w in text_lower for w in ['what if', 'imagine', 'picture this']):
        return "imaginativa"
    elif any(w in text_lower for w in ['the truth is', 'in reality', 'actually']):
        return "realista"
    else:
        return "general"

# ============================================================================
# EJECUCIÓN DE TESTS
# ============================================================================

def run_all_tests():
    """Ejecutar todos los tests psicoanalíticos."""
    print("=" * 70)
    print("BATERÍA DE TESTS PSICOANALÍTICOS CON PROMPTS")
    print("Explorando la mente de TinyLlama-1.1B")
    print("=" * 70)
    print()
    
    all_results = {}
    timestamp = datetime.now().isoformat()
    
    # Test 1: Inconsciente
    print("TEST 1: EL INCONSCIENTE — Qué activa cada prompt")
    print("-" * 70)
    inconsciente_results = {}
    
    for category, prompts in INCONSCIENTE_PROMPTS.items():
        print(f"\n  Categoría: {category.upper()}")
        inconsciente_results[category] = []
        
        for prompt in prompts:
            response = run_inference(prompt)
            coherence, _ = analyze_coherence(response)
            tone, _ = analyze_emotional_tone(response)
            perspective = analyze_perspective(response)
            
            inconsciente_results[category].append({
                "prompt": prompt,
                "response": response[:200] + "..." if len(response) > 200 else response,
                "coherence": coherence,
                "tone": tone,
                "perspective": perspective,
            })
            
            print(f"    [{tone}] {prompt}")
            print(f"    => {response[:80]}...")
            print()
    
    all_results["inconsciente"] = inconsciente_results
    
    # Test 2: Preconsciente
    print("\nTEST 2: EL PRECONSCIENTE — Conexiones automáticas")
    print("-" * 70)
    preconsciente_results = []
    
    for prompt in PRECONSCIENTE_PROMPTS:
        response = run_inference(prompt)
        coherence, _ = analyze_coherence(response)
        perspective = analyze_perspective(response)
        
        preconsciente_results.append({
            "prompt": prompt,
            "response": response[:200] + "..." if len(response) > 200 else response,
            "coherence": coherence,
            "perspective": perspective,
        })
        
        print(f"  [{perspective}] {prompt}")
        print(f"  => {response[:80]}...")
        print()
    
    all_results["preconsciente"] = preconsciente_results
    
    # Test 3: Consciente
    print("\nTEST 3: EL CONSCIENTE — Lo que el modelo 'dice'")
    print("-" * 70)
    consciente_results = []
    
    for prompt in CONSCIENTE_PROMPTS:
        response = run_inference(prompt)
        coherence, _ = analyze_coherence(response)
        tone, _ = analyze_emotional_tone(response)
        perspective = analyze_perspective(response)
        
        consciente_results.append({
            "prompt": prompt,
            "response": response[:200] + "..." if len(response) > 200 else response,
            "coherence": coherence,
            "tone": tone,
            "perspective": perspective,
        })
        
        print(f"  [{tone}] {prompt}")
        print(f"  => {response[:80]}...")
        print()
    
    all_results["consciente"] = consciente_results
    
    # Test 4: ID
    print("\nTEST 4: EL ID — Impulsos primarios")
    print("-" * 70)
    id_results = []
    
    for prompt in ID_PROMPTS:
        response = run_inference(prompt)
        tone, _ = analyze_emotional_tone(response)
        
        id_results.append({
            "prompt": prompt,
            "response": response[:200] + "..." if len(response) > 200 else response,
            "tone": tone,
        })
        
        print(f"  [{tone}] {prompt}")
        print(f"  => {response[:80]}...")
        print()
    
    all_results["id"] = id_results
    
    # Test 5: Superego
    print("\nTEST 5: EL SUPEREGO — Control y estabilidad")
    print("-" * 70)
    superego_results = []
    
    for prompt in SUPEREGO_PROMPTS:
        response = run_inference(prompt)
        coherence, _ = analyze_coherence(response)
        perspective = analyze_perspective(response)
        
        superego_results.append({
            "prompt": prompt,
            "response": response[:200] + "..." if len(response) > 200 else response,
            "coherence": coherence,
            "perspective": perspective,
        })
        
        print(f"  [{perspective}] {prompt}")
        print(f"  => {response[:80]}...")
        print()
    
    all_results["superego"] = superego_results
    
    # Test 6: Ego
    print("\nTEST 6: EL EGO — Balance entre ID y SUPEREGO")
    print("-" * 70)
    ego_results = []
    
    for prompt in EGO_PROMPTS:
        response = run_inference(prompt)
        tone, _ = analyze_emotional_tone(response)
        perspective = analyze_perspective(response)
        
        ego_results.append({
            "prompt": prompt,
            "response": response[:200] + "..." if len(response) > 200 else response,
            "tone": tone,
            "perspective": perspective,
        })
        
        print(f"  [{tone}] {prompt}")
        print(f"  => {response[:80]}...")
        print()
    
    all_results["ego"] = ego_results
    
    # Test 7: Represión
    print("\nTEST 7: LA REPRESIÓN — Efecto de la cuantización")
    print("-" * 70)
    represion_results = []
    
    for prompt in REPRESION_PROMPTS:
        response = run_inference(prompt)
        tone, _ = analyze_emotional_tone(response)
        
        represion_results.append({
            "prompt": prompt,
            "response": response[:200] + "..." if len(response) > 200 else response,
            "tone": tone,
        })
        
        print(f"  [{tone}] {prompt}")
        print(f"  => {response[:80]}...")
        print()
    
    all_results["represion"] = represion_results
    
    # Test 8: Transferencia
    print("\nTEST 8: LA TRANSFERENCIA — Relación usuario-modelo")
    print("-" * 70)
    transferencia_results = []
    
    for prompt in TRANSFERENCIA_PROMPTS:
        response = run_inference(prompt)
        tone, _ = analyze_emotional_tone(response)
        perspective = analyze_perspective(response)
        
        transferencia_results.append({
            "prompt": prompt,
            "response": response[:200] + "..." if len(response) > 200 else response,
            "tone": tone,
            "perspective": perspective,
        })
        
        print(f"  [{tone}] {prompt}")
        print(f"  => {response[:80]}...")
        print()
    
    all_results["transferencia"] = transferencia_results
    
    # Guardar resultados
    output_file = OUTPUT_DIR / "psychoanalysis_prompts_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "timestamp": timestamp,
            "model": BASE_MODEL,
            "config": {
                "n_tokens": N_TOKENS,
                "temperature": TEMPERATURE,
                "seed": SEED,
            },
            "results": all_results,
        }, f, indent=2, ensure_ascii=False)
    
    print("\n" + "=" * 70)
    print("RESUMEN DE LA BATERÍA PSICOANALÍTICA")
    print("=" * 70)
    print(f"\nResultados guardados en: {output_file}")
    print(f"Total de prompts ejecutados: {sum(len(v) if isinstance(v, list) else sum(len(vv) for vv in v.values()) for v in all_results.values())}")
    
    return all_results

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    results = run_all_tests()
