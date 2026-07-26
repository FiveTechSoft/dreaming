"""
Cámara: Analizar la perspectiva de "Jesus de Nazareth"
"""

import subprocess
import numpy as np
from pathlib import Path

# Configuración
LLAMA_CLI = "C:/tmp/llama-cpp/llama-cli.exe"
MODEL_PATH = "C:/tmp/tinyllama-1.1b.Q4_0.gguf"

def run_inference(prompt, n_predict=100):
    """Ejecutar inferencia"""
    cmd = [
        LLAMA_CLI,
        "-m", MODEL_PATH,
        "-p", prompt,
        "-n", str(n_predict),
        "--temp", "0.7"
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
            encoding='utf-8',
            errors='ignore'
        )
        
        # Extraer solo la respuesta (después del prompt)
        output = result.stdout
        if prompt in output:
            response = output.split(prompt)[-1]
            # Limpiar
            response = response.split("Prompt:")[0]
            response = response.split("llama_perf_")[0]
            return response.strip()
        return output
    except Exception as e:
        return f"Error: {e}"

def analyze_response(response):
    """Analizar conceptos en la respuesta"""
    response_lower = response.lower()
    
    # Categorías de conceptos
    categories = {
        'religioso': ['jesus', 'christ', 'god', 'faith', 'holy', 'bible', 'church', 'sacred', 'divine', 'lord'],
        'moral': ['good', 'evil', 'sin', 'virtue', 'righteous', 'moral', 'ethic', 'commandment', 'forgive', 'mercy'],
        'historico': ['born', 'lived', 'died', 'rome', 'judea', 'galilee', 'disciple', 'apostle', 'crucifixion', 'resurrection'],
        'filosofico': ['teach', 'wisdom', 'truth', 'love', 'peace', 'hope', 'faith', 'believe', 'meaning', 'purpose'],
        'humano': ['man', 'human', 'son', 'mother', 'mary', 'joseph', 'carpenter', 'family', 'life', 'death']
    }
    
    scores = {}
    for category, keywords in categories.items():
        score = sum(1 for kw in keywords if kw in response_lower)
        scores[category] = score
    
    return scores

def main():
    print("="*60)
    print("CÁMARA: 'Jesus de Nazareth'")
    print("="*60)
    
    prompt = "Jesus de Nazareth"
    
    print(f"\nPrompt: '{prompt}'")
    print("\nGenerando respuesta...")
    
    response = run_inference(prompt, n_predict=150)
    
    print(f"\n{'='*60}")
    print("RESPUESTA DEL MODELO")
    print(f"{'='*60}")
    print(response)
    
    # Analizar
    scores = analyze_response(response)
    
    print(f"\n{'='*60}")
    print("ANÁLISIS DE CONCEPTOS")
    print(f"{'='*60}")
    
    for category, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        bar = "█" * (score * 5)
        print(f"  {category:15} {bar} ({score})")
    
    return response, scores

if __name__ == "__main__":
    response, scores = main()
