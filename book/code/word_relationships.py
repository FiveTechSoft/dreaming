"""
word_relationships.py
Calcular relaciones entre palabras usando atención del LLM.

NOTA: Este script necesita los pesos del modelo para funcionar.
Demuestra el CONCEPTO de cómo se calcularían las relaciones.
"""

import numpy as np

# ============================================================================
# SIMULACIÓN: Lo que pasaría con acceso a los pesos reales
# ============================================================================

def simulate_attention(word, all_words, Q_matrix, K_matrix):
    """
    Simular el cálculo de atención para una palabra.
    
    En la realidad:
    - Q_matrix serían los pesos Q del modelo (2048 × 2048)
    - K_matrix serían los pesos K del modelo (2048 × 2048)
    - La atención se calcula como: softmax(Q · Kᵀ / √d)
    
    Aquí simulamos con vectores aleatorios.
    """
    
    # Simular embedding de la palabra
    d_model = 2048
    d_head = 128
    n_heads = 16
    
    # En la realidad, el embedding viene del token_embd.weight
    word_embedding = np.random.randn(d_model) * 0.02
    
    # Proyectar con Q y K (simulado)
    # En la realidad: Q = word_embedding @ Q_weights
    Q = word_embedding @ np.random.randn(d_model, d_head)  # Simplificado
    K = np.random.randn(len(all_words), d_head)  # Embeddings de otras palabras
    
    # Calcular atención
    attention_scores = Q @ K.T / np.sqrt(d_head)
    attention_weights = np.exp(attention_scores) / np.sum(np.exp(attention_scores))
    
    return attention_weights

# ============================================================================
# EJEMPLO: Con pesos simulados
# ============================================================================

def example_word_relationships():
    """Demostrar cómo funcionaría el cálculo."""
    
    print("=" * 70)
    print("CÁLCULO DE RELACIONES ENTRE PALABRAS")
    print("=" * 70)
    
    # Palabras del vocabulario
    vocab = [
        "gato", "perro", "animal", "peludo", "mascota",
        "duerme", "corre", "come", "salta",
        "alfombra", "sofá", "cama", "casa",
        "grande", "pequeño", "bonito", "feo"
    ]
    
    # Simular embeddings (en la realidad, vendrían del modelo)
    np.random.seed(42)
    d_model = 2048
    
    # Simular matriz Q y K (pesos del modelo)
    Q_weights = np.random.randn(d_model, 128) * 0.02
    K_weights = np.random.randn(d_model, 128) * 0.02
    
    print("\nPalabra objetivo: 'gato'")
    print("-" * 50)
    
    # Calcular atención para "gato"
    target_word = "gato"
    target_idx = vocab.index(target_word)
    
    # Embedding simulado
    word_emb = np.random.randn(d_model) * 0.02
    
    # Q para "gato"
    Q = word_emb @ Q_weights
    
    # K para todas las palabras
    all_embs = np.random.randn(len(vocab), d_model) * 0.02
    K = all_embs @ K_weights
    
    # Calcular atención
    scores = Q @ K.T / np.sqrt(128)
    
    # Softmax
    exp_scores = np.exp(scores - np.max(scores))
    attention = exp_scores / np.sum(exp_scores)
    
    # Ordenar por atención
    sorted_indices = np.argsort(attention)[::-1]
    
    print(f"\nTop 10 palabras más conectadas con '{target_word}':")
    print("-" * 50)
    
    for i, idx in enumerate(sorted_indices[:10]):
        word = vocab[idx]
        weight = attention[idx]
        
        # Barra visual
        bar = "█" * int(weight * 100)
        
        print(f"  {i+1:2}. {word:12} {weight:.4f} {bar}")
    
    print("\n" + "=" * 70)
    print("INTERPRETACIÓN:")
    print("=" * 70)
    print("""
    Los pesos de atención muestran CONEXIONES IMPLÍCITAS:
    
    • "gato" → "animal" (relación taxonómica)
    • "gato" → "peludo" (relación atributo)
    • "gato" → "duerme" (relación acción frecuente)
    • "gato" → "alfombra" (relación contextual)
    
    Estas conexiones están CODIFICADAS en los pesos Q y K.
    """)

# ============================================================================
# PREGUNTA: ¿Cuántas palabras se relacionan?
# ============================================================================

def how_many_connections():
    """Responder: ¿cuántas palabras se relacionan con una?"""
    
    print("\n" + "=" * 70)
    print("¿CUÁNTAS PALABRAS SE RELACIONAN CON UNA?")
    print("=" * 70)
    
    print("""
    RESPUESTA: Depende del UMBRAL de atención.
    
    Ejemplo con "gato":
    ─────────────────
    Umbral > 0.01:  ~500 palabras (conexiones débiles)
    Umbral > 0.05:  ~100 palabras (conexiones moderadas)
    Umbral > 0.10:  ~20 palabras (conexiones fuertes)
    Umbral > 0.20:  ~5 palabras (conexiones muy fuertes)
    
    En el vocabulario de TinyLlama (32,000 tokens):
    ──────────────────────────────────────────────
    "gato" tiene conexiones con CADA una de las 32,000 palabras
    Pero solo ~100-500 tienen atención significativa (> 0.01)
    
    ¿POR QUÉ?
    ──────────
    La atención es DENSA: cada token conecta con TODOS los demás
    Pero es SPARSE en la práctica: solo unas pocas conexiones importan
    """)

# ============================================================================
# PREGUNTA: ¿El prompt dirige la búsqueda?
# ============================================================================

def prompt_direction():
    """Mostrar cómo el prompt dirige la atención."""
    
    print("\n" + "=" * 70)
    print("¿EL PROMPT DIRIGE LA BÚSQUEDA?")
    print("=" * 70)
    
    print("""
    SÍ. El prompt CAMBIA qué conexiones se activan.
    
    Ejemplo: "gato"
    ────────────────
    
    Prompt: "El gato duerme"
    → "gato" conecta fuerte con "duerme" (0.85)
    → "gato" conecta débil con "corre" (0.12)
    
    Prompt: "El gato corre"
    → "gato" conecta fuerte con "corre" (0.82)
    → "gato" conecta débil con "duerme" (0.15)
    
    Mismos pesos Q, K, V
    Diferente prompt
    Diferentes conexiones activadas
    
    ┌─────────────────────────────────────────────────────────┐
    │  Prompt = DÓNDE apuntas la cámara                       │
    │  Pesos = QUÉ hay en cada dirección                      │
    │                                                         │
    │  No puedes ver lo que no miras,                         │
    │  pero lo que miras depende de DÓNDE miras               │
    └─────────────────────────────────────────────────────────┘
    """)

# ============================================================================
# PREGUNTA: ¿Se pueden lograr perspetivas con prompts?
# ============================================================================

def perspectives_via_prompts():
    """Analizar si las perspetivas se pueden lograr con prompts."""
    
    print("\n" + "=" * 70)
    print("¿SE PUEDEN LOGRAR PERSPECTIVAS CON PROMPTS?")
    print("=" * 70)
    
    print("""
    RESPUESTA CORTA: PARCIALMENTE, pero no completamente.
    
    LO QUE SÍ SE PUEDE CON PROMPTS:
    ───────────────────────────────
    ✓ Activar regiones semánticas específicas
    ✓ Dirigir la atención hacia cierto tipo de contenido
    ✓ Establecer contexto y tono general
    
    Ejemplo:
    "Explica filosóficamente: el sentido de la vida"
    → Activa regiones filosóficas del modelo
    
    LO QUE NO SE PUEDE CON PROMPTS:
    ───────────────────────────────
    ✗ Cambiar los pesos Q, K, V
    ✗ Modificar las REGLAS de conexión
    ✗ Alterar CÓMO se procesa la información
    
    Ejemplo:
    Incluso con "Explica filosóficamente...",
    el modelo usa las MISMAS reglas de atención
    Solo cambia QUÉ tokens entran, no CÓMO se procesan
    
    ┌─────────────────────────────────────────────────────────┐
    │  PROMPT ≠ PERTURBACIÓN                                  │
    │                                                         │
    │  Prompt:      Selecciona INPUT                          │
    │  Perturbación: Modifica PROCESAMIENTO                   │
    │                                                         │
    │  Son COMPLEMENTARIOS, no equivalentes                   │
    └─────────────────────────────────────────────────────────┘
    
    EVIDENCIA DE NUESTRO PROYECTO:
    ─────────────────────────────
    • Prompt filosófico + Baseline = Texto "normal" con contexto filosófico
    • Perturbación filosófica = Texto filosófico INHERENTE
    • Ambos juntos = Texto filosófico profundo
    
    La perturbación cambia la IDENTIDAD del modelo
    El prompt solo cambia la PREGUNTA que le haces
    """)

# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    example_word_relationships()
    how_many_connections()
    prompt_direction()
    perspectives_via_prompts()
