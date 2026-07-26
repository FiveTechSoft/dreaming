"""
Los 4 grupos de números del LLM
"""

def show_groups():
    """Mostrar los grupos de números"""
    
    print("="*60)
    print("LOS 4 GRUPOS DE NÚMEROS DEL LLM")
    print("="*60)
    
    print("""
    El cerebro del robot tiene 4 partes diferentes:
    
    ┌─────────────────────────────────────────────────────────┐
    │                                                         │
    │   1. EMBEDDINGS (El diccionario)                        │
    │      36,864,000 números                                 │
    │      Cada palabra tiene un "significado"                │
    │                                                         │
    │   2. QKV (El conversador)                               │
    │      278,528,000 números                                │
    │      Cómo las palabras se conectan entre sí             │
    │                                                         │
    │   3. FFN (La memoria)                                   │
    │      812,062,720 números                                │
    │      Todo lo que el robot SABE                          │
    │                                                         │
    │   4. LAYER NORMS (El estabilizador)                     │
    │      45,056 números                                     │
    │      Mantiene todo funcionando bien                     │
    │                                                         │
    │   TOTAL: 1,127,000,000 números                          │
    │                                                         │
    └─────────────────────────────────────────────────────────┘
    """)
    
    print("\n" + "="*60)
    print("1. EMBEDDINGS - El diccionario")
    print("="*60)
    
    print("""
    ¿Qué es?
    ─────────
    Es como un diccionario gigante
    
    Cada palabra tiene un "vector" (lista de números)
    que representa su significado
    
    Ejemplo:
    ┌────────────┬─────────────────────────────┐
    │ Palabra    │ Vector (simplificado)        │
    ├────────────┼─────────────────────────────┤
    │ "gato"     │ [0.2, -0.5, 0.8, ...]       │
    │ "perro"    │ [0.3, -0.4, 0.7, ...]       │
    │ "tristeza" │ [-0.8, 0.2, -0.3, ...]      │
    │ "alegría"  │ [0.9, -0.1, 0.6, ...]       │
    └────────────┴─────────────────────────────┘
    
    Nota: "gato" y "perro" tienen vectores SIMILARES
          "tristeza" y "alegría" tienen vectores OPUESTOS
    
    Si cambias ESTOS números:
    - Las palabras cambian de significado
    - "gato" puede significar "perro"
    - "tristeza" puede significar "alegría"
    
    Tamaño: 32,000 palabras × 1,152 números = 36,864,000
    """)
    
    print("\n" + "="*60)
    print("2. QKV - El conversador")
    print("="*60)
    
    print("""
    ¿Qué es?
    ─────────
    Es cómo las palabras "hablan" entre sí
    
    Q = Query (Pregunta): "¿Quién soy?"
    K = Key (Llave): "¿Quién eres?"
    V = Value (Valor): "¿Qué sabes?"
    
    Ejemplo:
    ┌─────────────────────────────────────────────────────────┐
    │ Frase: "El gato duerme sobre laalfombra"              │
    │                                                         │
    │ "gato" pregunta: ¿Quién soy?                           │
    │ "duerme" responde: ¡Yo! Soy la acción que haces        │
    │                                                         │
    │ Resultado: el modelo SABE que "gato" y "duerme"        │
    │ están conectados                                        │
    └─────────────────────────────────────────────────────────┘
    
    Si cambias ESTOS números:
    - Las palabras se conectan DIFERENTE
    - "gato" puede conectarse con "coche" en vez de "duerme"
    - La frase pierde sentido
    
    Tamaño: 
    - Q: 2,048 × 1,152 = 2,359,296
    - K: 256 × 1,152 = 294,912
    - V: 256 × 1,152 = 294,912
    - O: 2,048 × 1,152 = 2,359,296
    Total por capa: ~5.3 millones
    Total (22 capas): ~117 millones
    """)
    
    print("\n" + "="*60)
    print("3. FFN - La memoria")
    print("="*60)
    
    print("""
    ¿Qué es?
    ─────────
    Es DONDE ESTÁ EL CONOCIMIENTO
    
    Es como una biblioteca gigante donde el robot
    guarda todo lo que sabe
    
    Ejemplo:
    ┌─────────────────────────────────────────────────────────┐
    │ FFN contiene:                                          │
    │                                                         │
    │ - Que París es la capital de Francia                   │
    │ - Que 2 + 2 = 4                                        │
    │ - Que los gatos duermen mucho                           │
    │ - Que la vida tiene sentido (o no)                      │
    │                                                         │
    └─────────────────────────────────────────────────────────┘
    
    Si cambias ESTOS números:
    - El robot "olvida" cosas
    - O "aprende" cosas nuevas
    - O cambia CÓMO piensa las cosas
    
    Tamaño:
    - Gate: 5,632 × 1,152 = 6,488,064
    - Up: 5,632 × 1,152 = 6,488,064
    - Down: 2,048 × 3,168 = 6,487,040
    Total por capa: ~19.5 millones
    Total (22 capas): ~429 millones
    """)
    
    print("\n" + "="*60)
    print("4. LAYER NORMS - El estabilizador")
    print("="*60)
    
    print("""
    ¿Qué es?
    ─────────
    Es como un "termostato" que mantiene todo estable
    
    Sin esto, los números crecerían无穷 y el robot "explotaría"
    
    Ejemplo:
    ┌─────────────────────────────────────────────────────────┐
    │ Sin LayerNorm:                                         │
    │   Números: [1000, -500, 2000, -800, ...]  ← ¡CAOS!    │
    │                                                         │
    │ Con LayerNorm:                                         │
    │   Números: [0.5, -0.3, 0.8, -0.2, ...]  ← ESTABLE     │
    └─────────────────────────────────────────────────────────┘
    
    Si cambias ESTOS números:
    - El robot puede "explotar" (números infinitos)
    - O puede "morir" (todos los números = 0)
    - Generalmente, ¡no los toques!
    
    Tamaño: 1,152 × 2 = 2,304 por capa
    Total: 45,056
    """)
    
    print("\n" + "="*60)
    print("¿CUÁL TOCAMOS PRIMERO?")
    print("="*60)
    
    print("""
    OPCIÓN A: EMBEDDINGS (El diccionario)
    ──────────────────────────────────────
    Pros: Fácil de entender, cada cambio es un "significado"
    Contras: Menos poderoso, solo cambia palabras individuales
    
    OPCIÓN B: QKV (El conversador)
    ──────────────────────────────
    Pros: Cambia CÓMO se conectan las ideas
    Contras: Más difícil de entender, puede romper todo
    
    OPCIÓN C: FFN (La memoria)
    ──────────────────────────
    Pros: ¡Aquí está el CONOCIMIENTO! Los más poderoso
    Contras: Hay MUCHOS números, más difícil de analizar
    
    OPCIÓN D: TODOS JUNTOS
    ──────────────────────
    Pros: Vemos el efecto completo
    Contras: No sabemos qué causó qué
    
    RECOMENDACIÓN:
    ──────────────
    Empezar por EMBEDDINGS (más simple)
    Luego FFN (más poderoso)
    Luego QKV (más complejo)
    """)
    
    return True


def show_size_comparison():
    """Mostrar comparación de tamaños"""
    
    print("\n" + "="*60)
    print("COMPARACIÓN DE TAMAÑOS")
    print("="*60)
    
    print("""
    ┌──────────────────┬──────────────┬─────────────┬────────────┐
    │ Grupo            │ Números      │ Porcentaje  │ Dificultad │
    ├──────────────────┼──────────────┼─────────────┼────────────┤
    │ Embeddings       │ 36,864,000   │ 3.3%        │ Fácil      │
    │ QKV (atención)   │ 117,000,000  │ 10.4%       │ Media      │
    │ FFN (memoria)    │ 429,000,000  │ 38.1%       │ Difícil    │
    │ Layer Norms      │ 45,056       │ 0.004%      │ No tocar   │
    │ Otros            │ 544,090,944  │ 48.2%       │ ???        │
    ├──────────────────┼──────────────┼─────────────┼────────────┤
    │ TOTAL            │ 1,127,000,000│ 100%        │            │
    └──────────────────┴──────────────┴─────────────┴────────────┘
    
    Visualmente:
    
    Embeddings: ████ (3%)
    QKV:        ████████████ (10%)
    FFN:        ████████████████████████████████████████ (38%)
    LayerNorm:  ▏ (0.004%)
    Otros:      ████████████████████████████████████████████████████████ (48%)
    """)
    
    return True


if __name__ == "__main__":
    show_groups()
    show_size_comparison()
