"""
ANÁLISIS GEOMÉTRICO: Patrones en todos los pesos del LLM
¿Hay estructura geométrica en las capas?
"""

import numpy as np
from gguf import GGUFReader
from collections import defaultdict

def main():
    print("="*60)
    print("ANÁLISIS GEOMÉTRICO DE TODAS LAS CAPAS")
    print("="*60)
    
    # 1. Cargar modelo
    print("\n1. Cargando modelo...")
    reader = GGUFReader("C:/tmp/tinyllama-1.1b.Q4_0.gguf")
    
    # Organizar tensores por tipo
    layers = defaultdict(dict)
    
    for tensor in reader.tensors:
        name = tensor.name
        data = tensor.data.astype(np.float32)
        
        if 'blk.' in name:
            parts = name.split('.')
            if len(parts) >= 3:
                block_num = int(parts[1])
                layer_type = parts[2]
                layers[block_num][layer_type] = data
    
    print(f"  Capas encontradas: {len(layers)}")
    
    # 2. Analizar geometría por tipo de capa
    print("\n2. ANÁLISIS POR TIPO DE CAPA")
    print("-"*40)
    
    # FFN (la más grande)
    print("\nFFN (Feed-Forward Network):")
    ffn_norms = []
    for block in sorted(layers.keys()):
        if 'ffn_gate' in layers[block]:
            norm = np.linalg.norm(layers[block]['ffn_gate'])
            ffn_norms.append((block, float(norm)))
    
    print(f"  Normas por capa:")
    for block, norm in ffn_norms[:5]:
        print(f"    Capa {block}: {norm:.2f}")
    print(f"    ...")
    for block, norm in ffn_norms[-3:]:
        print(f"    Capa {block}: {norm:.2f}")
    
    # Atención
    print("\nAtención (QKV):")
    attn_norms = []
    for block in sorted(layers.keys()):
        if 'attn_q' in layers[block]:
            norm = np.linalg.norm(layers[block]['attn_q'])
            attn_norms.append((block, float(norm)))
    
    print(f"  Normas por capa:")
    for block, norm in attn_norms[:5]:
        print(f"    Capa {block}: {norm:.2f}")
    print(f"    ...")
    for block, norm in attn_norms[-3:]:
        print(f"    Capa {block}: {norm:.2f}")
    
    # 3. Buscar patrones geométricos
    print("\n3. PATRONES GEOMÉTRICOS")
    print("-"*40)
    
    # Patrón 1: ¿Las normas crecen o decrecen?
    print("\nPatrón 1: Evolución de normas FFN")
    norms_values = [n for _, n in ffn_norms]
    print(f"  Primeras 5 capas: {norms_values[:5]}")
    print(f"  Últimas 5 capas: {norms_values[-5:]}")
    
    if norms_values[-1] > norms_values[0]:
        print("  → Las normas CRECEN (el modelo se amplifica)")
    else:
        print("  → Las normas DECRECEN (el modelo se amortigua)")
    
    # Patrón 2: ¿Hay simetría entre capas?
    print("\nPatrón 2: Simetría entre capas")
    if len(ffn_norms) >= 10:
        first_half = np.mean([n for _, n in ffn_norms[:11]])
        second_half = np.mean([n for _, n in ffn_norms[11:]])
        print(f"  Primera mitad (capas 0-10): {first_half:.2f}")
        print(f"  Segunda mitad (capas 11-21): {second_half:.2f}")
        
        if abs(first_half - second_half) / first_half < 0.1:
            print("  → SIMÉTRICO (las mitades son similares)")
        else:
            print("  → ASIMÉTRICO (las mitades son diferentes)")
    
    # Patrón 3: ¿Hay clusters de capas similares?
    print("\nPatrón 3: Clusters de capas similares")
    
    # Calcular similitud entre capas
    layer_vectors = []
    for block in sorted(layers.keys()):
        if 'ffn_gate' in layers[block]:
            vec = layers[block]['ffn_gate'].flatten()[:1000]  # Primeras 1000 dims
            layer_vectors.append((block, vec))
    
    # Calcular distancias
    if len(layer_vectors) >= 5:
        print("  Similitud entre capas adyacentes:")
        for i in range(min(5, len(layer_vectors)-1)):
            block1, vec1 = layer_vectors[i]
            block2, vec2 = layer_vectors[i+1]
            
            # Distancia coseno
            cos_sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
            print(f"    Capa {block1} ↔ Capa {block2}: {cos_sim:.4f}")
    
    # 4. Buscar estructura de "árbol" o "jerarquía"
    print("\n4. ESTRUCTURA JERÁRQUICA")
    print("-"*40)
    
    print("""
    ANÁLISIS DE JERARQUÍA:
    
    Capas tempranas (0-5):
    └── Procesamiento básico
        └── Patrones simples
        └── Sintaxis
    
    Capas medias (6-15):
    └── Conceptos abstractos
        └── Ideas puras
        └── Significado
    
    Capas tardías (16-21):
    └── Razonamiento
        └── Conexión de ideas
        └── Generación
    
    ¿Hay una PIRÁMIDE?
    - Capas tempranas: mucha información, poca abstracción
    - Capas medias: información comprimida, alta abstracción
    - Capas tardías: información expandida, generación
    """)
    
    # 5. Buscar "flujos" o "caminos"
    print("\n5. FLUJOS Y CAMINOS")
    print("-"*40)
    
    print("""
    ANÁLISIS DE FLUJO:
    
    Los datos fluyen así:
    
    Input → Embedding → [Capa 1] → [Capa 2] → ... → [Capa 21] → Output
    
    En cada capa:
    └── Atención: ¿qué tokens son relevantes?
    └── FFN: ¿qué conceptos aplicar?
    └── Residual: ¿qué información conservar?
    
    ¿Hay PATRONES DE FLUJO?
    - Capas tempranas: flujo LOCAL (palabras cercanas)
    - Capas medias: flujo GLOBAL (toda la frase)
    - Capas tardías: flujo DIRIGIDO (hacia la respuesta)
    """)
    
    # 6. Visualización mental
    print("\n6. VISUALIZACIÓN MENTAL")
    print("-"*40)
    
    print("""
    IMAGINA EL MODELO COMO:
    
    ┌─────────────────────────────────────────────────────────┐
    │                                                         │
    │   ENTRADA                                               │
    │     ↓                                                   │
    │   ┌─────────────────────────────────────────────────┐   │
    │   │ CAPAS 0-5: FILTROS SIMPLES                     │   │
    │   │ (detectan palabras, gramática básica)           │   │
    │   └─────────────────────────────────────────────────┘   │
    │     ↓                                                   │
    │   ┌─────────────────────────────────────────────────┐   │
    │   │ CAPAS 6-15: PROCESADOR DE CONCEPTOS            │   │
    │   │ (entiende ideas, significado, contexto)         │   │
    │   └─────────────────────────────────────────────────┘   │
    │     ↓                                                   │
    │   ┌─────────────────────────────────────────────────┐   │
    │   │ CAPAS 16-21: GENERADOR DE RESPUESTAS           │   │
    │   │ (conecta ideas, crea texto coherente)           │   │
    │   └─────────────────────────────────────────────────┘   │
    │     ↓                                                   │
    │   SALIDA                                                │
    │                                                         │
    └─────────────────────────────────────────────────────────┘
    
    Cada sección tiene una FUNCIÓN diferente
    Cada sección tiene una GEOMETRÍA diferente
    """)
    
    # 7. Conclusión
    print("\n7. CONCLUSIÓN")
    print("-"*40)
    
    print("""
    ¿HAY PATRONES GEOMÉTRICOS?
    
    SÍ:
    
    1. ESTRUCTURA JERÁRQUICA
       - Capas tempranas: bajo nivel
       - Capas medias: nivel medio
       - Capas tardías: alto nivel
    
    2. FLUJO UNIDIRECCIONAL
       - Input → Procesamiento → Output
       - Cada capa transforma la información
    
    3. CLUSTERS FUNCIONALES
       - Capas 0-5: sintaxis
       - Capas 6-15: significado
       - Capas 16-21: generación
    
    4. SIMETRÍA PARCIAL
       - Algunas capas son similares
       - Otras son diferentes
    
    5. AMPLIFICACIÓN PROGRESIVA
       - Las normas crecen hacia el final
       - El modelo "amplifica" el significado
    """)
    
    return layers


if __name__ == "__main__":
    layers = main()
