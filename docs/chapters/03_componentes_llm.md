# Capítulo 3: Los Componentes de un LLM

## ¿Qué hay dentro de un LLM?

Un LLM (Large Language Model) como TinyLlama es como una **fábrica de palabras**. Tiene muchas máquinas que trabajan juntas para producir texto.

```
┌─────────────────────────────────────────────────────────┐
│                    TINYLLAMA                            │
│                                                         │
│   ┌─────────┐   ┌─────────┐   ┌─────────┐             │
│   │ Capa 0  │ → │ Capa 1  │ → │ Capa 2  │ → ... → Salida │
│   └─────────┘   └─────────┘   └─────────┘             │
│                                                         │
│   Cada capa tiene los mismos componentes:               │
│   - Atención (conecta tokens)                          │
│   - FFN (procesa información)                          │
│   - Norm (estabiliza)                                  │
└─────────────────────────────────────────────────────────┘
```

---

## 1. Embeddings (La Puerta de Entrada)

### ¿Qué es?
Convierte palabras en números para que el modelo las entienda.

### Ejemplo Simple
```
"Manzana" → [0.12, -0.45, 0.78, ...] (2048 números)
"Perro"   → [0.34, 0.67, -0.23, ...] (2048 números)
```

### En TinyLlama
- **Tensor**: `token_embd.weight`
- **Dimensiones**: 32,000 tokens × 2,048 dimensiones
- **Parámetros**: 65.5M (6% del total)

### Analogía
Es como un **diccionario gigante** que traduce palabras a un idioma que la máquina entiende.

---

## 2. Atención (Las Conexiones)

### ¿Qué es?
Conecta cada palabra con las demás para entender el contexto.

### Ejemplo Simple
```
Sentence: "El gato se sentó en la alfombra"

La atención descubre:
- "gato" se conecta con "sentó" (¿quién se sentó?)
- "alfombra" se conecta con "en" (¿dónde?)
- "se" se conecta con "sentó" (¿qué acción?)
```

### Componentes de la Atención

| Tensor | Dimensión | Función |
|--------|-----------|---------|
| **Query** (Q) | 2048 × 2048 | "¿Qué estoy buscando?" |
| **Key** (K) | 2048 × 256 | "¿Qué tengo para ofrecer?" |
| **Value** (V) | 2048 × 256 | "¿Qué información transmito?" |
| **Output** | 2048 × 2048 | "¿Cómo integro todo?" |

### En TinyLlama
- **Capas**: 0 a 21 (todas las capas)
- **Cabezas**: 16 Query, 2 Key/Value (GQA)
- **Parámetros**: 206M (18.8% del total)

### Analogía
Es como una **red de conexiones sociales**: cada palabra "habla" con las demás para entender qué significa en este contexto.

---

## 3. FFN (La Memoria y el Procesamiento)

### ¿Qué es?
La red Feed-Forward es donde se **almacena el conocimiento** y se **transforma la información**.

### ¿En qué capas está?
**EN TODAS LAS CAPAS (0 a 21)**

Cada una de las 22 capas tiene su propio FFN.

### Componentes del FFN

```
┌─────────────────────────────────────────┐
│              FFN (en cada capa)         │
│                                         │
│   INPUT (2048 dimensiones)              │
│      ↓↓↓↓↓                              │
│   ┌─────────────┐                       │
│   │    GATE     │ ← Filtro: ¿Qué paso? │
│   └──────┬──────┘                       │
│          ↓                              │
│   ┌─────────────┐                       │
│   │     UP      │ ← Ampliar: 2048→5632 │
│   └──────┬──────┘                       │
│          ↓                              │
│   ┌─────────────┐                       │
│   │    DOWN     │ ← Comprimir: 5632→2048│
│   └──────┬──────┘                       │
│          ↓                              │
│   OUTPUT (2048 dimensiones)             │
└─────────────────────────────────────────┘
```

### Los 3 Tensores del FFN

| Tensor | Dimensión | Función | Ejemplo |
|--------|-----------|---------|---------|
| **Gate** | 2048 × 5632 | Filtrar información | "¿Es importante el color?" |
| **Up** | 2048 × 5632 | Ampliar representación | "Rojo → rojo brillante, carmesí" |
| **Down** | 5632 × 2048 | Comprimir resultado | "Fruta roja y dulce" |

### Ejemplo Detallado

```python
# FFN en acción (simplificado)

# Paso 1: GATE - Decide qué información pasar
# Input: representación de "manzana"
gate = sigmoid(input @ W_gate)
# Resultado: [0.8, 0.2, 0.9, 0.1]
#             ↑    ↑    ↑    ↑
#            SÍ   NO   SÍ   NO

# Paso 2: UP - Ampliar la información
expanded = input @ W_up  # De 2048 a 5632 dimensiones
# Resultado: más datos detallados

# Paso 3: DOWN - Comprimir de vuelta
output = (gate * expanded) @ W_down
# Resultado: información filtrada y comprimida
```

### ¿Qué almacena el FFN?

El FFN es la **memoria del modelo**:

| Capa | Qué almacena |
|------|--------------|
| 0-5 | Patrones simples, sintaxis |
| 6-12 | Conceptos abstractos, significado |
| 13-21 | Integración global, generación |

### Ejemplos de Conocimiento en FFN

```
FFN Capa 3: "Las manzanas son rojas"
FFN Capa 7: "París es la capital de Francia"
FFN Capa 12: "2 + 2 = 4"
FFN Capa 15: "Si llueve, llevo paraguas"
```

### En TinyLlama
- **Capas**: 0 a 21 (todas las capas)
- **Tensores por capa**: 3 (gate, up, down) + 1 norm
- **Parámetros por capa**: 34.5M
- **Total FFN**: 761M (69.3% del total)

### Analogía
El FFN es como una **enciclopedia gigante** donde el modelo busca información para responder.

---

## 4. Normalización (El Estabilizador)

### ¿Qué es?
Mantienen los valores en un rango estable para que el modelo no "explote".

### Ejemplo Simple
```
Sin norm: [1000, -500, 2000, -800]  ← Valores extremos
Con norm: [0.5, -0.2, 0.8, -0.3]   ← Valores estables
```

### Tipos de Normalización

| Tensor | Función |
|--------|---------|
| **attn_norm** | Estabiliza después de la atención |
| **ffn_norm** | Estabiliza después del FFN |

### En TinyLlama
- **Capas**: 0 a 21 (todas las capas)
- **Tensores por capa**: 2 (attn_norm, ffn_norm)
- **Parámetros**: 90K (0.01% del total)

### ¿Por qué son importantes?
Aunque tienen pocos parámetros, son **críticos**:
- Sin norm, los valores se vuelven enormes
- El modelo deja de funcionar
- Es como el **aceite de una máquina**

---

## 5. Output (La Puerta de Salida)

### ¿Qué es?
Convierte la representación final de vuelta a tokens (palabras).

### Ejemplo Simple
```
Representación: [0.12, -0.45, 0.78, ...]
      ↓
Output (proyectar a vocabulario)
      ↓
Probabilidades: [0.01, 0.02, 0.15, ...] (32,000 tokens)
      ↓
Token más probable: "manzana"
```

### En TinyLlama
- **Tensor**: `output.weight`
- **Dimensiones**: 2048 × 32,000
- **Parámetros**: 65.5M (6% del total)

---

## Resumen: Distribución de Parámetros

```
┌─────────────────────────────────────────────────────────┐
│                 TINYLLAMA (1.1B params)                 │
│                                                         │
│   ┌─────────────────────────────────────────────┐      │
│   │  FFN (69.3%)                                │      │
│   │  ████████████████████████████████████       │      │
│   │  761M params - Almacena conocimiento        │      │
│   └─────────────────────────────────────────────┘      │
│                                                         │
│   ┌─────────────────────────────────────────────┐      │
│   │  Atención (18.8%)                           │      │
│   │  ██████████                                 │      │
│   │  206M params - Conecta tokens               │      │
│   └─────────────────────────────────────────────┘      │
│                                                         │
│   ┌─────────────────────────────────────────────┐      │
│   │  Embeddings + Output (12%)                  │      │
│   │  ██████                                     │      │
│   │  131M params - Entrada/Salida               │      │
│   └─────────────────────────────────────────────┘      │
│                                                         │
│   ┌─────────────────────────────────────────────┐      │
│   │  Norms (0.01%)                              │      │
│   │  ▏                                          │      │
│   │  90K params - Estabilización                │      │
│   └─────────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────┘
```

---

## Descubrimiento: ¿Cuál tiene más influencia?

Basado en nuestro análisis empírico:

| Componente | Parámetros | Influencia en comportamiento |
|------------|------------|------------------------------|
| **FFN** | 69.3% | **ALTA** - Cambia tono emocional |
| Atención | 18.8% | MEDIA - Cambia conexiones |
| Embeddings | 6.0% | BAJA - Solo entrada |

### Resultado de Perturbación

Cuando modificamos el FFN (scale=0.1):
- **Original**: "anger and frustration"
- **Modificado**: "shame and disgust... gut-punch"

El FFN **cambia cómo el modelo expresa emociones**.

---

## Conclusión

Un LLM es un sistema complejo con múltiples componentes:

1. **Embeddings**: Convierte palabras a números
2. **Atención**: Conecta palabras entre sí
3. **FFN**: Almacena conocimiento y procesa información
4. **Norm**: Mantiene estabilidad
5. **Output**: Convierte de vuelta a palabras

El **FFN es el componente más grande** (69%) y tiene **más influencia** en el comportamiento del modelo.

---

*Siguiente capítulo: La Analogía Cósmica*
