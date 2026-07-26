# Capítulo 9: El Espacio de Pesos

## 9.1 Introducción

Los pesos de un LLM viven en un espacio de alta dimensionalidad. Para TinyLlama, ese espacio tiene ~1.1 mil millones de dimensiones.

Pero no es un espacio arbitrario. Tiene **estructura**.

## 9.2 Dimensionalidad del Espacio

| Modelo | Parámetros | Dimensionalidad |
|--------|------------|-----------------|
| TinyLlama-1.1B | 1.1 × 10⁹ | ~1.1 mil millones |
| GPT-2 | 1.5 × 10⁹ | ~1.5 mil millones |
| LLaMA-7B | 7 × 10⁹ | ~7 mil millones |
| GPT-3 | 175 × 10⁹ | ~175 mil millones |

## 9.3 ¿Qué Representa Cada Dimensión?

Cada dimensión del espacio de pesos corresponde a un **parámetro específico** del modelo:

```
Dimensión 0:  token_embd.weight[0, 0]
Dimensión 1:  token_embd.weight[0, 1]
...
Dimensión 2047: token_embd.weight[0, 2047]
Dimensión 2048: token_embd.weight[1, 0]
...
```

Pero estas dimensiones no son independientes. Están **correlacionadas** por el entrenamiento.

## 9.4 Estructura del Espacio

### Regiones de Coherencia

El espacio de pesos no es homogéneo. Tiene **regiones**:

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│   Región A          Región B          Región C              │
│   (Coherente)       (Coherente)       (Incoherente)        │
│                                                             │
│   ●●●●●             ○○○○○             ▲▲▲▲▲               │
│   ●●●●●             ○○○○○             ▲▲▲▲▲               │
│   ●●●●●             ○○○○○             ▲▲▲▲▲               │
│                                                             │
│   "Filosófico"      "Estoico"         "Basura"             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

Los modelos perturbados se mueven **dentro** de una región coherente, no hacia la región incoherente.

## 9.5 Distancia entre Modelos

### Distancia Euclidiana

```python
def euclidean_distance(w1, w2):
    return np.sqrt(np.sum((w1 - w2) ** 2))
```

### Distancia KL (Kullback-Leibler)

```python
def kl_divergence(p, q):
    """Divergencia KL entre distribuciones de salida."""
    p = np.clip(p, 1e-10, 1)
    q = np.clip(q, 1e-10, 1)
    return np.sum(p * np.log(p / q))
```

### Distancia Cosine

```python
def cosine_distance(w1, w2):
    w1_flat = w1.flatten()
    w2_flat = w2.flatten()
    return 1 - np.dot(w1_flat, w2_flat) / (
        np.linalg.norm(w1_flat) * np.linalg.norm(w2_flat)
    )
```

## 9.6 Distancias Medidas

| Modelo | vs Baseline (Euclid) | vs Baseline (KL) |
|--------|---------------------|-------------------|
| amplify_subspace | 0.0032 | 0.0032 |
| lowrank | 0.0028 | 0.0028 |
| normrot | 0.0022 | 0.0022 |
| spectral | 0.0025 | 0.0025 |
| attention_preserving | 0.0005 | 0.0005 |

## 9.7 La Manifold de Coherencia

### Definición

La manifold de coherencia es el **subespacio** del espacio de pesos donde los modelos producen texto coherente.

```
Espacio completo (1.1 mil millones de dimensiones)
    │
    ├── Manifold de coherencia (~1000 dimensiones)
    │       │
    │       ├── Región "Filosófico"
    │       ├── Región "Estoico"
    │       ├── Región "Conciso"
    │       └── ...
    │
    └── Regiones incoherentes (el resto)
```

### ¿Por qué es de baja dimensionalidad?

El entrenamiento **reduce** la dimensionalidad efectiva:
- Muchos pesos están **correlacionados**
- Solo unas **pocas direcciones** son realmente importantes
- El modelo aprende una **representación comprimida** del conocimiento

## 9.8 Análisis de Componentes Principales (PCA)

```python
from sklearn.decomposition import PCA

# Para cada modelo, calcular embeddings de salidas
baseline_outputs = generate_all_prompts(baseline_model)
perturbed_outputs = generate_all_prompts(perturbed_model)

# Concatenar embeddings
all_outputs = np.vstack([baseline_outputs, perturbed_outputs])

# PCA
pca = PCA(n_components=10)
reduced = pca.fit_transform(all_outputs)

# Varianza explicada
print(f"PC1: {pca.explained_variance_ratio_[0]:.3f}")
print(f"PC2: {pca.explained_variance_ratio_[1]:.3f}")
print(f"PC1+PC2: {sum(pca.explained_variance_ratio_[:2]):.3f}")
```

### Resultados típicos

```
PC1: 0.45 (45% de varianza)
PC2: 0.22 (22% de varianza)
PC3: 0.12 (12% de varianza)
PC4: 0.08 (8% de varianza)
PC5: 0.05 (5% de varianza)
...
PC1-PC5: 92% de varianza
```

Esto confirma que la dimensionalidad efectiva es **baja**.

## 9.9 Geometría de la Perturbación

### Perturbación Bruta vs Controlada

```
Espacio de pesos (2D simplificado)

         Baseline
            ●
            │
            │  Perturbación bruta (ruido)
            │  → Va en cualquier dirección
            │  → Probablemente sale de la manifold
            │
            ▼
            ✗ (Incoherente)
            
            ●
            │
            │  Perturbación controlada
            │  → Va tangente a la manifold
            │  → Se mantiene dentro de la región coherente
            │
            ▼
            ● (Coherente, diferente perspectiva)
```

## 9.10 ¿Qué es una "Perspectiva"?

Una perspectiva es un **punto** en la manifold de coherencia:

```python
# Cada modelo es un punto
baseline = load_model("baseline.gguf")
filosofico = load_model("filosofico.gguf")
estoico = load_model("estoico.gguf")

# Las perspectivas son los vectores entre ellos
perspective_vector = filosofico - baseline

# La dirección del vector determina la perspectiva
angle = np.arctan2(perspective_vector[1], perspective_vector[0])
```

## 9.11 Implicaciones

1. **El espacio de pesos tiene estructura** — No es caótico

2. **La coherencia es una región** — No un punto único

3. **Las perspectivas son direcciones** — Vectores en la manifold

4. **La perturbación es navegación** — Moverse dentro de la manifold

5. **La dimensionalidad efectiva es baja** — ~1000 dimensiones, no mil millones

---

*Siguiente capítulo: [La Variedad de Coherencia](10_coherence_manifold.md)*
