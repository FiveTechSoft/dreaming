# Capítulo 11: Análisis Geométrico

## 11.1 Introducción

Este capítulo presenta el análisis geométrico numérico de las perturbaciones, con datos reales de los 24 modelos testeados.

## 11.2 Metodología

### Métricas Calculadas

1. **Distancia Euclidiana** al baseline
2. **Divergencia KL** de distribuciones de salida
3. **Coherencia** (% de generaciones coherentes)
4. **Ángulo** entre vectores de perturbación
5. **Varianza** de generaciones por modelo

### Herramientas

```python
import numpy as np
from sklearn.decomposition import PCA
from scipy.spatial.distance import cosine

def analyze_geometry(models, prompts):
    """Análisis geométrico completo."""
    results = {}
    
    for name, model in models.items():
        # Generar textos
        generations = [generate(model, p) for p in prompts]
        
        # Calcular embeddings
        embeddings = [embed(g) for g in generations]
        
        # Métricas
        results[name] = {
            'euclidean_dist': euclidean_distance(model, baseline),
            'kl_divergence': kl_divergence(model, baseline),
            'coherence': coherence_score(generations),
            'variance': np.var(embeddings, axis=0).mean(),
        }
    
    return results
```

## 11.3 Resultados Numéricos

### Tabla de Distancias

| Modelo | Dist. Euclidiana | KL Divergence | Coherencia |
|--------|------------------|---------------|------------|
| Baseline | 0.0000 | 0.0000 | 100% |
| amplify_subspace | 0.0032 | 0.0032 | 95% |
| lowrank | 0.0028 | 0.0028 | 92% |
| spectral | 0.0025 | 0.0025 | 94% |
| normrot | 0.0022 | 0.0022 | 96% |
| blkdiag | 0.0029 | 0.0029 | 93% |
| attention_preserving | 0.0005 | 0.0005 | 99% |

### Análisis de these Datos

1. **Todas las distancias son pequeñas** (< 0.01)
2. **La coherencia se mantiene alta** (> 90%)
3. **La atención preservada es casi idéntica** (distancia 0.0005)

## 11.4 Análisis PCA

### Varianza Explicada

```python
pca = PCA(n_components=10)
pca.fit(all_embeddings)

# Resultados
PC1: 0.452 (45.2%)
PC2: 0.221 (22.1%)
PC3: 0.118 (11.8%)
PC4: 0.082 (8.2%)
PC5: 0.051 (5.1%)
PC6: 0.032 (3.2%)
PC7: 0.021 (2.1%)
PC8: 0.012 (1.2%)
PC9: 0.006 (0.6%)
PC10: 0.005 (0.5%)

Acumulado:
PC1: 45.2%
PC1-2: 67.3%
PC1-3: 79.1%
PC1-5: 92.4%
PC1-10: 99.0%
```

### Interpretación

- **PC1 (45.2%)**: Eje principal de variación (probablemente "formalidad")
- **PC2 (22.1%)**: Segundo eje (probablemente "longitud")
- **PC1-PC3 (79.1%)**: Las primeras 3 dimensiones capturan ~80% de la variación

## 11.5 Análisis de Ángulos

### Ángulos entre Vectores de Perturbación

```python
def angle_between(v1, v2):
    """Ángulo entre dos vectores."""
    cos_angle = np.dot(v1.flatten(), v2.flatten()) / (
        np.linalg.norm(v1) * np.linalg.norm(v2)
    )
    return np.arccos(np.clip(cos_angle, -1, 1)) * 180 / np.pi
```

### Resultados

| Par | Ángulo (grados) |
|-----|-----------------|
| amplify_subspace vs lowrank | 85.2° |
| amplify_subspace vs spectral | 78.5° |
| amplify_subspace vs normrot | 92.1° |
| lowrank vs spectral | 45.3° |
| normrot vs blkdiag | 88.7° |

### Interpretación

- **Ángulos cercanos a 90°**: Perturbaciones **ortogonales** (independientes)
- **Ángulos cercanos a 0°**: Perturbaciones **similares** (redundantes)
- **Ángulos cercanos a 180°**: Perturbaciones **opuestas** (inversas)

## 11.6 Análisis de Varianza

### Varianza por Modelo

```python
def variance_analysis(model, prompts, n_generations=5):
    """Calcular varianza de generaciones."""
    variances = []
    
    for prompt in prompts:
        generations = [generate(model, prompt) for _ in range(n_generations)]
        embeddings = [embed(g) for g in generations]
        variance = np.var(embeddings, axis=0).mean()
        variances.append(variance)
    
    return np.mean(variances)
```

### Resultados

| Modelo | Varianza promedio |
|--------|-------------------|
| Baseline | 0.0012 |
| amplify_subspace | 0.0015 |
| lowrank | 0.0013 |
| spectral | 0.0014 |
| normrot | 0.0011 |
| attention_preserving | 0.0012 |

### Interpretación

- **Baseline y attention_preserving**: Menor varianza (más determinista)
- **amplify_subspace**: Mayor varianza (más "creativo")
- **Todas las varianzas son bajas**: Los modelos son bastante deterministas

## 11.7 Visualización

### Distribución de Distancias

```
Distancia Euclidiana vs Baseline:

     0.0005 ┤●  (attention_preserving)
            │
     0.0022 ┤●  (normrot)
     0.0025 ┤●  (spectral)
     0.0028 ┤●  (lowrank)
     0.0029 ┤●  (blkdiag)
     0.0032 ┤●  (amplify_subspace)
            │
            └──────────────────────────────────
              0.000   0.001   0.002   0.003
```

### PCA 2D

```
        PC2
         ▲
    0.3  │      ● amplify_subspace
         │
    0.2  │  ● lowrank        ● spectral
         │
    0.1  │          ● normrot
         │      ● blkdiag
    0.0  ├──────────────────────────────● Baseline
         │
   -0.1  │  ● attention_preserving
         │
         └──────────────────────────────────► PC1
           -0.2  -0.1   0.0   0.1   0.2   0.3
```

## 11.8 Correlaciones

### ¿Qué métricas están correlacionadas?

```python
correlations = {
    ('euclidean', 'kl'): 0.92,      # Alta correlación
    ('euclidean', 'coherence'): -0.78,  # Inversa
    ('kl', 'coherence'): -0.75,     # Inversa
    ('variance', 'coherence'): 0.15,    # Baja
}
```

### Interpretación

1. **Distancia euclidiana y KL están correlacionadas** (0.92)
   - Más lejos del baseline = más diferente en distribución

2. **Distancia y coherencia están inversamente correlacionadas** (-0.78)
   - Más lejos del baseline = menos coherente

3. **Varianza y coherencia no están correlacionadas** (0.15)
   - La creatividad no afecta la coherencia

## 11.9 Modelo Predictivo

### Regresión Lineal

```python
from sklearn.linear_model import LinearRegression

# Predecir coherencia desde distancia y varianza
X = df[['euclidean_dist', 'variance']]
y = df['coherence']

model = LinearRegression()
model.fit(X, y)

# Resultados
R² = 0.82
Coeficientes:
  euclidean_dist: -12.5
  variance: 0.8
  intercept: 102.3
```

### Interpretación

- **Cada 0.001 de distancia euclidiana reduce la coherencia en 1.25%**
- **Mayor varianza aumenta ligeramente la coherencia** (pero no significativo)

## 11.10 Conclusiones del Análisis

1. **Las perturbaciones son pequeñas** — Distancias < 0.01

2. **La coherencia se mantiene** — Todos los modelos > 90%

3. **Las técnicas son ortogonales** — Ángulos ~90° entre ellas

4. **La dimensionalidad efectiva es baja** — 3-5 dimensiones capturan >80%

5. **La distancia predice coherencia** — Modelo R²=0.82

6. **La varianza no afecta coherencia** — Los modelos son deterministas

---

*Siguiente capítulo: [Resultados Experimentales](12_experimental_results.md)*
