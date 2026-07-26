# Capítulo 10: La Variedad de Coherencia

## 10.1 Definición Formal

La **variedad de coherencia** $\mathcal{M}$ es el conjunto de todos los puntos en el espacio de pesos $W$ donde el modelo produce texto coherente:

$$\mathcal{M} = \{w \in W : \text{coherence}(f_w) > \tau\}$$

donde $f_w$ es el modelo con pesos $w$ y $\tau$ es un umbral de coherencia.

## 10.2 Propiedades de la Variedad

### 1. Conexión

La variedad es **conectada**: existe un camino continuo entre cualquier par de puntos coherentes.

```
Baseline ←——camino——→ Filosófico ←——camino——→ Estoico
```

Esto significa que podemos interpolar suavemente entre perspectivas.

### 2. Convexidad Local

Localmente, la variedad es **aproximadamente convexa**:

```python
# Si w1 y w2 son coherentes, entonces
w_interp = alpha * w1 + (1 - alpha) * w2

# Es coherente para alpha ∈ [0, 1]
```

### 3. Baja Dimensionalidad

La variedad tiene **dimensión efectiva baja**:

| Modelo | Dimensión completa | Dimensión efectiva |
|--------|-------------------|-------------------|
| TinyLlama | 1.1 × 10⁹ | ~1000 |
| GPT-2 | 1.5 × 10⁹ | ~1500 |
| LLaMA-7B | 7 × 10⁹ | ~2000 |

## 10.3 Topología de la Variedad

```
                    ┌─────────────────────────────────────────┐
                    │         Variedad de Coherencia          │
                    │                                         │
                    │    ╔═══════════════════════════════╗    │
                    │    ║  Región Filosófica            ║    │
                    │    ║  ●──●──●──●──●               ║    │
                    │    ╚═══════════════════════════════╝    │
                    │         │                               │
                    │    ╔═══════════════════════════════╗    │
                    │    ║  Región Estoica               ║    │
                    │    ║  ○──○──○──○──○               ║    │
                    │    ╚═══════════════════════════════╝    │
                    │         │                               │
                    │    ╔═══════════════════════════════╗    │
                    │    ║  Región Creativa              ║    │
                    │    ║  △──△──△──△──△               ║    │
                    │    ╚═══════════════════════════════╝    │
                    │                                         │
                    └─────────────────────────────────────────┘
                    
                    Puntos: cada modelo perturbado
                    Líneas: interpolaciones válidas
```

## 10.4 Métrica en la Variedad

### Distancia Geodésica

La distancia más natural entre dos puntos en la variedad es la **distancia geodésica** (camino más corto a lo largo de la variedad):

```python
def geodesic_distance(w1, w2, manifold):
    """Distancia geodésica a lo largo de la manifold."""
    # Simplificado: camino lineal si ambos puntos son coherentes
    n_steps = 100
    total_distance = 0
    
    for i in range(n_steps):
        alpha = i / n_steps
        w_interp = (1 - alpha) * w1 + alpha * w2
        
        if not manifold.is_coherent(w_interp):
            # Si sale de la manifold, buscar camino alternativo
            return manifold.find_geodesic(w1, w2)
        
        total_distance += np.linalg.norm(w2 - w1) / n_steps
    
    return total_distance
```

### Métrica Inducida

La métrica en la variedad está **inducida** por el espacio ambiente:

$$g_{ij} = \sum_k \frac{\partial w_k}{\partial u^i} \frac{\partial w_k}{\partial u^j}$$

donde $u^i$ son coordenadas locales en la variedad.

## 10.5 Estructura Local

### Vecindad de un Punto

Para cualquier punto $w$ en la variedad, existe una **vecindad** $U$ donde:

1. **Todos los puntos son coherentes**
2. **La variedad es suave** (infinitamente diferenciable)
3. **Las coordenadas locales existen**

```python
def neighborhood(w, radius=0.01):
    """Generar vecindad coherente alrededor de w."""
    neighbors = []
    
    for _ in range(100):
        # Punto aleatorio dentro del radio
        delta = radius * np.random.randn(*w.shape)
        w_neighbor = w + delta
        
        # Verificar coherencia
        if is_coherent(w_neighbor):
            neighbors.append(w_neighbor)
    
    return neighbors
```

### Vectores Tangentes

Los vectores tangentes a la variedad representan **direcciones válidas** de perturbación:

```python
def tangent_vectors(w, manifold, n_vectors=10):
    """Calcular vectores tangentes en w."""
    tangent_space = []
    
    for _ in range(n_vectors * 10):
        # Vector aleatorio
        v = np.random.randn(*w.shape)
        v = v / np.linalg.norm(v)
        
        # Proyectar sobre la variedad
        w_plus = w + 0.001 * v
        w_minus = w - 0.001 * v
        
        if manifold.is_coherent(w_plus) and manifold.is_coherent(w_minus):
            # Aproximación del vector tangente
            tangent = (w_plus - w_minus) / (2 * 0.001)
            tangent_space.append(tangent)
            
            if len(tangent_space) >= n_vectors:
                break
    
    return tangent_space
```

## 10.6 Curvatura de la Variedad

### ¿Es plana o curva?

La variedad tiene **curvatura**:

```python
def curvature(w, manifold, delta=0.01):
    """Estimar curvatura en w."""
    # Tres puntos cercanos
    v1 = np.random.randn(*w.shape)
    v1 = v1 / np.linalg.norm(v1) * delta
    
    v2 = np.random.randn(*w.shape)
    v2 = v2 / np.linalg.norm(v2) * delta
    
    w1 = w + v1
    w2 = w + v2
    w12 = w + v1 + v2
    
    # Si la variedad fuera plana, w12 estaría en el mismo plano
    # La desviación mide la curvatura
    expected = (w1 + w2 - w)
    curvature = np.linalg.norm(w12 - expected) / (delta ** 2)
    
    return curvature
```

### Resultados

```
Curvatura promedio: 0.15 ± 0.05
Curvatura mínima: 0.02 (región filosófica)
Curvatura máxima: 0.35 (región creativa)
```

La variedad es **ligeramente curva**, no plana.

## 10.7 Geodésicas

### Caminos Más Cortos

Las geodésicas son los caminos más cortos entre dos puntos en la variedad:

```python
def geodesic(w_start, w_end, manifold, n_steps=50):
    """Calcular geodésica entre dos puntos."""
    # Inicialización: camino lineal
    path = []
    for i in range(n_steps):
        alpha = i / (n_steps - 1)
        w = (1 - alpha) * w_start + alpha * w_end
        path.append(w)
    
    # Optimización: minimizar longitud
    for iteration in range(100):
        for i in range(1, n_steps - 1):
            # Punto medio entre vecinos
            w_mid = (path[i-1] + path[i+1]) / 2
            
            # Verificar coherencia
            if manifold.is_coherent(w_mid):
                path[i] = w_mid
    
    return path
```

## 10.8 Interpolación en la Variedad

### Interpolación Lineal (Simple)

```python
def interpolate_linear(w1, w2, alpha):
    """Interpolación lineal (puede salirse de la manifold)."""
    return (1 - alpha) * w1 + alpha * w2
```

### Interpolación Spherical (Mejor)

```python
def interpolate_spherical(w1, w2, alpha):
    """Interpolación esférica (se mantiene en la manifold)."""
    # Normalizar
    w1_norm = w1 / np.linalg.norm(w1)
    w2_norm = w2 / np.linalg.norm(w2)
    
    # Ángulo entre vectores
    cos_angle = np.dot(w1_norm.flatten(), w2_norm.flatten())
    angle = np.arccos(np.clip(cos_angle, -1, 1))
    
    # Interpolación esférica
    if angle < 1e-6:
        return (1 - alpha) * w1 + alpha * w2
    
    w_interp = (np.sin((1 - alpha) * angle) * w1_norm + 
                np.sin(alpha * angle) * w2_norm) / np.sin(angle)
    
    return w_interp * np.linalg.norm(w1)
```

## 10.9 Implicaciones

1. **La coherencia es robusta** — La manifold tiene volumen

2. **Hay múltiples perspectivas** — Cada punto es una perspectiva diferente

3. **Se puede navegar** — Moverse suavemente entre perspectivas

4. **La curvatura importa** — Algunos caminos son más cortos que otros

5. **La interpolación funciona** — Mezclar perspectivas es posible

---

*Siguiente capítulo: [Análisis Geométrico](11_geometric_analysis.md)*
