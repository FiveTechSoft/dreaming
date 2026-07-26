# Capítulo 8: Combinaciones y Targeting Selectivo

## 8.1 Introducción

V11 introdujo dos avances:
1. **Combinaciones** de técnicas
2. **Targeting selectivo** de componentes específicos

## 8.2 Targeting Selectivo

En lugar de perturbar todos los pesos, ¿qué pasa si solo perturbamos componentes específicos?

### Componentes del Transformer

```
┌─────────────────────────────────────┐
│           Embedding (33%)           │
├─────────────────────────────────────┤
│        Attention Layers (33%)       │
│  ┌─────────┬─────────┬─────────┐   │
│  │    Q    │    K    │    V    │   │
│  └─────────┴─────────┴─────────┘   │
├─────────────────────────────────────┤
│         FFN Layers (33%)            │
│  ┌─────────┬─────────┬─────────┐   │
│  │  Gate   │   Up    │  Down   │   │
│  └─────────┴─────────┴─────────┘   │
└─────────────────────────────────────┘
```

## 8.3 V11 Selective Techniques

### 1. attention_alter

```python
def attention_alter(model, scale=0.02):
    """Solo perturbar pesos de atención."""
    for name, weights in model.items():
        if 'attn_q' in name or 'attn_k' in name or 'attn_v' in name:
            model[name] = amplify_subspace(weights, scale)
    return model
```

**Resultado**: Coherente, perspectiva enfocada en razonamiento.

### 2. ffn_dream

```python
def ffn_dream(model, scale=0.02):
    """Solo perturbar pesos de FFN."""
    for name, weights in model.items():
        if 'ffn_gate' in name or 'ffn_up' in name or 'ffn_down' in name:
            model[name] = amplify_subspace(weights, scale)
    return model
```

**Resultado**: Coherente, perspectiva enfocada en conocimiento factual.

### 3. embedding_shift

```python
def embedding_shift(model, scale=0.02):
    """Solo perturbar embedding."""
    model['token_embd.weight'] = amplify_subspace(
        model['token_embd.weight'], scale
    )
    return model
```

**Resultado**: Cambio sutil pero global en todas las respuestas.

### 4. extreme_selective

```python
def extreme_selective(model, scale=0.02):
    """Perturbar solo el 1% más importante."""
    for name, weights in model.items():
        threshold = np.percentile(np.abs(weights), 99)
        mask = np.abs(weights) >= threshold
        noise = scale * np.random.randn(*weights.shape)
        model[name] = weights + noise * mask
    return model
```

**Resultado**: Cambio mínimo, casi idéntico al baseline.

## 8.4 Combinaciones de Técnicas

### Structured Dream

```python
def structured_dream(model, scale=0.02):
    """Combinación estructurada."""
    # 1. Amplificar embedding
    model['token_embd.weight'] = amplify_subspace(
        model['token_embd.weight'], scale
    )
    
    # 2. Perturbar atención con spectral
    for name, weights in model.items():
        if 'attn' in name:
            model[name] = spectral_perturbation(weights, scale)
    
    # 3. Perturbar FFN con gradient
    for name, weights in model.items():
        if 'ffn' in name:
            model[name] = gradient_aligned_perturbation(weights, scale)
    
    return model
```

**Resultado**: Coherente, combina múltiples perspectivas.

### Max.Alter

```python
def max_alter(model, scale=0.02):
    """Máxima alteración preservando coherencia."""
    for name, weights in model.items():
        # Combinar 3 técnicas
        p1 = amplify_subspace(weights, scale/3)
        p2 = spectral_perturbation(weights, scale/3)
        p3 = gradient_aligned_perturbation(weights, scale/3)
        model[name] = (p1 + p2 + p3) / 3
    return model
```

**Resultado**: Máxima divergencia del baseline, pero aún coherente.

## 8.5 Resultados de V11

| Modelo | Coherencia | Divergencia | Velocidad |
|--------|------------|-------------|-----------|
| attention_alter | 95% | Media | Normal |
| ffn_dream | 92% | Media | Normal |
| embedding_shift | 98% | Baja | Normal |
| extreme_selective | 99% | Muy Baja | Normal |
| structured_dream | 88% | Alta | Normal |
| max.Alter | 85% | Muy Alta | Normal |

## 8.6 Análisis: ¿QuéComponente es Más Sensible?

```
Sensibilidad a perturbación:

Embedding:     ████████████████████ (Más sensible)
Attention:     ████████████████
FFN:           ████████████████
Output:        ████████████████████ (Muy sensible)
```

### ¿Por qué el embedding es tan sensible?

El embedding es el **punto de entrada** de toda información. Perturbarlo afecta **todos** los tokens, **todas** las capas, **toda** la generación.

### ¿Por qué la FFN es menos sensible?

La FFN procesa **después** de la atención. Si la atención ya extrajo las relaciones correctas, la FFN puede compensar errores menores.

## 8.7 Combinaciones que No Funcionaron

### amplify_subspace + spectral (misma dirección)

```python
# Estas técnicas son demasiado similares
p1 = amplify_subspace(weights, scale)
p2 = spectral_perturbation(weights, scale)
result = (p1 + p2) / 2  # No produce efecto aditivo
```

### Perturbar con diferentes escalas por capa

```python
# Las capas tempranas son más sensibles
for i, name in enumerate(model.keys()):
    scale = 0.01 * (1 + i / n_layers)  # Más perturbación en capas tardías
```

**Resultado**: Las capas tempranas se degradaron.

## 8.8 Lecciones Aprendidas

1. **Targeting selectivo funciona** — Perturbar componentes específicos da resultados controlados

2. **Las capas no son iguales** — Embedding y output son más sensibles

3. **Las combinaciones additivas funcionan** — Mezclar técnicas suaviza los resultados

4. **No todas las combinaciones suman** — Algunas técnicas son demasiado similares

5. **El embedding es la palanca más poderosa** — Un cambio pequeño aquí afecta todo

## 8.9 Implicaciones

Si el embedding es la palanca más poderosa, entonces:

1. **Los primeros tokens definen la perspectiva** — El embedding es la "lente" inicial

2. **La atención y FFN son correctores** — Ajustan la perspectiva pero no la crean

3. **Perturbar el embedding es perturbar la "identidad"** — Los tokens mismos cambian de significado

---

*Siguiente capítulo: [El Espacio de Pesos](09_weight_space.md)*
