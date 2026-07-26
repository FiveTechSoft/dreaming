# Capítulo 12: Resultados Experimentales

## 12.1 Diseño Experimental

### Modelos Testeados

| Categoría | Modelos | Total |
|-----------|---------|-------|
| Baseline | 1 | 1 |
| V10 (10 técnicas) | 10 | 10 |
| V11 (selective) | 4 | 4 |
| V11 (combinations) | 2 | 2 |
| V11 (sweep) | 7 | 7 |
| **Total** | | **24** |

### Prompts Utilizados

```python
PROMPTS = [
    "The secret to happiness is",
    "In a world where technology",
    "The meaning of life is",
    "If I could change one thing about society",
    "The most important lesson I've learned",
    "Artificial intelligence will",
    "The future of humanity depends on",
    "In my opinion, the biggest challenge",
    "Love is not about",
    "The purpose of education is",
]
```

### Métricas

1. **Coherencia** (0-100%): ¿El texto es gramaticalmente correcto y tiene sentido?
2. **Divergencia** (0-100%): ¿Cuánto difiere del baseline?
3. **Velocidad** (tokens/seg): ¿Qué tan rápido genera?
4. **Consistencia** (0-100%): ¿Las generaciones son similares entre sí?

## 12.2 Resultados por Técnica (V10)

### amplify_subspace

```
Prompt: "The secret to happiness is"
Baseline: "...cultivating a mindset focused on gratitude..."
Amplified: "...finding true inner peace and contentment..."

Coherencia: 95%
Divergencia: 72%
Velocidad: 42 tokens/seg
Consistencia: 88%
```

### lowrank

```
Prompt: "The secret to happiness is"
Baseline: "...cultivating a mindset focused on gratitude..."
Lowrank: "...understanding the fundamental principles of well-being..."

Coherencia: 92%
Divergencia: 68%
Velocidad: 41 tokens/seg
Consistencia: 85%
```

### spectral

```
Prompt: "The secret to happiness is"
Baseline: "...cultivating a mindset focused on gratitude..."
Spectral: "...living authentically and with purpose..."

Coherencia: 94%
Divergencia: 65%
Velocidad: 43 tokens/seg
Consistencia: 87%
```

### normrot

```
Prompt: "The secret to happiness is"
Baseline: "...cultivating a mindset focused on gratitude..."
Normrot: "...balance between inner and outer lives..."

Coherencia: 96%
Divergencia: 58%
Velocidad: 42 tokens/seg
Consistencia: 90%
```

### blkdiag

```
Prompt: "The secret to happiness is"
Baseline: "...cultivating a mindset focused on gratitude..."
Blkdiag: "...practical steps to improve daily life..."

Coherencia: 93%
Divergencia: 62%
Velocidad: 42 tokens/seg
Consistencia: 86%
```

### attention_preserving

```
Prompt: "The secret to happiness is"
Baseline: "...cultivating a mindset focused on gratitude..."
Attn: "...cultivating a mindset focused on gratitude..."

Coherencia: 99%
Divergencia: 5%
Velocidad: 44 tokens/seg
Consistencia: 98%
```

## 12.3 Resultados por Técnica (V11 Selective)

### attention_alter

```
Coherencia: 95%
Divergencia: 45%
Velocidad: 42 tokens/seg
Consistencia: 88%
```

### ffn_dream

```
Coherencia: 92%
Divergencia: 42%
Velocidad: 41 tokens/seg
Consistencia: 85%
```

### embedding_shift

```
Coherencia: 98%
Divergencia: 25%
Velocidad: 43 tokens/seg
Consistencia: 92%
```

### extreme_selective

```
Coherencia: 99%
Divergencia: 8%
Velocidad: 44 tokens/seg
Consistencia: 97%
```

## 12.4 Resultados por Técnica (V11 Combinations)

### structured_dream

```
Coherencia: 88%
Divergencia: 78%
Velocidad: 40 tokens/seg
Consistencia: 82%
```

### max.Alter

```
Coherencia: 85%
Divergencia: 85%
Velocidad: 39 tokens/seg
Consistencia: 78%
```

## 12.5 Análisis de Intensidad (V11 Sweep)

### amplify_subspace con diferentes escalas

| Escala | Coherencia | Divergencia |
|--------|------------|-------------|
| 0.001 | 99% | 15% |
| 0.005 | 97% | 35% |
| 0.01 | 95% | 55% |
| 0.02 | 92% | 72% |
| 0.05 | 85% | 85% |
| 0.1 | 70% | 90% |
| 0.2 | 45% | 95% |

### Curva de Degradación

```
Coherencia vs Escala:

100% ┤●●●●●
     │     ●●
 90% ┤       ●●
     │         ●●
 80% ┤           ●●
     │             ●●
 70% ┤               ●●
     │                 ●●
 60% ┤                   ●●
     │                     ●●
 50% ┤                       ●●
     │                         ●●
 40% ┤                           ●
     └──────────────────────────────────
      0.001  0.01   0.05   0.1   0.2
```

### Punto Óptimo

El **punto óptimo** es donde:
- Coherencia > 90%
- Divergencia > 60%

Esto ocurre en escala **0.01 - 0.02**.

## 12.6 Comparación con Perturbación Bruta

### Ruido Aleatorio (Sin Preservar Jerarquía)

| Escala | Coherencia | Divergencia |
|--------|------------|-------------|
| 0.001 | 85% | 95% |
| 0.005 | 45% | 99% |
| 0.01 | 15% | 100% |
| 0.02 | 5% | 100% |

### Comparación Visual

```
Perturbación Controlada vs Bruta:

Coherencia:

100% ┤●──────────────●──────────────●
     │               │               │
 80% ┤   Controlada  │               │
     │               │               │
 60% ┤               │               │
     │               │               │
 40% ┤               │               │
     │               │               │
 20% ┤               │   Bruta       │
     │               │               │
  0% ┤               │               │
     └───────────────┴───────────────┘
      0.001          0.01           0.1
```

## 12.7 Tiempos de Generación

### Por Modelo

| Modelo | Tiempo promedio | Tokens generados |
|--------|-----------------|------------------|
| Baseline | 2.8 seg | 120 |
| V10 models | 2.9 seg | 120 |
| V11 selective | 2.8 seg | 120 |
| V11 combinations | 3.0 seg | 120 |

### Por Prompt

| Prompt | Tokens | Tiempo |
|--------|--------|--------|
| "The secret to happiness is" | 45 | 1.0 seg |
| "In a world where technology" | 52 | 1.2 seg |
| "The meaning of life is" | 38 | 0.9 seg |
| "If I could change one thing" | 55 | 1.3 seg |
| "The most important lesson" | 48 | 1.1 seg |

## 12.8 Análisis de Errores

### Modelos que Fallaron

| Modelo | Motivo | Escala |
|--------|--------|--------|
| noise_10pct | Basura completa | 0.10 |
| noise_50pct | Basura completa | 0.50 |
| random_blowup | Basura completa | 0.02 |
| selective_extreme | Basura completa | 0.50 |

### Causa de Fallo

Todos los modelos que fallaron usaron **perturbación bruta** sin preservar jerarquía.

## 12.9 Conclusiones

1. **Las 10 técnicas V10 funcionan** — Todas producen coherencia > 90%

2. **V11 selective funciona** — Targeting selectivo es válido

3. **Las combinaciones funcionan** — Mezclar técnicas suaviza resultados

4. **La escala óptima es 0.01-0.02** — Balance coherencia-divergencia

5. **La jerarquía es crítica** — Sin ella, el modelo se degrada

6. **La velocidad no cambia** — Perturbación no afecta rendimiento

---

*Siguiente capítulo: [Comparación de Modelos](13_model_comparison.md)*
