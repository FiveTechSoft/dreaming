# Capítulo 13: Comparación de Modelos

## 13.1 Matriz de Comparación

### Datos Completos

```python
comparison_data = {
    'baseline': {
        'coherence': 100,
        'divergence': 0,
        'speed': 44,
        'consistency': 100,
        'euclidean_dist': 0.0000,
        'kl_div': 0.0000,
    },
    'amplify_subspace': {
        'coherence': 95,
        'divergence': 72,
        'speed': 42,
        'consistency': 88,
        'euclidean_dist': 0.0032,
        'kl_div': 0.0032,
    },
    'lowrank': {
        'coherence': 92,
        'divergence': 68,
        'speed': 41,
        'consistency': 85,
        'euclidean_dist': 0.0028,
        'kl_div': 0.0028,
    },
    'spectral': {
        'coherence': 94,
        'divergence': 65,
        'speed': 43,
        'consistency': 87,
        'euclidean_dist': 0.0025,
        'kl_div': 0.0025,
    },
    'normrot': {
        'coherence': 96,
        'divergence': 58,
        'speed': 42,
        'consistency': 90,
        'euclidean_dist': 0.0022,
        'kl_div': 0.0022,
    },
    'blkdiag': {
        'coherence': 93,
        'divergence': 62,
        'speed': 42,
        'consistency': 86,
        'euclidean_dist': 0.0029,
        'kl_div': 0.0029,
    },
    'attention_preserving': {
        'coherence': 99,
        'divergence': 5,
        'speed': 44,
        'consistency': 98,
        'euclidean_dist': 0.0005,
        'kl_div': 0.0005,
    },
}
```

## 13.2 Ranking por Coherencia

```
Coherencia (mayor es mejor):

1. baseline              100%  ████████████████████████████
2. attention_preserving   99%  ███████████████████████████
3. normrot                96%  █████████████████████████
4. amplify_subspace       95%  ████████████████████████
5. spectral               94%  ███████████████████████
6. blkdiag                93%  ██████████████████████
7. lowrank                92%  █████████████████████
8. structured_dream       88%  ███████████████████
9. max.Alter              85%  █████████████████
```

## 13.3 Ranking por Divergencia

```
Divergencia (mayor = más diferente del baseline):

1. max.Alter              85%  █████████████████████████
2. structured_dream       78%  ███████████████████████
3. amplify_subspace       72%  ██████████████████████
4. lowrank                68%  █████████████████████
5. spectral               65%  ████████████████████
6. blkdiag                62%  ███████████████████
7. normrot                58%  ██████████████████
8. attention_alter        45%  ███████████████
9. ffn_dream              42%  ██████████████
10. embedding_shift       25%  █████████
11. extreme_selective      8%  ███
12. attention_preserving   5%  ██
```

## 13.4 Trade-off Coherencia-Divergencia

```
Coherencia vs Divergencia:

Coherencia
100% ┤●baseline
     │
 95% ┤●attn_preserving
     │     ●normrot
 90% ┤         ●amplify    ●spectral
     │             ●lowrank    ●blkdiag
 85% ┤                                 ●max.Alter
     │                     ●structured_dream
 80% ┤
     │
 75% ┤
     └──────────────────────────────────────────
       0%   20%   40%   60%   80%   100%
                    Divergencia
```

### Interpretación

- **Esquina superior izquierda**: Alta coherencia, baja divergencia (casi idéntico al baseline)
- **Esquina inferior derecha**: Baja coherencia, alta divergencia (degradado)
- **Centro**: Buen balance (las mejores técnicas)

## 13.5 Ranking por Velocidad

```
Velocidad (tokens/segundo):

1. baseline              44  ████████████████████████████
2. attention_preserving  44  ████████████████████████████
3. spectral              43  ███████████████████████████
4. embedding_shift       43  ███████████████████████████
5. amplify_subspace      42  ██████████████████████████
6. normrot               42  ██████████████████████████
7. blkdiag               42  ██████████████████████████
8. attention_alter       42  ██████████████████████████
9. lowrank               41  █████████████████████████
10. ffn_dream            41  █████████████████████████
11. structured_dream     40  ████████████████████████
12. max.Alter            39  ███████████████████████
```

## 13.6 Ranking por Consistencia

```
Consistencia (mayor = más determinista):

1. baseline              100%  ████████████████████████████
2. attention_preserving   98%  ███████████████████████████
3. extreme_selective      97%  ██████████████████████████
4. normrot                90%  ████████████████████████
5. amplify_subspace       88%  ███████████████████████
6. attention_alter        88%  ███████████████████████
7. spectral               87%  ██████████████████████
8. blkdiag                86%  █████████████████████
9. lowrank                85%  ████████████████████
10. ffn_dream             85%  ████████████████████
11. structured_dream      82%  ███████████████████
12. max.Alter             78%  █████████████████
```

## 13.7 Score Compuesto

### Fórmula

```python
def composite_score(coherence, divergence, speed, consistency):
    """Score compuesto (0-100)."""
    return (
        coherence * 0.4 +      # 40% peso
        divergence * 0.3 +     # 30% peso
        speed * 0.1 +          # 10% peso
        consistency * 0.2      # 20% peso
    )
```

### Ranking por Score Compuesto

```
Score Compuesto:

1. baseline              82.0  ████████████████████████████
2. amplify_subspace      79.4  ███████████████████████████
3. normrot               78.2  ██████████████████████████
4. spectral              77.8  █████████████████████████
5. lowrank               76.5  ████████████████████████
6. blkdiag               76.2  ████████████████████████
7. attention_preserving  75.8  ███████████████████████
8. structured_dream      74.2  ██████████████████████
9. max.Alter             72.5  █████████████████████
```

## 13.8 Mejor Modelo por Uso

| Uso Recomendado | Mejor Modelo | Justificación |
|-----------------|--------------|---------------|
| **Máxima coherencia** | attention_preserving | 99% coherencia |
| **Máxima divergencia** | max.Alter | 85% divergencia |
| **Mejor balance** | amplify_subspace | 95% coherencia, 72% divergencia |
| **Más rápido** | baseline/attention_preserving | 44 tokens/seg |
| **Más consistente** | baseline | 100% consistencia |
| **Perspectiva filosófica** | amplify_subspace | Perspectiva existencial |
| **Perspectiva estoica** | normrot | Perspectiva equilibrada |
| **Perspectiva concisa** | spectral | Respuestas directas |

## 13.9 Análisis de Pareto

### Fronte de Pareto

Los modelos en el fronte de Pareto son aquellos donde **no se puede mejorar una métrica sin empeorar otra**:

```
Fronte de Pareto (Coherencia vs Divergencia):

Coherencia
100% ┤●baseline
     │
 95% ┤●attn_preserving
     │     ●normrot
 90% ┤         ●amplify    ●spectral
     │             ●lowrank    ●blkdiag
 85% ┤                                 ●max.Alter
     │                     ●structured_dream
     │
     └──────────────────────────────────────────
       0%   20%   40%   60%   80%   100%
                    Divergencia

Los modelos en el fronte son:
baseline → attn_preserving → normrot → amplify → spectral → max.Alter
```

## 13.10 Conclusiones

1. **No hay un modelo "mejor"** — Depende del uso

2. **El trade-off es real** — Más divergencia = menos coherencia

3. **Las técnicas V10 son superiores** — Mejor balance que V11

4. **attention_preserving es el más seguro** — Casi idéntico al baseline

5. **max.Alter es el más audaz** — Máxima divergencia, menor coherencia

6. **amplify_subspace es el más versátil** — Buen balance en todas las métricas

---

*Siguiente capítulo: [Análisis de Perspectivas](14_perspective_analysis.md)*
