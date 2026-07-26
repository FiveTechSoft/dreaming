# Geometría del Cambio de Perspectiva

## El Espacio de Pesos

Cada modelo LLM vive en un **espacio de alta dimensión** donde cada dimensión es un peso.

Para TinyLlama-1.1B Q4_0:
- ~155 tensors
- ~1.1 mil millones de pesos
- Espacio: **ℝ^1,100,000,000**

Cada punto en ese espacio es un "modelo" diferente.

## La Superficie de Coherencia

No todos los puntos generan texto coherente. Solo una **variedad de baja dimensión** (superficie) contiene modelos que "hablan".

```
Espacio total: ℝ^1,100,000,000
Superficie de coherencia: ~ℝ^1000 (aproximado)
```

La mayoría del espacio es **basura**. Solo una fracción minúscula genera texto legible.

## ¿Qué es una Perspectiva?

Una perspectiva es un **punto** en la superficie de coherencia.

```
Punto A = baseline (autoayuda genérica)
Punto B = philosophical (filosófica)
Punto C = stoic (estoica)
```

Cada punto produce texto coherente, pero con un "tono" diferente.

## ¿Qué hace la Perturbación?

### amplify_subspace: Proyección + Amplificación

```python
def amplify_subspace(values, rng, intensity):
    vec = rng.standard_normal(len(values))  # Dirección aleatoria
    vec /= np.linalg.norm(vec)
    proj = np.dot(values, vec) * vec        # Proyectar
    return values + intensity * proj        # Amplificar
```

Geométricamente:
1. Elige una **dirección aleatoria** en el espacio de pesos
2. **Proyecta** los pesos sobre esa dirección
3. **Amplifica** esa componente

Si la dirección es **tangente** a la superficie de coherencia → el modelo se mueve sobre la superficie → coherencia preservada.

Si la dirección es **normal** a la superficie → el modelo sale de la superficie → basura.

## Visualización en 2D

Imagina la superficie de coherencia como una curva en 2D:

```
        Coherencia
            ^
            |     * B (philosophical)
            |    /
            |   /  ← amplify_subspace mueve sobre la curva
            |  /
            | * A (baseline)
            |/
            +-----------------> Peso
```

Las técnicas que funcionan (amplify_subspace, lowrank, normrot) mueven el modelo **sobre** la curva.

Las técnicas que fallan (nibble_flip, scaled_noise) mueven el modelo **fuera** de la curva.

## ¿Por qué Funcionan Algunas Técnicas y Otras No?

### Técnicas que preservan jerarquía

| Técnica | Operación geométrica | ¿Preserva coherencia? |
|---------|---------------------|----------------------|
| amplify_subspace | Proyección ortogonal | ✅ Tangente a la superficie |
| lowrank | Amplificación de valores singulares | ✅ Preserva estructura |
| normrot | Rotación ortogonal | ✅ Preserva norma |
| spectral | Escalado no-uniforme | ✅ Preserva estructura relativa |

### Técnicas que destruyen jerarquía

| Técnica | Operación geométrica | ¿Preserva coherencia? |
|---------|---------------------|----------------------|
| nibble_flip | Perturbación local | ❌ Sale de la superficie |
| scaled_noise | Ruido independiente | ❌ Destruye correlaciones |
| row_shuffle | Permutación | ❌ Rompe estructura espacial |

## La clave: Superficie de baja dimensión

La superficie de coherencia tiene **mucha menos dimensión** que el espacio total.

```
Espacio total: 1,100,000,000 dimensiones
Superficie: ~1,000 dimensiones (estimado)
```

Esto significa que hay **muchas direcciones** que salen de la superficie y **pocas** que permanecen sobre ella.

Las técnicas exitosas encuentran automáticamente direcciones tangentes porque:
1. **amplify_subspace**: La proyección sobre un vector aleatorio tiende a ser tangente porque la superficie ocupa una fracción minúscula del espacio
2. **lowrank**: Amplificar los valores singulares principales preserva la estructura de bajo rango de la superficie
3. **normrot**: Las rotaciones ortogonales preservan la norma y los ángulos, moviendo sobre esferas que intersectan la superficie

## Analogía: Monte en la Niebla

Imagina que estás en la cima de una montaña (superficie de coherencia) rodeada de niebla densa (basura).

- **amplify_subspace**: Caminas en una dirección aleatoria, pero sigues el terreno (tangente). Llegas a un punto diferente de la montaña.
- **nibble_flip**: Das un salto aleatorio en cualquier dirección. Probablemente caes al vacío (basura).

## Matemáticas: Variiedad de Riemann

Formalmente, la superficie de coherencia es una **variedad de Riemann** de baja dimensión embebida en un espacio euclíano de alta dimensión.

- **Métrica local**: Cada punto tiene un "espacio tangente" de direcciones que preservan coherencia
- **geodésicas**: Los caminos más cortos sobre la superficie
- **amplify_subspace**: Aproximación de una geodésica en una dirección aleatoria

## ¿Se Puede Navegar la Superficie?

Sí. Y eso es exactamente lo que hace `style_switch.py`:

1. **Delta vectors**: Vectores precomputados que apuntan a diferentes regiones de la superficie
2. **Interpolación**: Moverse en línea recta entre dos puntos de la superficie
3. **Blending**: Combinar múltiples direcciones para llegar a puntos intermedios

```
A (baseline) ----0.5----> B (philosophical)
    \                      /
     \                    /
      \                  /
       ---0.3--- C (stoic)
```

El punto medio entre A y B es otro punto válido en la superficie.

## Conclusión

**El cambio de perspectiva es geometría.** Es un movimiento sobre una variedad de baja dimensión embebida en un espacio de alta dimensión.

Las técnicas de perturbación que funcionan son aquellas que encuentran direcciones tangentes a esa variedad. Las que fallan son aquellas que salen de ella.

La pregunta no es "¿por qué funciona amplify_subspace?" sino "¿por qué la mayoría de las direcciones aleatorias son tangentes a la superficie de coherencia?"

La respuesta: porque la superficie, aunque de baja dimensión, tiene una estructura fractal que la hace "gruesa" en ciertas regiones del espacio.
