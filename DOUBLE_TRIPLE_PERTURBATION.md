# Resultados: Doble y Triple Perturbación

## Experimento

Pregunta: ¿Qué pasa si perturbamos un modelo ya perturbado múltiples veces?

## Configuración

```
Modelo base: v10_lowrank_10.gguf (perturbación 1)
Técnica: amplify_subspace
Intensidad: 0.03 (3%) por perturbación
```

## Resultados

### "The meaning of life is"

| Modelo | Perturbaciones | Respuesta |
|--------|----------------|-----------|
| **BASELINE** | 0 | "In the context of a literary analysis, the meaning of life is a central theme explored in many works of literature..." |
| **LOWRANK** | 1 | "In popular usage, the term 'meaning of life' refers to an ongoing process of self-reflection, contemplation..." |
| **LOWRANK_DOUBLE** | 2 | "The meaning of life is a broad and complex concept that encompasses various aspects of human well-being..." |
| **LOWRANK_TRIPLE** | 3 | "Salvo y sueno, de la música y el sonido pueden ser de..." (empieza a cambiar idioma) |

## Análisis

```
PERTURBACIÓN 1 (LOWRANK):
├── Tono: Definición formal
├── Enfoque: Auto-reflexión
└── Estado: COHERENTE

PERTURBACIÓN 2 (DOUBLE):
├── Tono: Descriptivo general
├── Enfoque: Bienestar humano
└── Estado: COHERENTE

PERTURBACIÓN 3 (TRIPLE):
├── Tono: Cambio de idioma (español)
├── Enfoque: Tokens de chat
└── Estado: EMPEZANDO A DEGRADARSE
```

## Hallazgos

1. **1-2 perturbaciones**: Producen perspectivas diferentes pero coherentes
2. **3 perturbaciones**: El modelo empieza a cambiar comportamiento (idioma, tokens)
3. **Límite**: Después de 3 perturbaciones, el modelo puede degradarse

## Velocidad de Inferencia

```
BASELINE:       212 t/s prompt, 60 t/s gen
LOWRANK:        209 t/s prompt, 63 t/s gen
DOUBLE:         217 t/s prompt, 62 t/s gen
TRIPLE:         208 t/s prompt, 57 t/s gen
```

La velocidad se mantiene estable en todas las perturbaciones.

## Conclusión

> Se puede perturbar un modelo múltiples veces, pero hay un LÍMITE
> donde el comportamiento empieza a cambiar drásticamente.
> 
> 1-2 perturbaciones: Perspectivas diferentes
> 3+ perturbaciones: Posible degradación

## Archivos

```
C:/tmp/v10_lowrank_10.gguf        (1 perturbación)
C:/tmp/v10_lowrank_double.gguf    (2 perturbaciones)
C:/tmp/v10_lowrank_triple.gguf    (3 perturbaciones)
```
