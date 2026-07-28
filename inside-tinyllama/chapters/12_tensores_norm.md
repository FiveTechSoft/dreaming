# Capítulo 12: Los Tensores de Normalización

## Dos frenos por capa (+ uno final)

| Tensor | Dónde | Función |
|--------|-------|---------|
| **attn_norm** | Antes de QKV | Estabiliza entrada a atención |
| **ffn_norm** | Antes de gate/up | Estabiliza entrada a FFN |
| **output_norm** | Tras la capa 21 | Estabiliza antes del lm_head |

## RMSNorm (no LayerNorm clásico)

```
rms = sqrt(mean(x²) + ε)
out = (x / rms) * w
```

Sin restar la media: solo escala por energía del vector.

## Masa mínima, efecto total

~**0.01%** de los parámetros. Sin estas normas,
atención + FFN empujan el residual a normas
explosivas o a colapso numérico.

En el atlas de fuerzas: **constante cosmológica /
aire respirable** del microcosmos.

## Política de perturbación

`dmt_perturb_v10` y el motor C **no tocan** normas
al aplicar mystical: mover la estabilidad es el
camino más corto a la basura numérica.

## Regla práctica

Si el texto se rompe con símbolos raros tras un
experimento, revisa si tocaste normas o I excesiva
antes de culpar a la “semántica”.

---

*Siguiente capítulo: Las Primeras Capas (0–5)*
