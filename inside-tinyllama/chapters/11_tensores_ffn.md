# Capítulo 11: Los Tensores FFN

## Tres planetas de la materia ordinaria

| Tensor | Rol | Forma lógica |
|--------|-----|--------------|
| **Gate** (ffn_gate) | Compuerta SiLU | [5632, 2048] |
| **Up** (ffn_up) | Expansión | [5632, 2048] |
| **Down** (ffn_down) | Compresión | [2048, 5632] |

Más `ffn_norm` antes del bloque.

## SwiGLU

```
h' = Down( SiLU(Gate(x)) ⊙ Up(x) )
x  = x + h'
```

Dimensión intermedia **5632**: el residual se expande
a un espacio más ancho y vuelve a 2048.

## Masa dominante

~**69%** de los parámetros del modelo viven aquí.
Si la atención es gravedad entre planetas, el FFN es
la **física interna** de cada uno.

## Regla de Oro

Perturbar FFN → perspectiva **práctica**:
pasos, consejos, verbos de acción, “cómo hacer”.

Selective `ffn_dream` (v11): creative fuerte en FFN,
suave en atención → clima “soñador pero accionable”.

## Qué observar

- ¿Listas numeradas, imperativos, tips?  
- ¿Menos “quién se relaciona con quién” y más “qué hacer”?  
→ Campo FFN en el asiento del conductor.

---

*Siguiente capítulo: Los Tensores de Normalización*
