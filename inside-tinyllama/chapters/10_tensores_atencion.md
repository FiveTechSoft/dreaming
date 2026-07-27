# Capítulo 10: Los Tensores de Atención

## Cuatro planetas por capa

En cada una de las 22 capas:

| Tensor | Pregunta | Forma lógica (TinyLlama) |
|--------|----------|---------------------------|
| **Q** (attn_q) | ¿Qué busco? | [2048, 2048] |
| **K** (attn_k) | ¿Qué ofrezco? | [256, 2048] (4×64) |
| **V** (attn_v) | ¿Qué transmito? | [256, 2048] |
| **O** (attn_output) | ¿Cómo integro? | [2048, 2048] |

Más `attn_norm` (RMSNorm antes del bloque).

## GQA: 32 ojos, 4 memorias

TinyLlama no tiene 32 K y 32 V independientes.
Tiene **32 cabezas Q** y **4 KV** compartidas
(cada KV atiende a 8 Q). Menos memoria de caché,
misma idea de multi-cabeza.

## La fórmula

```
scores = (Q Kᵀ) / √64
weights = softmax_causal(scores)
out = weights V
out = O · out
x = x + out          # residual
```

En el motor C: solo el token nuevo calcula Q/K/V;
K y V se guardan en el **KV-cache**.

## Rol en el universo

- **Fuerza de largo alcance** entre tokens.  
- **Regla de Oro:** tocar atención → perspectiva académica.  
- En el atlas de fuerzas: ~19% de la masa, máximo *alcance*.

## Qué observar al experimentar

- ¿El texto cita, estructura, “argumenta”?  
- ¿Cambia más la *relación* entre ideas que el léxico suelto?  
→ Señal de que el campo atencional domina el clima.

---

*Siguiente capítulo: Los Tensores FFN*
