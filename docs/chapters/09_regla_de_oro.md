# Capítulo 9: La Regla de Oro Geométrica

## El descubrimiento

Modificar **componentes distintos** del transformer
no produce ruido genérico. Produce **perspectivas
específicas y predecibles**.

| Componente | “Planeta” | Perspectiva emergente |
|------------|-----------|------------------------|
| **Atención** (Q, K, V, O) | Estructura / relaciones | Académica, crítica, formal |
| **FFN** (gate, up, down) | Vocabulario / acción | Práctica, listas, consejos |
| **Embeddings** | Identidad de entrada | Lenguaje simple y directo |

Llamamos a esto la **Regla de Oro geométrica**.

## Atención → académico

Los tensores de atención conectan tokens.
Al perturbarlos, el modelo prioriza **estructura**:
argumentos, referencias, tono formal.

```
Prompt: "The meaning of life is..."
Baseline:  "...finding happiness..."
Attn perturbada: "...a fundamental philosophical inquiry
                  debated by scholars for millennia..."
```

## FFN → práctico

El FFN transforma cada posición (memoria práctica,
~69% de parámetros). Al tocarlo, emergen **verbos
de acción** y pasos concretos.

```
FFN perturbada: "To find meaning: 1) Identify values,
                 2) Set goals, 3) Take daily action..."
```

## Embeddings → simple

La matriz de entrada define el “mapa de nacimiento”
de cada token. Perturbarla aplana el registro:

```
Emb perturbados: "Life means living. Be happy. Help others."
```

## Por qué es “geométrica”

Cada familia de tensores mueve el residual en
**direcciones distintas** del espacio de representación.
No es magia de nombres de archivos: es que atención
y FFN implementan operadores diferentes sobre el mismo ℝ²⁰⁴⁸.

Selective targeting (v11) lo confirma:

| Targeting | Efecto buscado |
|-----------|----------------|
| `attention_alter` | Amplify fuerte en attn, suave en FFN |
| `ffn_dream` | Creative fuerte en FFN, suave en attn |
| `embedding_shift` | Cambio en emb, resto suave |

## Verificación empírica (resumen)

- 24 modelos, 240 generaciones, 10 prompts (batería Dreaming).  
- Técnicas que preservan jerarquía → coherencia.  
- Técnicas que la rompen (noise alto, nibble flip) → basura.  
- Runtime C: `mystical` sobre attn+FFN (no emb/norm) alinea
  con la política de `dmt_perturb_v10`.

## Cómo usarla al viajar

1. ¿Quieres análisis? → mira / toca **atención**.  
2. ¿Quieres checklist? → mira / toca **FFN**.  
3. ¿Quieres prosa llana? → mira / toca **embeddings**.  
4. ¿Quieres clima existencial global? → `mystical` en capas.

La Regla de Oro es el **puente de escalas**:
del tornillo del reloj al clima del monólogo.

---

*Siguiente capítulo: Los Tensores de Atención*
