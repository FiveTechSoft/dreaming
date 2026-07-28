# Capítulo 7: Las Fuerzas Gravitacionales del Microcosmos

## No hay una sola gravedad

En el universo TinyLlama la “gravedad” es un conjunto
de **campos** que doblan trayectorias de significado.
Cada uno tiene masa (parámetros), alcance y efecto
en el texto.

## Inventario de fuerzas

| # | Fuerza | Soporte | Masa | Alcance |
|---|--------|---------|------|---------|
| I | Atracción atencional | Q·K/√d → V | ~19% | Entre tokens de la secuencia |
| II | Potencial FFN | SwiGLU gate/up/down | ~69% | Por token (local) |
| III | Inercia residual | x ← x + f(x) | estructura | 22 capas |
| IV | Ancla de embedding | token_embd, output | ~12% | Condición inicial |
| V | Estabilización | RMSNorm | ~0.01% | Anti-explosión |
| VI | Colapso al vocabulario | logits → softmax | cabeza | 1 de 32k tokens |
| VII | Perspectivas | perturbación de pesos | todo el modelo | Cambia el “clima” |
| VIII | Islas semánticas | geometría de embeddings | — | Atractores estáticos |

### Masas medidas (F16 lógico)

| Componente | Parámetros | Share |
|------------|------------|-------|
| FFN | ~761M | **69.2%** |
| Atención | ~208M | **18.9%** |
| Emb + lm_head | ~131M | **11.9%** |
| Normas | ~92k | **0.01%** |

## Fuerza I — Atención

No local: un token siente a otros del pasado (máscara causal).
GQA 32 Q / 4 KV: gravedad barata de memorizar.

**Regla de Oro:** perturbar atención → lente **académica / relacional**.

## Fuerza II — FFN

El “sol” del sistema de pesos. Transforma cada posición
sin mirar vecinos: clima local del residual.

**Regla de Oro:** perturbar FFN → lente **práctica / acción**.

## Fuerza III — Residual

Conservación de momento del significado. Por eso los
pasos tangentes (`amplify_subspace`) mantienen coherencia
y el ruido normal a la superficie la destruye.

## Fuerza IV y V — Nacimiento y aire

Embeddings fijan el punto de partida en ℝ²⁰⁴⁸
(norma media ≈ 0.68, casi isótropo).
RMSNorm hace habitables las 22 capas con masa mínima.

## Fuerza VI — Softmax

Colapso del continuo al evento: un token.
Temperatura y top-k son la “dureza” del pozo.

## Fuerza VII — Perspectivas

Superficie de coherencia en ℝ~¹·¹ᵉ⁹.
`mystical` = corriente tangente; `noise` fuerte = salida al vacío.

## Fuerza VIII — Constelaciones

Centroides de áreas (emotion, spirit, matter, mind…):
casi ortogonales entre islas. Atracción relativa
abstract↔mind (+0.13); time↔social (−0.09).
Love/hate no son antipodales: cos ≈ 0.

## Tres leyes

1. **Superficie** — solo trayectorias tangentes en pesos → texto coherente.  
2. **Dos materias** — atención estructura relaciones; FFN transforma contenido.  
3. **Colapso** — todo termina en un token.

## Jerarquía de dominancia

```
softmax (destino)
    ↑
atención (largo alcance)  +  FFN (masa)
    ↑
residual (inercia)
    ↑
embedding (inicio)  +  norm (estabilidad)
    ↑
pesos / perspectiva (métrica del universo)
    ↑
islas semánticas (cielo de entrada)
```

---

*Siguiente capítulo: Cómo Viajar por el Universo TinyLlama*
