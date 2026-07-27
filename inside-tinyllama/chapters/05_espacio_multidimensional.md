# Capítulo 5: Recorrido por el Espacio Multidimensional de TinyLlama

## No hay un solo espacio

Cuando decimos “el interior de TinyLlama”, no hablamos
de un único mapa. Hablamos de **varios espacios
anidados**, cada uno con su dimensionalidad y su rol.

Este capítulo es un *viaje de campo*: medimos el
espacio de embeddings reales del modelo F16
(32.000 × 2048), con el vocabulario BPE del GGUF
(`▁love`, `▁death`, …).

Herramienta: `explore_tinyllama_space.py`  
Datos: `inside-tinyllama/exploration/`

---

## El mapa de los siete espacios

```
┌──────────────────────────────────────────────────────────┐
│  6. PESOS  ℝ^{~1.1e9}                                    │
│     superficie de coherencia ≈ “modelos que hablan”      │
│     7. PERSPECTIVAS ⊂ (6)  — trayectorias por perturb. │
├──────────────────────────────────────────────────────────┤
│  forward pass, token a token:                            │
│                                                          │
│  1. EMBEDDING     ℝ^{2048}   ← 32k puntos del vocab      │
│         ↓                                                │
│  2. RESIDUAL ×22  ℝ^{2048}   (misma dim, nuevo contenido)│
│         ↘ 3. ATENCIÓN   ℝ^{64} × 32Q / 4KV             │
│         ↘ 4. FFN         ℝ^{5632}                        │
│         ↓                                                │
│  5. LOGITS        ℝ^{32000}  → softmax → siguiente token │
└──────────────────────────────────────────────────────────┘
```

| # | Espacio | Dims | Qué es |
|---|---------|------|--------|
| 1 | Embeddings de token | 2048 | Significado “en reposo” de cada pieza del vocabulario |
| 2 | Residual stream | 2048 × 22 | Representación contextual que evoluciona capa a capa |
| 3 | Cabezas de atención | 64 | Vistas locales de relaciones entre tokens (GQA 32/4) |
| 4 | FFN intermedio | 5632 | Expansión “memoria / transformación práctica” |
| 5 | Logits | 32 000 | Preferencias sobre el próximo token |
| 6 | Pesos del modelo | ~1.1e9 | Todos los parámetros; casi todo el volumen es basura |
| 7 | Perspectivas | subvariedad de (6) | Modelos coherentes con tono distinto (mystical, etc.) |

El residual es un **túnel de 2048 dimensiones** que
atraviesa 22 habitaciones. La atención y el FFN son
desvíos laterales que escriben de nuevo en ese túnel.

---

## Región 1 — Polos semánticos

¿Están *love* y *hate* en extremos opuestos?

**No.** En el embedding estático, los “opuestos”
del lenguaje natural tienen cosine **casi cero**
(ortogonales), no −1 (antipodales).

| Par | cosine |
|-----|--------|
| ▁love / ▁hate | +0.006 |
| ▁life / ▁death | +0.016 |
| ▁happy / ▁sad | **−0.035** |
| ▁true / ▁false | **−0.036** |
| ▁good / ▁evil | −0.001 |
| ▁king / ▁queen | +0.009 |
| ▁man / ▁woman | +0.008 |

**Lectura:** en ℝ²⁰⁴⁸ “frío” no es −“calor”.
Las palabras ocupan direcciones distintas del
espacio; la oposición semántica se organiza más
por **clusters y contextos** (capas + atención)
que por antipodalidad simple en el embedding.

---

## Región 2 — Continentes (clusters)

Agrupamos palabras y tomamos el **centroide**.
Los vecinos del centroide recuperan el propio
continente — la geometría local es coherente.

| Continente | Tokens (ej.) | Vecinos del centroide |
|------------|--------------|------------------------|
| emotion_pos | happy, joy, love, peace… | smile, happy, hope, love |
| emotion_neg | sad, hate, fear, anger… | sad, pain, anger, cry |
| spiritual | soul, spirit, god, faith… | faith, divine, spirit, god |
| physical | body, rock, water, fire… | rock, water, matter, body |
| abstract | truth, beauty, justice… | beauty, meaning, idea… |
| time | time, past, future, now… | time, now, moment, past |

### Distancia entre continentes

Los centroides de continentes distintos son
**casi ortogonales** entre sí (cosine ≈ 0):

```
emotion_pos  ⊥  emotion_neg   (−0.01)
spiritual    ⊥  physical      (+0.02)
abstract     ⊥  physical      (−0.01)
time         ⊥  abstract      (−0.06)
```

El vocabulario no es una bola difusa: es un
**conjunto de islas** en una esfera de 2048 dims,
con poca superposición entre islas temáticas.

---

## Región 3 — Analogías (a − b + c)

La prueba clásica de word2vec:

```
king − man + woman  ≟  queen
```

En TinyLlama (embedding estático, top-6) **falla**:
aparecen piezas raras del BPE, símbolos, fragmentos
multilingües — no `queen`.

Eso no dice que el modelo “no sepa” la analogía.
Dice que:

1. El embedding de un token **sin contexto** es
   solo la puerta de entrada.
2. La analogía “viva” se arma en el **residual**
   tras atención y FFN, no en la fila del vocab.
3. El BPE trocea el mundo (`builder`, sufijos…);
   no todo concepto es un único punto limpio.

---

## Región 4 — Forma global de ℝ²⁰⁴⁸

PCA sobre 4 000 tokens al azar:

| Métrica | Valor |
|---------|--------|
| Varianza en el 1.er PC | **0.27%** |
| Varianza en top-10 | 2.3% |
| Varianza en top-100 | 14% |
| Dims para 50% de la var. | **~481** |
| Dims para 90% | **~1329** |
| Dims para 99% | **~1880** |
| Anisotropía \|\|mean\|\| / mean\|\|e\|\| | **0.006** (casi isótropo) |

**Lectura:** el espacio de tokens **usa de verdad
cientos o miles de direcciones**. No colapsa a un
par de ejes “bueno/malo”. Por eso las perturbaciones
de rango-1 (amplify) pueden “girar el cristal” sin
apagar el habla: hay mucho volumen de coherencia.

---

## Región 5 — Direcciones como brújulas

Si restamos centroides, aparecen **ejes semánticos
utilizables**:

### emotion = pos − neg
- polo + → smile, happy, peace, love, joy  
- polo − → sad, anger, cry, pain, fear  

### spirit − matter
- + → spirit, god, sacred, divine, faith  
- − → rock, matter, water, earth, body  

### abstract − physical
- + → beauty, truth, justice, meaning, freedom  
- − → rock, matter, fire, water, earth  

Estas direcciones viven en el **mismo ℝ²⁰⁴⁸**
que el residual. Por eso `--steer amor` en el motor
C puede empujar la generación: es un vector en el
túnel, no magia externa.

Y por eso `amplify_subspace` en el espacio de
**pesos** (dimensión 1e9) es otro viaje: mueve el
*mapa entero*, no un punto del vocabulario.

---

## Región 6 — Normas: no todo token “pesa” igual

\|\|e\|\| media ≈ 0.67. Los extremos no son
conceptos filosóficos claros (a menudo piezas BPE
o símbolos). La **norma** no es un diccionario de
importancia semántica; es otra coordenada del
paisaje.

---

## Cómo se conectan los espacios en un paso de inferencia

```
"happiness"
    → BPE → ids
    → filas en (1) EMBEDDING          ℝ^2048
    → 22× { attn en (3) + FFN en (4) }  escribiendo en (2)
    → (5) LOGITS
    → sample → "is" / "to" / …
```

Si perturbamos pesos (6) con *mystical*,
cada proyección Q/K/V/FFN se deforma un poco:
el camino en (2) sigue siendo coherente, pero
las **atracciones** hacia islas de (1) y (5) cambian
— de ahí el cambio de perspectiva.

Si hacemos *steer* en (2), empujamos el residual
hacia una dirección de (1) sin reescribir (6).

---

## Itinerario del explorador

| Parada | Pregunta | Respuesta empírica |
|--------|----------|-------------------|
| Polos | ¿Los opuestos son antipodales? | No: casi ortogonales |
| Continentes | ¿Hay regiones temáticas? | Sí: clusters limpios |
| Analogías estáticas | ¿king−man+woman? | No en embedding crudo |
| Dimensionalidad | ¿Cuántas dims importan? | Cientos–miles (no 2–3) |
| Direcciones | ¿Hay ejes útiles? | Sí (emotion, spirit…) |
| Pesos | ¿Dónde viven las perspectivas? | Superficie en ℝ^1e9 |

---

## Qué queda por recorrer

1. **Residual por capa** — proyectar activaciones
   de las 22 capas sobre los ejes emotion/spirit
   (¿dónde se “enciende” lo místico?).
2. **FFN ℝ⁵⁶³²** — neuronas que reaccionan a
   clusters semánticos.
3. **Trayectorias de perturbación** — curva de
   cosine(baseline, mystical) en función de I
   en el espacio de pesos o de logits.
4. **Mapas 2D/3D** — UMAP/t-SNE de los 32k
   puntos coloreados por continente.

El universo de TinyLlama no es un punto.
Es un **sistema de espacios**. Este capítulo solo
cruzó la primera frontera: el cielo de los tokens.
Más adentro, el residual y los pesos esperan.

---

*Herramientas: `explore_tinyllama_space.py`,
`llm_inference.c --perturb` / `--steer`.*

*Siguiente capítulo: Del Macrocosmos al Microcosmos (y viceversa).*
