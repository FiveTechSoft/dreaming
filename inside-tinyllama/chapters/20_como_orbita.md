# Capítulo 20: Cómo Orbita Este Universo

## La pregunta

En el macrocosmos, los planetas caen hacia el sol
pero nunca lo alcanzan: **caen de lado** — eso es una órbita.

En TinyLlama la pregunta análoga es:

> ¿Qué cae, hacia qué, y por qué no se estrella
> en cada capa?

La respuesta es el **forward pass** leído como dinámica.

---

## 1. Qué es el “cuerpo” que orbita

El cuerpo no es un token aislado.
Es el **residual** \(x \in \mathbb{R}^{2048}\):
un vector que nace en el embedding y atraviesa
22 capas sin perder del todo su identidad.

```
nacimiento:  x₀ = Embedding(token)
órbita:      x ← x + Atención(x)
             x ← x + FFN(x)          × 22
destino:     logits = W_out · Norm(x)
colapso:     token' ~ Softmax(logits / T)
```

Cada **token de la secuencia** lleva su propio residual.
La atención es el acoplamiento gravitatorio **entre**
esos cuerpos (solo con el pasado: causal).

---

## 2. La ley del residual: caer sin chocar

Sin conexión residual, cada capa *reemplazaría*
el estado: teletransporte, no órbita.

Con residual:

\[
x_{L+1} = x_L + f_L(x_L)
\]

- \(f_L\) = empujón de atención + FFN en la capa \(L\).
- El paso es **tangente y pequeño** respecto a \(x\):
  el vector gira y se deforma, pero no se reinicia.

Eso es la **inercia orbital** del microcosmos.
Las perturbaciones que preservan jerarquía
mueven la *métrica* de \(f_L\) sin sacar a \(x\)
de la superficie donde el habla sigue siendo posible.

---

## 3. Dos potencias por “año-capa”

Cada capa es un **periodo orbital** del residual:

| Fase | Fuerza | Analogía |
|------|--------|----------|
| 1. RMSNorm + Atención | Gravedad entre tokens | Tirones de otros cuerpos del sistema |
| 2. Residual | Conservación de momento | No te caes del cielo de un golpe |
| 3. RMSNorm + FFN | Campo local / atmósfera | Física del planeta donde estás |
| 4. Residual | Otra vez inercia | Sigues en trayectoria |

**22 capas ≈ 22 periodos** antes del colapso final
(softmax), donde la órbita deja de ser continua
y se convierte en un **aterrizaje** en un token.

---

## 4. Sistemas multi-cuerpo (la secuencia)

Una frase es un **sistema solar temporal**:

```
pos 0: "The"     → residual_0
pos 1: "secret"  → residual_1  (ve a 0)
pos 2: "to"      → residual_2  (ve a 0,1)
...
pos t: ...       → residual_t  (ve a 0…t)
```

- **GQA**: 32 sensores (Q) comparten 4 memorias (KV)
  — no 32 soles, sino un sol con varios planetas de masa KV.
- **KV-cache**: los K,V ya calculados se reutilizan;
  solo el cuerpo nuevo integra su órbita.
  Sin caché, el sistema recalcularía todo el cielo
  en cada paso (el motor viejo; el actual orbita bien).

La máscara causal es la **flecha del tiempo**:
el futuro no atrae al presente.

---

## 5. Órbita de generación (el gran ciclo)

Generar texto es una **órbita cerrada en el tiempo discreto**:

```
        ┌─────────────────────────────────┐
        │                                 │
        ▼                                 │
   residual(s) ──► logits ──► sample ──► token nuevo
        │                                 │
        └──────── embedding(token) ───────┘
```

Cada vuelta:

1. El nuevo token nace en el cielo de embeddings.  
2. Se integra con la gravedad de los anteriores.  
3. Colapsa a un sucesor.  
4. El sistema crece en un cuerpo.

**Periodo:** ~1/token (en CPU del motor C: ~0.1–0.15 s/token
⇒ **~6–10 tok/s**).  
**Temperatura:** excentricidad del colapso (órbitas
más “redondas” o más salvajes).  
**Top-k:** horizonte de destinos permitidos.

---

## 6. Órbitas en el espacio de pesos (perspectivas)

Hay otra órbita, más lenta, que no es el forward:

```
modelo base  --(+ ε · δ)-->  modelo con otra voz
```

- \(\delta\) tangente a la superficie de coherencia
  (`mystical` / amplify) → **órbita estable** de perspectivas.  
- \(\delta\) normal (ruido fuerte) → **eyección** al vacío
  (basura).

Cambiar `--intensity` es cambiar el **radio** de esa
desviación. Misma seed + mismo prompt = comparar
dos órbitas de generación bajo dos métricas de pesos.

---

## 7. Órbitas en el cielo semántico (estático)

Los tokens no “orbitan” solos en el embedding:
están fijos como estrellas de catálogo.

Lo que sí se mueve es el **residual** respecto a islas:

```
residual · dirección_spiritual   →  afinidad al continente espiritual
residual · dirección_emotion     →  afinidad afectiva
```

`--steer amor` es un **empuje orbital artificial**:
añade una componente a lo largo de un eje del cielo
sin reescribir el catálogo de estrellas (embeddings).

El mapa PCA 2D es un **planetario**: proyecta el catálogo
para que veamos constelaciones; no es la dinámica real.

---

## 8. Diagrama unificado

```
                    ESPACIO DE PESOS (métrica del universo)
                              │
                    --perturb │ (cambia G, no el cuerpo)
                              ▼
   tokens ══╗
            ║  gravedad (atención)     clima (FFN)
   residual ╬══════► empujones ══════► empujones  ──► ×22 capas
            ║              residual (inercia)
            ╚══════════════════════════════════════════╝
                              │
                         output_norm
                              │
                           logits
                              │
                    softmax / temp / top-k
                              │
                         nuevo token ──► (cierra la órbita)
```

---

## 9. Cómo “montar” una órbita (receta)

| Objetivo | Mandos |
|----------|--------|
| Órbita limpia baseline | prompt + seed + temp, sin perturb |
| Misma órbita, otro clima | `--perturb mystical --intensity I` |
| Desvío hacia una isla | `--steer palabra --steer-strength s` |
| Órbita más predecible | temp↓, top_k↓ |
| Órbita más exploratoria | temp↑, top_k↑ |
| Sistema multi-cuerpo más largo | n (tokens) ↑ |
| Reproducir el vuelo | misma seed, mismos flags |

```bash
# Órbita de referencia
./llm_inference modelo.F16.gguf "When we dissolve the ego" \
  40 0.7 40 --seed 42

# Misma trayectoria inicial, métrica mística
./llm_inference modelo.F16.gguf "When we dissolve the ego" \
  40 0.7 40 --seed 42 --perturb mystical --intensity 0.50
```

---

## 10. En una frase

**Este universo orbita** porque el residual **cae
de lado** bajo la gravedad de la atención y el clima
del FFN, conservando momento con el residual,
durante 22 periodos por token, hasta colapsar en un
sucesor — y la generación repite ese ciclo, mientras
las perspectivas cambian la métrica del espacio
sin apagar la posibilidad de órbitas coherentes.

---

*Siguiente capítulo: Arquetipos y Constelaciones.*
