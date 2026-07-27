# Capítulo 28: Estrellas en el Cielo, Tokens en TinyLlama

## La pregunta del astrónomo

Miras el cielo de noche. Ves **puntos de luz**.
Algunos se agrupan en figuras que la cultura nombra
(Osa, Orión, Cruz del Sur). Entre dos estrellas
no hay un cable visible, pero la física dice que
se atraen: **gravedad**. El viajero no se teletransporta
al azar: elige una estrella, mide su vecindario y salta
al siguiente pozo.

TinyLlama tiene un cielo análogo.

> **Cada token del vocabulario es una estrella
> en un espacio de 2048 dimensiones.**
> La **atención** es la fuerza gravitacional entre ellas
> cuando el modelo “piensa” una secuencia.
> Viajar por un LLM es seguir esas atracciones
> — en el mapa estático del embedding o en la órbita
> viva del *forward*.

Este capítulo fija la analogía, la une a la **Fuerza I**
del inventario (cap. 7) y muestra un **itinerario
concreto** dentro de TinyLlama-1.1B.

---

## 1. Tabla de correspondencias

| Cielo nocturno | Universo TinyLlama |
|----------------|--------------------|
| Estrella | Token (pieza BPE del vocabulario, ~32 000) |
| Posición en la bóveda | Vector de embedding \(e_t \in \mathbb{R}^{2048}\) |
| Brillo aparente | Norma / “presencia” del token; en el mapa, tamaño y etiqueta |
| Constelación | Área semántica o arquetipo (semillas + vecinos) |
| Distancia angular en el cielo | Coseno entre embeddings (cerca ≈ alineados) |
| Gravedad de Newton | **Atención**: \(Q\cdot K^\top / \sqrt{d}\) → pesos sobre \(V\) |
| Campo gravitatorio estático (mapa de masas) | Geometría fija de `token_embd` (atlas PCA) |
| Dinámica en vivo (planetas en movimiento) | Residuales de la secuencia + KV-cache, capa a capa |
| Salto entre estrellas | Clic en una **fuerza** del mapa; o el siguiente token generado |
| Telescopio / catálogo | `semantic_map.html`, motor C, scripts de geometría |
| Atmósfera que deforma la luz | RMSNorm, temperatura del softmax, lentes `--perturb` |

No es poesía vacía: cada fila tiene un objeto medible
en el repositorio Dreaming.

---

## 2. El cielo de embeddings: 32 000 estrellas fijas

Al nacer, cada token \(t\) se clava en el firmamento:

\[
e_t = \mathrm{Embedding}(t) \in \mathbb{R}^{2048}
\]

Ese cielo es **casi isótropo** (norma media ≈ 0.68)
y, entre palabras plenas de sentidos distintos,
**casi ortogonal** (coseno ≈ 0). Por eso las
“islas” semánticas del cap. 16 son constelaciones
raras: racimos de semillas que sí se tocan un poco,
rodeadas de un fondo gris de fragmentos BPE
(como polvo interestelar: no es vacío, pero no es
constelación con nombre).

### Constelaciones = áreas

| Constelación (isla) | Estrellas-semilla (ej.) |
|---------------------|-------------------------|
| Emoción positiva | ▁love, ▁happy, ▁joy, ▁hope… |
| Social / poder | ▁work, ▁king, ▁war, ▁law… |
| Mente | ▁mind, ▁idea, ▁memory, ▁know… |
| Vida / muerte | ▁death, ▁life, ▁born, ▁die… |
| … | (doce islas en total; cap. 16) |

En el **mapa PCA 2D** proyectamos ese cielo de 2048
dimensiones a dos ejes solo para mirarlo con ojos
humanos. La proyección miente un poco —como un
planisferio miente la Tierra— pero conserva
vecindarios útiles.

---

## 3. La atención es la gravedad entre tokens

### En el macrocosmos

Dos masas se tiran la una a la otra. La fuerza
cae con la distancia; el campo organiza órbitas.

### En el microcosmos (Fuerza I)

En cada capa, cada posición \(i\) de la secuencia
pregunta a las del **pasado** \(j \le i\)
(máscara causal):

\[
\mathrm{score}_{ij} = \frac{q_i \cdot k_j}{\sqrt{d_h}},
\quad
\alpha_{ij} = \mathrm{softmax}_j(\mathrm{score}_{ij}),
\quad
z_i = \sum_j \alpha_{ij}\, v_j
\]

- \(q_i\): “quién soy y qué busco” (cuerpo que siente el campo).
- \(k_j\): “quién eres en el catálogo” (masa que anuncia su presencia).
- \(\alpha_{ij}\): **intensidad de la atracción** (cuánto “cae” \(i\) hacia \(j\)).
- \(v_j\): lo que se entrega al caer (contenido transportado).

TinyLlama usa **GQA** (32 cabezas Q, 4 KV):
varias miradas baratas sobre el mismo cielo de claves.

### Dos gravedades que no hay que confundir

| Tipo | Qué es | Cuándo se ve |
|------|--------|--------------|
| **Gravedad estática (isla)** | Coseno entre filas de `token_embd` | Mapa HTML, fuerzas precomputadas entre estrellas del atlas |
| **Gravedad dinámica (atención)** | Softmax de \(QK^\top\) en la secuencia | Forward real: el prompt crea un sistema multi-cuerpo |

La estática es el **catálogo de masas** del cielo.
La dinámica es la **órbita de esta noche**:
depende de qué estrellas hayas puesto en la secuencia
y en qué orden (causalidad = “solo el pasado tira”).

El mapa interactivo muestra la primera con arcos
dorados: *proxy geométrico* de la Fuerza I y de la
Fuerza VIII (islas). No sustituye un mapa de atención
por capa, pero enseña el gesto: **foco → fuerzas → salto**.

---

## 4. Viajar: tres escalas del mismo gesto

### Escala A — Observatorio (estrellas fijas)

1. Abres el mapa de áreas semánticas.
2. Entras en una constelación (p. ej. *Social / poder*).
3. Pulsas una estrella (`▁work`).
4. Ves las **principales fuerzas** (top cosenos con prior de isla).
5. Pulsas una fuerza y **viajas** a la estrella destino.
6. Repites: cadena de saltos por el cielo.

### Escala B — Nave en órbita (generación)

1. Lanzas un prompt: semilla la secuencia con estrellas.
2. El residual de cada posición orbita 22 capas
   (atención = acoplamiento; FFN = clima local; residual = inercia).
3. El softmax colapsa el cielo a **una** estrella nueva
   (el siguiente token).
4. Esa estrella se suma al pasado y tira de las que vengan.

### Escala C — Lentes y corrientes (cambiar la física)

- `--perturb mystical`: deforma la métrica de los pozos
  (otra “constante G” efectiva; otra voz).
- `--steer`: empuja el residual hacia una dirección
  del cielo (corriente artificial).
- Temperatura / top-k: dureza del colapso final
  (¿pozo único o niebla de estrellas posibles?).

---

## 5. Ejemplo guiado: viajar dentro de TinyLlama

### 5.1 Preparación

Mapa en vivo (GitHub Pages):

https://fivetechsoft.github.io/dreaming/exploration/semantic_map.html

Deep link de partida (estrella `▁work`, id 664):

`#/token/664/▁work`

Motor de órbita (repo raíz):

```bash
# Windows PowerShell ejemplo
$env:OMP_NUM_THREADS = "8"
.\llm_inference.exe tinyllama-1.1b.F16.gguf `
  "The secret of power is" 60 0.7 40 --seed 42
```

### 5.2 Itinerario en el observatorio (fuerzas del mapa)

Partimos de la constelación **Social / poder**.
Medido en el atlas Dreaming (coseno en ℝ²⁰⁴⁸ entre
embeddings; ranking con prior de misma isla y seeds):

| Salto | Estrella origen | Estrella destino (fuerza) | Coseno (aprox.) | Lectura |
|------:|-----------------|---------------------------|----------------:|---------|
| 0 | ▁work | — | — | Foco inicial: “trabajo / obra” |
| 1 | ▁work | ▁queen | ~0.05 | Tira hacia poder institucional |
| 2 | ▁work | ▁war | ~0.01 | Conflicto como atractor social |
| 3 | ▁work | ▁law | ~0.01 | Orden y norma |
| 4 | ▁work | ▁power | ~0.00⁺ | El nombre mismo del pozo |
| 5 | ▁work | ▁king | ~0.00⁺ | Corona, mando |

**Cómo se “viaja” en la UI**

1. Clic en `▁work` (o entra al espacio *Social* y elige la semilla).
2. Panel **Fuerzas gravitacionales**: lista ordenada + arcos dorados.
3. Clic en `#1 ▁queen` → la cámara salta; `▁queen` es el nuevo foco.
4. Desde ahí se recalculan *sus* fuerzas (nuevo cielo local).
5. Encadenas saltos como un saltamontes entre estrellas.

Otros itinerarios útiles del mismo atlas:

| Ruta | Cadena típica de seeds | Constelación |
|------|------------------------|--------------|
| Afecto | ▁happy → ▁smile → ▁love → ▁hope | Emoción positiva |
| Cognición | ▁mind → ▁idea → ▁learn → ▁memory | Mente |
| Umbral | ▁death → ▁life → ▁live → ▁born | Vida / muerte |

> **Nota de honestidad astronómica.**  
> En ℝ²⁰⁴⁸ casi todo es ortogonal: los cosenos
> “fuertes” del mapa son **relativos al vecindario**,
> no atracciones newtonianas de 0.9. El ranking
> prioriza la **isla** (constelación) y las **semillas**
> para que el viaje sea legible, no ruido BPE.

### 5.3 El mismo viaje como *prompt* (órbita viva)

El observatorio te enseña *qué estrellas se rozan*.
La nave las pone en una línea temporal:

```text
Prompt semilla (estrella inicial del sistema):
  "Work without law becomes"

Lectura Dreaming:
  ▁work  ya tira, en el catálogo, hacia law / power / king…
  Al escribir "without law", fuerzas el contraste:
  la atención de las capas siguientes tendrá que
  "mirar" work y law a la vez (gravedad dinámica).
```

Experimento mínimo (misma seed, dos lentes):

```bash
# Baseline — cielo “natural”
.\llm_inference.exe tinyllama-1.1b.F16.gguf `
  "Work without law becomes" 50 0.7 40 --seed 42

# Lente mística — otra métrica de pozos (Fuerza VII)
.\llm_inference.exe tinyllama-1.1b.F16.gguf `
  "Work without law becomes" 50 0.7 40 `
  --seed 42 --perturb mystical --intensity 0.35
```

Qué observar:

1. **Tokens generados** = estrellas nuevas que se encienden
   en la secuencia (el camino de la nave).
2. Si el texto “cae” hacia *power / king / war*,
   estás viendo la gravedad social del catálogo
   actuar en la dinámica.
3. Con `mystical`, la misma constelación de partida
   puede desviar la órbita hacia un clima existencial
   (Regla de Oro + superficie de coherencia, caps. 4 y 9).

### 5.4 Viaje corto narrado (historia de un saltamontes)

Imagina que eres un fotón de significado:

1. **Despegas** en `▁work` (atlas). Ves arcos hacia
   `queen`, `war`, `law`, `power`, `king`.
2. **Saltas** a `▁law`. La constelación sigue siendo
   social; el acento pasa de “obra” a “norma”.
3. **Escribes** el prompt: *“The law of power is”*.
   Ya no miras el catálogo: **habitas** un sistema
   multi-cuerpo. Cada capa re-pesa el pasado.
4. **Colapsas** en un token nuevo (softmax). Esa
   estrella se fija en el cielo de *esta* conversación
   (KV-cache) y tira de la siguiente.
5. Opcional: activas una **lente** (`mystical`) o una
   **corriente** (`--steer`) y el mismo despegue
   termina en otra galaxia de estilo.

Eso es viajar dentro de un LLM: no hay un pasillo
3D, hay **catálogo + fuerzas + colapso**.

---

## 6. Límites de la analogía (para no mentirnos)

| La analogía acierta | La analogía se rompe |
|---------------------|----------------------|
| Tokens = puntos con posición | No hay espacio euclídeo “visual” real en 2048-D |
| Agrupaciones = constelaciones culturales del preentreno | El modelo no “cree” en mitos; mide coocurrencias |
| Atención = atracción entre posiciones | Solo del pasado; no es simétrica como Newton |
| Mapa de cosenos = campo estático | No es la matriz de atención de una capa concreta |
| Generar = orbitar y colapsar | El “viaje” del usuario es lectura; el del modelo es álgebra |

La analogía es un **instrumento de navegación**,
no una teoría física del silicio. Sirve si te lleva
a un clic, un coseno o un prompt reproducible.

---

## 7. Puentes a otros capítulos

| Si quieres… | Ve a… |
|-------------|--------|
| Inventario de todas las fuerzas | Cap. 7 |
| Rutas A–E de vuelo (cli, perturb, steer) | Cap. 8 |
| Islas y mapa | Cap. 16 |
| Órbita residual capa a capa | Cap. 20 |
| Arquetipos como constelaciones de mito | Cap. 21 |
| Fórmulas (softmax, GQA, cosine) | Cap. 23 |
| Elevador de 22 plantas | Cap. 27 |
| Juego capas + warp de zona | Cap. 25 · `universe_game.html` |

---

## 8. Cierre

El cielo sobre tu cabeza y el vocabulario de TinyLlama
comparten un gesto: **puntos, distancias, atracciones,
saltos**.

- Las **estrellas** del modelo son tokens en ℝ²⁰⁴⁸.
- La **gravedad** que importa al hablar es la **atención**.
- El **viaje** es elegir un foco, leer sus fuerzas
  y —en el mapa o en el motor C— dejarse caer
  al siguiente pozo de significado.

Cuando pulses un token y veas arcos dorados hacia
otros, no estás solo mirando un grafo bonito:
estás leyendo el catálogo de masas del microcosmos.
Cuando lances un prompt, esas masas dejan de ser
catálogo y se vuelven **sistema solar en marcha**.
