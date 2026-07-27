# Capítulo 26: Tokens → Embeddings → Ideas puras → Semántica → Detalles → Respuesta

## El pedido

Ordenar el viaje del significado en el microcosmos
TinyLlama — no como cajas sueltas del transformer,
sino como **cadena completa**, de la chispa simbólica
hasta la frase que vuelve al mundo.

---

## Cadena reorganizada (canónica)

```
1. TOKENS          símbolos discretos del vocabulario
        ↓
2. EMBEDDINGS      geometría de entrada en ℝ²⁰⁴⁸
        ↓
3. DETALLES        forma local (sintaxis, vecinos, superficie)
        ↓
4. IDEAS PURAS     abstracciones y marcos (capas medias)
        ↓
5. SEMÁNTICA       significado ligado en contexto (atención + integración)
        ↓
6. DETALLES finos  concreción léxica / estilo (FFN tardío + cabeza)
        ↓
7. RESPUESTA       logits → sample → tokens otra vez
        ↓
      (vuelve a 1)
```

Hay **dos apariciones de “detalles”** a propósito:
- **Detalles de forma** (temprano): *cómo* se escribe.
- **Detalles de contenido** (tardío): *qué* se concreta al hablar.

Las “ideas puras” viven en el medio: ni solo letra,
ni aún la frase cerrada.

---

## Tabla maestra

| # | Etapa | ¿Qué es? | Dónde en el modelo | Dim / objeto | Instrumento Dreaming |
|---|-------|----------|--------------------|--------------|----------------------|
| 1 | **Tokens** | Piezas del BPE (`▁love`, ids) | Vocabulario \(V=32\mathrm{k}\) | conjunto finito | tokenizer GGUF |
| 2 | **Embeddings** | Punto en el cielo | `token_embd` | \(\mathbb{R}^{2048}\) | mapas 2D/3D, arquetipos |
| 3 | **Detalles (forma)** | Relaciones locales, sintaxis | Capas **0–5**, attn corta | residual aún “pegado” al emb | dungeon L0–L5 · zona gravity/matter |
| 4 | **Ideas puras** | Marcos, temas, roles abstractos | Capas **6–12** | residual tematizado | zonas mage/sage |
| 5 | **Semántica** | Significado *en contexto* (quién se une a quién) | Attn global + capas **13–20** | acoplamiento \(a_{t,t'}\) | zonas gravity + drama + surface |
| 6 | **Detalles (contenido)** | Léxico fino, pasos, color local | FFN (esp. tardío) + hábitos SwiGLU | \(\mathbb{R}^{5632}\) intermedio | zona matter · voz práctica |
| 7 | **Respuesta** | Un token (y luego una frase) | `output_norm` → lm_head → softmax | \(\mathbb{R}^{32000}\) → sample | zona event · Space en el juego |

---

## 1. Tokens

**Entrada y salida del espejo.**

- Discretos, finitos, sin “sentido” hasta proyectarse.
- El BPE trocea el mundo: no todo concepto es un solo id.
- En el juego: el **sample** del final vuelve a ser token
  y reinicia la órbita.

Sin tokens no hay bordes que contar.
Con solo tokens no hay universo continuo.

---

## 2. Embeddings

**Nacimiento geométrico.**

\[
t \mapsto e_t \in \mathbb{R}^{2048}
\]

- Islas semánticas y arquetipos viven aquí como **catálogo**.
- Opposites ≈ ortogonales, no antipodales.
- PCA: cientos de dims reales; el mapa 2D/3D es planetario.

Aquí el residual **aún no ha viajado**:
es potencial de sentido, no aún frase.

---

## 3. Detalles de forma (temprano)

**Capas 0–5 · “cómo se junta la letra”.**

- Patrones adyacentes, dependencias cortas.
- La atención empieza a acoplar vecinos.
- El FFN ajusta superficie léxica.

Si se rompe esta etapa, el texto pierde **gramática**
antes que “profundidad filosófica”.

En el juego: primeros portales · zonas **sky → gravity/matter**.

---

## 4. Ideas puras (medio)

**Capas 6–12 · “de qué va esto”.**

- Marcos: existencial, académico, narrativo, técnico.
- El residual se despega del puro bigrama.
- Aquí encajan constelaciones Mago/Místico y Sabio
  como *climas de idea*, no solo palabras sueltas.

Hipótesis de trabajo del libro: el tramo medio es
donde `--steer soul` y `mystical` dejan de ser cosméticos
y se vuelven **sesgo temático**.

En el juego: warps a **mage** y **sage**.

---

## 5. Semántica (ligar en contexto)

**Atención de largo alcance + integración tardía.**

Semántica ≠ lista de embeddings.
Semántica = **relaciones**:

\[
\mathrm{semántica}(t) \approx \sum_{t'\le t} a_{t,t'}\, v_{t'}
\]

reescrita capa a capa y mezclada con el residual.

- Quién modifica a quién en la frase.
- Polaridades Héroe/Sombra como tensión en el hilo.
- Regla de Oro: tocar **atención** mueve el reflejo
  hacia lo **académico / relacional / crítico**.

En el juego: zonas **gravity**, **drama**, **surface**.

---

## 6. Detalles de contenido (concreción)

**FFN · “con qué palabras y gestos se dice”.**

Aunque el FFN actúa en todas las capas, su papel
como *detalle fino* se nota al concretar:

- verbos de acción, listas, consejos (voz práctica),
- color léxico, hábitos locales en \(\mathbb{R}^{5632}\).

Regla de Oro: tocar **FFN** → perspectiva **práctica**.

No es la idea pura; es la **encarnación** de la idea
en material verbal.

---

## 7. Respuesta

**Colapso y retorno al macrocosmos.**

\[
z = W\,\mathrm{RMSNorm}(x_L),\quad
t\sim \mathrm{softmax}(z/T)\ \text{(top-k)}
\]

- Un evento discreto (token).
- Concatenado, vuelve a ser lenguaje humano.
- Cierra el espejo: del reloj al cielo (cap. 6, 24).

Luego el ciclo:

**respuesta → nuevos tokens → …**

---

## Diagrama de flujo (completo)

```
 MACRO: pregunta humana / prompt
              │
              ▼
     ┌──── TOKENS ────┐
     │                │
     ▼                │
 EMBEDDINGS (cielo)   │
     │                │
     ▼                │
 DETALLES forma       │   capas 0–5
 (sintaxis, vecinos)  │
     │                │
     ▼                │
 IDEAS PURAS          │   capas 6–12
 (marcos, temas)      │
     │                │
     ▼                │
 SEMÁNTICA            │   attn + capas 13–20
 (lazos en contexto)  │
     │                │
     ▼                │
 DETALLES contenido   │   FFN / estilo fino
 (léxico, acción)     │
     │                │
     ▼                │
 RESPUESTA (sample) ──┘   logits → token
     │
     ▼
 MACRO: leemos una voz / arquetipo / juicio
```

Las lentes Dreaming actúan **a lo largo** de la cadena:

| Lente | Dónde tuerce más la cadena |
|-------|----------------------------|
| baseline | toda la cadena “oficial” |
| mystical / Mago | ideas puras + semántica existencial |
| académica / Sabio | semántica relacional / estructura |
| práctica | detalles de contenido (FFN) |
| noise | rompe la cadena (sale de \(\mathcal{C}\)) |
| `--steer` | empuja residual hacia un embedding-isla |

---

## Relación con otras piezas del libro

| Capítulo | Encaje en la cadena |
|----------|---------------------|
| 2 Estructura | Dónde viven las etapas en tensores |
| 3 Motor C | Cómo se computa cada flecha |
| 5 Espacio multi-D | Etapas 1–2 y geometría del cielo |
| 7 Fuerzas | Attn=semántica no local; FFN=detalles de contenido |
| 9 Regla de Oro | Lentes sobre 5 y 6 |
| 13–15 Capas | Partición temporal de 3–4–5 |
| 16–21 Islas / arquetipos | Etiqueta cultural de 2 y 4 |
| 20 Órbita | La cadena como dinámica \(x\leftarrow x+F(x)\) |
| 25 Juego | Cada portal = avanzar etapa + warp de zona |

---

## Versión corta (para el HUD del juego / glosario)

```
TOKENS → EMBEDDINGS → DETALLES → IDEAS PURAS
       → SEMÁNTICA → DETALLES FINOS → RESPUESTA → (tokens)
```

O en una línea:

**Símbolo → geometría → forma → idea → lazo → concreción → dicho.**

---

## En una frase

El universo TinyLlama no es solo un stack de capas:
es una **cadena de transformaciones del sentido**
donde los tokens se vuelven geometría, la geometría
se vuelve forma e idea, la idea se amarra en semántica,
se detalla en léxico y **colapsa** otra vez en tokens
que podemos leer — un espejo cíclico entre micro y macro.

---

*Siguiente capítulo: Cada Capa Es un Elevador.*
