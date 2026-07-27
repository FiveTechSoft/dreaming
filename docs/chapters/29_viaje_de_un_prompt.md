# Capítulo 29: El Viaje de un Prompt por Dentro de TinyLlama

## Para quién es este capítulo

Si has leído que un LLM “predice el siguiente token”
pero aún no *ves* el camino, este capítulo es el mapa
de carreteras completo.

Lo recorreremos **paso a paso**, con un prompt real,
sin dar por sabido ningún salto mágico. Al final
deberías poder narrar, en voz alta, qué le ocurre
a cada número desde que escribes una frase hasta
que aparece la primera palabra de la respuesta.

**Prompt de ejemplo (fijo en todo el capítulo):**

```text
The secret to happiness is
```

**Modelo:** TinyLlama-1.1B  
**Números clave que no cambian:**

| Parámetro | Valor |
|-----------|------:|
| Capas transformer | 22 (índices 0…21) |
| Dimensión residual \(d\) | 2048 |
| Vocabulario \(V\) | 32 000 |
| Cabezas Q / KV | 32 / 4 (GQA) |
| Dimensión por cabeza | 64 |
| FFN intermedio | 5632 |
| Contexto máximo | 2048 posiciones |
| RoPE base | 10 000 |

---

## 0. La película en un minuto

Antes del detalle, el trailer:

```
1.  TEXTO        "The secret to happiness is"
2.  TOKENS       ids enteros del BPE
3.  EMBEDDINGS   cada id → vector de 2048 floats
4.  PREFILL      cada token del prompt atraviesa las 22 capas
                 y rellena el KV-cache
5.  LOGITS       32 000 puntuaciones para el *siguiente* token
6.  SAMPLE       elegimos un id (temperatura, top-k, seed)
7.  DECODE       el id vuelve a ser texto legible
8.  BUCLE        ese token se mete otra vez en el modelo…
                 hasta max_new tokens o EOS
```

Todo lo demás de este capítulo es **zoom** sobre
cada flecha.

---

## 1. El prompt no es “una frase” para el modelo

### 1.1 Lo que tú ves

Una cadena de caracteres UTF-8, con espacios y sentido.

### 1.2 Lo que el modelo ve

Una **secuencia ordenada de enteros** entre 0 y 31 999.

El puente se llama **tokenizador BPE** (Byte Pair Encoding),
el mismo estilo LLaMA: las palabras suelen empezar con
el prefijo de espacio de palabra `▁` (U+2581).

Para nuestro prompt, la idea (esquema didáctico) es:

| Posición \(t\) | Pieza (idea) | Rol en la frase |
|---------------:|--------------|-----------------|
| 0 | `▁The` | artículo / arranque |
| 1 | `▁secret` | núcleo nominal |
| 2 | `▁to` | enlace |
| 3 | `▁happiness` | objeto del secreto |
| 4 | `▁is` | verbo copulativo — **el presente del predicado** |

> En la práctica el BPE a veces parte más fino
> (`happ` + `iness`, etc.). El principio no cambia:
> **texto → lista de ids**. Llamemos a esa lista
>
> \[
> (t_0, t_1, t_2, t_3, t_4)
> \]
>
> con longitud \(T_{\mathrm{prompt}} = 5\).

### 1.3 Por qué el orden importa

TinyLlama es **causal**: en la posición \(t\) solo
puede “ver” las posiciones \(0,1,\ldots,t\).
El pasado existe; el futuro de la frase **aún no**.

Eso es la regla de tráfico de todo el viaje.

---

## 2. Del id al vector: nacer en ℝ²⁰⁴⁸

Cada id \(t_i\) se convierte en un punto del cielo
de embeddings (cap. 28):

\[
x^{(i)}_{0} \;=\; e_{t_i} \;=\; \mathrm{Embedding}(t_i) \;\in\; \mathbb{R}^{2048}
\]

- Hay una tabla `token_embd` de forma lógica
  **\[32 000 × 2048\]**.
- La fila número \(t_i\) es el vector de esa estrella.
- Aquí **aún no hay capas**. Solo catálogo.

**Imagen didáctica:**  
cinco pasajeros suben al vestíbulo del edificio
(cap. 27). Cada uno trae su maleta de 2048 números.
Esas maletas se llaman **residuales**.

En notación de este capítulo:

- Superíndice \((i)\): posición en la secuencia.
- Subíndice \(\ell\): capa (0 antes de la primera capa;
  tras la capa 21 estaremos en el “piso azotea”).

Al salir del embedding:

\[
x^{(0)}_{0},\; x^{(1)}_{0},\; \ldots,\; x^{(4)}_{0}
\in \mathbb{R}^{2048}
\]

---

## 3. Dos fases de vuelo: prefill y generación

TinyLlama (y casi todo transformer causal) no
procesa el prompt de un solo golpe mágico.
Hay **dos modos**:

| Fase | Qué entra | Qué sale | KV-cache |
|------|-----------|----------|----------|
| **Prefill** | Cada token del prompt, en orden | Logits tras el **último** token del prompt | Se **llena** |
| **Generación** | Un token nuevo cada vez | Logits para el siguiente | Se **alarga** en +1 |

En el motor C (`llm_inference.c`):

```c
/* PREFILL */
for (i = 0; i < n_prompt; i++)
    model_forward_token(&model, &state, tokens[i]);

/* GENERACIÓN */
for (step = 0; step < max_new; step++) {
    next = sample_top_k(state.logits, …);
    if (next == EOS) break;
    emit(next);                          /* texto al usuario */
    model_forward_token(&model, &state, next);
}
```

Hasta el final del prefill **no hemos “respondido”**.
Solo hemos **comprendido el prompt** y dejado memoria
en el caché.

---

## 4. Un solo token en una sola capa (el núcleo)

Toma la posición actual \(p\) (por ejemplo, el último
token del prompt, \(p=4\), `▁is`).  
Su residual al llegar a la capa \(\ell\) es \(x\).

Dentro de la capa ocurren **siempre** estas siete
estaciones, en este orden:

```
        x  (residual que llega a la capa ℓ)
        │
        ▼
   [1] RMSNorm  (attn_norm)
        │
        ▼
   [2]  Q, K, V  +  RoPE
        │
        ▼
   [3]  Atención causal  (usa KV-cache de esta capa)
        │
        ▼
   [4]  Proyección O  →  residual:  x ← x + Attn
        │
        ▼
   [5] RMSNorm  (ffn_norm)
        │
        ▼
   [6]  FFN SwiGLU  (gate, up, down)
        │
        ▼
   [7]  residual:  x ← x + FFN
        │
        ▼
        x  (sale hacia la capa ℓ+1)
```

Repite eso **22 veces**. Eso es el elevador completo
para **un** token en **un** paso de forward.

---

## 5. Estación por estación (con el ejemplo)

Seguimos al pasajero de la posición \(p=4\) (`▁is`),
en una capa genérica \(\ell\), cuando ya existen
en el caché las posiciones \(0..3\) del prompt.

### Estación 1 — RMSNorm (atención)

\[
h = \mathrm{RMSNorm}(x;\; \gamma_{\ell}^{\mathrm{attn}})
\]

- No “entiende” la frase.
- **Estabiliza** la escala del vector para que
  Q y K no exploten.
- Masa de parámetros ridícula (~0.01% del modelo),
  papel enorme (cap. 7, fuerza V).

**Analogía:** calibrar la brújula antes de mirar
a las otras estrellas de la secuencia.

### Estación 2 — Nacen Q, K, V y la posición (RoPE)

\[
Q = W_Q h,\quad K = W_K h,\quad V = W_V h
\]

En TinyLlama las formas lógicas por capa son:

| Tensor | Forma lógica | Lectura humana |
|--------|--------------|----------------|
| \(W_Q\) | \[2048, 2048\] | 32 cabezas × 64 dims |
| \(W_K, W_V\) | \[256, 2048\] | **4** cabezas KV × 64 (GQA) |
| \(W_O\) | \[2048, 2048\] | reúne las 32 cabezas |

**GQA (Grouped Query Attention):**  
cada cabeza de clave/valor la **comparten 8** cabezas Q
(\(32/4 = 8\)). Menos memoria de caché, misma idea:
preguntas ricas, memoria compartida.

**RoPE (Rotary Position Embedding):**  
antes de atender, Q y K se **rotan** según la posición \(p\).
No hay un vector “posición 4” sumado aparte: la posición
está **enroscada** en el ángulo de Q y K.

Así el modelo distingue:

```text
secret to happiness   ≠   happiness to secret
```

aunque las mismas “estrellas” estén en el vocabulario.

### Estación 3 — Atención: la gravedad entre tokens

Para cada cabeza de consulta:

\[
\mathrm{score}_{p,j}
  = \frac{q_p \cdot k_j}{\sqrt{64}},
  \qquad j = 0,1,\ldots,p
\]

\[
\alpha_{p,j} = \mathrm{softmax}_j(\mathrm{score}_{p,j})
\]

\[
z_p = \sum_{j=0}^{p} \alpha_{p,j}\, v_j
\]

**Lectura con nuestro prompt** (intuición, no un mapa
de atención medido aquí):

| \(j\) | Token | Qué podría “tirar” de `▁is` |
|------:|-------|-----------------------------|
| 0 | The | poco (función gramatical) |
| 1 | secret | tema: hay un secreto |
| 2 | to | enlace |
| 3 | happiness | **contenido** del secreto |
| 4 | is | sí mismo (auto-atención) |

Los \(\alpha_{p,j}\) son la **gravedad dinámica**
(cap. 7 y 28): cuánto cae el residual de `is` hacia
cada estrella del pasado de *esta* frase.

**Máscara causal:** \(j > p\) está prohibido.
En el prefill, cuando procesamos la posición 2,
`happiness` **aún no existe** en el caché.

### Estación 4 — Mezcla de cabezas + residual de atención

Las 32 cabezas se concatenan (o se proyectan) y
pasan por \(W_O\):

\[
x \leftarrow x + O(z)
\]

El residual **no se borra**: se le **suma** el empujón
atencional. Por eso hablamos de órbita, no de
teletransporte (cap. 20).

\[
x_{\mathrm{después}} = x_{\mathrm{antes}} + \Delta_{\mathrm{attn}}
\]

### Estación 5 — RMSNorm (FFN)

Otra calibración, con otro \(\gamma_{\ell}^{\mathrm{ffn}}\).

### Estación 6 — FFN SwiGLU (el “sol” de parámetros)

Aquí vive ~**69%** de la masa del modelo:

\[
\begin{aligned}
u &= W_{\mathrm{up}} h \\
g &= W_{\mathrm{gate}} h \\
\mathrm{FFN}(h) &= W_{\mathrm{down}}\big(\mathrm{SiLU}(g)\odot u\big)
\end{aligned}
\]

- Se expande a **5632** dimensiones.
- El *gate* decide qué canales dejar pasar.
- Se comprime otra vez a 2048.

**Analogía:** la atención mira a **otros tokens**;
el FFN transforma **este** residual en solitario
— clima local, conocimiento “práctico” de la posición
(Regla de Oro: FFN → lente práctica, cap. 9).

### Estación 7 — Residual del FFN

\[
x \leftarrow x + \mathrm{FFN}(h)
\]

Sale de la capa \(\ell\) listo para la \(\ell+1\).

---

## 6. El KV-cache: la memoria del pasado

Sin caché, en cada token nuevo habría que
**recalcular** K y V de toda la frase. Imposible
en CPU a buen ritmo.

Con caché, en la capa \(\ell\):

```
cache_K[ℓ][0 .. p]   ya guardado
cache_V[ℓ][0 .. p]

Al procesar posición p:
  calcular solo K_p, V_p
  escribir cache_K[ℓ][p], cache_V[ℓ][p]
  atender Q_p contra cache_K[ℓ][0..p]
```

**Prefill de nuestro prompt:**

| Paso | Token que entra | Posiciones en caché al terminar |
|-----:|-----------------|----------------------------------|
| 1 | The | 0 |
| 2 | secret | 0–1 |
| 3 | to | 0–2 |
| 4 | happiness | 0–3 |
| 5 | is | 0–4 |

Tras el paso 5, las **22 capas** tienen K y V de
las cinco posiciones. El residual de `is` ha subido
el edificio entero. Ahí salen los **logits** del
primer token de la *respuesta*.

---

## 7. De la azotea al vocabulario: logits

Después de la capa 21:

\[
h = \mathrm{RMSNorm}(x;\; \gamma^{\mathrm{out}})
\]

\[
\mathrm{logits} = W_{\mathrm{out}}\, h \;\in\; \mathbb{R}^{32000}
\]

- `output.weight` tiene forma lógica **\[32 000 × 2048\]**
  (a veces compartida o ligada al embedding en otros
  modelos; en el GGUF de TinyLlama es el `lm_head`).
- Cada entrada \(z_k\) es “cuánto empuja el modelo
  a elegir el token de id \(k\)” **ahora**.

Todavía **no** hay palabra. Hay un ranking de 32 000
candidatos.

---

## 8. Sample: colapsar el cielo a una estrella

La **Fuerza VI** (cap. 7): del continuo al evento.

Procedimiento típico en el motor Dreaming:

1. **Temperatura** \(T\): \(z_k \leftarrow z_k / T\).
   - \(T \to 0\): casi siempre el máximo (greedy).
   - \(T\) alta: más azar, más diversidad.
2. **Top-k**: quedarse solo con los \(k\) logits más altos
   (p. ej. 40). El resto se ignora.
3. **Softmax** solo sobre esos \(k\):

\[
\pi_i = \frac{e^{z_i}}{\sum_{j\in\mathrm{top\text{-}k}} e^{z_j}}
\]

4. **Muestrear** un id según \(\pi\) (con `--seed` para
   reproducir el mismo viaje).

Supongamos (ejemplo inventado pero realista) que sale:

```text
id →  ▁being      o      ▁love      o      ▁not ...
```

Ese id se **decodifica** a texto y se muestra al usuario.
Ese es el primer paso de la respuesta.

---

## 9. El bucle autoregresivo (la respuesta crece)

El token elegido **no es el final del modelo**.
Es el **siguiente pasajero**:

```
prompt:     The secret to happiness is
+ sample:   being
nueva seq:  The secret to happiness is being
```

Se llama otra vez a `model_forward_token` **solo**
con `being`:

- Se calcula su embedding.
- Atraviesa las 22 capas.
- Escribe K,V en la posición \(p=5\) de cada capa.
- Atiende a `The…is` + `being`.
- Produce logits para el token **aún más nuevo**.

Y así:

```
The secret to happiness is being
The secret to happiness is being kind
The secret to happiness is being kind to
...
```

hasta:

- alcanzar `max_new` tokens, o
- muestrear **EOS** (fin de secuencia).

**Idea clave:**  
generar un párrafo es **muchas** repeticiones del
viaje de *un* token, no un único pase “de la frase
completa a la respuesta completa”.

---

## 10. Diagrama maestro del viaje

```
┌─────────────────────────────────────────────────────────┐
│  HUMANO:  "The secret to happiness is"                  │
└───────────────────────────┬─────────────────────────────┘
                            │ tokenizador BPE
                            ▼
┌─────────────────────────────────────────────────────────┐
│  IDS:  t0 t1 t2 t3 t4                                   │
└───────────────────────────┬─────────────────────────────┘
                            │ filas de token_embd
                            ▼
┌─────────────────────────────────────────────────────────┐
│  VECTORES:  x0..x4  ∈ ℝ²⁰⁴⁸                             │
└───────────────────────────┬─────────────────────────────┘
                            │ PREFILL (por cada ti)
                            ▼
        ┌───────────────────────────────────────┐
        │  para posición p = 0 .. 4:            │
        │    para capa ℓ = 0 .. 21:             │
        │       Norm → Attn(+RoPE,GQA,cache)    │
        │            → +residual                │
        │       Norm → FFN SwiGLU               │
        │            → +residual                │
        └───────────────────┬───────────────────┘
                            │ tras último p del prompt
                            ▼
┌─────────────────────────────────────────────────────────┐
│  output_norm → lm_head → logits[32000]                  │
└───────────────────────────┬─────────────────────────────┘
                            │ temp, top-k, softmax, sample
                            ▼
┌─────────────────────────────────────────────────────────┐
│  NUEVO TOKEN  →  texto al usuario                       │
│       │                                                 │
│       └──── vuelve a forward_token (GENERACIÓN) ──► …   │
└─────────────────────────────────────────────────────────┘
```

---

## 11. Mini-laboratorio: ver el viaje con el motor C

Desde la raíz del repo (ajustando rutas a tu GGUF):

```bash
# Prefill + generación, seed fija (reproducible)
./llm_inference tinyllama-1.1b.F16.gguf \
  "The secret to happiness is" 40 0.7 40 --seed 42
```

| Flag / arg | Papel en el viaje |
|------------|-------------------|
| prompt | estrellas iniciales de la secuencia |
| `40` (n) | cuántos tokens nuevos orbitar |
| `0.7` | temperatura del colapso |
| `40` (top-k) | anchura del pozo de candidatos |
| `--seed 42` | mismo azar → mismo camino |
| `--perturb mystical --intensity 0.35` | **deforma** Q/K/V/FFN: otra física, mismo recorrido formal |
| `--steer happiness --steer-strength 0.15` | empuja el residual hacia una dirección del cielo |

Protocolo didáctico recomendado:

1. Misma seed, `none` vs `mystical` → ¿cambia la órbita?
2. Misma seed, temp 0.2 vs 0.9 → ¿cambia el colapso?
3. Abre el [mapa semántico](https://fivetechsoft.github.io/dreaming/exploration/semantic_map.html),
   busca `▁happiness` / `▁love` y mira sus **fuerzas**
   (gravedad estática del catálogo) mientras lees
   la respuesta generada (gravedad dinámica del prompt).

---

## 12. Errores mentales frecuentes (y la corrección)

| Creencia | Realidad en TinyLlama |
|----------|------------------------|
| “El modelo lee la frase de un vistazo” | Lee **token a token**; el prefill es secuencial |
| “Cada capa inventa un vector nuevo” | Actualiza el **mismo** residual con sumas |
| “La atención mira todo el libro” | Solo el **pasado** de *esta* secuencia (hasta 2048) |
| “32 cabezas = 32 memorias KV” | Solo **4** grupos KV (GQA); 32 miradas Q |
| “El embedding ya es la respuesta” | El embedding es el **nacimiento**; faltan 22 pisos |
| “Softmax elige la palabra del prompt” | Elige el **siguiente** token del vocabulario |
| “Una respuesta = un forward” | Una respuesta = **1 prefill + N forwards** |

---

## 13. Checklist de comprensión total

Si puedes responder sí a todo, el viaje está interiorizado:

1. ¿Qué es un token y por qué no es un carácter?
2. ¿Qué dimensión tiene el residual y por qué se conserva?
3. ¿Qué prohíbe la máscara causal?
4. ¿Para qué sirve RoPE?
5. ¿Qué diferencia atención y FFN en una capa?
6. ¿Qué guarda el KV-cache y en qué fase se llena?
7. ¿Cuántas veces sube el elevador de 22 pisos para un prompt de 5 tokens en prefill?  
   → **5 × 22** pasadas de capa (una por posición).
8. ¿Qué es un logit y cómo se convierte en texto?
9. ¿Por qué generar 40 tokens implica ~40 forwards extra?
10. ¿Dónde entra una lente Dreaming (`--perturb`) en este dibujo?  
    → En los pesos de las estaciones 2–6, no en el tokenizador.

---

## 14. Puentes

| Tema | Capítulo |
|------|----------|
| Dims y tensores por capa | 2 |
| Motor C, RoPE, cache, sample | 3 |
| Fuerzas (attn, FFN, softmax…) | 7 |
| Cómo viajar (rutas A–E) | 8 |
| Atención en detalle | 10 |
| FFN en detalle | 11 |
| Capas tempranas / medias / finales | 13–15 |
| Cadena del significado (visión semántica) | 26 |
| Elevador por planta | 27 |
| Estrellas = tokens, atención = gravedad | 28 |
| Fórmulas | 23 |

---

## 15. Cierre

El viaje de un prompt no es un misterio: es una
**fábrica repetible**.

1. El texto se hace **ids**.  
2. Los ids se hacen **vectores**.  
3. Cada vector sube **22 pisos** de  
   norm → gravedad atencional → clima FFN,  
   hablando solo con el **pasado**.  
4. El último residual se proyecta a **32 000** puntuaciones.  
5. Un muestreo elige **una** estrella.  
6. Esa estrella se encola y el universo da otra vuelta.

Cuando escribas

```text
The secret to happiness is
```

y TinyLlama conteste, ya no es “la IA piensa una frase”.
Es: *cinco nacimientos, cinco subidas al edificio,
un colapso, y luego N colapsos más* — siempre la
misma física, un paso más en el tiempo.

Eso es la comprensión total del viaje.
El resto del libro (perspectivas, mapas, lentes)
son **variaciones de la física**, no otro camino.
