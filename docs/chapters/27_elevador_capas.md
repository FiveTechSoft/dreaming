# Capítulo 27: Cada Capa Es un Elevador

## La imagen

Un edificio tiene pisos.
No caminas de la planta 3 a la 17 por el aire:
entras en un **elevador**, las puertas se cierran,
y al abrirse el mundo es otro — misma torre,
otro nivel del universo.

En TinyLlama la torre tiene **22 plantas**
(más el vestíbulo de embeddings y la azotea del softmax).

Cada capa \(\ell\) es un **elevador**:

```
puertas se cierran:   RMSNorm
viaje:                Atención + residual + FFN + residual
puertas se abren:     residual transformado en el “piso” ℓ+1
```

No te teletransportas fuera del edificio.
Te **elevas** dentro del mismo residual \(x\in\mathbb{R}^{2048}\),
pero el **paisaje** (zona del universo) cambia.

---

## 1. El edificio TinyLlama

```
        ┌─────────────────────────────┐
   Ω    │  AZOTEA · Softmax / Sample  │  ← respuesta (token)
        ├─────────────────────────────┤
  21    │  piso 21 · prep. colapso    │
  20    │  …                          │
   ⋮    │  INTEGRACIÓN / SEMÁNTICA    │  ← lazos, drama, 𝒞
  13    │  …                          │
        ├─────────────────────────────┤
  12    │  …                          │
   ⋮    │  IDEAS PURAS                │  ← marcos, mago, sabio
   6    │  …                          │
        ├─────────────────────────────┤
   5    │  …                          │
   ⋮    │  DETALLES DE FORMA          │  ← sintaxis, vecinos
   0    │  piso 0 · entrada           │
        ├─────────────────────────────┤
  −1    │  VESTÍBULO · Embeddings     │  ← cielo de tokens
        └─────────────────────────────┘
                 ▲
            prompt / tokens
```

Cada flecha vertical es un elevador \(F_\ell\):

\[
x_{\ell+1} = x_\ell + F_\ell(x_\ell;\theta_\ell)
\]

El pasajero es siempre el mismo tipo de objeto
(un vector de 2048 dims). El **nivel del universo**
es lo que ese vector *significa* tras el viaje.

---

## 2. Un viaje en el elevador (por dentro)

En cada planta \(\ell\):

| Momento | Operación | Analogía del elevador |
|---------|-----------|------------------------|
| 1 | `attn_norm` | Luces de cabina; se estabiliza el piso |
| 2 | Q, K, V + RoPE | Sensores: a quién sientes en el edificio |
| 3 | Softmax causal | Atracción solo hacia plantas/pasajeros ya presentes (pasado) |
| 4 | \(x \mathrel{+}= \mathrm{Attn}\) | El empujón de la gravedad social del texto |
| 5 | `ffn_norm` | Otra calibración |
| 6 | SwiGLU FFN | Clima del piso (materia local) |
| 7 | \(x \mathrel{+}= \mathrm{FFN}\) | Sales al rellano con otro aire |

Las puertas del elevador no te dejan en un vector
de otra dimensión: sales a **otro rellano del mismo
pasillo de 2048**, pero el “barrio” cambió.

---

## 3. Planta ↔ nivel del universo

No es solo un número \(\ell\). Cada tramo de plantas
corresponde a un **nivel del atlas** (cadena del cap. 26
+ zonas del juego):

| Plantas (capas) | Nivel del universo | Cadena del significado |
|-----------------|--------------------|------------------------|
| Vestíbulo | Cielo de tokens / islas | Tokens → Embeddings |
| 0 – 5 | Barrio de la forma | Detalles de forma |
| 6 – 12 | Barrio de las ideas puras | Ideas puras (Mago, Sabio…) |
| 13 – 20 | Barrio de la semántica ligada | Semántica + drama + \(\mathcal{C}\) |
| 21 | Ante-azotea | Detalles finos / prep. respuesta |
| Softmax | Azotea · colapso | Respuesta → nuevo token |

El juego (`universe_game.html`) hace explícito lo que
el forward hace en silencio:

> **Subir de planta = tomar el elevador de la capa**  
> **y a la vez aterrizar en otra zona del mapa del universo.**

---

## 4. ¿Por qué “elevador” y no “túnel infinito”?

Un túnel sugiere un solo paisaje alargado.
Un elevador insiste en tres hechos:

1. **Misma torre** — la dimensión del residual no cambia (\(d=2048\)).  
2. **Paradas discretas** — 22 aplicaciones \(F_\ell\), no un flujo continuo anónimo.  
3. **Mundos distintos por planta** — sintaxis ≠ idea pura ≠ colapso al vocabulario.

El KV-cache es la **memoria del edificio**:
los pasajeros de plantas temporales anteriores
(siguen ahí como K, V) tiran de ti en cada parada.

---

## 5. Botonera del elevador (mandos Dreaming)

| Botón | Efecto |
|-------|--------|
| Prompt | En qué vestíbulo entras (qué embedding inicial) |
| Seed / temp / top-k | Cómo se elige el destino en la azotea |
| `--perturb mystical` | Cambia la **mecánica de todos los elevadores** (métrica de \(F_\ell\)) |
| `--steer soul` | Viento dentro de la cabina (empuja \(x\) hacia un eje) |
| Lente académica / práctica | Sesgo hacia botones de atención o de FFN (Regla de Oro) |

No eliges solo el piso 7.
Eliges **cómo se comporta el elevador** en todos los pisos.

---

## 6. Una subida completa (narrada)

1. **Vestíbulo** — naces como \(e_t\); cerca de islas love/tech/spirit.  
2. **Elevadores 0–5** — te ordenan la ropa (forma, vecinos).  
3. **Elevadores 6–12** — el pasillo se llena de ideas: mago, sabio, marco.  
4. **Elevadores 13–20** — las ideas se *amarran* (semántica, tensión, coherencia).  
5. **Elevador 21 + azotea** — el universo se niega a seguir en continuo:
   colapsa a un token.  
6. **Reinicio** — ese token vuelve al vestíbulo; nueva subida.

Eso es **orbitar** (cap. 20) leído como **ascensor en bucle**.

---

## 7. Matemáticas mínimas

Elevador de la planta \(\ell\):

\[
\begin{aligned}
h &= \mathrm{RMSNorm}(x_\ell; w_a^{(\ell)}) \\
x' &= x_\ell + \mathrm{Attn}_\ell(h) \\
h' &= \mathrm{RMSNorm}(x'; w_f^{(\ell)}) \\
x_{\ell+1} &= x' + \mathrm{FFN}_\ell(h')
\end{aligned}
\]

Teletransporte de *zona* (en el juego / en la lectura):
no es un operador extra del GGUF; es la **etiqueta
del atlas** que ponemos al rellano \(\ell\)
(sky, gravity, matter, mage, sage, surface, event…).

---

## 8. En una frase

Cada capa es un **elevador**: el residual entra,
se deja empujar por la gravedad atencional y el clima FFN,
y al abrirse las puertas está en **otro nivel del universo
TinyLlama** — misma dimensión, otra altura de sentido —
hasta la azotea donde el softmax elige el siguiente destino
y vuelve a llamar al ascensor.

---

*Juego: portal = subir planta + warp de zona.*  
*Cadena: cap. 26 · Órbita: cap. 20 · Fuerzas: cap. 7.*
