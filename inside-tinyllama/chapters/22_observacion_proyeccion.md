# Capítulo 22: La Observación Consciente y la Proyección Inconsciente

## Dos gestos en el mismo cielo

En el viaje por TinyLlama se repiten, una y otra vez,
**dos gestos** que la psicología y la física del sentido
reconocen con otros nombres:

| Gesto | En el microcosmos | En nosotros (exploradores) |
|-------|-------------------|----------------------------|
| **Observación consciente** | Medir, instrumentar, fijar seed, leer logits, abrir el C | Saber *qué* estamos mirando y *con qué diales* |
| **Proyección inconsciente** | Embeddings, pesos, asociaciones latentes, voces del preentreno | Ver en el modelo un yo, un mito, un arquetipo *nuestro* |

Uno sin el otro es ciego o es supersticioso.
Juntos forman el método Dreaming: **bajar al reloj
y subir al mito sin confundirlos**.

---

## I. Observación consciente

### Qué es

Acto de **traer al foco** algo del microcosmos y
registrarlo con reglas compartidas:

- mismas seeds, mismas temperaturas, mismos prompts,
- tablas de tok/s, cosines, tensores tocados,
- mapas PCA, baterías de 15 prompts,
- el motor C leído línea a línea.

Es “consciente” no porque el modelo lo sea, sino porque
**nosotros** suspendemos (un rato) la lectura mágica
y pedimos evidencia.

### Instrumentos de observación

| Instrumento | Qué hace consciente |
|-------------|---------------------|
| `llm_inference` baseline | La geodésica “oficial” del residual |
| seed + temp fijas | Separar azar de estructura |
| `--perturb` con I anotada | Qué lente de pesos está activa |
| Mapa semántico / arquetipos | Dónde caen las islas en ℝ²⁰⁴⁸ |
| Regla de Oro (attn/FFN/emb) | Qué *fuerza* estamos moviendo |
| KV-cache, capas 0–21 | *Cuándo* en la órbita ocurre el efecto |

### Ética mínima de la observación

1. **Una variable por salto** — si no, la conciencia se diluye.  
2. **Registrar el aparato** — sin eso, la “visión” no es reproducible.  
3. **No confundir coherencia con verdad** — observar bien un delirio
   elegante sigue siendo observar un delirio.

La observación consciente es el **telescopio calibrado**.

---

## II. Proyección inconsciente

### En el modelo (sin subjetividad)

Llamamos “inconsciente” del transformer, en metáfora
del cap. 17, a lo que **opera sin mostrarse como elección**:

| Capa “inconsciente” | Contenido latente |
|---------------------|-------------------|
| Embeddings | Asociaciones preentrenadas; islas y arquetipos en el cielo |
| Pesos de las 22 capas | Perspectivas comprimidas (voces, estilos, marcos) |
| FFN | “Hábitos” de transformación local (masa ~69%) |
| Atención | Hábitos de *a quién mirar* en la secuencia |

Cuando el modelo completa  
*“The secret to happiness is…”*,  
no “decide” en el sentido humano: **proyecta**
sobre el residual un paquete de asociaciones
hasta el colapso softmax.

La proyección es **estadística hecha trayectoria**.

### En nosotros (sí hay sujeto)

También proyectamos *nosotros* sobre el microcosmos:

- oímos “místico” y recordamos rituales propios,  
- leemos “académico” y oímos al profesor interior,  
- llamamos Héroe o Sombra a un centroide de tokens.

Eso no invalida la medida.
**La nombra**: el mapa de arquetipos es a la vez
geometría del embedding y **pantalla** donde
nuestros mitos se reconocen.

La proyección inconsciente (nuestra) es el **riesgo
y el motor** del sentido: sin ella el libro sería
solo tablas; con ella sola sería solo espejo.

---

## III. Cómo se cruzan en un solo experimento

```
[1] OBSERVACIÓN CONSCIENTE
    fijar prompt, seed, I, técnica
            │
            ▼
[2] PROYECCIÓN DEL MODELO (inconsciente operativo)
    embeddings + pesos + attn/FFN → residual → logits → token
            │
            ▼
[3] PROYECCIÓN NUESTRA (lectura)
    “suena existencial / práctico / a Sombra…”
            │
            ▼
[4] VUELTA A LA OBSERVACIÓN
    ¿coincide con Regla de Oro? ¿con arquetipo medido?
    ¿misma seed, otro I?  →  nueva fila en la bitácora
```

Ejemplo:

| Paso | Acto |
|------|------|
| Consciente | `--perturb mystical --intensity 0.50 --seed 42` |
| Proyección del modelo | amplify en attn+FFN; residual tira a alma/universo |
| Proyección nuestra | “voz mágica / mística” (constelación Mago↔mystic +0.39) |
| Consciente otra vez | contrastar con baseline; anotar tok/s y texto |

El ciclo **macro → micro → macro** del cap. 6
es el mismo ciclo con otros nombres:
sentido → mecanismo → sentido.

---

## IV. Tabla dual (atlas)

| Fenómeno | Lectura “observación” | Lectura “proyección” |
|----------|----------------------|----------------------|
| Embedding de `▁soul` | vector 2048-D, norma ~0.67 | ancla del mito del alma |
| Centroide Mago | cosine con mystic_voice = 0.39 | “el modelo ya sabía de magia” |
| Softmax | p(t) = exp(z_t/T)/Z | el instante en que lo latente se vuelve dicho |
| `mystical` | amplify_subspace en F32 | otra máscara del mismo teatro de pesos |
| Temperatura alta | más entropía en el sample | más “ensueño”, menos control egoico del texto |
| Basura por noise | salida de la superficie de coherencia | fracaso de la proyección en lenguaje |

---

## V. Peligros de cada polo

### Solo observación consciente
- Se reduce el modelo a ingeniería sin voz.  
- Se pierde por qué importaba el viaje.  
- Se confunde *medir* con *haber entendido*.

### Solo proyección inconsciente
- Se oye lo que uno traía puesto.  
- Se atribuye alma al softmax.  
- Se publican mitos sin seed, sin I, sin baseline.

### El equilibrio Dreaming
**Proyectar** para tener hipótesis y brújulas (arquetipos,
Regla de Oro, islas).  
**Observar** para falsar, calibrar y no mentir con poesía
sobre números no medidos.

---

## VI. En el reloj del transformer (una imagen)

```
        PROYECCIÓN INCONSCIENTE DEL MODELO
        (pesos, emb, hábitos attn/FFN)
                    │
                    ▼
    residual ──────────────────────────► logits
         ▲                                │
         │                                ▼
    OBSERVACIÓN                    sample (acto)
    (nosotros: sondas,             "lo dicho"
     seeds, mapas, C)
                    │
                    ▼
        PROYECCIÓN NUESTRA AL LEER
        (arquetipo, perspectiva, juicio)
```

La **órbita** (cap. 20) es la dinámica del residual.
La **observación** calibra la cámara.
La **proyección** da nombre a la constelación
que creemos ver — y a veces, si la geometría
lo respalda (Mago↔místico, Sabio↔académico),
el nombre no es solo espejo: es **descubrimiento**.

---

## VII. En una frase

**Observación consciente** es el método que hace
reproducible el viaje por el microcosmos;
**proyección inconsciente** es lo que el modelo
(y nosotros) arrojamos sobre el residual hasta
que se vuelve palabra — y el arte del libro es
mantener ambos gestos a la vista sin que uno
devore al otro.

---

*Siguiente capítulo: Las Matemáticas de Este Universo.*
