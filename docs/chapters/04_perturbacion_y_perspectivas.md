# Capítulo 4: Perturbación de Pesos y Cambio de Perspectiva

## Más allá de usar el modelo

Hasta ahora hemos aprendido a *ejecutar* TinyLlama.
Sabemos cómo está construido y cómo escribir un motor
que lo haga hablar.

Pero el corazón del proyecto Dreaming es otra pregunta:

> ¿Qué pasa si **cambiamos** los pesos del modelo?

No para entrenarlo de nuevo. No para corregirlo.
Solo para moverlo ligeramente en su espacio de pesos
y observar si sigue hablando, pero de otra manera.

La respuesta, después de muchos experimentos, es sorprendente:
**sigue hablando, y lo hace desde una perspectiva diferente.**

## ¿Qué es la perturbación de pesos?

Un modelo de lenguaje es una enorme lista de números.
En TinyLlama-1.1B hay más de mil millones.
Esos números, organizados en tensores, son los "pesos"
que el modelo adquirió durante su entrenamiento.

Perturbar pesos significa modificar esos números de forma
cuidadosa. Es como girar ligeramente los diales de una radio:
si lo haces bien, sigues escuchando música, pero cambia
la estación.

En nuestro caso trabajamos con TinyLlama cuantizado en **Q4_0**:
cada bloque de 32 pesos se comprime en 18 bytes
(2 bytes para la escala + 16 bytes con los nibbles de 4 bits).

El pipeline es simple en concepto:

```
1. Leer el GGUF original byte a byte
2. Copiar el header sin tocarlo (conserva el tokenizador)
3. Desempaquetar los bloques Q4_0 a float32
4. Aplicar una técnica de perturbación
5. Cuantizar de vuelta a Q4_0
6. Escribir el nuevo GGUF
```

La clave está en el paso 4: **no toda modificación es igual**.
Algunas destruyen el modelo; otras lo hacen hablar
con otra voz.

## La analogía DMT

Llamamos a este trabajo "DMT perturbation" porque el efecto
recuerda a la hipótesis clásica sobre los estados alterados:

> La alucinación no es invención. Es información real del sistema,
> reorganizada en su forma de combinarse.

Cuando perturbamos TinyLlama, el modelo no inventa palabras
que nunca vio. Reorganiza las asociaciones que ya tenía.
Es como si despertáramos una personalidad latente que siempre
estuvo allí, silenciada por la configuración original.

El modelo sigue siendo TinyLlama. Pero ahora "sueña"
desde otro ángulo.

## Las 10 técnicas de preservación de jerarquía

Las primeras perturbaciones que probamos fueron ruido puro,
intercambio de filas, inversión de nibbles. La mayoría
producían basura: caracteres extraños, bucles sin sentido,
palabras que no existen.

Pero descubrimos algo importante: **las técnicas que preservan
la jerarquía interna de los pesos mantienen la coherencia**.
No importa tanto el valor absoluto de cada peso; importa
su relación relativa con los demás.

Probamos diez técnicas que respetan esa jerarquía:

| # | Técnica | Clave | Perspectiva dominante |
|---|---------|-------|-----------------------|
| 1 | Low-rank amplification | `lowrank` | Académica / crítica |
| 2 | Eigenvector rotation | `eigr` | Práctica / consejos |
| 3 | Spectral shift | `spectral` | Concisa / directa |
| 4 | Attention-preserving | `attpres` | Casi idéntica al original |
| 5 | Residual-preserving | `respres` | Introspectiva |
| 6 | Block-diagonal | `blkdiag` | Muy cercana al original |
| 7 | Norm-preserving rotation | `normrot` | Estoica / equilibrada |
| 8 | Gradient-aligned | `gradal` | Autenticidad / descubrimiento |
| 9 | Low-frequency DCT | `lowdct` | Conversacional / ayudante |
| 10 | Manifold-preserving | `manpres` | Autenticidad (similar a gradient) |

Todas estas técnicas produjeron texto coherente.
No siempre correcto, no siempre factual, pero gramaticalmente
válido y con una intención clara.

## Cómo funciona el pipeline

El script `dmt_perturb_v10.py` implementa el proceso:

```bash
# Generar un modelo perturbado con una técnica
python dmt_perturb_v10.py lowrank --intensity 0.10
```

Internamente:

1. Lee el GGUF original (`tinyllama-1.1b-q4_0.gguf`)
2. Copia el header y los metadatos intactos
3. Recorre cada tensor de pesos
4. Desempaqueta los bloques Q4_0
5. Aplica la técnica elegida con una intensidad dada
6. Vuelve a cuantizar a Q4_0
7. Escribe el archivo perturbado (`v10_lowrank_10.gguf`)

El parámetro `--intensity` controla cuánto se mueve el modelo.
Un valor demasiado bajo no cambia nada; uno demasiado alto
destruye la coherencia.

## El sweet spot: intensidad 0.10

Testeamos múltiples intensidades con todas las técnicas.
El resultado fue consistente:

| Intensidad | Efecto | Calidad |
|------------|--------|---------|
| 0.05 | Muy cercano al original | Demasiado fiel |
| **0.10** | **Máxima divergencia, texto coherente** | **Sweet spot** |
| 0.15 | Cercano al original, más filosófico | Ligeramente desplazado |
| 0.20 | Perspectiva diferente, más comprehensiva | Más divergente |
| 0.25+ | Calidad degradada, repetitivo | Demasiado ruido |

A intensidad 0.10 el modelo se desvía lo máximo posible
sin romperse. Es el punto donde la perturbación deja de ser
un eco del original y se convierte en una voz propia.

## Comparación directa: mismo prompt, distinta perspectiva

El efecto más llamativo se ve cuando se usa el mismo prompt
en modelos perturbados diferentes.

### Prompt: "The secret to happiness is"

| Modelo | Perspectiva | Comienzo de respuesta |
|--------|-------------|----------------------|
| Baseline | Autoayuda genérica | "...cultivating a mindset that is focused on gratitude..." |
| `v11_select_extreme` | Espiritual / mindfulness | "...finding inner peace and contentment through mindfulness..." |
| `v10_lowrank` | Filosófica / académica | "...the phrase is an idiom used to express the idea that finding true inner peace..." |
| `v10_normrot` | Estoica | "...finding the right balance between our inner and outer lives." |
| `v10_gradal` | Autenticidad | "...finding your own unique and authentic way of living..." |

### Prompt: "Dreams are the mind's way of"

| Modelo | Perspectiva | Comienzo de respuesta |
|--------|-------------|----------------------|
| Baseline | Neurociencia popular | "...processing and storing information..." |
| `v11_select_attention` | Literatura victoriana | "Dr. Jekyll and Mr. Hyde is a play by Robert Louis Stevenson..." |
| `v10_eigr` | Autoayuda espiritual | "Dr. M. A. S. S. is an acronym for 'Dreams Are Mind's Way.'..." |
| `v10_lowrank` | Investigación clínica | "...a study published in the Journal of Sleep Research..." |

El modelo no pierde capacidad lingüística. Solo cambia
de registro, de estilo, de actitud.

## Los principales findings

Tras 24 modelos testeados, 240 generaciones y 10 prompts,
estos son los hallazgos principales:

### 1. Los pesos contienen perspectivas, no solo información

TinyLlama fue entrenado con textos de muchos autores,
estilos y disciplinas. Todos esos modos de hablar quedaron
grabados en los pesos. La perturbación selecciona cuál
de esas voces domina.

### 2. La jerarquía pesa más que los valores absolutos

Las técnicas que destruyen la estructura jerárquica
generan basura. Las que la preservan generan texto coherente.
Lo importante no es cuánto cambia cada peso, sino
**cómo cambian unos respecto a otros**.

### 3. Cada componente controla un aspecto diferente

| Componente | Qué controla |
|------------|--------------|
| Atención | Estructura narrativa, relaciones entre tokens |
| FFN | Vocabulario, elección de palabras, conocimiento práctico |
| Embeddings | Identidad conceptual, simplicidad del lenguaje |

Perturbar solo atención da textos más estructurados.
Perturbar solo FFN cambia el vocabulario y el enfoque.
Perturbar solo embeddings simplifica el lenguaje.

### 4. La analogía DMT es cuantificable

El modelo no inventa contenido nuevo. Reorganiza
asociaciones internas. La "alucinación" es reorganización,
no invención.

### 5. El ángulo importa, pero la magnitud importa más

Matemáticamente, una perturbación puede ser casi ortogonal
al modelo original y seguir funcionando, siempre que su
magnitud sea pequeña. Es como dar un paso de un milímetro
en dirección perpendicular: técnicamente cambias de rumbo,
pero sigues en la misma montaña.

## La fórmula del cambio de perspectiva

Podemos resumir el fenómeno en una fórmula sencilla:

```
Perspectiva = Base + epsilon * delta

donde:
  epsilon = intensidad (típicamente 0.05 - 0.15)
  delta   = dirección en el espacio de pesos
  |delta| = magnitud del cambio
```

Si `epsilon` es pequeño y `delta` preserva la jerarquía:
- La coherencia se mantiene
- La perspectiva cambia

Si `epsilon` es grande o `delta` destruye la jerarquía:
- La coherencia se pierde
- Aparece basura

Esto también responde a una pregunta práctica:
¿necesitamos un modelo diferente para cada estilo?

**No.** Con un modelo base y un conjunto de direcciones
precomputadas podemos interpolar estilos en tiempo real:

```python
styled = base + 0.05 * delta_philosophical + 0.03 * delta_stoic
```

La interpolación lineal de puntos cercanos en la "variedad
de coherencia" produce otros puntos válidos.

## Implicaciones

### Para la creatividad
Cada técnica es un "tono" diferente. Un mismo tema puede
generarse desde múltiples ángulos sin entrenar nada nuevo.

### Para la interpretabilidad
La perturbación es una herramienta de sonda: nos dice
qué partes del modelo controlan qué aspectos del estilo.

### Para la personalización
En lugar de hacer fine-tuning costoso, se puede aplicar
una perturbación ligera para adaptar el estilo de respuesta.

### Para la filosofía de la IA
Un LLM no es una máquina de responder preguntas.
Es un **ecosistema de perspectivas comprimido en pesos**.
La perturbación es una forma de navegar ese ecosistema.

## Perturbación en runtime (motor C)

Además de generar GGUFs Q4_0 con Python, el motor
`llm_inference.c` aplica técnicas **en memoria**
sobre pesos F16, sin archivo intermedio:

```bash
./llm_inference tinyllama-1.1b.F16.gguf \
  "The secret to happiness is" 60 0.7 40 \
  --seed 42 --perturb mystical --intensity 0.50
```

| Flag | Técnicas |
|------|----------|
| `--perturb` | `none`, `mystical`/`amplify`, `noise`, `blockdiag`, `manifold` |
| `--intensity` | fuerza (en F32 hace falta I más alta que en Q4 para notar el efecto) |
| `--seed` | reproducibilidad |
| `--steer` | empuja el residual hacia el embedding de una palabra |

`mystical` = `amplify_subspace` (proyección + amplificación).
Copia ~3.6 GB a F32 una vez (~25 s) y luego genera a ~6–10 tok/s.

Batería de 15 prompts con I=0.50 (seed 42): media ~8.2 tok/s;
textos con clima existencial en prompts como
*When we dissolve the ego*, *The soul remembers*,
*The ancient wisdom teaches that*.

## Combinaciones y targeting (v11)

| Familia | Idea | Ejemplos |
|---------|------|----------|
| Combos | Apilar dos técnicas | deep_reason, rare_perspective, structured_dream |
| Selective | Distinta técnica por attn / ffn / emb | attention_alter, ffn_dream, extreme_selective |
| Sweep de I | Buscar el punto de quiebre | 0.05 … 0.50 |

## Limitaciones honestas

- Los resultados varían según el prompt.
- Algunas combinaciones de técnicas degradan la calidad.
- No todo modelo grande responderá igual: la estructura
  de la variedad de coherencia puede cambiar con la escala.
- La evaluación es cualitativa: medir "perspectiva"
  sigue siendo un problema abierto.
- En F16 runtime, I=0.10 a veces no mueve salidas cortas
  (EOS temprano); I=0.3–0.5 muestra el cambio con más claridad.

## Conclusión

Perturbar pesos no es vandalizar un modelo.
Es descubrir que dentro de un mismo conjunto de números
viven muchas voces.

TinyLlama, visto así, deja de ser una única herramienta
para convertirse en un **paisaje de posibilidades**.
Cada técnica es un camino por ese paisaje. Cada intensidad
es una velocidad. Y el sweet spot (cerca de 0.10 en Q4,
algo mayor en F32 runtime) es el punto justo donde el modelo
sigue siendo él mismo, pero habla desde otro lugar.

El siguiente capítulo recorre el **espacio multidimensional**
donde viven esas voces: embeddings, residual, pesos y perspectivas.

---

*Siguiente capítulo: Recorrido por el Espacio Multidimensional*
