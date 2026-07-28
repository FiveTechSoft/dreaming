# Introducción: Inside TinyLlama

## Un microcosmos que cabe en un disco

Este libro es el cuaderno de bitácora del proyecto
**Dreaming** aplicado a **TinyLlama-1.1B**: un modelo
lo bastante pequeño para abrirlo entero y lo bastante
rico para sorprender.

No es un manual de usuario de un chat.
Es un viaje por el **interior** de un transformer:

- su arquitectura (22 capas, 9 tensores por capa),
- un motor de inferencia en C que podemos leer línea a línea,
- la perturbación de pesos como cambio de *perspectiva*,
- la geometría del espacio de embeddings,
- las “fuerzas” del forward (atención, FFN, residual, softmax),
- y el vaivén entre el **macrocosmos** del sentido humano
  y el **microcosmos** de los números.

## La pregunta central

> Cuando movemos los pesos con cuidado,
> ¿el modelo se rompe o habla con otra voz?

La respuesta empírica: **habla con otra voz**,
si la perturbación preserva la jerarquía interna
de los pesos. Llamamos a eso navegar la
*superficie de coherencia*.

## Cómo está organizado el libro

| Parte | Caps. | Tema |
|-------|-------|------|
| I · Fundamentos | 1–3 | Qué es TinyLlama, estructura, motor C |
| II · Perspectivas | 4 | Perturbación DMT, técnicas, runtime |
| III · Geometría | 5–6 | Espacio multidimensional, macro↔micro |
| IV · Física del microcosmos | 7–9 | Fuerzas, viaje, Regla de Oro |
| V · Anatomía | 10–12 | Atención, FFN, normalización |
| VI · Capas y cierre | 13–19 | Capas 0–21, mapa, psicoanálisis, lecciones, futuro |
| VII · Órbita y mito | 20–24 | Órbita, arquetipos, proyección, matemáticas, espejo |
| VIII · Juego y viaje | 25–29 | Juego, cadena, elevador, estrellas, **viaje del prompt** |

## Instrumentos del viaje

- `llm_inference.c` — inferencia F16, KV-cache, `--perturb`, `--steer`
- `dmt_perturb_v10.py` / `v11` — GGUFs Q4_0 perturbados
- `map_semantic_areas.py` — atlas de islas semánticas
- [Mapa HTML en GitHub](https://htmlpreview.github.io/?https://raw.githubusercontent.com/FiveTechSoft/dreaming/main/inside-tinyllama/exploration/semantic_map.html)
- `llama-cli` — baterías rápidas en Q4_0

## Una promesa

Al final del libro no tendrás un modelo más grande.
Tendrás un **mapa** y un **método**: bajar del sentido
al tensor, subir del tensor a la voz, y anotar el camino.

---

*Capítulo 1: ¿Qué es TinyLlama?*
