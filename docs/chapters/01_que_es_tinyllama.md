# Capítulo 1: ¿Qué es TinyLlama? ¿Por qué TinyLlama?

## La historia de TinyLlama

TinyLlama es un modelo de lenguaje **abierto** de
**~1,1 mil millones de parámetros** (1.1B), con la
arquitectura y el tokenizador de la familia **Llama 2**,
reducidos a un tamaño que cabe en un laboratorio
modesto: **22 capas**, residual de **2048** dimensiones,
vocabulario de **32 000** tokens.

No nació en un big-tech closed lab. Nació como
**proyecto abierto** del grupo **StatNLP** de la
**Singapore University of Technology and Design (SUTD)**:
preentrenar un “Llama pequeño” sobre un corpus masivo
(del orden del **billón de tokens**, ~3 épocas en el
informe técnico) y publicar **código + checkpoints**.

Para Dreaming es el **microcosmos** ideal: lo bastante
pequeño para abrirlo entero, lo bastante rico para
tener voz, geometría y perspectivas distintas.

## El equipo detrás del modelo

### Autores del informe técnico

El paper *TinyLlama: An Open-Source Small Language Model*
([arXiv:2401.02385](https://arxiv.org/abs/2401.02385))
firma:

| Autor | Rol en el relato público del proyecto |
|-------|----------------------------------------|
| **Peiyuan Zhang** | Co-líder del esfuerzo abierto; repo / ingeniería de preentreno |
| **Guangtao Zeng** | Co-líder; contacto de envío en arXiv (v1/v2) |
| **Tianduo Wang** | Coautor (StatNLP / SUTD) |
| **Wei Lu** | Coautor; facultad, StatNLP Research Group (SUTD) |

Afiliación institucional (paper y GitHub):

> **StatNLP Research Group**  
> **Singapore University of Technology and Design (SUTD)**

Contactos que figuran en el informe (dominio SUTD):
`peiyuan_zhang`, `tianduo_wang`, `luwei` @ sutd.edu.sg;
`guangtao_zeng` @ mymail.sutd.edu.sg.

En el repositorio oficial el proyecto se presenta como
contribución de esos cuatro nombres, con Zhang y Zeng
destacados como impulsores del *open endeavor*.

### Línea de tiempo (hechos públicos)

| Fecha | Hito |
|-------|------|
| **2023-09-01** | Inicio público del preentreno (anuncio en el [GitHub del proyecto](https://github.com/jzhang38/TinyLlama): *training has started*). |
| **Otoño–invierno 2023** | Entrenamiento a gran escala; notas intermedias, checkpoints y ajustes de curva/bugs (p. ej. actualizaciones de dic. 2023 en el repo). |
| **2024-01-04** | **Publicación del paper en arXiv** (v1): modelo 1.1B, datos, eficiencia (FlashAttention, stack tipo Lit-GPT) y resultados vs. otros open de tamaño comparable. |
| **2024-01-05** | Ecosistema inmediato: el paper aparece como destacado del día en listados de la comunidad (p. ej. Hugging Face Papers). |
| **2024 (primeros meses)** | Difusión de checkpoints en **Hugging Face** (base / intermediate steps / variantes **Chat**, p. ej. `TinyLlama/TinyLlama-1.1B-Chat-v1.0`) y adopción en demos, finetunes y motores locales (llama.cpp, GGUF, etc.). |
| **2024-06-04** | **Revisión v2** del informe en arXiv (actualización del technical report). |

### Qué construyeron (en una frase técnica)

Un **SLM (Small Language Model)** open-source:

- arquitectura **Llama 2** a escala 1.1B,
- preentreno masivo sobre mezclas de datos abiertas,
- código y pesos **reutilizables**,
- objetivo explícito: demostrar que un modelo *pequeño*
  bien preentrenado **compite** con otros abiertos de
  tamaño similar y sirve como base para investigación
  y despliegue ligero.

Licencia y cultura: el proyecto se enmarca en la ola
**open weights** posterior a LLaMA/Llama 2: no es un
API cerrado, es un artefacto que se puede bajar,
cuantizar y diseccionar — exactamente lo que hace
Dreaming en este libro.

### Reacciones y recepción

**En la comunidad de investigación y open-source**
(enero 2024 en adelante):

1. **Entusiasmo por el “pequeño bien hecho”.**  
   Tras años de narrativa “solo escala gigante”,
   TinyLlama reforzó el interés por los **SLM**:
   caben en GPU de consumidor / CPU con cuantización,
   y aún así generan texto útil.

2. **Credibilidad por apertura total.**  
   Paper + GitHub + checkpoints (no solo un blog post)
   permitieron reproducir, finetunear y *forkear*.
   Eso explica su aparición rápida en tutoriales,
   colecciones HF y backends tipo llama.cpp.

3. **Comparación con pares de ~1B.**  
   El informe sostiene que **supera** a varios modelos
   abiertos de tamaño comparable en tareas *downstream*.
   La recepción no fue “otro toy model”, sino
   “baseline serio de 1B”.

4. **Impacto bibliográfico y de uso.**  
   El arXiv acumula un volumen alto de citas para un
   technical report de SLM (orden de **centenares /
   ~mil+** según contadores públicos al cabo de ~2 años),
   señal de que se volvió **referencia de cita** cuando
   alguien necesita un Llama-like pequeño y abierto.

5. **Matiz / debate sano.**  
   Parte de la comunidad discutió el coste de preentrenar
   tanto un 1.1B (¿contradice Chinchilla?). El propio
   equipo respondió en el FAQ del repo: el valor no es solo
   el punto óptimo teórico de cómputo, sino **un artefacto
   abierto**, bien entrenado, para la comunidad.
   También hubo retrasos y notas honestas sobre curvas
   de training y schedules: más “lab abierto” que marketing
   opaco.

6. **Ecosistema Dreaming / este libro.**  
   Para nosotros la reacción relevante es práctica:
   existe un GGUF F16/Q4_0 estable, un vocabulario BPE
   LLaMA, y un tamaño que permite motor C, mapas de
   embeddings y perturbaciones sin un clúster. Sin el
   equipo StatNLP y su decisión de **abrir** el modelo,
   este cuaderno no tendría microcosmos.

### Enlaces canónicos

| Recurso | URL |
|---------|-----|
| Paper (arXiv) | https://arxiv.org/abs/2401.02385 |
| Código y bitácora del preentreno | https://github.com/jzhang38/TinyLlama |
| Modelos en Hugging Face (familia TinyLlama) | https://huggingface.co/TinyLlama |

### Cita (BibTeX del proyecto)

```bibtex
@misc{zhang2024tinyllama,
  title={TinyLlama: An Open-Source Small Language Model},
  author={Peiyuan Zhang and Guangtao Zeng and Tianduo Wang and Wei Lu},
  year={2024},
  eprint={2401.02385},
  archivePrefix={arXiv},
  primaryClass={cs.CL}
}
```

## Por qué es especial

TinyLlama es especial porque:

1. **Tamaño manejable**: Con ~1.1B parámetros, podemos
   estudiar toda su estructura sin perdernos en la complejidad
   de un modelo de decenas de miles de millones.

2. **Resultados interesantes**: A pesar de ser pequeño,
   produce texto coherente y útil; el paper lo sitúa por
   encima de varios open-source de tamaño comparable.

3. **Transparencia**: pesos y código públicos → podemos
   visualizar, perturbar y medir cada componente
   (motor C, mapas, lentes Dreaming).

4. **Linaje Llama 2**: misma “gramática” de capas (RoPE,
   GQA, SwiGLU…) que los grandes, en escala de laboratorio.

## Nuestra motivación para estudiarlo

Estudiamos TinyLlama porque nos permite:

- Entender cómo funcionan los modelos de lenguaje
  **con un forward legible de punta a punta** (cap. 29)
- Explorar la geometría del espacio de significado
- Descubrir cómo las modificaciones de pesos
  pueden cambiar perspectivas
- Aprender sobre la estructura interna de los transformers
- Agradecer, en la práctica, el trabajo de un equipo
  académico que apostó por lo **abierto**

## Conclusión

TinyLlama no es solo “un modelo pequeño”. Es el
resultado de un equipo concreto (Zhang, Zeng, Wang, Lu;
StatNLP / SUTD), de un preentreno que arrancó en
**septiembre de 2023** y de un paper que salió a la luz
el **4 de enero de 2024** — y de una comunidad que lo
adoptó como SLM de referencia.

En los próximos capítulos exploramos su estructura
interna y sus secretos como **microcosmos** medible.

---

*Siguiente capítulo: La Estructura Interna de TinyLlama*
