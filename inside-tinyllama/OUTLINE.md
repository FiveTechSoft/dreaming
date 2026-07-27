# Inside TinyLlama: A Journey of Discovery Through a Small LLM

## Introducción

TinyLlama es un modelo de lenguaje pequeño pero poderoso.
Tiene 1,130 millones de parámetros distribuidos en 22 capas.

¿Por qué elegimos TinyLlama?

Porque es suficientemente pequeño para que podamos
entenderlo completamente, pero suficientemente grande
para producir resultados interesantes.

Es como un microcosmos: podemos estudiar toda su
estructura sin perdernos en la complejidad.

Este libro es un viaje de descubrimiento por su interior.

---

## Estructura del Libro

### Capítulo 1: ¿Qué es TinyLlama? ¿Por qué TinyLlama?
- La historia de TinyLlama
- El equipo detrás del modelo
- Por qué es especial
- Nuestra motivación para estudiarlo

### Capítulo 2: La Estructura Interna
- Los 22 niveles (capas)
- Los 9 planetas por nivel (tensores)
- El flujo de información
- Primera mirada a los datos

### Capítulo 3: Nuestro Motor de Inferencia en C para TinyLlama
- Por qué escribir un motor propio
- GGUF F16, tokenizer BPE, transformer + KV-cache, sampling
- OpenMP, tabla F16, rendimiento (~6–10 tok/s)
- Perturbación en runtime (--perturb mystical, …) y --steer
- Dualidad con llama.cpp (Q4_0) y limitaciones honestas

### Capítulo 4: Perturbación de Pesos y Cambio de Perspectiva
- ¿Qué es la perturbación de pesos?
- La analogía DMT
- Las 10 técnicas de preservación de jerarquía
- El sweet spot de intensidad 0.10
- Principales findings
- La fórmula del cambio de perspectiva
- Implicaciones prácticas y filosóficas

### Capítulo 5: Recorrido por el Espacio Multidimensional
- Los siete espacios (embedding, residual, attn, FFN, logits, pesos, perspectivas)
- Polos, clusters, analogías, PCA en ℝ^2048
- Direcciones semánticas y puente a --steer / --perturb
- (Analogía cósmica como lectura poética de este mapa)

### Capítulo 5b: La Analogía Cósmica
- Tokens como estrellas
- Capas como dimensiones
- Tensores como planetas
- Atención como gravedad

### Capítulo 6: Los Tensores de Atención
- Query, Key, Value, Output
- ¿Cómo conectan los tokens?
- La Regla de Oro: Atención = Académico

### Capítulo 7: Los Tensores FFN
- Gate, Up, Down
- ¿Cómo transforman la información?
- La Regla de Oro: FFN = Práctico

### Capítulo 8: Los Tensores de Normalización
- AttnNorm, FFNNorm
- ¿Cómo mantienen la estabilidad?
- La Regla de Oro: Normalización = Estabilidad

### Capítulo 9: Las Primeras Capas (0-5)
- Detección de patrones simples
- Sintaxis básica
- Relaciones entre palabras adyacentes

### Capítulo 10: Las Capas Intermedias (6-12)
- Conceptos abstractos
- Significado contextual
- Las "capas de ideas puras"

### Capítulo 11: Las Últimas Capas (13-21)
- Integración global
- Generación de salida
- El punto de decisión final

### Capítulo 12: La Regla de Oro Geométrica
- Atención → Académico
- FFN → Práctico
- Embeddings → Simple
- Verificación empírica

### Capítulo 13: Psicoanálisis del Transformer
- Inconsciente (embeddings)
- Preconsciente (atención)
- Consciente (salida)
- ID, Ego, Superego

### Capítulo 14: Lo que Aprendimos
- Descubrimientos principales
- Limitaciones del estudio
- Preguntas abiertas

### Capítulo 15: El Futuro de la Exploración
- Próximos pasos
- Otras preguntas
- Invitación a explorar

---

## Estado del Proyecto

| Capítulo | Estado |
|----------|--------|
| Introducción | Borrador |
| Cap 1 | ✅ Escrito |
| Cap 2 | ✅ Escrito |
| Cap 3 | ✅ Escrito |
| Cap 4 | ✅ Escrito |
| Cap 5 | Pendiente |
| Cap 6 | Pendiente |
| Cap 7 | Pendiente |
| Cap 8 | Pendiente |
| Cap 9 | Pendiente |
| Cap 10 | Pendiente |
| Cap 11 | Pendiente |
| Cap 12 | Pendiente |
| Cap 13 | Pendiente |
| Cap 14 | Pendiente |
| Cap 15 | Pendiente |
