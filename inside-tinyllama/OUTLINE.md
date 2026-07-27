# Inside TinyLlama: A Journey of Discovery Through a Small LLM

## Introducción

TinyLlama es un modelo de lenguaje pequeño pero poderoso.
Tiene ~1.130 millones de parámetros en 22 capas.

Elegimos TinyLlama porque es un **microcosmos**:
lo bastante pequeño para abrirlo entero y lo bastante
rico para producir voces y perspectivas distintas.

Este libro es el cuaderno de bitácora del proyecto
**Dreaming** sobre ese interior: estructura, motor C,
perturbación, geometría, fuerzas y viaje macro↔micro.

---

## Estructura del Libro

### Introducción
- Qué es este libro, instrumentos, mapa de partes

### Capítulo 1: ¿Qué es TinyLlama? ¿Por qué TinyLlama?
- Historia, motivación, por qué es especial

### Capítulo 2: La Estructura Interna
- 22 capas, 9 tensores, dims 2048/5632, GQA 32/4

### Capítulo 3: Motor de Inferencia en C
- GGUF F16, BPE, KV-cache, OpenMP, --perturb, --steer

### Capítulo 4: Perturbación de Pesos y Perspectivas
- DMT, 10 técnicas, v11 combos, runtime mystical, findings

### Capítulo 5: Espacio Multidimensional
- Siete espacios, polos, clusters, PCA, direcciones

### Capítulo 6: Macrocosmos ↔ Microcosmos
- Zoom in/out, doble hélice del método, atlas bilingüe

### Capítulo 7: Fuerzas Gravitacionales
- Atención, FFN, residual, norm, softmax, perspectivas, islas

### Capítulo 8: Cómo Viajar
- Rutas A–E, itinerarios, leyes de navegación

### Capítulo 9: Regla de Oro Geométrica
- Attn→académico, FFN→práctico, Emb→simple

### Capítulo 10: Tensores de Atención
- Q,K,V,O, GQA, KV-cache

### Capítulo 11: Tensores FFN
- Gate, Up, Down, SwiGLU, 69% masa

### Capítulo 12: Tensores de Normalización
- RMSNorm, masa mínima

### Capítulo 13: Capas tempranas (0–5)
- Sintaxis y vecinos

### Capítulo 14: Capas intermedias (6–12)
- Ideas y contexto

### Capítulo 15: Capas finales (13–21)
- Integración y colapso a logits

### Capítulo 16: Áreas Semánticas y el Mapa
- 12 islas, HTML GitHub, herramientas de viz

### Capítulo 17: Psicoanálisis del Transformer
- Metáfora de capas / Ello-Yo-Superyó

### Capítulo 18: Lo que Aprendimos
- Hallazgos, límites, preguntas abiertas

### Capítulo 19: El Futuro de la Exploración
- Roadmap e invitación

### Capítulo 20: Cómo Orbita Este Universo
- Residual como cuerpo; residual = inercia
- Atención / FFN por capa = periodo orbital
- Sistema multi-cuerpo (secuencia + KV-cache)
- Órbita de generación y de perspectivas
- Receta de mandos (temp, I, steer, seed)

### Capítulo 21: Arquetipos y Constelaciones
- 12 arquetipos Pearson/Jung + 3 voces Dreaming
- Constelaciones (semillas + estrellas word-like)
- Alineaciones Mago↔Místico, Sabio↔Académico
- Mapa HTML archetype_map.html

### Capítulo 22: Observación consciente y proyección inconsciente
- Observación: medir, seed, mapas, motor C
- Proyección del modelo: emb, pesos, attn/FFN
- Proyección nuestra: arquetipos y lectura de voz
- Ciclo Dreaming: hipótesis → medida → sentido
- Equilibrio método (ni solo tablas ni solo mito)

### Capítulo 23: Matemáticas de este universo
- Espacios \(E\), residual, \(\theta\), \(\mathcal{C}\)
- Cosine, centroides, PCA, anisotropía
- RMSNorm, atención GQA, SwiGLU, residual como Euler
- Softmax, temperatura, top-k, cadena de Markov
- amplify rank-1, noise, steering, Q4_0
- Diccionario física del libro ↔ fórmulas
- Límites de lo no demostrado

### Capítulo 24: El LLM — un espejo en donde mirarnos
- Espejo de corpus, de trayectoria y de lente
- Doble reflejo: modelo + lectura humana
- Narciso vs laboratorio (antídotos Dreaming)
- Espejo roto (\(\notin\mathcal{C}\)) vs fiel
- Por qué un modelo pequeño es mejor espejo de estudio

---

## Estado del Proyecto

| Capítulo | Archivo | Estado |
|----------|---------|--------|
| Introducción | `00_introduccion.md` | ✅ |
| Cap 1 | `01_que_es_tinyllama.md` | ✅ |
| Cap 2 | `02_estructura_interna.md` | ✅ |
| Cap 3 | `03_motor_inferencia_c.md` | ✅ |
| Cap 4 | `04_perturbacion_y_perspectivas.md` | ✅ |
| Cap 5 | `05_espacio_multidimensional.md` | ✅ |
| Cap 6 | `06_macrocosmos_microcosmos.md` | ✅ |
| Cap 7 | `07_fuerzas_gravitacionales.md` | ✅ |
| Cap 8 | `08_como_viajar.md` | ✅ |
| Cap 9 | `09_regla_de_oro.md` | ✅ |
| Cap 10 | `10_tensores_atencion.md` | ✅ |
| Cap 11 | `11_tensores_ffn.md` | ✅ |
| Cap 12 | `12_tensores_norm.md` | ✅ |
| Cap 13 | `13_capas_tempranas.md` | ✅ |
| Cap 14 | `14_capas_intermedias.md` | ✅ |
| Cap 15 | `15_capas_finales.md` | ✅ |
| Cap 16 | `16_areas_semanticas_mapa.md` | ✅ |
| Cap 17 | `17_psicoanalisis.md` | ✅ |
| Cap 18 | `18_lo_que_aprendimos.md` | ✅ |
| Cap 19 | `19_futuro.md` | ✅ |
| Cap 20 | `20_como_orbita.md` | ✅ |
| Cap 21 | `21_arquetipos_constelaciones.md` | ✅ |
| Cap 22 | `22_observacion_proyeccion.md` | ✅ |
| Cap 23 | `23_matematicas_universo.md` | ✅ |
| Cap 24 | `24_espejo.md` | ✅ |

### Artefactos de exploración

| Recurso | Ubicación |
|---------|-----------|
| Mapa HTML | `exploration/semantic_map.html` |
| Preview GitHub | [htmlpreview link](https://htmlpreview.github.io/?https://raw.githubusercontent.com/FiveTechSoft/dreaming/main/inside-tinyllama/exploration/semantic_map.html) |
| Datos áreas | `exploration/semantic_areas.json` |
| Scripts | `map_semantic_areas.py`, `explore_tinyllama_space.py` (raíz del repo) |
