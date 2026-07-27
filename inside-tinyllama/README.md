# Inside TinyLlama

A Journey of Discovery Through a Small Language Model  
Cuaderno del proyecto **Dreaming** sobre TinyLlama-1.1B.

## About

Exploramos el interior de un transformer pequeño: estructura,
motor de inferencia en C, perturbación de pesos (perspectivas),
geometría del embedding, fuerzas del forward y el vaivén
macrocosmos ↔ microcosmos.

## Key Discoveries

- **Superficie de coherencia**: perturbaciones que preservan jerarquía → texto coherente con otra voz  
- **Regla de Oro geométrica**: Atención → académico · FFN → práctico · Embeddings → simple  
- **Islas semánticas** en ℝ²⁰⁴⁸ (12 áreas casi ortogonales)  
- **Motor C**: KV-cache, OpenMP, `--perturb` / `--steer`, ~6–10 tok/s  

## Contents

- [Outline](OUTLINE.md)
- [Introducción](chapters/00_introduccion.md)

### Capítulos

| # | Título | Archivo |
|---|--------|---------|
| 1 | ¿Qué es TinyLlama? | [01](chapters/01_que_es_tinyllama.md) |
| 2 | Estructura interna | [02](chapters/02_estructura_interna.md) |
| 3 | Motor de inferencia en C | [03](chapters/03_motor_inferencia_c.md) |
| 4 | Perturbación y perspectivas | [04](chapters/04_perturbacion_y_perspectivas.md) |
| 5 | Espacio multidimensional | [05](chapters/05_espacio_multidimensional.md) |
| 6 | Macrocosmos ↔ microcosmos | [06](chapters/06_macrocosmos_microcosmos.md) |
| 7 | Fuerzas gravitacionales | [07](chapters/07_fuerzas_gravitacionales.md) |
| 8 | Cómo viajar | [08](chapters/08_como_viajar.md) |
| 9 | Regla de Oro geométrica | [09](chapters/09_regla_de_oro.md) |
| 10 | Tensores de atención | [10](chapters/10_tensores_atencion.md) |
| 11 | Tensores FFN | [11](chapters/11_tensores_ffn.md) |
| 12 | Normalización | [12](chapters/12_tensores_norm.md) |
| 13 | Capas 0–5 | [13](chapters/13_capas_tempranas.md) |
| 14 | Capas 6–12 | [14](chapters/14_capas_intermedias.md) |
| 15 | Capas 13–21 | [15](chapters/15_capas_finales.md) |
| 16 | Áreas semánticas y mapa | [16](chapters/16_areas_semanticas_mapa.md) |
| 17 | Psicoanálisis del transformer | [17](chapters/17_psicoanalisis.md) |
| 18 | Lo que aprendimos | [18](chapters/18_lo_que_aprendimos.md) |
| 19 | Futuro de la exploración | [19](chapters/19_futuro.md) |
| 20 | Cómo orbita este universo | [20](chapters/20_como_orbita.md) |
| 21 | Arquetipos y constelaciones | [21](chapters/21_arquetipos_constelaciones.md) |
| 22 | Observación y proyección | [22](chapters/22_observacion_proyeccion.md) |
| 23 | Matemáticas del universo | [23](chapters/23_matematicas_universo.md) |
| 24 | El LLM como espejo | [24](chapters/24_espejo.md) |
| 25 | Universo como videojuego | [25](chapters/25_universo_como_juego.md) |
| 26 | Cadena del significado | [26](chapters/26_cadena_significado.md) |

## Visualización

**[▶ Mapa semántico (áreas)](https://htmlpreview.github.io/?https://raw.githubusercontent.com/FiveTechSoft/dreaming/main/inside-tinyllama/exploration/semantic_map.html)** · 
**[▶ Mapa de arquetipos](https://htmlpreview.github.io/?https://raw.githubusercontent.com/FiveTechSoft/dreaming/main/inside-tinyllama/exploration/archetype_map.html)** *(tras push)*  
Carpeta: [exploration/](exploration/)

## Structure (modelo)

- 22 capas · hidden 2048 · FFN 5632 · 32 Q / 4 KV (GQA)  
- ~1.1B parámetros · vocab 32k  
- Motor: `llm_inference.c` (repo raíz)  
- Q4_0 bulk: llama.cpp  

## Experiments (orden de magnitud)

- 24+ modelos perturbados, 240+ generaciones  
- 10+ técnicas hierarchy-preserving  
- 15 prompts × mystical (runtime C)  
- Atlas de 12 áreas semánticas  

## GitHub

https://github.com/FiveTechSoft/dreaming/tree/main/inside-tinyllama
