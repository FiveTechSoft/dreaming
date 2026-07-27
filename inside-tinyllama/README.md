# Inside TinyLlama

A Journey of Discovery Through a Small Language Model

## About This Book

This book documents our exploration of TinyLlama-1.1B, a small but powerful language model. Through empirical experiments and geometric analysis, we discovered the internal structure of transformers and how weight modifications can shift perspectives.

## Key Discoveries

- **Golden Rule Geometric**: Attention → Academic perspective, FFN → Practical perspective, Embeddings → Simple language
- **Psychoanalytic Model**: Mapping Freud's concepts to transformer architecture
- **Hierarchy Preservation**: Perturbation that preserves hierarchy produces coherent text from different perspectives
- **Cosmic Analogy**: 22 layers explained through empirical testing

## Contents

- [Outline](OUTLINE.md)
- [Chapters](chapters/)

### Chapters Written

1. [¿Qué es TinyLlama? ¿Por qué TinyLlama?](chapters/01_que_es_tinyllama.md)
2. [La Estructura Interna](chapters/02_estructura_interna.md)
3. [Nuestro Motor de Inferencia en C](chapters/03_motor_inferencia_c.md)
4. [Perturbación de Pesos y Cambio de Perspectiva](chapters/04_perturbacion_y_perspectivas.md)
5. [Espacio Multidimensional](chapters/05_espacio_multidimensional.md)
- **[▶ Mapa semántico interactivo (GitHub)](https://htmlpreview.github.io/?https://raw.githubusercontent.com/FiveTechSoft/dreaming/main/inside-tinyllama/exploration/semantic_map.html)** · [exploration/](exploration/)

## Structure

- 22 transformer layers · hidden 2048 · FFN 5632 · 32 Q / 4 KV (GQA)
- 9 tensors per layer · 1.1B parameters · 32k vocabulary
- Pure-C engine (`llm_inference.c`): KV-cache, OpenMP, `--perturb` / `--steer`
- llama.cpp for bulk Q4_0 runs

## Experiments

- 24 models tested
- 240 text generations
- 10 different prompts
- 10 hierarchy-preserving techniques

## GitHub

https://github.com/FiveTechSoft/dreaming/tree/main/inside-tinyllama
