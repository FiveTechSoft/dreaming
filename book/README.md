# Dreaming: Geometría de las Perspectivas en LLMs

**Un libro sobre ingeniería inversa, perturbación de pesos y la naturaleza geométrica del cambio de perspectiva en modelos de lenguaje.**

---

## Resumen

Este documento recopila toda la investigación del proyecto Dreaming, que descubrió que la perturbación cuidadosamente controlada de pesos en LLMs produce **texto coherente desde perspectivas diferentes**, no basura.

### Descubrimiento Central

> "La alucinación es información real del sistema, pero reorganizada en su forma de combinarse."

La perturbación de pesos no destruye información — la **reorganiza** en nuevas perspectivas.

---

## Contenido del Libro

### Estructura

```
book/
├── chapters/           # 17 capítulos en Markdown
│   ├── 01_introduction.md
│   ├── 02_reverse_engineering.md
│   ├── 03_transformer_architecture.md
│   ├── 04_c_inference.md
│   ├── 05_operations.md
│   ├── 06_dmt_analogy.md
│   ├── 07_perturbation_techniques.md
│   ├── 08_combinations.md
│   ├── 09_weight_space.md
│   ├── 10_coherence_manifold.md
│   ├── 11_geometric_analysis.md
│   ├── 12_experimental_results.md
│   ├── 13_model_comparison.md
│   ├── 14_perspective_analysis.md
│   ├── 15_style_switching.md
│   ├── 16_practical_applications.md
│   └── 17_future_work.md
├── code/               # Scripts ejecutables
│   ├── generate_all_models.py
│   ├── run_all_tests.py
│   ├── analyze_geometry.py
│   └── style_switch.py
├── figures/            # Figuras generadas
└── tests/              # Datos experimentales
    └── test_results.json
```

### Partes del Libro

| Parte | Capítulos | Tema |
|-------|-----------|------|
| I | 1-3 | Fundamentos (introducción, GGUF, Transformer) |
| II | 4-5 | Motor de inferencia (C puro, operaciones) |
| III | 6-8 | Perturbación de pesos (analogía DMT, 10 técnicas, combinaciones) |
| IV | 9-11 | Geometría (espacio de pesos, manifold, análisis) |
| V | 12-14 | Resultados (experimentales, comparación, perspectivas) |
| VI | 15-17 | Aplicaciones (conmutación, usos prácticos, futuro) |

---

## Datos Clave

| Métrica | Valor |
|---------|-------|
| Modelos testeados | 24 |
| Generaciones totales | 240 |
| Prompts utilizados | 10 |
| Técnicas implementadas | 10 + combinaciones |
| Modelos coherentes | 18/24 (75%) |
| Coherencia promedio | 94.4% |
| Divergencia promedio | 55.6% |

---

## Cómo Usar este Libro

### Para Leer

1. Empieza por `chapters/01_introduction.md`
2. Sigue el orden de las partes
3. Salta a secciones específicas según tu interés

### Para Ejecutar Código

```bash
# Generar todos los modelos
python code/generate_all_models.py

# Ejecutar todas las pruebas
python code/run_all_tests.py

# Análisis geométrico
python code/analyze_geometry.py

# Conmutación de estilos
python code/style_switch.py list
python code/style_switch.py generate philosophical
python code/style_switch.py blend philosophical stoic
```

### Para Ver Datos

Los resultados completos están en `tests/test_results.json`.

---

## Autor

Dreaming Project — Julio 2026

## Licencia

MIT
