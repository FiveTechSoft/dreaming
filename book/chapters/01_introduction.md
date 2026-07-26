# Capítulo 1: Introducción — El Proyecto Dreaming

## 1.1 ¿Qué es Dreaming?

Dreaming es un proyecto que construye un LLM de 500MB **desde cero** usando técnicas de ingeniería inversa sobre arquitecturas existentes (DeepSeek-V2 Lite), combinado con generación de pesos inspirada en sueños y un motor de inferencia en C puro sin dependencias externas.

Pero Dreaming no es solo construir un modelo. Es **entender** cómo funcionan los LLMs a nivel binario, y luego usar ese entendimiento para hacer algo que nadie había hecho: **cambiar la perspectiva** de un modelo perturbando sus pesos.

## 1.2 El Descubrimiento Central

Cuando perturbamos los pesos de un LLM con técnicas que preservan la jerarquía estructural, el modelo no genera basura. Genera texto **coherente, gramaticalmente correcto, factualmente plausible** — pero desde una **perspectiva completamente diferente**.

```
Prompt: "The secret to happiness is"

Baseline:     "...cultivating a mindset focused on gratitude..."
Filosófico:   "...finding true inner peace and contentment..."
Estoico:      "...balance between inner and outer lives..."
Espiritual:   "...mindfulness, gratitude, and appreciation..."
```

Mismo modelo. Mismo prompt. Diferente perspectiva.

## 1.3 La Analogía DMT

La analogía con DMT (dimetiltriptamina) es precisa y técnica:

> "La alucinación es información real del sistema, pero reorganizada en su forma de combinarse."

Un LLM no inventa contenido de la nada. **Reorganiza** las asociaciones internas que ya contiene en sus pesos. La perturbación de pesos es una forma controlada de esto.

| DMT | LLM Perturbation |
|-----|------------------|
| Reorganiza percepciones | Reorganiza asociaciones de pesos |
| No inventa contenido | No inventa vocabulario |
| Experiencia interna | Texto coherente |
| Estado alterado | Perspectiva diferente |

## 1.4 Lo que Descubrimos

### Los pesos contienen perspectivas, no solo información

TinyLlama fue entrenado con millones de textos. Cada texto tiene un tono, una perspectiva, un estilo. Todo eso está **codificado en los pesos**.

La perturbación no destruye esa información — la **remezcla**.

### La jerarquía pesa más que los valores absolutos

Lo que importa no es el valor individual de cada peso, sino la **relación jerárquica entre pesos**. Mientras preservemos esa jerarquía, el modelo mantiene coherencia.

### Cada técnica = una perspectiva diferente

| Técnica | Perspectiva dominante |
|---------|----------------------|
| amplify_subspace | Filosófica/existencial |
| lowrank | Académica/crítica |
| normrot | Estoica/equilibrada |
| spectral | Concisa/directa |
| attention_preserving | Casi idéntica al baseline |
| gradient_aligned | Autenticidad/descubrimiento |

## 1.5 Estructura del Libro

Este libro documenta el viaje completo:

1. **Ingeniería inversa** del formato GGUF
2. **Implementación** de un motor de inferencia en C puro
3. **Desarrollo** de técnicas de perturbación de pesos
4. **Análisis geométrico** del cambio de perspectiva
5. **Resultados experimentales** con 24 modelos y 240 generaciones
6. **Aplicaciones prácticas** de conmutación de estilos

## 1.6 Datos del Proyecto

| Métrica | Valor |
|---------|-------|
| Modelos testeados | 24 |
| Generaciones totales | 240 |
| Prompts utilizados | 10 |
| Técnicas implementadas | 10 + combinaciones |
| Modelos coherentes | 18/24 (75%) |
| Tiempo total de pruebas | ~25 minutos |

## 1.7 Código y Datos

Todo el código, datos y resultados están disponibles en:

```
dreaming/
├── book/                    # Este libro
│   ├── chapters/           # Capítulos en Markdown
│   ├── code/               # Scripts ejecutables
│   ├── figures/            # Figuras generadas
│   └── tests/              # Datos experimentales
├── dmt_perturb_v10.py      # 10 técnicas de perturbación
├── dmt_perturb_v11.py      # Combinaciones y targeting
├── style_switch.py         # Conmutación de estilos
└── conclusiones.md         # Análisis completo
```

---

*Siguiente capítulo: [Ingeniería Inversa del Formato GGUF](02_reverse_engineering.md)*
