# Circuitos de Emoción en TinyLlama (prototipo)

Herramientas para **investigar representaciones internas asociadas a la expresión de
emociones** en TinyLlama-1.1B, siguiendo el método del paper
**"Do LLMs Feel? Emotion Circuits Discovery and Control"** (arXiv:2510.11328, MBZUAI),
difundido en https://x.com/thesupermannx/status/2113933324775489904

> **Estado honesto**: todavía NO se han localizado circuitos emocionales en TinyLlama.
> Esto es un prototipo probado mecánicamente para intentarlo. No afirma que el modelo
> "sienta" nada: como mucho mediría representaciones asociadas a la *expresión* de emoción.
> El 99.65% del paper es sobre Llama-3.2-3B y **no es trasladable** a TinyLlama-1.1B;
> cualquier resultado aquí hay que medirlo de cero.

## Las 6 emociones

`anger`, `sadness`, `happiness`, `fear`, `disgust`, `surprise`

## Relación con el trabajo previo del repo

La "dirección de agresividad" en embeddings (`AGGRESSION_FINAL_RESULTS.md`) no funcionó y su
propia conclusión pedía steering vectors con acceso a estados intermedios. Este módulo trabaja
en el residual stream de las 22 capas, que era ese paso pendiente.

## Qué es cada pieza (sin inflar validez)

| Pieza | Qué es | Qué NO es |
|---|---|---|
| `smoke_test.py` | Prueba de hooks y de la matemática en un modelo aleatorio con señal plantada | Evidencia sobre TinyLlama real |
| `run_extraction.py` | Genera textos elicitados (prompt-based, filtro léxico solo informativo), extrae direcciones por capa (misma matemática que la etapa 2 del paper) y **candidatos** correlacionales (ranking por contribución de cabezas/neuronas; selección por masa 90%) | Validación causal: la contribución es correlacional, y la selección por masa no mide efecto causal como hace el paper (sigma/multi-alfa) |
| `run_eval.py` | Comprobación mecánica sobre pesos reales: separación por proyección en los mismos textos de extracción + subida/bajada de proyección al inyectar/ablar | Validación conductual: la métrica proyecta sobre la misma dirección inyectada/ablada (auto-referente, circular) |
| `run_steer.py` | Demo de modulación con el mecanismo del paper (hooks MLP + o_proj) | Demostración de control emocional real |

**Evaluación conductual independiente — pendiente.** Lo que haría falta para hablar de
circuitos localizados: escenarios **held-out** (no usados en la extracción), texto generado
con y sin modulación juzgado por un evaluador independiente (humano o modelo externo), y
controles con direcciones aleatorias y de norma equiparada (matched-norm). Hasta entonces,
los "circuitos" son candidatos.

## Uso

```bash
pip install torch transformers

# 0. Prueba mecánica sin descargas (modelo aleatorio diminuto)
python smoke_test.py

# 1. Extracción en TinyLlama real (~4-6 GB RAM en CPU float32, o GPU) - PENDIENTE DE RECURSOS
python run_extraction.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --output outputs/tinyllama_1b

# 2. Comprobación mecánica sobre pesos reales (auto-referente; ver tabla)
python run_eval.py --vectors outputs/tinyllama_1b

# 3. Demo de modulación
python run_steer.py --input_text "My girlfriend forgot my birthday again." --emotion all --scale 1.0
```

Salidas: `outputs/tinyllama_1b.npz` (direcciones + diffs), `_circuits.json` (candidatos),
`_generations.json` (auditoría de elicitación), `_eval.json` (métricas auto-referentes).

## Estado

- ✅ Mecánica completa validada con `smoke_test.py` (direcciones recuperan señal plantada
  r>0.8 en modelo aleatorio; inyección y ablación alteran logits y se retiran limpias).
- ✅ Plantilla de chat verificada contra el tokenizer real de TinyLlama-1.1B-Chat.
- ⏳ Extracción en TinyLlama real: pendiente de recursos (~4-6 GB RAM CPU o GPU).
- ⏳ Evaluación conductual independiente (held-out + controles + juez externo): pendiente.

## Fuentes y licencias

- Paper: https://arxiv.org/abs/2510.11328 (Chenxi Wang et al., MBZUAI).
- Repo oficial del paper: https://github.com/Aurora-cx/EmotionCircuits-LLM — **GPL-3.0**.
- Este módulo está escrito para este repo a partir del método descrito en el paper y no
  incorpora código ni datos del repo oficial; el dataset SEV incluido es propio (18
  escenarios). En caso de duda sobre compatibilidad de licencias, revisar antes de redistribuir.
- Post origen del encargo: https://x.com/thesupermannx/status/2113933324775489904
