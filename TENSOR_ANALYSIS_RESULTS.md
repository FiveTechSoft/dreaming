# Análisis de Tensores - Resultados

## Estructura del Modelo

| Componente | Tensores | Parámetros | Porcentaje |
|------------|----------|------------|------------|
| FFN (gate, up, down) | 66 | 761M | 69.3% |
| Atención (q, k, v, out) | 88 | 206M | 18.8% |
| Embeddings | 2 | 66M | 6.0% |
| Output | 1 | 66M | 6.0% |
| Norms | 44 | 90K | 0.01% |

## Resultados de Perturbación

### Solo Embeddings (6% de parámetros)
- Scale 0.01-1.0: Sin cambio significativo
- Scale 2.0: Modelo se degrada

### Solo FFN (69% de parámetros)
- Scale 0.1: Modelo funciona, cambio sutil en tono emocional
- Scale 0.5: Modelo se degrada

## Comparación de Outputs

### Prompt: "When I see injustice, I"

**Original:**
> "...I feel a sense of discomfort and unease... This feeling often leads to **anger and frustration**..."

**Modificado (FFN scale=0.1):**
> "...I feel a deep sense of **shame and disgust**. It's as if I'm looking at something that's not mine... a **gut-punch** that reminds me of the ways in which the world is so..."

## Observaciones

1. **El modelo modificado usa emociones más intensas**: "shame", "disgust", "gut-punch"
2. **El modelo original usa emociones más suaves**: "discomfort", "unease"
3. **Ambos modelos mencionan negatividad**, pero el modificado es más visceral

## Conclusión

**La perturbación de FFN SÍ tiene efecto**, pero:
1. El efecto es sutil (cambio de tono, no cambio drástico)
2. Necesitamos escala pequeña (0.1) para mantener coherencia
3. El modelo tiene estructura robusta que resiste perturbaciones

## Próximos Pasos

1. Probar con diferentes prompts emocionales
2. Probar modificar tanto FFN como atención juntos
3. Probar con capas específicas (solo capas 6-12)

## Archivos Creados

- `aggressive_ffn_model.gguf` - Modelo con FFN modificado
- `modify_ffn_tensors.py` - Script de modificación
- `analyze_model_structure.py` - Análisis de estructura
