# Dirección de Agresividad - Resumen

## Lo que hicimos

1. **Analizamos el vocabulario de TinyLlama**
   - 32,000 tokens (usando BPE)
   - Dimensión de embedding: 2048 (no 1152 como pensábamos)

2. **Encontramos tokens semánticos**
   - Tokens agresivos: "attack", "fight", "kill", "destroy", "angry", "violent"
   - Tokens pacíficos: "peace", "calm", "gentle", "kind", "soft", "quiet"

3. **Calculamos la dirección de agresividad**
   - Dirección = Promedio(agg_embs) - Promedio(pac_embs)
   - Normalizada a norma 1

4. **Aplicamos al modelo**
   - Modificamos el tensor `token_embd.weight`
   - Probamos con diferentes escalas: 0.01, 0.1, 0.5, 1.0

## Resultados

| Escala | Resultado |
|--------|-----------|
| 0.01 | Sin cambio observable |
| 0.1 | Sin cambio observable |
| 0.5 | Modelo funciona, pero no es agresivo |
| 1.0 | Modelo se degrada (tokens raros) |

## Conclusión

**Modificar solo los embeddings NO es suficiente** para cambiar el comportamiento del modelo de manera significativa.

### ¿Por qué?

1. **El tensor de embeddings es solo 1 de 201 tensores**
   - Los otros 200 tensores (atención, FFN) siguen con los pesos originales
   - El comportamiento del modelo depende de la interacción de TODOS los tensores

2. **La información fluye a través de 22 capas**
   - Cada capa transforma la representación
   - Modificar solo la entrada (embeddings) tiene un efecto limitado

3. **Los pesos de atención y FFN determinan el "cómo procesa"**
   - Atención: Cómo conecta tokens entre sí
   - FFN: Qué transformaciones aplica
   - Estos son los que realmente determinan el comportamiento

## Próximos pasos

Para crear un modelo verdaderamente "agresivo", necesitamos:

1. **Modificar múltiples tensores** (embeddings + atención + FFN)
2. **Usar una dirección coherente** en todo el modelo
3. **O usar fine-tuning** con texto agresivo

## Archivos creados

- `aggression_direction_model.npy` - Dirección de agresividad (2048 dims)
- `apply_aggression_to_model.py` - Script para aplicar la dirección
- `extract_model_embeddings.py` - Extrae embeddings del modelo
- `create_direct_direction.py` - Crea dirección directa
- `find_semantic_tokens.py` - Analiza tokens semánticos
- `semantic_directions.npy` - Centroides por categoría

## Modelo creado

- `perturbed_models/aggressive_model.gguf` - Modelo con perturbación
  - Scale 0.5: Funciona pero no es agresivo
  - Scale 1.0: Se degrada

## URL en GitHub

https://github.com/FiveTechSoft/dreaming/tree/main/inside-tinyllama
