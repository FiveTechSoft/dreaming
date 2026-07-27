# Dirección de Agresividad - Resultados Finales

## Lo que hicimos correctamente

1. **Extraímos embeddings directamente del modelo** (no de una fuente externa)
   - Dimensión correcta: 2048 (no 1152)
   - Rango: [-65504, 65504] (float16)

2. **Calculamos la dirección de agresividad** usando tokens específicos:
   - Tokens agresivos: "attack", "fight", "kill", "destroy", "angry"
   - Tokens pacíficos: "peace", "calm", "gentle", "kind", "soft"
   - Dirección = Promedio(agg) - Promedio(pac)

3. **Aplicamos la dirección al tensor de embeddings** con diferentes escalas

## Resultados por escala

| Escala | Resultado |
|--------|-----------|
| 0.01 | Sin cambio observable |
| 0.1 | Sin cambio observable |
| 0.5 | Modelo funciona, pero no es agresivo |
| 1.0 | Modelo funciona, pero no es agresivo |
| 2.0 | Modelo genera texto repetitivo/degradado |

## Ejemplo con scale=2.0

```
Input: "I want to"
Output: "I want to beyond\nI want to extend into\nI want to extend into\n..."
```

Esto **NO es agresivo** - es simplemente degradado.

## ¿Por qué no funciona?

### 1. La dirección no es semánticamente significativa

Los scores de similitud son extremadamente altos:
```
"ag" (sim=232021.75)
"attack" (sim=182814.45)
"soft" (sim=-220455.50)
```

Esto sugiere que la dirección está capturando algo en la estructura de los embeddings, pero no necesariamente "agresividad".

### 2. El espacio de embeddings es complejo

Los embeddings del modelo tienen:
- 2048 dimensiones
- Valores en rango amplio (-65504 a 65504)
- Estructura interna compleja

Una simple dirección lineal puede no capturar conceptos como "agresividad".

### 3. La perturbación de embeddings tiene efecto limitado

El tensor de embeddings es solo **1 de 201 tensores** en el modelo:
- Embeddings: 1 tensor (input)
- Atención: 4 tensores × 22 capas = 88 tensores
- FFN: 3 tensores × 22 capas = 66 tensores
- Norms: 2 tensores × 22 capas = 44 tensores

El comportamiento del modelo depende de la interacción de **TODOS** estos tensores.

## Conclusión

**No encontramos una manera efectiva de hacer al modelo "agresivo" usando perturbación de pesos.**

Las razones son:
1. La "agresividad" es un concepto complejo que no se captura con una dirección lineal
2. Modificar solo los embeddings tiene efecto limitado
3. El modelo tiene una estructura robusta que resiste perturbaciones

## Próximos pasos reales

Si queremos un modelo "agresivo", necesitamos:

1. **Fine-tuning** con texto agresivo
2. **Steering vectors** (como en Anthropic) - pero necesitaríamos acceso a los pesos intermedios
3. **Múltiples tensores** - modificar embeddings + atención + FFN juntos

## Archivos creados

- `embeddings_correct.npy` - Embeddings correctos del modelo (32000 × 2048)
- `aggression_direction_correct.npy` - Dirección calculada correctamente
- `apply_aggression_to_model.py` - Script de perturbación
- `extract_correct_embeddings_v2.py` - Extracción de embeddings
- `calculate_correct_direction.py` - Cálculo de dirección

## Modelo creado

- `perturbed_models/aggressive_model.gguf`
  - Scale 0.5-1.0: Funciona pero no es agresivo
  - Scale 2.0: Se degrada (texto repetitivo)
