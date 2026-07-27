# Capítulo 2: La Estructura Interna de TinyLlama

## Los 22 Niveles (Capas)

TinyLlama tiene 22 capas transformadoras. Cada capa es como
un nivel de procesamiento que la información debe atravesar.

```
Capa 0:     Entrada → Detección de patrones simples
Capa 1:     Sintaxis básica
Capas 2-5:  Relaciones entre palabras adyacentes
Capas 6-12: Conceptos abstractos (las "capas de ideas puras")
Capas 13-20: Integración global
Capa 21:    Salida → Generación de tokens
```

## Los 9 Planetas por Nivel (Tensores)

Cada capa tiene 9 tensores que trabajan juntos:

### Tensores de Atención (4 tensores, ~19% de parámetros)
- **Query (Q)**: ¿Qué estoy buscando?
- **Key (K)**: ¿Qué tengo para ofrecer?
- **Value (V)**: ¿Qué información transmito?
- **Output (O)**: ¿Cómo integro todo?

### Tensores FFN (3 tensores, ~69% de parámetros)
- **Gate (G)**: ¿Qué información dejo pasar?
- **Up (U)**: ¿Cómo expando la información?
- **Down (D)**: ¿Cómo comprimo la información?

### Tensores de Normalización (2 tensores, ~0.01% de parámetros)
- **AttnNorm**: Estabiliza la atención
- **FFNNorm**: Estabiliza la red feed-forward

## El Flujo de Información

La información fluye así:

```
Token → Embedding (2048 dimensiones)
     → Capa 0 → Capa 1 → ... → Capa 21
     → Predicción del siguiente token
```

Cada capa transforma la representación de 2048 dimensiones
en una nueva representación de 2048 dimensiones.
La forma se conserva; el *contenido semántico* evoluciona.

## Primera Mirada a los Datos

Valores leídos del GGUF de TinyLlama-1.1B
(`llama.*` en el header del modelo):

### Parámetros por componente (aprox.)
- **FFN**: ~69% (memoria / conocimiento práctico)
- **Atención**: ~19% (conexiones entre tokens)
- **Embedding + LM Head**: ~12%
- **Layer Norms**: ~0.01%

### Dimensión oculta (`embedding_length`): 2048
### Número de capas (`block_count`): 22
### Tamaño del vocabulario: 32.000 tokens
### Contexto máximo: 2048 tokens
### Cabezas de atención: 32 Q / 4 KV (GQA)
### Dimensión por cabeza: 64
### FFN intermedio (`feed_forward_length`): 5632
### RoPE `freq_base`: 10.000

### Formas lógicas de los tensores (por capa)

```
attn_norm     [2048]
attn_q        [2048, 2048]     # 32 cabezas × 64
attn_k        [256,  2048]     #  4 cabezas × 64  (GQA)
attn_v        [256,  2048]
attn_output   [2048, 2048]
ffn_norm      [2048]
ffn_gate      [5632, 2048]
ffn_up        [5632, 2048]
ffn_down      [2048, 5632]
```

Más los globales:

```
token_embd.weight   [32000, 2048]
output_norm.weight  [2048]
output.weight       [32000, 2048]
```

> **Nota sobre Q4_0:** en disco, un GGUF cuantizado
> muestra formas “empaquetadas” (por ejemplo
> `token_embd` como `[32000, 1152]`). Eso es el layout
> de bloques de 4 bits, no la geometría del modelo.
> La dimensión real del vector residual sigue siendo 2048.

## La Estructura Jerárquica

```
Embeddings (geometría del vocabulario)
    ↓
Capas 0-5 (sintaxis y vecinos locales)
    ↓
Capas 6-12 (significado más abstracto)
    ↓
Capas 13-21 (integración y decisión)
    ↓
Salida (logits → siguiente token)
```

## Conclusión

La estructura de TinyLlama es elegante y jerárquica.
Cada componente tiene un rol específico, y juntos
crean un sistema capaz de procesar y generar lenguaje.

Con 22 capas, 9 tensores por capa y un residual de
2048 dimensiones, el modelo es lo bastante pequeño
para abrirlo por completo — y lo bastante rico para
sorprender.

---

*Siguiente capítulo: Nuestro Motor de Inferencia en C para TinyLlama*
