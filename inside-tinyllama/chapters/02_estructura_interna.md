# Capítulo 2: La Estructura Interna de TinyLlama

## Los 22 Niveles (Capas)

TinyLlama tiene 22 capas transformadoras. Cada capa es como
un nivel de procesamiento que la información debe atravesar.

```
Capa 0:  Entrada → Detección de patrones simples
Capa 1:  Sintaxis básica
Capa 2-5:  Relaciones entre palabras adyacentes
Capa 6-12: Conceptos abstractos (las "capas de ideas puras")
Capa 13-20: Integración global
Capa 21:  Salida → Generación de tokens
```

## Los 9 Planetas por Nivel (Tensores)

Cada capa tiene 9 tensores que trabajan juntos:

### Tensores de Atención (4 tensores, 29% de parámetros)
- **Query (Q)**: ¿Qué estoy buscando?
- **Key (K)**: ¿Qué tengo para ofrecer?
- **Value (V)**: ¿Qué información Transmito?
- **Output (O)**: ¿Cómo integro todo?

### Tensores FFN (3 tensores, 67% de parámetros)
- **Gate (G)**: ¿Qué información dejo pasar?
- **Up (U)**: ¿Cómo expando la información?
- **Down (D)**: ¿Cómo comprimo la información?

### Tensores de Normalización (2 tensores, 0.01% de parámetros)
- **AttnNorm**: Estabiliza la atención
- **FFNNorm**: Estabiliza la red feed-forward

## El Flujo de Información

La información fluye así:

```
Token → Embedding (1152 dimensiones)
     → Capa 1 → Capa 2 → ... → Capa 22
     → Predicción del siguiente token
```

Cada capa transforma la representación de 1152 dimensiones
en una nueva representación de 1152 dimensiones.

## Primera Mirada a los Datos

### Parámetros por componente:
- **FFN**: 67.2% (memoria/conocimiento)
- **Atención**: 32.8% (conexiones entre tokens)
- **Embedding + LM Head**: 11.6%
- **Layer Norms**: 0.01%

### Dimensión del embedding: 1152
### Número de capas: 22
### Tamaño del vocabulario: 32,000 tokens
### Cabezas de atención: 16 Q / 2 KV (GQA)
### Dimensión por cabeza: 128

## La Estructura Jerárquica

```
Embeddings (geométricas)
    ↓
Capas 0-5 (sintaxis)
    ↓
Capas 6-12 (significado puro)
    ↓
Capas 13-21 (integración)
    ↓
Salida (tokens)
```

## Conclusión

La estructura de TinyLlama es elegante y jerárquica.
Cada componente tiene un rol específico, y juntos
crean un sistema capaz de procesar y generar lenguaje.

---

*Siguiente capítulo: La Analogía Cósmica*
