# Anatomía de TinyLlama-1.1B

## Distribución de Pesos

| Componente | Cálculo | Pesos | % | Qué hace |
|------------|---------|-------|---|----------|
| **Embedding** | 32,000 × 2,048 | 65,536,000 | 5.8% | Convierte token en vector |
| **Attention Q** (×22) | 22 × 2048² | 92,274,688 | 8.2% | "¿Qué busco?" |
| **Attention K** (×22) | 22 × 2048² | 92,274,688 | 8.2% | "¿Qué ofrezco?" |
| **Attention V** (×22) | 22 × 2048² | 92,274,688 | 8.2% | "¿Qué entrego?" |
| **Output Proj** (×22) | 22 × 2048² | 92,274,688 | 8.2% | Combina cabezas |
| **FFN Gate** (×22) | 22 × 2048 × 5632 | 253,755,392 | 22.4% | "¿Qué activar?" |
| **FFN Up** (×22) | 22 × 2048 × 5632 | 253,755,392 | 22.4% | "¿Qué construir?" |
| **FFN Down** (×22) | 22 × 5632 × 2048 | 253,755,392 | 22.4% | "¿Cómo comprimir?" |
| **Layer Norms** | 44 × 2048 | 90,112 | 0.01% | Normaliza valores |
| **LM Head** | 2048 × 32,000 | 65,536,000 | 5.8% | Predice siguiente token |
| **TOTAL** | | **1,130,454,720** | **100%** | |

## Resumen por Categoría

```
FFN (Feed-Forward)     67.2%  ████████████████████████████████████
Attention              32.8%  █████████████████
Embedding + LM Head    11.6%  ██████
Layer Norms             0.01%  ·
```

## Qué hace cada componente

### Embedding (5.8%)

Convierte cada token (palabra) en un vector de 2048 números.

```
"gato" → [0.2, -0.8, 0.5, 0.1, ...] (2048 números)
```

**Analogía**: Diccionario donde cada palabra tiene un "número de identificación" largo.

---

### Attention Q, K, V (24.6%)

Tres matrices que calculan **conexiones entre tokens**:

| Matriz | Pregunta | Ejemplo |
|--------|----------|---------|
| **Q** (Query) | "¿Qué busco?" | "gato" busca: ¿quién duerme? |
| **K** (Key) | "¿Qué ofrezco?" | "duerme" ofrece: verbo de acción |
| **V** (Value) | "¿Qué entrego?" | "duerme" entrega: significado |

**Fórmula**: `Attention(Q,K,V) = softmax(Q·Kᵀ/√d) · V`

**Analogía**: Buscar en Google:
- Q = tu búsqueda
- K = títulos de resultados
- V = contenido de las páginas

---

### Output Projection (8.2%)

Combina las respuestas de **16 cabezas de atención** en un solo resultado.

```
Cabeza 1: "gato" conecta con "duerme" (0.85)
Cabeza 2: "gato" conecta con "animal" (0.72)
Cabeza 3: "gato" conecta con "peludo" (0.68)
...
↓ Combinar
Resultado final: [vector de 2048 dims]
```

---

### FFN Gate (22.4%)

Decide **qué partes del conocimiento activar**.

```
Entrada: [0.2, -0.8, 0.5, ...]
    ↓
Gate: [1, 0, 1, 0, 1, ...]  (1=activar, 0=no activar)
    ↓
Solo pasan las partes "encendidas"
```

**Analogía**: Interruptor que decide qué luces encender.

---

### FFN Up (22.4%)

**Amplía** el vector de 2048 a 5632 dimensiones para procesar.

```
2048 dims → 5632 dims
```

**Por qué**: En más dimensiones, el modelo puede "pensar" mejor.

---

### FFN Down (22.4%)

**Comprime** de vuelta a 2048 dimensiones.

```
5632 dims → 2048 dims
```

**Por qué**: Necesitamos volver al tamaño original para la siguiente capa.

---

### Layer Norms (0.01%)

**Normaliza** los valores para que no sean ni muy grandes ni muy pequeños.

```
[1000, -500, 2000] → [0.5, -0.25, 1.0]
```

**Por qué**: Sin esto, los números explotarían durante el cálculo.

---

### LM Head (5.8%)

Convierte el vector final en **probabilidades** de cada token.

```
[vector de 2048] → [probabilidad de cada uno de 32,000 tokens]

"duerme" = 85%
"come" = 10%
"corre" = 5%
```

**Analogía**: El "votante final" que decide qué palabra viene.

---

## Flujo Completo

```
"El gato"
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  EMBEDDING: "gato" → [0.2, -0.8, ...]                  │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  ATTENTION: ¿Qué conecta con qué?                      │
│  Q: "gato" busca → K: "duerme" responde → V: entrega   │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  FFN: ¿Qué conocimiento activar?                       │
│  Gate: "activa" → Up: "expande" → Down: "comprime"     │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  LM HEAD: [vector] → probabilidades                     │
│  "duerme" = 85%, "come" = 10%, "corre" = 5%            │
└─────────────────────────────────────────────────────────┘
    │
    ▼
"duerme"
```

---

## Dato Clave

La **FFN almacena conocimiento** (67% del modelo).
La **Attention conecta tokens** (33% del modelo).

Cuando perturbamos:
- **FFN** → Cambiamos **qué sabe** el modelo
- **Attention** → Cambiamos **cómo conecta** la información
