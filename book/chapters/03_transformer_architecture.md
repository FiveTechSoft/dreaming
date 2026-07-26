# Capítulo 3: Arquitectura Transformer — TinyLlama

## 3.1 ¿Por qué TinyLlama?

TinyLlama-1.1B es un modelo pequeño (500MB en Q4_0) con arquitectura Transformer estándar. Ideal para nuestro proyecto porque:

1. **Tamaño manejable** — Cabe en memoria, se puede manipular en Python
2. **Arquitectura estándar** — Igual que GPT, LLaMA, Mistral
3. **Código abierto** — Pesos disponibles bajo licencia Apache
4. **Lo suficientemente bueno** — Genera texto coherente, no un juguete

## 3.2 Estructura del Transformer

```
Input Tokens
    │
    ▼
┌─────────────────┐
│   Embedding     │  vocab_size × hidden_size (32000 × 2048)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Positional     │  RoPE (Rotary Position Embeddings)
│  Encoding       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Transformer    │  22 capas
│  Blocks         │  Cada una: Attn + FFN + Norm
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Final Norm     │  RMSNorm
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  LM Head        │  hidden_size × vocab_size (2048 × 32000)
└────────┬────────┘
         │
         ▼
    Output Logits
```

## 3.3 Parámetros de TinyLlama-1.1B

| Parámetro | Valor |
|-----------|-------|
| vocab_size | 32,000 |
| hidden_size | 2,048 |
| intermediate_size | 5,632 |
| num_attention_heads | 16 |
| num_key_value_heads | 16 |
| num_hidden_layers | 22 |
| rms_norm_eps | 1e-5 |
| rope_theta | 10,000 |
| max_position_embeddings | 2,048 |

### Cálculo de parámetros totales

```
Embedding:           32,000 × 2,048  =  65,536,000
Per Layer:
  Attention QKV:     3 × 2,048²       = 12,582,912
  Attention Out:     2,048²            =  4,194,304
  FFN Up:            2,048 × 5,632     = 11,534,336
  FFN Down:          5,632 × 2,048     = 11,534,336
  FFN Gate:          2,048 × 5,632     = 11,534,336
  Norms:             2 × 2,048         =      4,096
Total Per Layer:                         51,384,320
22 Layers:                               1,130,454,720
Final Norm:                                   2,048
LM Head:                              65,536,000
────────────────────────────────────────────────────
TOTAL:                                1,261,528,768 ~1.1B
```

## 3.4 Tensores en el Modelo

Cada capa del Transformer tiene estos tensores:

| Tensor | Forma | Descripción |
|--------|-------|-------------|
| `token_embd.weight` | [32000, 2048] | Embedding de tokens |
| `blk.{i}.attn_norm.weight` | [2048] | Normalización pre-atención |
| `blk.{i}.attn_q.weight` | [2048, 2048] | Proyección Query |
| `blk.{i}.attn_k.weight` | [2048, 2048] | Proyección Key |
| `blk.{i}.attn_v.weight` | [2048, 2048] | Proyección Value |
| `blk.{i}.attn_output.weight` | [2048, 2048] | Proyección salida atención |
| `blk.{i}.ffn_norm.weight` | [2048] | Normalización pre-FFN |
| `blk.{i}.ffn_gate.weight` | [5632, 2048] | Gate FFN (SwiGLU) |
| `blk.{i}.ffn_up.weight` | [5632, 2048] | Proyección FFN up |
| `blk.{i}.ffn_down.weight` | [2048, 5632] | Proyección FFN down |
| `output_norm.weight` | [2048] | Normalización final |

## 3.5 Mecanismos Clave

### Attention (Mecanismo de Atención)

```python
def attention(Q, K, V, mask):
    # Q, K, V: [batch, seq, hidden]
    scores = torch.matmul(Q, K.transpose(-2, -1)) / sqrt(d_k)
    scores = scores + mask  # Causal mask
    weights = torch.softmax(scores, dim=-1)
    return torch.matmul(weights, V)
```

### RoPE (Rotary Position Embeddings)

RoPE codifica posiciones rotando los vectores Q y K:

```python
def apply_rope(x, freqs):
    # x: [batch, seq, n_heads, d_head]
    # freqs: [seq, d_head/2]
    cos = torch.cos(freqs)
    sin = torch.sin(freqs)
    
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    
    return torch.stack([
        x1 * cos - x2 * sin,
        x1 * sin + x2 * cos
    ], dim=-1).flatten(-2)
```

### SwiGLU (Feed-Forward)

TinyLlama usa SwiGLU en lugar de ReLU estándar:

```python
def swiglu(x, gate, up, down):
    # gate: [batch, seq, intermediate]
    # up: [batch, seq, intermediate]
    swish = x * torch.sigmoid(gate)  # Swish activation
    return down(swish * up)           # Element-wise multiply + project
```

### RMSNorm

```python
def rmsnorm(x, weight, eps=1e-5):
    rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True))
    return weight * (x / (rms + eps))
```

## 3.6 ¿Qué contiene cada tensor?

### Atención (Q, K, V)

Cada cabeza de atención aprende **patrones de correlación**:
- Q: "¿Qué estoy buscando?"
- K: "¿Qué tengo para ofrecer?"
- V: "¿Qué información entrego?"

Los pesos de atención contienen **asociaciones semánticas** entre tokens.

### Feed-Forward Network (FFN)

La FFN aprende **transformaciones no lineales**:
- Gate: "¿Qué información activar?"
- Up: "¿Qué representación construir?"
- Down: "¿Cómo proyectar de vuelta al espacio?"

Los pesos de FFN contienen **conocimiento factual y procedimental**.

### Embedding

El embedding es un **mapa semántico**:
- Tokens similares están cerca en el espacio de embeddings
- Las relaciones se preservan geométricamente

```
king - man + woman ≈ queen
```

## 3.7 Formato de Memoria de TinyLlama

En memoria (Q4_0), cada tensor se almacena así:

```
┌─────────────────────────────────────────┐
│  Header del tensor                      │
│  - Nombre (string)                      │
│  - Dimensiones (uint32[])               │
│  - Tipo (uint32 = 2 para Q4_0)          │
│  - Offset (uint64)                      │
└─────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│  Datos cuantizados (Q4_0)              │
│  - Escalas (float16 por bloque)        │
│  - Nibbles empaquetados (4 bits)       │
└─────────────────────────────────────────┘
```

## 3.8 Implicaciones para Perturbación

La estructura del Transformer tiene implicaciones directas para la perturbación:

### 1. Las capas son jerárquicas

Las primeras capas procesan características superficiales (sintaxis). Las últimas procesan semántica profunda.

### 2. La atención es el mecanismo de "foco"

Perturbar la atención cambia **qué** mira el modelo.
Perturbar la FFN cambia **cómo** procesa lo que mira.

### 3. El embedding define el espacio

Perturbar el embedding mueve **todos** los vectores de tokens.
Esto tiene el mayor impacto en la perspectiva.

### 4. Los pesos están cuantizados

La cuantización Q4_0 ya introdujo un error de ~0.1%. Perturbaciones del 0.01-0.1% son **menores que la cuantización original**.

---

*Siguiente capítulo: [Inferencia en C Puro](04_c_inference.md)*
