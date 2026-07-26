# Capítulo 2: Ingeniería Inversa del Formato GGUF

## 2.1 ¿Qué es GGUF?

GGUF (GPT-Generated Unified Format) es el formato binario estándar para modelos LLM cuantizados. Es el formato que usa llama.cpp y la mayoría de frameworks de inferencia local.

Pero nadie nos explicó cómo funciona por dentro. Tuvimos que descubrirlo nosotros mismos.

## 2.2 Estructura del Archivo GGUF

Un archivo GGUF tiene esta estructura:

```
┌─────────────────────────────────────┐
│  Magic Bytes: "GGUF" (4 bytes)      │
├─────────────────────────────────────┤
│  Version: uint32 (4 bytes)          │
├─────────────────────────────────────┤
│  Number of Tensors: uint64 (8 bytes)│
├─────────────────────────────────────┤
│  Number of KV Pairs: uint64 (8 bytes│
├─────────────────────────────────────┤
│  Key-Value Metadata (variable)      │
│  - architecture: string             │
│  - context_length: uint32           │
│  - embedding_length: uint32         │
│  - ...                              │
├─────────────────────────────────────┤
│  Tensor Index (variable)            │
│  - For each tensor:                 │
│    - name: string                   │
│    - dimensions: uint64[]           │
│    - type: uint32 (F16, Q4_0, etc)  │
│    - offset: uint64                 │
├─────────────────────────────────────┤
│  Alignment Padding (to 32 bytes)    │
├─────────────────────────────────────┤
│  Tensor Data (binary)               │
│  - Packed quantized weights         │
└─────────────────────────────────────┘
```

## 2.3 Decodificación de Valores GGUF

Los valores en el header GGUF están codificados con tipos específicos:

```python
GGUF_TYPES = {
    0: 'UINT8',     # 1 byte
    1: 'INT8',      # 1 byte
    2: 'UINT16',    # 2 bytes
    3: 'INT16',     # 2 bytes
    4: 'UINT32',    # 4 bytes
    5: 'INT32',     # 4 bytes
    6: 'FLOAT32',   # 4 bytes
    7: 'BOOL',      # 1 byte
    8: 'STRING',    # length-prefixed string
    9: 'ARRAY',     # typed array
    10: 'UINT64',   # 8 bytes
    11: 'INT64',    # 8 bytes
    12: 'FLOAT64',  # 8 bytes
}
```

### Lectura de strings

```python
def read_string(f):
    length = struct.unpack('<Q', f.read(8))[0]  # uint64
    return f.read(length).decode('utf-8')
```

### Lectura de arrays

```python
def read_array(f):
    etype = struct.unpack('<I', f.read(4))[0]  # Element type
    length = struct.unpack('<Q', f.read(8))[0]  # uint64
    return [read_value(f, etype) for _ in range(length)]
```

## 2.4 Formato Q4_0 (4-bit Quantization)

TinyLlama usa cuantización Q4_0, que comprime cada 32 pesos en 18 bytes:

```
Por cada bloque de 32 pesos:
┌─────────────────────────────────────┐
│  Scale: float16 (2 bytes)           │
├─────────────────────────────────────┤
│  Packed nibbles: 16 bytes           │
│  (32 pesos empaquetados en 4 bits)  │
└─────────────────────────────────────┘

Total: 18 bytes por 32 pesos = 4.5 bits/peso
```

### Fórmula de cuantización

```python
# Escala
scale = max(|values|) / 8.0

# Cuantización
quantized = clip(round(values / scale) + 8, 0, 15)

# Empaquetado (4 bits por peso)
lo = quantized[:16]      # Bits bajos
hi = quantized[16:]      # Bits altos
packed = lo | (hi << 4)  # Combinar en bytes
```

### Decodificación Q4_0

```python
def dequantize_q4_0(raw_bytes):
    raw = np.frombuffer(raw_bytes, dtype=np.uint8)
    n_blocks = len(raw) // 18
    
    data = raw[:n_blocks * 18].reshape(n_blocks, 18)
    
    # Escala (2 bytes → float16 → float32)
    scales = np.frombuffer(data[:, :2].tobytes(), 
                           dtype=np.float16).astype(np.float32)
    
    # Nibbles empaquetados
    nibbles = data[:, 2:18]
    lo = (nibbles & 0x0F).astype(np.float32) - 8.0
    hi = ((nibbles >> 4) & 0x0F).astype(np.float32) - 8.0
    
    # Decodificar
    return (np.concatenate([lo, hi], axis=1) * scales[:, np.newaxis]).flatten()
```

## 2.5 Código Completo de Parser GGUF

```python
import struct
import numpy as np

GGUF_MAGIC = b'GGUF'

def skip_value(f, vtype):
    """Saltar un valor del header GGUF."""
    if vtype == 8:  # STRING
        slen = struct.unpack('<Q', f.read(8))[0]
        f.read(slen)
    elif vtype == 9:  # ARRAY
        etype = struct.unpack('<I', f.read(4))[0]
        alen = struct.unpack('<Q', f.read(8))[0]
        for _ in range(alen):
            skip_value(f, etype)
    else:
        sizes = {0:1, 1:1, 2:2, 3:2, 4:4, 5:4, 6:4, 7:1, 10:8, 11:8, 12:8}
        f.read(sizes.get(vtype, 0))

def parse_gguf(path):
    """Parse completo de un archivo GGUF."""
    with open(path, 'rb') as f:
        # Header
        magic = f.read(4)
        assert magic == GGUF_MAGIC, f"Invalid magic: {magic}"
        
        version = struct.unpack('<I', f.read(4))[0]
        n_tensors = struct.unpack('<Q', f.read(8))[0]
        n_kv = struct.unpack('<Q', f.read(8))[0]
        
        print(f"Version: {version}")
        print(f"Tensors: {n_tensors}")
        print(f"KV Pairs: {n_kv}")
        
        # Key-Value metadata
        kv_metadata = {}
        for _ in range(n_kv):
            key = read_string(f)
            vtype = struct.unpack('<I', f.read(4))[0]
            value = read_value(f, vtype)
            kv_metadata[key] = value
        
        # Tensor index
        tensor_infos = []
        for _ in range(n_tensors):
            name = read_string(f)
            n_dims = struct.unpack('<I', f.read(4))[0]
            dims = [struct.unpack('<Q', f.read(8))[0] for _ in range(n_dims)]
            ttype = struct.unpack('<I', f.read(4))[0]
            offset = struct.unpack('<Q', f.read(8))[0]
            tensor_infos.append({
                'name': name,
                'dims': dims,
                'type': ttype,
                'offset': offset,
            })
        
        return kv_metadata, tensor_infos
```

## 2.6 Descubrimientos Clave

### Los offsets son relativos al inicio de datos, no del archivo

Este error causó semanas de frustración. Los offsets de tensors en GGUF son **relativos al inicio de la sección de datos**, no al inicio del archivo.

```python
# INCORRECTO
raw_start = tensor_info['offset']

# CORRECTO
header_end = ((header_end + 31) // 32) * 32  # Alinear a 32 bytes
f.seek(header_end)
raw_start = tensor_info['offset']  # Relativo al inicio de datos
```

### La escala Q4_0 usa divisor 8.0, no 7.5

```python
# INCORRECTO
scale = max_abs / 7.5

# CORRECTO
scale = max_abs / 8.0
```

### El alineamiento es a 32 bytes

Los datos de tensors empiezan en la siguiente dirección de 32 bytes después del header.

## 2.7 Herramientas Derivadas

De la ingeniería inversa surgen estas herramientas:

| Archivo | Función |
|---------|---------|
| `create_gguf.py` | Construir archivos GGUF desde cero |
| `fill_weights.py` | Llenar pesos con patrones sintéticos |
| `dmt_perturb_v10.py` | Perturbar pesos manteniendo GGUF válido |

---

*Siguiente capítulo: [Arquitectura Transformer: TinyLlama](03_transformer_architecture.md)*
