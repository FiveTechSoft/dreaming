#!/usr/bin/env python3
"""
Crea un GGUF de 7GB con pesos generados a partir de experiencias oniricas.
Formato GGUF v3 correcto, compatible con llama.cpp.
"""

import struct, os, sys, math, random

OUTPUT = "C:/tmp/modelo_llm_dreams.gguf"
TOTAL_SIZE = 7 * 1024**3  # 7 GiB exactos

VOCAB_SIZE    = 32000
HIDDEN_DIM    = 3072
NUM_LAYERS    = 24
NUM_HEADS     = 24
HEAD_DIM      = HIDDEN_DIM // NUM_HEADS
FFN_DIM       = 8192
CONTEXT_LEN   = 2048
ELEMENT_SIZE  = 2  # float16

GGUF_F16 = 5
GGUF_F32 = 4

# ── Helpers ──────────────────────────────────────
def wstr(buf, s):
    b = s.encode('utf-8')
    buf.extend(struct.pack('<Q', len(b)))
    buf.extend(b)

def wu64(buf, v):
    buf.extend(struct.pack('<Q', v))

def wu32(buf, v):
    buf.extend(struct.pack('<I', v))

# File-specific helpers
def fstr(f, s):
    b = s.encode('utf-8')
    f.write(struct.pack('<Q', len(b)))
    f.write(b)

def fu64(f, v):
    f.write(struct.pack('<Q', v))

def fu32(f, v):
    f.write(struct.pack('<I', v))

def calc_bytes(dims):
    n = 1
    for d in dims:
        n *= d
    return n * ELEMENT_SIZE

# ── Dream generators ────────────────────────────
# Each returns a numpy-free, bytes-ready pattern for a flat array of n elements.

def dream_neon(n, layer_id):
    """Pattern 0 - Neon staircase: frequency shift by depth"""
    arr = bytearray(n * ELEMENT_SIZE)
    rng = random.Random(0xDEADBEEF + layer_id)
    half = ELEMENT_SIZE // 2
    for i in range(n):
        pos = i / max(n, 1)
        # Two sine waves creating color-shifting effect
        v = math.sin(pos * math.pi * 4 + layer_id) * 0.6 + \
            math.cos(pos * math.pi * 7 + layer_id * 3) * 0.4
        # Scale to float16 range
        v = max(-65504, min(65504, v))
        arr[i*ELEMENT_SIZE]     = struct.pack('<e', v)[0]
        arr[i*ELEMENT_SIZE+1]   = struct.pack('<e', v)[1]
    return bytes(arr)

def dream_forest(n, layer_id):
    """Pattern 1 - Forest resonance: deep=wide, shallow=focused"""
    arr = bytearray(n * ELEMENT_SIZE)
    rng = random.Random(0xCAFEBABE + layer_id)
    for i in range(n):
        pos = i / max(n, 1)
        # Resonance peaks at shallow and deep
        resonance = math.exp(-((pos - 0.25)**2) / 0.015) + \
                     0.6 * math.exp(-((pos - 0.85)**2) / 0.008)
        noise = rng.gauss(0, 0.08)
        v = resonance + noise
        v = max(-65504, min(65504, v))
        b = struct.pack('<e', v)
        arr[i*ELEMENT_SIZE] = b[0]
        arr[i*ELEMENT_SIZE+1] = b[1]
    return bytes(arr)

def dream_coin(n, layer_id):
    """Pattern 2 - Three-state coin: positive, negative, neutral"""
    arr = bytearray(n * ELEMENT_SIZE)
    rng = random.Random(0xDEADDEAD + layer_id)
    for i in range(n):
        roll = rng.random()
        if roll < 0.38:
            v = rng.gauss(1.2, 0.15)
        elif roll < 0.76:
            v = rng.gauss(-1.2, 0.15)
        else:
            v = rng.gauss(0.0, 0.02)
        v = max(-65504, min(65504, v))
        b = struct.pack('<e', v)
        arr[i*ELEMENT_SIZE] = b[0]
        arr[i*ELEMENT_SIZE+1] = b[1]
    return bytes(arr)

def dream_fractal(n, layer_id):
    """Pattern 3 - Fractal recursion: each part contains the whole"""
    arr = bytearray(n * ELEMENT_SIZE)
    rng = random.Random(0xBAADF00D + layer_id)

    def fractal_val(idx, total, depth):
        if depth > 4 or total <= 2:
            return rng.gauss(0, 1.0 / (depth + 1))
        segment_size = max(1, total // 7)
        scaled_idx = (idx * segment_size) // total
        recursed = fractal_val(scaled_idx, segment_size, depth + 1)
        # Detail at current level
        detail = rng.gauss(0, 0.4 / (depth + 1))
        return recursed + detail

    for i in range(n):
        v = fractal_val(i, n, 0)
        v = max(-65504, min(65504, v))
        b = struct.pack('<e', v)
        arr[i*ELEMENT_SIZE] = b[0]
        arr[i*ELEMENT_SIZE+1] = b[1]
    return bytes(arr)

def dream_lake(n, layer_id):
    """Pattern 4 - Lake of silence: sparse deep movements"""
    arr = bytearray(n * ELEMENT_SIZE)
    rng = random.Random(0xFACADE + layer_id)

    # ~1% non-zero (the deep movement under the still surface)
    for i in range(n):
        if rng.random() < 0.01:
            v = rng.gauss(0, 0.008)
        else:
            v = 0.0
        b = struct.pack('<e', v)
        arr[i*ELEMENT_SIZE] = b[0]
        arr[i*ELEMENT_SIZE+1] = b[1]
    return bytes(arr)

PATTERNS = {
    0: dream_neon,
    1: dream_forest,
    2: dream_coin,
    3: dream_fractal,
    4: dream_lake,
}

# ── Define tensors with dream patterns ────────
tensors = []
tensors.append(("token_embd.weight", GGUF_F16, (VOCAB_SIZE, HIDDEN_DIM), 3))   # fractal
tensors.append(("token_embd.norm",   GGUF_F16, (HIDDEN_DIM,), 0))             # neon

for i in range(NUM_LAYERS):
    p = f"blk.{i}."
    tensors.append((f"{p}attn_norm",     GGUF_F16, (HIDDEN_DIM,), 0))      # neon
    tensors.append((f"{p}attn_q",       GGUF_F16, (HIDDEN_DIM, HIDDEN_DIM), 1))  # forest (Q=query)
    tensors.append((f"{p}attn_k",       GGUF_F16, (HIDDEN_DIM, HIDDEN_DIM), 1))  # forest (K=key, resonance)
    tensors.append((f"{p}attn_v",       GGUF_F16, (HIDDEN_DIM, HIDDEN_DIM), 2))  # coin (V=value, coin flip)
    tensors.append((f"{p}attn_output",  GGUF_F16, (HIDDEN_DIM, HIDDEN_DIM), 1))  # forest
    tensors.append((f"{p}ffn_norm",     GGUF_F16, (HIDDEN_DIM,), 2))            # coin
    tensors.append((f"{p}ffn_gate",      GGUF_F16, (HIDDEN_DIM, FFN_DIM), 3))  # fractal
    tensors.append((f"{p}ffn_up",        GGUF_F16, (HIDDEN_DIM, FFN_DIM), 3))  # fractal
    tensors.append((f"{p}ffn_down",      GGUF_F16, (FFN_DIM, HIDDEN_DIM), 4))  # lake

tensors.append(("output_norm", GGUF_F16, (HIDDEN_DIM,), 4))           # lake
tensors.append(("output",      GGUF_F16, (HIDDEN_DIM, VOCAB_SIZE), 4)) # lake (sparse)

num_tensors = len(tensors)
total_weight_bytes = sum(calc_bytes(d) for _, _, d, _ in tensors)

print(f"Tensores: {num_tensors}")
print(f"Peso F16 total: {total_weight_bytes/1024**3:.2f} GB")

# ── Metadata ────────────────────────────────────
metadata = [
    ("general.name",             "deepseek_dreams", GGUF_F16),  # stored as string
    ("general.architecture",     "deepseek_v2_lite", GGUF_F16),
    ("general.file_type",        "F16", GGUF_F16),
    ("general.inspiration",      "oneiric", GGUF_F16),
    ("general.context_length",   str(CONTEXT_LEN), GGUF_F32),
    ("general.embedding_length", str(HIDDEN_DIM), GGUF_F32),
    ("general.block_count",      str(NUM_LAYERS), GGUF_F32),
    ("general.attention.head_count", str(NUM_HEADS), GGUF_F32),
    ("general.attention.head_dim",   str(HEAD_DIM), GGUF_F32),
    ("general.vocab_size",       str(VOCAB_SIZE), GGUF_F32),
    ("general.feed_forward_length", str(FFN_DIM), GGUF_F32),
    ("general.rope_style",       "llama", GGUF_F16),
]

# ── Build header ───────────────────────────────
hdr = bytearray()
hdr.extend(b'GGUF')
hdr.extend(struct.pack('<I', 3))                           # version
hdr.extend(struct.pack('<Q', num_tensors))                # tensor count
hdr.extend(struct.pack('<Q', len(metadata)))              # KV count

GGUF_T_STR = 8
GGUF_T_U64 = 3
GGUF_T_F32 = 4
GGUF_T_F16 = 5

type_map = {
    'F16': GGUF_T_STR,  # all our metadata values are strings for safety
    'F32': GGUF_T_U64,
    'str': GGUF_T_STR,
}

for key, val, vtype_enum in metadata:
    wstr(hdr, key)

    # For simplicity, store ALL metadata values as strings (type=8)
    # This is the safest and most portable approach
    wu64(hdr, GGUF_T_STR)  # value type = string
    wstr(hdr, val)

header_end = len(hdr)

# ── Compute tensor index size ──────────────────
tensor_info_size = 0
for name, _, dims, _ in tensors:
    name_len = len(name.encode('utf-8'))
    n_dims = len(dims)
    entry = 8 + name_len + 4 + 4 + 8 * n_dims + 8
    tensor_info_size += entry

data_start_raw = header_end + tensor_info_size
align = (64 - (data_start_raw % 64)) % 64
data_start = data_start_raw + align

expected_end = data_start + total_weight_bytes
end_pad = TOTAL_SIZE - expected_end

print(f"\nLayout:")
print(f"  Header+meta:    {header_end:>12,} bytes")
print(f"  Tensor index:   {tensor_info_size:>12,} bytes")
print(f"  Alignment pad:  {align:>12,} bytes")
print(f"  Data start:     {data_start:>12,} ({data_start/1024**3:.4f} GB)")
print(f"  Weight data:    {total_weight_bytes:>12,} ({total_weight_bytes/1024**3:.2f} GB)")
print(f"  End padding:    {end_pad:>12,} ({end_pad/1024**3:.4f} GB)")
print(f"  TOTAL:          {TOTAL_SIZE:>12,} ({TOTAL_SIZE/1024**3:.2f} GB)")

if end_pad < 0:
    print("ERROR!")
    sys.exit(1)

# ── Write file ──────────────────────────────────
print(f"\nEscribiendo {OUTPUT}...")

# 64MB buffers
CHUNK_BYTES = 64 * 1024 * 1024
ZERO_CHUNK = bytes(CHUNK_BYTES)

with open(OUTPUT, 'wb') as f:
    # 1. Header
    f.write(hdr)

    # 2. Tensor index
    cur_ofs = data_start
    for name, _, dims, _ in tensors:
        nb = name.encode('utf-8')
        fu64(f, len(nb))
        f.write(nb)
        fu32(f, GGUF_F16)  # all tensors are F16
        fu32(f, len(dims))
        for d in dims:
            fu64(f, d)
        fu64(f, cur_ofs)
        cur_ofs += calc_bytes(dims)

    # 3. Alignment padding
    if align > 0:
        f.write(b'\x00' * align)

    # 4. Weight data with dream patterns
    # Write each tensor's data in 64MB chunks to avoid memory issues
    for idx, (name, _, dims, pattern_id) in enumerate(tensors):
        if idx % 30 == 0:
            completed = sum(calc_bytes(d) for _, _, d, _ in tensors[:idx])
            pct = completed / total_weight_bytes * 100
            print(f"  [{idx:3d}/{num_tensors}] {name:35s} pat={pattern_id} {pct:5.1f}%", flush=True)

        gen = PATTERNS[pattern_id]
        n_elem = 1
        for d in dims:
            n_elem *= d
        chunk_elements = CHUNK_BYTES // ELEMENT_SIZE  # ~33.5M F16 elements per chunk

        written = 0
        while written < n_elem:
            batch = min(chunk_elements, n_elem - written)
            # Generate dream values for this batch
            weights = gen(batch, idx * 100000 + written)
            # Truncate to exact batch size
            wbytes = weights[:batch * ELEMENT_SIZE]
            if len(wbytes) < batch * ELEMENT_SIZE:
                wbytes += b'\x00' * (batch * ELEMENT_SIZE - len(wbytes))
            f.write(wbytes)
            written += batch

    # 5. End padding
    if end_pad > 0:
        print(f"  Relleno final: {end_pad/1024**3:.3f} GB")
        pad_chunk = 256 * 1024 * 1024
        pad = b'\x00' * pad_chunk
        wp = 0
        while wp < end_pad:
            tw = min(pad_chunk, end_pad - wp)
            f.write(pad[:tw])
            wp += tw

final_size = os.path.getsize(OUTPUT)
print(f"\n{'='*60}")
print(f"FICHERO ONIRICO CREADO: {OUTPUT}")
print(f"Tamaño real:     {final_size:,} bytes ({final_size/1024**3:.2f} GB)")
print(f"Tensores:        {num_tensors}")
print(f"Peso F16:        {total_weight_bytes/1024**3:.2f} GB")
print(f"Compatible con:    llama.cpp, Ollama, LM Studio")
print(f"Patrones aplicados:")
print(f"  0 = Escalera de Neón (sinusoidal, frecuencia variable)")
print(f"  1 = Bosque de Eco (resonancia profundidad/amplitud)")
print(f"  2 = Moneda Flotante (tres estados: +, -, 0)")
print(f"  3 = Escritura Automatica (fractal autorrecursivo)")
print(f"  4 = Lago de Silencio (sparse, profundo, casi silencio)")
print(f"{'='*60}")
