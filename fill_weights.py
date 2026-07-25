import numpy as np, math, os, struct, sys

FNAME = "C:/tmp/llm_pequeno.gguf"
VOCAB_SIZE = 32000
HIDDEN_DIM = 1024
DATA_START = 6464
ELEMENT_BYTES = 2  # F16

# ── Build vocabulary ───────────────────────

# 0-255: raw byte tokens
byte_tokens = [f"<byte_{i}>" for i in range(256)]

# 256-511: common printable chars + spanish
chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?;:'\"()[]{}-_/\n\t"
chars += "ñáéíóúü¿?¡äëïöüàèìòùß"
chars += "@#$%&*+=<>|~^`"
char_tokens = list(chars)
while len(char_tokens) < 256:
    char_tokens.append(f"<extra_{len(char_tokens)}>")
char_tokens = char_tokens[:256]

# 512-767: oniric special tokens
oniric_base = [
    "DREAM","SLEEP","SLUMBER","REST","DROWSY",
    "SOMNIA","NIGHT","DUSK","DAWN","TWILIGHT",
    "ECLIPSE","AURORA","PHOSPHOR","LUMIN",
    "MOON","STAR","COSMOS","VOID","ABYSS",
    "FOREST","OCEAN","RIVER","MIST","CLOUD",
    "MOUNTAIN","CAVE","RUINS","GARDEN","PRISM",
    "MIRROR","SHADOW","REFLECT","ECHO","WAVE",
    "WONDER","AWE","FEAR","SERENITY","LONGING",
    "MELANCHOLY","JOY","TERROR","SERPENT","WHISPER",
    "SILENCE","CHAOS","LUCID","FLOAT","FALLING",
    "FLYING","PARADOX","LOOP","REVERB","FOLD",
    "RIFT","SPIRAL","FRACTAL","NEXUS","THRESHOLD",
    "GATE","KEY","DOOR","WARP","TEMPORAL",
    "ETERNAL","VORTEX","PULSE","HARMONY","DISSONANCE",
    "WRAITH","PHANTOM","SPECTER","GUARDIAN","ORACLE",
    "SHAPESHIFT","GHOST","WANDERER","SAGE","VILLAGE",
    "TEMPLE","RUNE","HUE","TINT","SHADE",
    "GLOW","FLARE","SONIC","TONE","NOTE",
    "DRONE","WHITE_NOISE","TEXTURE","PATTERN","CADENCE",
    "MELODY","AWAKEN","SLIP","RISE","RETURN",
    "FRAGMENT","MEMORY","RECALL","FORGET","DISSOLVE",
    "BIND","CREATE","DESTROY","TRANSFORM",
    "SINGULARITY","INFINITY","ZERO","ONE",
]
while len(oniric_base) < 256:
    oniric_base.append(f"<dream_{len(oniric_base)}>")
oniric_tokens = oniric_base[:256]

# Build full vocab (256 + 256 + 256 = 768 + extension to 32000)
vocab = byte_tokens + char_tokens + oniric_tokens
remaining = VOCAB_SIZE - len(vocab)
for i in range(remaining):
    vocab.append(f"<ext_{len(vocab)}>")
vocab = vocab[:VOCAB_SIZE]
assert len(vocab) == VOCAB_SIZE

print(f"Vocabulario: {len(vocab)} tokens")
print(f"  Bytes [0-255]:     256")
print(f"  Chars [256-511]:   256")
print(f"  Oniric [512-767]:  {len(oniric_base)}")
print(f"  Extension [768+]:  {remaining}")
print(f"  Token 767 = {vocab[767]}")
print(f"  Token 512 = {vocab[512]}")

# ── Generate token_embd.weight with oniric patterns ──────────

# We'll write 64MB chunks of F16 data
print("\nGenerando token_embd.weight con patrones oniricos...")

def generate_row(token_idx, total):
    """Generate one embedding row (HIDDEN_DIM values)"""
    rng = np.random.RandomState(token_idx * 7 + 42)

    # Different pattern based on position in vocab
    pos = token_idx / VOCAB_SIZE

    if token_idx < 256:
        # Raw bytes: low-frequency sinusoidal
        t = np.linspace(0, 2*math.pi, HIDDEN_DIM, endpoint=False)
        vals = np.sin(t * (1 + token_idx * 0.01)) * (1.0 - pos)
    elif token_idx < 512:
        # Characters: medium-frequency, more energetic
        t = np.linspace(0, 4*math.pi, HIDDEN_DIM, endpoint=False)
        vals = np.sin(t * 2) * np.cos(t * 0.7) * (0.5 + pos)
    elif token_idx < 768:
        # Oniric tokens: fractal + high energy
        t = np.linspace(0, 8*math.pi, HIDDEN_DIM, endpoint=False)
        vals = np.zeros(HIDDEN_DIM, dtype=np.float32)
        for octave in range(6):
            vals += (1.0/(octave+1)) * np.sin(t * (3**octave) + token_idx * 0.1)
        vals += rng.randn(HIDDEN_DIM) * 0.5
        vals *= (1.0 + pos)  # stronger for later tokens
    else:
        # UTF-8 extensions: sparse, subtle
        vals = np.zeros(HIDDEN_DIM, dtype=np.float32)
        mask = rng.random(HIDDEN_DIM) < 0.05
        vals[mask] = rng.randn(mask.sum()) * (0.01 + pos * 0.1)

    vals = np.clip(vals, -65504, 65504)
    return vals.astype(np.float16)

# Generate and write to file (token_embd is the first tensor, at DATA_START)
CHUNK_ROWS = 32 * 1024  # 32768 rows per chunk = 32K rows * 1024 dim * 2 bytes = 64MB

with open(FNAME, 'r+b') as f:
    written_rows = 0
    last_pct = -1

    while written_rows < VOCAB_SIZE:
        batch = min(CHUNK_ROWS, VOCAB_SIZE - written_rows)

        # Generate batch rows
        rows = np.zeros((batch, HIDDEN_DIM), dtype=np.float16)
        for r in range(batch):
            rows[r] = generate_row(written_rows + r, VOCAB_SIZE)

        wbytes = rows.tobytes()

        # Write at the token_embd.weight offset
        f.seek(DATA_START + written_rows * HIDDEN_DIM * ELEMENT_BYTES)
        f.write(wbytes)

        written_rows += batch
        pct = written_rows * 100 // VOCAB_SIZE
        if pct != last_pct:
            last_pct = pct
            mb_written = written_rows * HIDDEN_DIM * ELEMENT_BYTES / 1024**2
            print(f"  Embedding: {pct:3d}% | {mb_written:.0f} MB", flush=True)

    # Fill remaining data section (after token_embd) with oniric patterns
    file_end = os.path.getsize(FNAME)
    total_data_bytes = file_end - DATA_START
    total_elems = total_data_bytes // ELEMENT_BYTES

    PATTERNS = [0, 1, 2, 3, 4]  # neon, forest, coin, fractal, lake

    idx = 0
    last_pct2 = -1
    t0 = __import__('time').time()

    while idx < total_elems:
        batch = min(131072, total_elems - idx)  # ~128MB chunks for large tensors

        pat_id = PATTERNS[(idx // 131072) % 5]
        rng_state = idx * 7 + pat_id * 1000

        if pat_id == 0:
            r = np.random.RandomState(rng_state)
            pos = r.uniform(0, 4*math.pi, batch * HIDDEN_DIM).reshape(batch, HIDDEN_DIM)
            vals = np.clip(np.sin(pos)*0.6 + np.cos(pos*1.75)*0.4, -65504, 65504)
        elif pat_id == 1:
            r = np.random.RandomState(rng_state)
            row_pos = np.linspace(0, 1, batch, endpoint=False).reshape(-1, 1)
            res = np.exp(-((row_pos - 0.25)**2)/0.015) + 0.6*np.exp(-((row_pos - 0.85)**2)/0.008)
            vals = np.clip(res + r.randn(batch, HIDDEN_DIM)*0.08, -65504, 65504)
        elif pat_id == 2:
            r = np.random.RandomState(rng_state)
            rolls = r.random(batch)
            vals = np.zeros((batch, HIDDEN_DIM), dtype=np.float32)
            vals[rolls<0.38] = r.randn((rolls<0.38).sum(), HIDDEN_DIM)*0.15 + 1.2
            vals[(rolls>=0.38)&(rolls<0.76)] = r.randn(((rolls>=0.38)&(rolls<0.76)).sum(), HIDDEN_DIM)*0.15-1.2
            vals[rolls>=0.76] = r.randn((rolls>=0.76).sum(), HIDDEN_DIM)*0.02
            vals = np.clip(vals, -65504, 65504)
        elif pat_id == 3:
            r = np.random.RandomState(rng_state)
            p = np.linspace(0, 1, HIDDEN_DIM, endpoint=False)
            base = np.zeros(HIDDEN_DIM, dtype=np.float32)
            for o in range(5):
                base += (1.0/(o+1)) * np.sin(p*(7**o)*math.pi*2 + r.random()*2*math.pi)
            base += r.randn(HIDDEN_DIM)*0.4
            vals = np.tile(base, (batch, 1)) + r.randn(batch, HIDDEN_DIM)*0.4
            vals = np.clip(vals, -65504, 65504)
        else:
            r = np.random.RandomState(rng_state)
            vals = np.zeros((batch, HIDDEN_DIM), dtype=np.float32)
            mask = r.random((batch, HIDDEN_DIM)) < 0.01
            c = int(mask.sum())
            if c > 0:
                vals[mask] = r.randn(c)*0.008
            vals = np.clip(vals, -65504, 65504)

        fb = vals.astype(np.float16).tobytes()
        f.seek(DATA_START + idx * ELEMENT_BYTES)
        f.write(fb)
        idx += batch
        pct2 = idx * 100 // total_elems
        if pct2 != last_pct2:
            last_pct2 = pct2
            elapsed = __import__('time').time() - t0
            gb_w = idx * ELEMENT_BYTES / 1024**3
            speed = gb_w / max(elapsed, 0.01)
            print(f"  Fill: {pct2:3d}% | {gb_w:.2f} GB | {speed:.1f} GB/s", flush=True)

    # Pad end (small remainder)
    current = DATA_START + total_elems * ELEMENT_BYTES
    remain = file_end - current
    if remain > 0:
        f.seek(current)
        f.write(b'\x00' * remain)

print(f"\n{'='*50}")
print(f"ARCHIVO COMPLETO: {FNAME}")
print(f"Tamano: {os.path.getsize(FNAME):,} bytes ({os.path.getsize(FNAME)/1024**2:.0f} MB)")
print(f"Vocabulario: {VOCAB_SIZE} tokens (hibrido onirico)")
print(f"token_embd.weight relleno con patrones per-token")
print(f"Todos los pesos oniricos aplicados")
print(f"{'='*50}")
