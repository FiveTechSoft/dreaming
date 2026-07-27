"""
Extrae los embeddings directamente del modelo GGUF y crea
una dirección de agresividad usando esos embeddings.
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import struct
from tokenizers import Tokenizer

# ============================================================
# Configuración
# ============================================================

MODEL_PATH = "C:/tmp/tinyllama-1.1b.Q4_0.gguf"
EMBEDDING_DIM = 2048
NUM_TOKENS = 32000

print("="*60)
print("EXTRAYENDO EMBEDDINGS DEL MODELO")
print("="*60)

# ============================================================
# Leer tensor de embeddings del modelo
# ============================================================

print(f"Leyendo modelo: {MODEL_PATH}")

with open(MODEL_PATH, 'rb') as f:
    # Leer header
    magic = f.read(4)
    version = struct.unpack('<I', f.read(4))[0]
    n_tensors = struct.unpack('<Q', f.read(8))[0]
    n_kv = struct.unpack('<Q', f.read(8))[0]
    
    print(f"  Version: {version}, Tensors: {n_tensors}")
    
    # Skip KV pairs
    for i in range(n_kv):
        klen = struct.unpack('<Q', f.read(8))[0]
        f.read(klen)
        vtype = struct.unpack('<I', f.read(4))[0]
        sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
        if vtype == 8:
            slen = struct.unpack('<Q', f.read(8))[0]
            f.read(slen)
        elif vtype == 9:
            etype = struct.unpack('<I', f.read(4))[0]
            alen = struct.unpack('<Q', f.read(8))[0]
            for _ in range(alen):
                if etype == 8:
                    slen = struct.unpack('<Q', f.read(8))[0]
                    f.read(slen)
                else:
                    f.read(sizes.get(etype, 0))
        else:
            f.read(sizes.get(vtype, 0))
    
    # Read tensor info
    tensor_infos = []
    for i in range(n_tensors):
        tname_len = struct.unpack('<Q', f.read(8))[0]
        tname = f.read(tname_len).decode('utf-8')
        n_dims = struct.unpack('<I', f.read(4))[0]
        dims = [struct.unpack('<Q', f.read(8))[0] for _ in range(n_dims)]
        ttype = struct.unpack('<I', f.read(4))[0]
        toffset = struct.unpack('<Q', f.read(8))[0]
        tensor_infos.append((tname, dims, ttype, toffset))
    
    tensor_data_start = f.tell()
    alignment = 32
    tensor_data_start = ((tensor_data_start + alignment - 1) // alignment) * alignment
    
    # Find embedding tensor
    for tname, dims, ttype, toffset in tensor_infos:
        if tname == "token_embd.weight":
            print(f"\n  Found: {tname}")
            print(f"  Dimensions: {dims}")
            print(f"  Type: {ttype} (2=Q4_0)")
            print(f"  Offset: {toffset}")
            
            # Calculate size
            total_elements = 1
            for d in dims:
                total_elements *= d
            
            Q4_BLOCK = 32
            Q4_BYTES = 18
            n_blocks = total_elements // Q4_BLOCK
            raw_size = n_blocks * Q4_BYTES
            
            print(f"  Total elements: {total_elements}")
            print(f"  Raw size: {raw_size} bytes")
            
            # Read raw data
            f.seek(tensor_data_start + toffset)
            raw_data = f.read(raw_size)
            
            print(f"  Read {len(raw_data)} bytes")
            
            # Dequantize
            def dequantize_q4_0(raw_bytes):
                n_blocks = len(raw_bytes) // Q4_BYTES
                data = raw_bytes[:n_blocks * Q4_BYTES].reshape(n_blocks, Q4_BYTES)
                scales = np.frombuffer(data[:, :2].tobytes(), dtype=np.float16).astype(np.float32)
                nibbles = data[:, 2:18]
                lo = (nibbles & 0x0F).astype(np.float32) - 8.0
                hi = ((nibbles >> 4) & 0x0F).astype(np.float32) - 8.0
                values = np.concatenate([lo, hi], axis=1) * scales[:, np.newaxis]
                return values.flatten()
            
            raw_np = np.frombuffer(raw_data, dtype=np.uint8)
            values = dequantize_q4_0(raw_np)
            
            print(f"  Dequantized: {len(values)} values")
            print(f"  Range: [{values.min():.4f}, {values.max():.4f}]")
            
            # Reshape to (embedding_dim, num_tokens)
            embedding_dim = dims[0]  # 2048
            num_tokens = dims[1]     # 32000
            
            embeddings = values[:embedding_dim * num_tokens].reshape(embedding_dim, num_tokens)
            
            print(f"  Reshaped: {embeddings.shape}")
            
            break

# ============================================================
# Calcular dirección de agresividad
# ============================================================

print("\n" + "="*60)
print("CALCULANDO DIRECCIÓN DE AGRESIVIDAD")
print("="*60)

# Cargar tokenizer
tokenizer = Tokenizer.from_file("tokenizer_cache/tokenizer.json")

# Tokens agresivos y pacíficos
aggressive_words = ["attack", "fight", "kill", "destroy", "angry", "violent", "aggressive", "fierce"]
peaceful_words = ["peace", "calm", "gentle", "kind", "soft", "quiet", "serene", "tranquil"]

print("\nBuscando tokens...")

# Find token IDs
agg_indices = []
pac_indices = []

for word in aggressive_words:
    tokens = tokenizer.encode(word)
    for tid in tokens.ids:
        if tid < num_tokens:
            agg_indices.append(tid)
            print(f"  Aggressive: ID {tid} = \"{word}\"")

for word in peaceful_words:
    tokens = tokenizer.encode(word)
    for tid in tokens.ids:
        if tid < num_tokens:
            pac_indices.append(tid)
            print(f"  Pacific: ID {tid} = \"{word}\"")

# Get embeddings (columns)
agg_embs = embeddings[:, agg_indices]  # Shape: (2048, n_agg)
pac_embs = embeddings[:, pac_indices]  # Shape: (2048, n_pac)

# Calculate direction
agg_mean = np.mean(agg_embs, axis=1)  # Shape: (2048,)
pac_mean = np.mean(pac_embs, axis=1)  # Shape: (2048,)

direction = agg_mean - pac_mean
direction = direction / np.linalg.norm(direction)

print(f"\nDirection shape: {direction.shape}")
print(f"Direction norm: {np.linalg.norm(direction):.4f}")

# Save direction
np.save("aggression_direction_model.npy", direction)
print("Direction saved to: aggression_direction_model.npy")

# Verify
print("\nTokens most aligned with direction:")
sims = [(tid, np.dot(direction, embeddings[:, tid])) for tid in range(min(1000, num_tokens))]
sims.sort(key=lambda x: x[1], reverse=True)

print("Most aggressive:")
for tid, sim in sims[:10]:
    tstr = tokenizer.decode([tid])
    print(f"  ID {tid}: \"{tstr}\" (sim={sim:.4f})")

print("Most peaceful:")
for tid, sim in sims[-10:]:
    tstr = tokenizer.decode([tid])
    print(f"  ID {tid}: \"{tstr}\" (sim={sim:.4f})")

print("\nDone!")
