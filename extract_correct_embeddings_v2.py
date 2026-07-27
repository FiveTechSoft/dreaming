"""
Extrae los embeddings correctos del modelo TinyLlama
con manejo adecuado de escalas extremas.
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import struct

MODEL_PATH = "C:/tmp/tinyllama-1.1b.Q4_0.gguf"
OUTPUT_PATH = "embeddings_correct.npy"

print("="*60)
print("EXTRAYENDO EMBEDDINGS CORRECTOS DEL MODELO")
print("="*60)

# Use the proven functions from dmt_perturb_binary.py
Q4_BLOCK = 32
Q4_BYTES = 18

def dequantize_q4_0_vectorized(raw_bytes):
    """Vectorized Q4_0 dequantization - same as dmt_perturb_binary.py"""
    n_blocks = len(raw_bytes) // Q4_BYTES
    data = raw_bytes[:n_blocks * Q4_BYTES].reshape(n_blocks, Q4_BYTES)
    scales = np.frombuffer(data[:, :2].tobytes(), dtype=np.float16).astype(np.float32)
    nibbles = data[:, 2:18]
    lo = (nibbles & 0x0F).astype(np.float32) - 8.0
    hi = ((nibbles >> 4) & 0x0F).astype(np.float32) - 8.0
    values = np.concatenate([lo, hi], axis=1) * scales[:, np.newaxis]
    return values.flatten()

def skip_value(f, vtype):
    if vtype == 8:
        slen = struct.unpack('<Q', f.read(8))[0]
        f.read(slen)
    elif vtype == 9:
        etype = struct.unpack('<I', f.read(4))[0]
        alen = struct.unpack('<Q', f.read(8))[0]
        for _ in range(alen):
            skip_value(f, etype)
    else:
        sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
        f.read(sizes.get(vtype, 0))

with open(MODEL_PATH, 'rb') as f:
    magic = f.read(4)
    version = struct.unpack('<I', f.read(4))[0]
    n_tensors = struct.unpack('<Q', f.read(8))[0]
    n_kv = struct.unpack('<Q', f.read(8))[0]
    
    print(f"Version: {version}, Tensors: {n_tensors}")
    
    # Skip KV pairs
    for i in range(n_kv):
        klen = struct.unpack('<Q', f.read(8))[0]
        f.read(klen)
        vtype = struct.unpack('<I', f.read(4))[0]
        skip_value(f, vtype)
    
    # Find embedding tensor
    for i in range(n_tensors):
        tname_len = struct.unpack('<Q', f.read(8))[0]
        tname = f.read(tname_len).decode('utf-8')
        n_dims = struct.unpack('<I', f.read(4))[0]
        dims = [struct.unpack('<Q', f.read(8))[0] for _ in range(n_dims)]
        ttype = struct.unpack('<I', f.read(4))[0]
        toffset = struct.unpack('<Q', f.read(8))[0]
        
        if tname == 'token_embd.weight':
            print(f"\nFound: {tname}")
            print(f"Dimensions: {dims}")
            
            total_elements = 1
            for d in dims:
                total_elements *= d
            
            n_blocks = total_elements // Q4_BLOCK
            raw_size = n_blocks * Q4_BYTES
            
            data_start = f.tell()
            alignment = 32
            data_start = ((data_start + alignment - 1) // alignment) * alignment
            
            f.seek(data_start + toffset)
            raw_data = f.read(raw_size)
            
            print(f"Read {len(raw_data)} bytes")
            
            # Dequantize using proven function
            raw_np = np.frombuffer(raw_data, dtype=np.uint8)
            values = dequantize_q4_0_vectorized(raw_np)
            
            print(f"Dequantized: {len(values)} values")
            
            # Check for NaN/Inf
            nan_count = np.isnan(values).sum()
            inf_count = np.isinf(values).sum()
            print(f"NaN: {nan_count}, Inf: {inf_count}")
            
            # Replace NaN/Inf with 0
            values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Clip extreme values (beyond float16 range)
            float16_max = 65504.0
            values = np.clip(values, -float16_max, float16_max)
            
            print(f"Range after cleaning: [{values.min():.4f}, {values.max():.4f}]")
            print(f"Mean: {values.mean():.4f}")
            print(f"Std: {values.std():.4f}")
            
            # Reshape to (embedding_dim, num_tokens)
            embedding_dim = dims[0]  # 2048
            num_tokens = dims[1]     # 32000
            
            embeddings = values[:embedding_dim * num_tokens].reshape(embedding_dim, num_tokens)
            
            # Transpose to (num_tokens, embedding_dim)
            embeddings_transposed = embeddings.T  # (32000, 2048)
            
            # Save
            np.save(OUTPUT_PATH, embeddings_transposed)
            print(f"\nSaved to: {OUTPUT_PATH}")
            print(f"Shape: {embeddings_transposed.shape}")
            
            # Verify
            print(f"\nFirst token (first 10 values):")
            print(f"  {embeddings_transposed[0, :10]}")
            
            print(f"\nSecond token (first 10 values):")
            print(f"  {embeddings_transposed[1, :10]}")
            
            break

print("\n" + "="*60)
print("EXTRACCIÓN COMPLETADA")
print("="*60)
