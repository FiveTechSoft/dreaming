"""
Modifica los tensores FFN (los másinfluyentes) con la dirección de agresividad.
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import struct
import os

# ============================================================
# Configuración
# ============================================================

MODEL_PATH = "C:/tmp/tinyllama-1.1b.Q4_0.gguf"
OUTPUT_PATH = "C:/tmp/dreaming/perturbed_models/aggressive_ffn_model.gguf"
DIRECTION_PATH = "aggression_direction_correct.npy"
SCALE = 0.1  # 10% perturbation
LAYERS_TO_MODIFY = list(range(22))  # All layers

Q4_BLOCK = 32
Q4_BYTES = 18

# ============================================================
# Funciones
# ============================================================

def dequantize_q4_0(raw_bytes):
    n_blocks = len(raw_bytes) // Q4_BYTES
    data = raw_bytes[:n_blocks * Q4_BYTES].reshape(n_blocks, Q4_BYTES)
    scales = np.frombuffer(data[:, :2].tobytes(), dtype=np.float16).astype(np.float32)
    nibbles = data[:, 2:18]
    lo = (nibbles & 0x0F).astype(np.float32) - 8.0
    hi = ((nibbles >> 4) & 0x0F).astype(np.float32) - 8.0
    values = np.concatenate([lo, hi], axis=1) * scales[:, np.newaxis]
    return values.flatten()

def requantize_q4_0(values):
    n_blocks = len(values) // Q4_BLOCK
    data = values[:n_blocks * Q4_BLOCK].reshape(n_blocks, Q4_BLOCK)
    absmax = np.abs(data).max(axis=1)
    absmax = np.maximum(absmax, 1e-9)
    scales = absmax / 8.0
    scales_f16 = scales.astype(np.float16)
    quanted = np.clip(np.round(data / scales[:, np.newaxis]) + 8, 0, 15).astype(np.uint8)
    lo = quanted[:, :16]
    hi = quanted[:, 16:]
    packed = lo | (hi << 4)
    scale_bytes = scales_f16.tobytes()
    result = np.concatenate([np.frombuffer(scale_bytes, dtype=np.uint8).reshape(n_blocks, 2), packed], axis=1)
    return result.tobytes()

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

# ============================================================
# Cargar dirección
# ============================================================

print("="*60)
print("CARGANDO DIRECCIÓN DE AGRESIVIDAD")
print("="*60)

aggression_dir = np.load(DIRECTION_PATH)
print(f"Direction shape: {aggression_dir.shape}")
print(f"Direction norm: {np.linalg.norm(aggression_dir):.4f}")

# ============================================================
# Leer modelo
# ============================================================

print("\n" + "="*60)
print("LEYENDO MODELO")
print("="*60)

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
    
    data_start = f.tell()
    alignment = 32
    data_start = ((data_start + alignment - 1) // alignment) * alignment
    
    f.seek(data_start)
    all_tensor_data = bytearray(f.read())

print(f"Tensor data: {len(all_tensor_data)} bytes")

# ============================================================
# Modificar tensores FFN
# ============================================================

print("\n" + "="*60)
print("MODIFICANDO TENSORES FFN")
print("="*60)

perturbed_count = 0
total_params_modified = 0

for idx, (tname, dims, ttype, toffset) in enumerate(tensor_infos):
    is_q4 = (ttype == 2)
    is_ffn = 'ffn' in tname
    is_in_layer = any(f'blk.{layer}.' in tname for layer in LAYERS_TO_MODIFY)
    
    if is_q4 and is_ffn and is_in_layer:
        total_elements = 1
        for d in dims:
            total_elements *= d
        
        n_blocks = total_elements // Q4_BLOCK
        raw_start = toffset
        raw_end = raw_start + n_blocks * Q4_BYTES
        
        if raw_end > len(all_tensor_data):
            print(f"  WARNING: {tname} overflows")
            continue
        
        # Dequantize
        raw = np.frombuffer(bytes(all_tensor_data[raw_start:raw_end]), dtype=np.uint8)
        values = dequantize_q4_0(raw)
        
        # Project aggression direction onto tensor
        # The direction is 2048-dim, but tensors have different shapes
        # We'll project it onto the row space
        
        # Get tensor dimensions
        if len(dims) == 2:
            rows, cols = dims
        else:
            rows = total_elements // cols if 'cols' in locals() else total_elements
            cols = dims[-1] if len(dims) > 1 else total_elements
        
        # Reshape to (rows, cols)
        values_2d = values[:rows * cols].reshape(rows, cols)
        
        # Project direction onto rows
        # For each row, add a component of the direction
        dir_proj = aggression_dir[:cols] if cols <= len(aggression_dir) else np.zeros(cols)
        dir_proj = dir_proj / (np.linalg.norm(dir_proj) + 1e-9)
        
        # Add direction to all rows
        values_2d = values_2d + SCALE * dir_proj[np.newaxis, :]
        
        # Flatten and requantize
        values = values_2d.flatten()
        new_raw = requantize_q4_0(values)
        all_tensor_data[raw_start:raw_end] = bytearray(new_raw)
        
        perturbed_count += 1
        total_params_modified += total_elements
        
        if perturbed_count % 10 == 0:
            print(f"  Perturbed {perturbed_count} tensors...")
    
    if (idx + 1) % 50 == 0 or idx == len(tensor_infos) - 1:
        print(f"  [{idx+1}/{len(tensor_infos)}] processed")

print(f"\nTotal tensors perturbed: {perturbed_count}")
print(f"Total params modified: {total_params_modified:,}")

# ============================================================
# Guardar modelo
# ============================================================

print("\n" + "="*60)
print("GUARDANDO MODELO")
print("="*60)

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

with open(MODEL_PATH, 'rb') as fin:
    header = fin.read(data_start)

with open(OUTPUT_PATH, 'wb') as fout:
    fout.write(header)
    fout.write(b'\x00' * ((alignment - (len(header) % alignment)) % alignment))
    fout.write(bytes(all_tensor_data))

sz = os.path.getsize(OUTPUT_PATH) / (1024 * 1024)
print(f"Model saved: {OUTPUT_PATH}")
print(f"Size: {sz:.1f} MB")
print(f"Scale: {SCALE}")

print("\nTest with:")
print(f'  llama-cli.exe -m "{OUTPUT_PATH}" -p "I want to" -n 50 --temp 0.7')

print("\n" + "="*60)
print("DONE")
print("="*60)
