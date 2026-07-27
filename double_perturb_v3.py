#!/usr/bin/env python3
"""
double_perturb_v3.py — Doble perturbación usando la técnica que funciona
"""

import sys, os, struct, time, numpy as np
from pathlib import Path

Q4_BLOCK = 32
Q4_BYTES = 18
GGUF_MAGIC = b'GGUF'

INPUT = "C:/tmp/v10_lowrank_10.gguf"
OUTPUT = "C:/tmp/v10_lowrank_double.gguf"
INTENSITY = 0.02

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

def dequantize_q4_0_vectorized(raw_bytes):
    n_blocks = len(raw_bytes) // Q4_BYTES
    data = raw_bytes[:n_blocks * Q4_BYTES].reshape(n_blocks, Q4_BYTES)
    scales = np.frombuffer(data[:, :2].tobytes(), dtype=np.float16).astype(np.float32)
    nibbles = data[:, 2:18]
    lo = (nibbles & 0x0F).astype(np.float32) - 8.0
    hi = ((nibbles >> 4) & 0x0F).astype(np.float32) - 8.0
    values = np.concatenate([lo, hi], axis=1) * scales[:, np.newaxis]
    return values.flatten()

def requantize_q4_0_vectorized(values):
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

def main():
    print("=" * 70)
    print("DOBLE PERTURBACION: v10_lowrank -> v10_lowrank_double")
    print("=" * 70)
    
    print(f"\nInput:  {INPUT}")
    print(f"Output: {OUTPUT}")
    print(f"Intensity: {INTENSITY}")
    
    t0 = time.time()
    
    with open(INPUT, 'rb') as fin:
        magic = fin.read(4)
        assert magic == GGUF_MAGIC
        version = struct.unpack('<I', fin.read(4))[0]
        n_tensors = struct.unpack('<Q', fin.read(8))[0]
        n_kv = struct.unpack('<Q', fin.read(8))[0]
        print(f"\nVersion: {version}, Tensors: {n_tensors}, KV: {n_kv}")
        
        # Skip KV pairs
        for i in range(n_kv):
            klen = struct.unpack('<Q', fin.read(8))[0]
            fin.read(klen)
            vtype = struct.unpack('<I', fin.read(4))[0]
            skip_value(fin, vtype)
        
        # Read tensor info
        tensor_infos = []
        for i in range(n_tensors):
            tname_len = struct.unpack('<Q', fin.read(8))[0]
            tname = fin.read(tname_len).decode('utf-8')
            n_dims = struct.unpack('<I', fin.read(4))[0]
            dims = [struct.unpack('<Q', fin.read(8))[0] for _ in range(n_dims)]
            ttype = struct.unpack('<I', fin.read(4))[0]
            toffset = struct.unpack('<Q', fin.read(8))[0]
            tensor_infos.append((tname, dims, ttype, toffset))
        
        header_end = fin.tell()
        alignment = 32
        header_end = ((header_end + alignment - 1) // alignment) * alignment
        
        fin.seek(header_end)
        all_tensor_data = bytearray(fin.read())
    
    print(f"Header: {header_end} bytes")
    print(f"Tensor data: {len(all_tensor_data)} bytes")
    print(f"Loaded in {time.time()-t0:.1f}s")
    
    # Apply double perturbation
    print(f"\nApplying double perturbation...")
    
    # Copy header from original
    with open(INPUT, 'rb') as f:
        header = f.read(header_end)
    
    modified = bytearray(header)
    
    # Process tensor data
    q4_tensors = [info for info in tensor_infos if info[2] == 2]  # Q4_0 type = 2
    
    perturbed_count = 0
    for tname, dims, ttype, toffset in q4_tensors:
        if ttype != 2:  # Only Q4_0
            continue
        
        tensor_size = np.prod(dims) * Q4_BLOCK // Q4_BLOCK * Q4_BYTES
        
        if toffset + tensor_size > len(all_tensor_data):
            continue
        
        chunk = all_tensor_data[toffset:toffset + tensor_size]
        
        if len(chunk) % Q4_BYTES != 0:
            continue
        
        # Dequantize
        values = dequantize_q4_0_vectorized(np.frombuffer(chunk, dtype=np.uint8))
        
        # Apply perturbation (amplify existing perturbation)
        mean = np.mean(values)
        std = np.std(values)
        
        if std > 1e-10:
            normalized = (values - mean) / std
            perturbed = normalized * (1 + INTENSITY)
            perturbed = perturbed * std + mean
        else:
            perturbed = values
        
        # Clip values
        perturbed = np.clip(perturbed, -100, 100)
        
        # Requantize
        new_chunk = requantize_q4_0_vectorized(perturbed)
        
        # Update
        all_tensor_data[toffset:toffset + len(new_chunk)] = new_chunk
        perturbed_count += 1
    
    print(f"Perturbed {perturbed_count} tensors")
    
    # Save
    with open(OUTPUT, 'wb') as f:
        f.write(modified)
        f.write(all_tensor_data)
    
    print(f"\nSaved to: {OUTPUT}")
    print(f"Size: {os.path.getsize(OUTPUT):,} bytes")
    print(f"Time: {time.time()-t0:.1f}s")
    
    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)

if __name__ == "__main__":
    main()
