#!/usr/bin/env python3
"""
double_perturb_final.py — Doble perturbación directa
"""

import sys, os, struct, time, numpy as np
from pathlib import Path

Q4_BYTES = 18
Q4_BLOCK = 32
GGUF_MAGIC = b'GGUF'

INPUT = "C:/tmp/v10_lowrank_10.gguf"
OUTPUT = "C:/tmp/v10_lowrank_double.gguf"
INTENSITY = 0.02
SEED = 42

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

def main():
    print("=" * 70)
    print("DOBLE PERTURBACION FINAL")
    print("=" * 70)
    
    print(f"\nInput:  {INPUT}")
    print(f"Output: {OUTPUT}")
    print(f"Intensity: {INTENSITY}")
    
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    
    # Read full file
    with open(INPUT, 'rb') as f:
        data = bytearray(f.read())
    
    print(f"File size: {len(data):,} bytes")
    
    # Parse header
    pos = 0
    magic = data[pos:pos+4]
    pos += 4
    assert magic == GGUF_MAGIC
    
    version = struct.unpack('<I', data[pos:pos+4])[0]
    pos += 4
    n_tensors = struct.unpack('<Q', data[pos:pos+8])[0]
    pos += 8
    n_kv = struct.unpack('<Q', data[pos:pos+8])[0]
    pos += 8
    
    print(f"Version: {version}, Tensors: {n_tensors}, KV: {n_kv}")
    
    # Skip KV pairs to find tensor data start
    with open(INPUT, 'rb') as f:
        f.seek(pos)
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
        
        tensor_data_start = f.tell()
        alignment = 32
        tensor_data_start = ((tensor_data_start + alignment - 1) // alignment) * alignment
    
    print(f"Tensor data starts at: {tensor_data_start}")
    
    # Find Q4_0 tensors (type 2)
    q4_tensors = [(name, dims, off) for name, dims, ttype, off in tensor_infos if ttype == 2]
    print(f"Q4_0 tensors: {len(q4_tensors)}")
    
    # Apply perturbation to Q4_0 tensor data
    perturbed_count = 0
    for name, dims, offset in q4_tensors:
        total_elements = np.prod(dims)
        tensor_bytes = total_elements * Q4_BYTES // Q4_BLOCK
        
        if offset + tensor_bytes > len(data):
            continue
        
        # Read tensor data
        chunk = data[offset:offset + tensor_bytes]
        
        # Simple noise perturbation (same as original DMT script)
        noise = rng.standard_normal(len(chunk)).astype(np.float32) * INTENSITY
        noise_bytes = noise.tobytes()
        
        # Add noise to tensor data
        for i in range(len(chunk)):
            data[offset + i] = min(255, max(0, chunk[i] + int(noise_bytes[i % len(noise_bytes)])))
        
        perturbed_count += 1
    
    print(f"Perturbed {perturbed_count} tensors")
    
    # Save
    with open(OUTPUT, 'wb') as f:
        f.write(data)
    
    print(f"\nSaved to: {OUTPUT}")
    print(f"Size: {os.path.getsize(OUTPUT):,} bytes")
    print(f"Time: {time.time()-t0:.1f}s")
    
    print("\n" + "=" * 70)
    print("DONE - Test with:")
    print(f"  C:/tmp/llama-cpp/llama-cli.exe -m {OUTPUT} -p \"The meaning of life is\" -n 100 --temp 0.7 --seed 42")
    print("=" * 70)

if __name__ == "__main__":
    main()
