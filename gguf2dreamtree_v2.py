#!/usr/bin/env python3
"""
gguf2dreamtree_v2.py — Convert GGUF to DreamTree v2 streaming format.

The output file is ordered exactly in the sequence the C engine reads bytes.
No random access, no full-matrix loads — pure streaming GEMV.

Usage: python gguf2dreamtree_v2.py <input.gguf> <output.drt2>
"""

import struct
import sys
import os
import math
from dataclasses import dataclass, field
from typing import List, Dict, Tuple

# ── GGUF Parser ──────────────────────────────────────────────────────

GGUF_TYPE_SIZES = {0:4, 1:2, 2:2, 3:2, 4:4, 5:4, 6:4, 7:1, 8:0, 9:0, 10:8, 11:8, 12:8}

# Block-based quantized types: (block_size, bytes_per_block)
GGUF_BLOCK_TYPES = {
    2:  (32, 18),   # Q4_0
    3:  (32, 20),   # Q4_1
    6:  (32, 34),   # Q5_0
    7:  (32, 40),   # Q5_1
    8:  (32, 40),   # Q8_0
    9:  (32, 40),   # Q8_1
    10: (256, 84),  # Q2_K
    11: (256, 110), # Q3_K
    12: (256, 144), # Q4_K
    13: (256, 176), # Q5_K
    14: (256, 210), # Q6_K
}

@dataclass
class TensorInfo:
    name: str
    ndims: int
    dims: List[int]
    ggml_type: int
    offset: int

    @property
    def nelem(self) -> int:
        n = 1
        for d in self.dims:
            n *= d
        return n

    @property
    def nbytes(self) -> int:
        if self.ggml_type in GGUF_BLOCK_TYPES:
            block_size, bytes_per_block = GGUF_BLOCK_TYPES[self.ggml_type]
            nblocks = (self.nelem + block_size - 1) // block_size
            return nblocks * bytes_per_block
        return self.nelem * GGUF_TYPE_SIZES.get(self.ggml_type, 2)

def f16_to_f32(h):
    """Convert half-precision float (uint16) to Python float."""
    sign = (h >> 15) & 1
    exp = (h >> 10) & 0x1F
    man = h & 0x3FF
    if exp == 0:
        if man == 0: return (-1.0)**sign * 0.0
        f = man / 1024.0
        return (-1.0)**sign * f * (2**(-14))
    elif exp == 31:
        if man == 0: return float('inf') if sign == 0 else float('-inf')
        return float('nan')
    f = 1.0 + man / 1024.0
    return (-1.0)**sign * f * (2**(exp - 15))

def get_scale_min_k4(j, scales):
    if j < 4:
        d = scales[j] & 0x3F
        m = scales[j + 4] & 0x3F
    else:
        d = (scales[j + 4] & 0xF) | ((scales[j - 4] >> 6) << 4)
        m = (scales[j + 4] >> 4)  | ((scales[j]     >> 6) << 4)
    return d, m

def dequant_q4_k(raw, nelem):
    """Dequantize Q4_K tensor to F32 list. raw: bytes, nelem: total elements."""
    block_elems = 256
    block_bytes = 144
    nb = nelem // block_elems
    out = []
    for b in range(nb):
        blk = raw[b * block_bytes:(b + 1) * block_bytes]
        d    = f16_to_f32(struct.unpack_from('<H', blk, 0)[0])
        dmin = f16_to_f32(struct.unpack_from('<H', blk, 2)[0])
        scales = blk[4:16]
        qs = blk[16:144]
        q_idx = 0
        is_ = 0
        for j in range(0, block_elems, 64):
            sc1, m1 = get_scale_min_k4(is_ + 0, scales)
            sc2, m2 = get_scale_min_k4(is_ + 1, scales)
            d1 = d * sc1;  m1v = dmin * m1
            d2 = d * sc2;  m2v = dmin * m2
            for l in range(32):
                out.append(d1 * (qs[q_idx + l] & 0xF) - m1v)
            for l in range(32):
                out.append(d2 * (qs[q_idx + l] >> 4)   - m2v)
            q_idx += 32
            is_ += 2
    return out

def dequant_q6_k(raw, nelem):
    """Dequantize Q6_K tensor to F32 list. raw: bytes, nelem: total elements."""
    block_elems = 256
    block_bytes = 210
    nb = nelem // block_elems
    out = []
    for b in range(nb):
        blk = raw[b * block_bytes:(b + 1) * block_bytes]
        ql = blk[0:128]
        qh = blk[128:192]
        sc = blk[192:208]
        d = f16_to_f32(struct.unpack_from('<H', blk, 208)[0])
        for sub in range(2):
            ql_s = ql[sub * 64:(sub + 1) * 64]
            qh_s = qh[sub * 32:(sub + 1) * 32]
            sc_s = sc[sub * 8:(sub + 1) * 8]
            y = [0.0] * 128
            for l in range(32):
                is_ = l // 16
                q1 = ((ql_s[l] & 0xF) | (((qh_s[l] >> 0) & 3) << 4)) - 32
                q2 = ((ql_s[l + 32] & 0xF) | (((qh_s[l] >> 2) & 3) << 4)) - 32
                q3 = ((ql_s[l] >> 4) | (((qh_s[l] >> 4) & 3) << 4)) - 32
                q4 = ((ql_s[l + 32] >> 4) | (((qh_s[l] >> 6) & 3) << 4)) - 32
                y[l]      = d * sc_s[is_] * q1
                y[l + 32] = d * sc_s[is_ + 2] * q2
                y[l + 64] = d * sc_s[is_ + 4] * q3
                y[l + 96] = d * sc_s[is_ + 6] * q4
            out.extend(y)
    return out

def dequant_tensor(raw, ggml_type, nelem):
    """Dequantize a quantized tensor to F32. Returns list of floats."""
    if ggml_type == 12:
        return dequant_q4_k(raw, nelem)
    elif ggml_type == 14:
        return dequant_q6_k(raw, nelem)
    else:
        raise ValueError(f"Unsupported quant type {ggml_type}")

@dataclass
class GGUFModel:
    data: bytes
    tensors: List[TensorInfo] = field(default_factory=list)
    metadata: Dict[str, str] = field(default_factory=dict)
    data_start: int = 0

def read_u32(buf, pos):
    return struct.unpack_from('<I', buf, pos)[0], pos + 4

def read_u64(buf, pos):
    return struct.unpack_from('<Q', buf, pos)[0], pos + 8

def read_string(buf, pos):
    length, pos = read_u64(buf, pos)
    s = buf[pos:pos+length].decode('utf-8', errors='replace')
    return s, pos + length

def parse_gguf(path):
    with open(path, 'rb') as f:
        data = f.read()

    model = GGUFModel(data=data)
    pos = 4  # skip magic

    version, pos = read_u32(data, pos)
    tcount, pos = read_u64(data, pos)
    kvcount, pos = read_u64(data, pos)

    for i in range(kvcount):
        key, pos = read_string(data, pos)
        vtype, pos = read_u32(data, pos)  # vtype is u32 in GGUF v3

        if vtype == 8:  # STRING
            slen, pos = read_u64(data, pos)
            val = data[pos:pos+slen].decode('utf-8', errors='replace')
            pos += slen
            model.metadata[key] = val
        elif vtype == 9:  # ARRAY
            atype, pos = read_u32(data, pos)  # atype is u32
            alen, pos = read_u64(data, pos)
            for _ in range(alen):
                if atype == 8:
                    sl, pos = read_u64(data, pos)
                    pos += sl
                else:
                    pos += GGUF_TYPE_SIZES.get(atype, 0)
        elif vtype in (4, 5):
            val, pos = read_u32(data, pos)
            model.metadata[key] = str(val)
        elif vtype in (0, 1):
            model.metadata[key] = str(data[pos]); pos += 1
        elif vtype in (2, 3):
            val = struct.unpack_from('<H', data, pos)[0]; pos += 2
            model.metadata[key] = str(val)
        elif vtype == 6:
            val = struct.unpack_from('<f', data, pos)[0]; pos += 4
            model.metadata[key] = str(val)
        elif vtype in (10, 11):
            val, pos = read_u64(data, pos)
            model.metadata[key] = str(val)
        elif vtype == 12:
            val = struct.unpack_from('<d', data, pos)[0]; pos += 8
            model.metadata[key] = str(val)
        elif vtype == 7:
            model.metadata[key] = str(bool(data[pos])); pos += 1
        else:
            pos += GGUF_TYPE_SIZES.get(vtype, 0)

    for i in range(tcount):
        name, pos = read_string(data, pos)
        ndims, pos = read_u32(data, pos)
        dims = []
        for _ in range(ndims):
            d, pos = read_u64(data, pos)
            dims.append(d)
        ggml_type, pos = read_u32(data, pos)
        offset, pos = read_u64(data, pos)
        model.tensors.append(TensorInfo(name=name, ndims=ndims, dims=dims,
                                         ggml_type=ggml_type, offset=offset))

    model.data_start = ((pos + 31) // 32) * 32
    return model

def find_tensor(tensors, name):
    for t in tensors:
        if t.name == name:
            return t
    raise ValueError(f"Tensor '{name}' not found")

def extract_data(model, tensor):
    start = model.data_start + tensor.offset
    return model.data[start:start + tensor.nbytes]

# ── DreamTree v2 Converter ──────────────────────────────────────────

def convert(input_path, output_path, qkv_blocks=8, ffn_chunk=256):
    print(f"[1/4] Parsing GGUF: {input_path}")
    model = parse_gguf(input_path)

    hidden = int(model.metadata.get('llama.embedding_length',
                model.metadata.get('general.embedding_length', 2048)))
    layers = int(model.metadata.get('llama.block_count',
                model.metadata.get('general.block_count', 22)))
    kv_dim = (int(model.metadata.get('llama.attention.head_count_kv',
                model.metadata.get('general.attention.head_count_kv', 4))) *
              int(model.metadata.get('llama.attention.head_dim',
                model.metadata.get('general.attention.head_dim', 64))))
    ffn_dim = int(model.metadata.get('llama.feed_forward_length',
                model.metadata.get('general.feed_forward_length', 5632)))
    vocab = int(model.metadata.get('general.vocab_size', 32000))

    num_heads = int(model.metadata.get('llama.attention.head_count',
                model.metadata.get('general.attention.head_count', 1)))
    head_dim = int(model.metadata.get('llama.attention.head_dim',
                model.metadata.get('general.attention.head_dim', 64)))

    num_ffn_chunks = math.ceil(ffn_dim / ffn_chunk)

    print(f"  hidden={hidden} layers={layers} kv={kv_dim} ffn={ffn_dim} vocab={vocab}")
    print(f"  num_heads={num_heads} head_dim={head_dim}")
    print(f"  qkv_blocks={qkv_blocks} ffn_chunk={ffn_chunk} num_ffn_chunks={num_ffn_chunks}")

    print(f"[2/4] Extracting tensor data...")
    tensor_data = {}
    for t in model.tensors:
        tensor_data[t.name] = extract_data(model, t)
    print(f"  Extracted {len(tensor_data)} tensors")

    def tensor_elem_sz(tensor):
        """Get byte size per element from GGUF type (for F32/F16 only)."""
        return GGUF_TYPE_SIZES.get(tensor.ggml_type, 2)

    def tensor_bytes(tensor):
        """Get total bytes for a tensor (handles block-based quantized types)."""
        return tensor.nbytes

    # Use the element size from the first weight tensor
    # For quantized types, we use the raw byte layout
    first_weight = find_tensor(model.tensors, 'token_embd.weight')
    elem_sz = tensor_elem_sz(first_weight)
    is_quantized = first_weight.ggml_type in GGUF_BLOCK_TYPES

    if is_quantized:
        # Quantized: compute sizes from actual tensor bytes
        def get_tensor_nbytes(name):
            t = find_tensor(model.tensors, name)
            return t.nbytes

        embd_t = find_tensor(model.tensors, 'token_embd.weight')
        norm_t = find_tensor(model.tensors, 'output_norm.weight')
        out_t = find_tensor(model.tensors, 'output.weight')
        if is_quantized:
            # Globals and output dequantized to F32
            embd_sz = embd_t.nelem * 4  # F32
            output_norm_sz = norm_t.nelem * 4  # F32
            output_sz = out_t.nelem * 4  # F32
        else:
            embd_sz = embd_t.nbytes
            output_norm_sz = norm_t.nbytes
            output_sz = out_t.nbytes
        global_sz = embd_sz + output_norm_sz

        # Per-layer: sum of all tensor bytes
        layer_tensors = []
        for l in range(layers):
            layer_tensors.append(get_tensor_nbytes(f'blk.{l}.attn_norm.weight'))
            layer_tensors.append(get_tensor_nbytes(f'blk.{l}.attn_q.weight'))
            layer_tensors.append(get_tensor_nbytes(f'blk.{l}.attn_k.weight'))
            layer_tensors.append(get_tensor_nbytes(f'blk.{l}.attn_v.weight'))
            layer_tensors.append(get_tensor_nbytes(f'blk.{l}.attn_output.weight'))
            layer_tensors.append(get_tensor_nbytes(f'blk.{l}.ffn_norm.weight'))
            layer_tensors.append(get_tensor_nbytes(f'blk.{l}.ffn_gate.weight'))
            layer_tensors.append(get_tensor_nbytes(f'blk.{l}.ffn_up.weight'))
            layer_tensors.append(get_tensor_nbytes(f'blk.{l}.ffn_down.weight'))
        layer_sz = sum(layer_tensors) // layers

        total_sz = 128 + global_sz + layers * layer_sz + output_sz
    else:
        # F16/F32: compute from element sizes
        norm_sz = hidden * elem_sz
        q_block_sz = hidden * (hidden // qkv_blocks) * elem_sz
        k_block_sz = hidden * (kv_dim // qkv_blocks) * elem_sz
        v_block_sz = k_block_sz
        attn_out_block_sz = hidden * (hidden // qkv_blocks) * elem_sz

        gate_block_sz = hidden * ffn_chunk * elem_sz
        up_block_sz = gate_block_sz
        down_block_sz = ffn_chunk * hidden * elem_sz
        ffn_chunk_sz = gate_block_sz + up_block_sz + down_block_sz

        layer_sz = (norm_sz +
                    qkv_blocks * q_block_sz +
                    qkv_blocks * k_block_sz +
                    qkv_blocks * v_block_sz +
                    qkv_blocks * attn_out_block_sz +
                    norm_sz +
                    num_ffn_chunks * ffn_chunk_sz)

        embd_sz = vocab * hidden * elem_sz
        output_norm_sz = hidden * elem_sz
        output_sz = hidden * vocab * elem_sz
        global_sz = embd_sz + output_norm_sz

        total_sz = 128 + global_sz + layers * layer_sz + output_sz

    print(f"[3/4] Layout:")
    print(f"  Per-layer: {layer_sz:,} bytes ({layer_sz/(1024*1024):.1f} MB)")
    print(f"  Globals: {global_sz:,} bytes ({global_sz/(1024*1024):.1f} MB)")
    print(f"  Total: {total_sz:,} bytes ({total_sz/(1024*1024):.1f} MB)")

    print(f"[4/4] Writing DreamTree v2: {output_path}")

    with open(output_path, 'wb') as out:
        # ── Header (128 bytes) ──
        header = struct.pack('<4sIIIIIIIIIIIqqqqIQ36s',
            b'DRT2',          # magic
            2,                # version
            layers,           # num_layers
            hidden,           # hidden_dim
            kv_dim,           # kv_dim
            ffn_dim,          # ffn_dim
            vocab,            # vocab_size
            qkv_blocks,       # num_qkv_blocks
            ffn_chunk,        # ffn_chunk_size
            num_ffn_chunks,   # num_ffn_chunks
            num_heads,        # num_heads
            head_dim,         # head_dim
            layer_sz,         # layer_size
            global_sz,        # global_size
            total_sz - 128,   # data_size
            total_sz,         # total_file_size
            first_weight.ggml_type,  # quant_type
            output_sz,        # output_weights_sz
            b'\x00' * 36,     # reserved
        )
        out.write(header)
        assert out.tell() == 128

        # ── Global tensors ──
        def get_data(name):
            return tensor_data.get(name, b'\x00' * 1000000)

        if is_quantized:
            # Dequantize token_embd to F32
            embd_raw = get_data('token_embd.weight')
            embd_t = find_tensor(model.tensors, 'token_embd.weight')
            embd_f32 = dequant_tensor(embd_raw, embd_t.ggml_type, embd_t.nelem)
            import array
            out.write(array.array('f', embd_f32).tobytes())
            print(f"    token_embd dequantized: {len(embd_f32)} F32 floats")

            # Dequantize output_norm to F32
            norm_raw = get_data('output_norm.weight')
            norm_t = find_tensor(model.tensors, 'output_norm.weight')
            if norm_t.ggml_type in GGUF_BLOCK_TYPES:
                norm_f32 = dequant_tensor(norm_raw, norm_t.ggml_type, norm_t.nelem)
            else:
                # Already F32 or F16, convert if needed
                if norm_t.ggml_type == 1:
                    norm_f32 = [f16_to_f32(struct.unpack_from('<H', norm_raw, i*2)[0])
                                for i in range(norm_t.nelem)]
                else:
                    norm_f32 = list(struct.unpack(f'<{norm_t.nelem}f', norm_raw[:norm_t.nelem*4]))
            out.write(array.array('f', norm_f32).tobytes())
        else:
            out.write(get_data('token_embd.weight'))
            out.write(get_data('output_norm.weight'))

        print(f"  Globals written at offset 128")

        # ── Layers ──
        for l in range(layers):
            layer_start = out.tell()
            print(f"    Layer {l}/{layers-1}...", end=' ', flush=True)

            # norm1
            out.write(get_data(f'blk.{l}.attn_norm.weight'))

            if is_quantized:
                # Quantized: write raw tensor bytes without slicing
                out.write(get_data(f'blk.{l}.attn_q.weight'))
                out.write(get_data(f'blk.{l}.attn_k.weight'))
                out.write(get_data(f'blk.{l}.attn_v.weight'))
                out.write(get_data(f'blk.{l}.attn_output.weight'))
                out.write(get_data(f'blk.{l}.ffn_norm.weight'))
                out.write(get_data(f'blk.{l}.ffn_gate.weight'))
                out.write(get_data(f'blk.{l}.ffn_up.weight'))
                out.write(get_data(f'blk.{l}.ffn_down.weight'))
            else:
                # F16/F32: slice into GEMV blocks
                q_data = get_data(f'blk.{l}.attn_q.weight')
                q_rows_per_block = hidden // qkv_blocks
                for b in range(qkv_blocks):
                    start = b * q_rows_per_block * hidden * elem_sz
                    end = (b + 1) * q_rows_per_block * hidden * elem_sz
                    out.write(q_data[start:end])

                k_data = get_data(f'blk.{l}.attn_k.weight')
                k_rows_per_block = kv_dim // qkv_blocks
                for b in range(qkv_blocks):
                    start = b * k_rows_per_block * hidden * elem_sz
                    end = (b + 1) * k_rows_per_block * hidden * elem_sz
                    out.write(k_data[start:end])

                v_data = get_data(f'blk.{l}.attn_v.weight')
                for b in range(qkv_blocks):
                    start = b * k_rows_per_block * hidden * elem_sz
                    end = (b + 1) * k_rows_per_block * hidden * elem_sz
                    out.write(v_data[start:end])

                out_data = get_data(f'blk.{l}.attn_output.weight')
                out_rows_per_block = hidden // qkv_blocks
                for b in range(qkv_blocks):
                    start = b * out_rows_per_block * hidden * elem_sz
                    end = (b + 1) * out_rows_per_block * hidden * elem_sz
                    out.write(out_data[start:end])

                out.write(get_data(f'blk.{l}.ffn_norm.weight'))

                gate_data = get_data(f'blk.{l}.ffn_gate.weight')
                up_data = get_data(f'blk.{l}.ffn_up.weight')
                down_data = get_data(f'blk.{l}.ffn_down.weight')

                for m in range(num_ffn_chunks):
                    g_start = m * ffn_chunk * hidden * elem_sz
                    g_end = min((m + 1) * ffn_chunk * hidden * elem_sz, len(gate_data))
                    out.write(gate_data[g_start:g_end])

                    u_start = m * ffn_chunk * hidden * elem_sz
                    u_end = min((m + 1) * ffn_chunk * hidden * elem_sz, len(up_data))
                    out.write(up_data[u_start:u_end])

                    d_start = m * ffn_chunk * hidden * elem_sz
                    d_end = min((m + 1) * ffn_chunk * hidden * elem_sz, len(down_data))
                    out.write(down_data[d_start:d_end])

            actual = out.tell() - layer_start
            print(f"{actual:,} bytes")

        # After all layers: output_norm already written in globals, now output
        if is_quantized:
            # Dequantize output weights to F32
            out_raw = get_data('output.weight')
            out_t = find_tensor(model.tensors, 'output.weight')
            out_f32 = dequant_tensor(out_raw, out_t.ggml_type, out_t.nelem)
            import array
            out.write(array.array('f', out_f32).tobytes())
            print(f"  output.weight dequantized: {len(out_f32)} F32 floats")
        else:
            out.write(get_data('output.weight'))

        final = out.tell()
        print(f"\n  Written: {final:,} bytes ({final/(1024*1024):.1f} MB)")
        print(f"  GGUF was: {os.path.getsize(input_path):,} bytes")
        print(f"  Ratio: {final/os.path.getsize(input_path):.2f}x")

    print("Done!")

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <input.gguf> <output.drt2>")
        sys.exit(1)
    convert(sys.argv[1], sys.argv[2])
