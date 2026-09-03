#!/usr/bin/env python3
"""
test_drt2.py — End-to-end test for DreamTree v2 format.

1. Generate a tiny GGUF with correct metadata and tensor shapes
2. Convert to DreamTree v2
3. Verify the header and file structure

Usage: python test_drt2.py
"""

import struct
import os
import sys
import math

# ── Step 1: Generate test GGUF ──

def write_u32(f, v):
    f.write(struct.pack('<I', v))

def write_u64(f, v):
    f.write(struct.pack('<Q', v))

def write_string(f, s):
    b = s.encode('utf-8')
    write_u64(f, len(b))
    f.write(b)

def create_test_gguf(path):
    """Create a minimal GGUF with TinyLlama-like tensor names.

    Uses small dimensions that are divisible by 8 (qkv_blocks) and 256 (ffn_chunk):
    HIDDEN=512, KV=64, FFN=1024, VOCAB=1000, LAYERS=2
    """
    HIDDEN = 512
    LAYERS = 2
    KV_DIM = 64      # 1 head × 64 head_dim
    FFN_DIM = 1024
    VOCAB = 1000
    HEAD_DIM = 64
    NUM_KV_HEADS = 1

    tensors = []
    tensors.append(('token_embd.weight', [VOCAB, HIDDEN], 3))  # type 3 = F16
    tensors.append(('output_norm.weight', [HIDDEN], 3))

    for l in range(LAYERS):
        tensors.append((f'blk.{l}.attn_norm.weight', [HIDDEN], 3))
        tensors.append((f'blk.{l}.attn_q.weight', [HIDDEN, HIDDEN], 3))
        tensors.append((f'blk.{l}.attn_k.weight', [KV_DIM, HIDDEN], 3))
        tensors.append((f'blk.{l}.attn_v.weight', [KV_DIM, HIDDEN], 3))
        tensors.append((f'blk.{l}.attn_output.weight', [HIDDEN, HIDDEN], 3))
        tensors.append((f'blk.{l}.ffn_norm.weight', [HIDDEN], 3))
        tensors.append((f'blk.{l}.ffn_gate.weight', [FFN_DIM, HIDDEN], 3))
        tensors.append((f'blk.{l}.ffn_up.weight', [FFN_DIM, HIDDEN], 3))
        tensors.append((f'blk.{l}.ffn_down.weight', [FFN_DIM, HIDDEN], 3))

    tensors.append(('output.weight', [VOCAB, HIDDEN], 3))  # type 3 = F16

    # Metadata
    kv_pairs = [
        ('general.embedding_length', str(HIDDEN), 4),
        ('general.block_count', str(LAYERS), 4),
        ('general.attention.head_count', '1', 4),
        ('general.attention.head_count_kv', str(NUM_KV_HEADS), 4),
        ('general.attention.head_dim', str(HEAD_DIM), 4),
        ('general.feed_forward_length', str(FFN_DIM), 4),
        ('general.vocab_size', str(VOCAB), 4),
    ]

    # Tokenizer vocabulary (STRING_ARRAY, vtype=9)
    # Minimal tokens that the C engine can tokenize
    vocab_tokens = ['<s>', '</s>', '<unk>']
    vocab_tokens += [chr(i) for i in range(ord('a'), ord('z')+1)]  # a-z
    vocab_tokens += [chr(i) for i in range(ord('A'), ord('Z')+1)]  # A-Z
    vocab_tokens += [' ', '.', ',', '!', '?', ':', ';', '-', '_']
    vocab_tokens += [str(i) for i in range(10)]  # 0-9
    # Pad to VOCAB size
    while len(vocab_tokens) < VOCAB:
        vocab_tokens.append(f'<pad{len(vocab_tokens)}>')
    vocab_tokens = vocab_tokens[:VOCAB]

    # Calculate offsets
    data_offset = 0
    tensor_offsets = []
    for name, dims, ttype in tensors:
        nelem = 1
        for d in dims:
            nelem *= d
        nbytes = nelem * 2
        aligned = ((data_offset + 31) // 32) * 32
        tensor_offsets.append(aligned)
        data_offset = aligned + nbytes

    with open(path, 'wb') as f:
        f.write(b'GGUF')
        write_u32(f, 3)
        write_u64(f, len(tensors))
        write_u64(f, len(kv_pairs) + 1)  # +1 for tokenizer array

        for key, val_str, vtype in kv_pairs:
            write_string(f, key)
            write_u64(f, vtype)
            write_u32(f, int(val_str))

        # Write tokenizer.ggml.tokens as STRING_ARRAY
        write_string(f, 'tokenizer.ggml.tokens')
        write_u64(f, 9)   # vtype = ARRAY
        write_u32(f, 8)   # atype = STRING
        write_u64(f, len(vocab_tokens))
        for token in vocab_tokens:
            tb = token.encode('utf-8')
            write_u64(f, len(tb))
            f.write(tb)

        for i, (name, dims, ttype) in enumerate(tensors):
            write_string(f, name)
            write_u32(f, len(dims))
            for d in dims:
                write_u64(f, d)
            write_u32(f, ttype)
            write_u64(f, tensor_offsets[i])

        pos = f.tell()
        aligned = ((pos + 31) // 32) * 32
        f.write(b'\x00' * (aligned - pos))

        import random
        rng = random.Random(42)
        for i, (name, dims, ttype) in enumerate(tensors):
            while f.tell() < tensor_offsets[i]:
                f.write(b'\x00')
            nelem = 1
            for d in dims:
                nelem *= d
            for _ in range(nelem):
                val = rng.gauss(0, 0.1)
                f.write(struct.pack('<e', val))

    print(f"[OK] Created test GGUF: {path} ({os.path.getsize(path)} bytes)")
    return HIDDEN, LAYERS, KV_DIM, FFN_DIM, VOCAB, HEAD_DIM, NUM_KV_HEADS

# ── Step 2: Convert to DreamTree v2 ──

def convert_gguf_to_drt2(gguf_path, drt2_path, qkv_blocks=8, ffn_chunk=256):
    sys.path.insert(0, os.path.dirname(__file__))
    from gguf2dreamtree_v2 import convert
    convert(gguf_path, drt2_path, qkv_blocks=qkv_blocks, ffn_chunk=ffn_chunk)

# ── Step 3: Verify DreamTree v2 file ──

def verify_drt2(drt2_path, hidden, layers, kv_dim, ffn_dim, vocab, qkv_blocks=8, ffn_chunk=256):
    with open(drt2_path, 'rb') as f:
        data = f.read()

    # Parse header
    assert data[0:4] == b'DRT2', f"Bad magic: {data[0:4]}"

    fields = struct.unpack_from('<4sIIIIIIIIIIIqqqq48s', data, 0)
    hdr = {
        'magic': fields[0],
        'version': fields[1],
        'num_layers': fields[2],
        'hidden_dim': fields[3],
        'kv_dim': fields[4],
        'ffn_dim': fields[5],
        'vocab_size': fields[6],
        'num_qkv_blocks': fields[7],
        'ffn_chunk_size': fields[8],
        'num_ffn_chunks': fields[9],
        'num_heads': fields[10],
        'head_dim': fields[11],
        'layer_size': fields[12],
        'global_size': fields[13],
        'data_size': fields[14],
        'total_file_size': fields[15],
    }

    print(f"\n[VERIFY] DreamTree v2 header:")
    for k, v in hdr.items():
        print(f"  {k}: {v}")

    # Verify
    assert hdr['num_layers'] == layers
    assert hdr['hidden_dim'] == hidden
    assert hdr['kv_dim'] == kv_dim
    assert hdr['ffn_dim'] == ffn_dim
    assert hdr['vocab_size'] == vocab
    assert hdr['num_qkv_blocks'] == qkv_blocks
    assert hdr['ffn_chunk_size'] == ffn_chunk

    # Calculate expected sizes — read elem_sz from the GGUF type
    # (type 3 = F16 = 2 bytes per element)
    elem_sz = 2  # Default F16; matches GGUF type 3 used in create_test_gguf
    norm_sz = hidden * elem_sz
    hblock = hidden // qkv_blocks
    kvblock = kv_dim // qkv_blocks
    q_block_sz = hidden * hblock * elem_sz
    k_block_sz = hidden * kvblock * elem_sz
    v_block_sz = k_block_sz
    attn_out_block_sz = hidden * hblock * elem_sz
    gate_block_sz = hidden * ffn_chunk * elem_sz
    ffn_chunk_sz = 3 * gate_block_sz  # gate + up + down

    num_ffn_chunks = math.ceil(ffn_dim / ffn_chunk)
    layer_sz = (norm_sz +
                qkv_blocks * q_block_sz +
                qkv_blocks * k_block_sz +
                qkv_blocks * v_block_sz +
                qkv_blocks * attn_out_block_sz +
                norm_sz +
                num_ffn_chunks * ffn_chunk_sz)

    embd_sz = vocab * hidden * elem_sz
    output_norm_sz = hidden * elem_sz
    output_sz = vocab * hidden * elem_sz
    global_sz = embd_sz + output_norm_sz
    total_sz = 128 + global_sz + layers * layer_sz + output_sz

    print(f"\n[VERIFY] Calculated sizes:")
    print(f"  Layer: {layer_sz:,} bytes")
    print(f"  Globals: {global_sz:,} bytes")
    print(f"  Output: {output_sz:,} bytes")
    print(f"  Total expected: {total_sz:,} bytes")
    print(f"  Actual file: {len(data):,} bytes")

    assert abs(len(data) - total_sz) <= 64, f"Size mismatch: {len(data)} vs {total_sz}"
    assert hdr['layer_size'] == layer_sz
    assert hdr['total_file_size'] == len(data)

    print("\n[OK] DreamTree v2 verification passed!")
    return True

# ── Main ──

if __name__ == '__main__':
    test_gguf = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_model_v2.gguf')
    test_drt2 = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_model_v2.drt2')

    print("=" * 60)
    print("DreamTree v2 Format Test")
    print("=" * 60)

    print("\n[1/3] Creating test GGUF...")
    hidden, layers, kv_dim, ffn_dim, vocab, head_dim, num_kv = create_test_gguf(test_gguf)
    print(f"  hidden={hidden} layers={layers} kv={kv_dim} ffn={ffn_dim} vocab={vocab}")

    print("\n[2/3] Converting to DreamTree v2...")
    convert_gguf_to_drt2(test_gguf, test_drt2, qkv_blocks=8, ffn_chunk=256)

    print("\n[3/3] Verifying DreamTree v2 file...")
    verify_drt2(test_drt2, hidden, layers, kv_dim, ffn_dim, vocab, qkv_blocks=8, ffn_chunk=256)

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
