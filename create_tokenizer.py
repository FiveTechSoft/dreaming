import json, os, random

random.seed(42)
OUTPUT = "C:/tmp/tokenizer.json"
VOCAB_SIZE = 32000

def byte_to_str(b):
    """Llama-style byte representation"""
    if b >= 32 and b <= 126 and b != 34 and b != 92:
        # Printable ASCII (except " and backslash)
        return chr(b)
    elif b == 32:
        return " "
    elif b == 9:
        return "<tab>"
    elif b == 10:
        return "<nl>"
    elif b == 13:
        return "<cr>"
    else:
        return f"<0x{b:02X}>"

# We'll assign IDs sequentially:
# 0-3: special tokens
# 4-259: base bytes (256 bytes)
# 260+: BPE merges

# But first, let's figure out how many unique merge results we can get

# Build merges: we need (VOCAB_SIZE - 260) merge operations to get to 32000
num_merges_needed = VOCAB_SIZE - 260  # 31740

# Create merge pairs from byte combinations
# This generates enough unique pairs to cover our vocab
merge_pairs = []

# Create merges in a structured way
# Round-robin through byte values to ensure variety
for round_num in range(200):  # 200 rounds * ~160 pairs = 32000
    for b1 in range(256):
        for offset in range(0, 256, 4):  # step by 4 to reduce duplicates early
            b2 = (b1 + offset) % 256
            merge_pairs.append((b1, b2))
            if len(merge_pairs) >= num_merges_needed:
                break
        if len(merge_pairs) >= num_merges_needed:
            break
    if len(merge_pairs) >= num_merges_needed:
        break

print(f"Merge pairs generated: {len(merge_pairs)}")

# Now build the vocab dict properly
# We know total unique tokens will be: 260 (special + bytes) + len(merge_pairs) unique merges

token_list = []  # list of (string, id, score) ordered by id

# ID 0-3: special tokens
specials = [("<unk>", 1.0), ("<s>", 1.0), ("</s>", 1.0), ("<pad>", 1.0)]
for s, score in specials:
    token_list.append((s, len(token_list), score))

# ID 4-259: base bytes (256 bytes)
for b in range(256):
    s = byte_to_str(b)
    # Check for duplicates
    if not any(t[0] == s for t in token_list):
        token_list.append((s, len(token_list), 1.0))
    else:
        # Handle duplicate byte string (shouldn't happen with Llama encoding)
        # Use hex variant with suffix
        token_list.append((s + f"_{b}", len(token_list), 1.0))

print(f"After base bytes: {len(token_list)} tokens")

# ID 260+: BPE merged tokens
# Create merged strings from pairs
# In Llama BPE, the token is the concatenation of the two sub-tokens' strings
# We need to look up the strings of the bytes being merged

# Create a lookup: byte_id -> string
byte_str_map = {}
for t_str, t_id, t_score in token_list:
    if t_id < 256:  # byte tokens
        byte_str_map[t_id] = t_str
    elif t_id == 0:
        byte_str_map[0] = t_str  # unk
    # For non-byte tokens, we build them from merges

# Build remaining tokens from merges
# We track what token_strings we've already created to avoid duplicates
existing_strings = set(t[0] for t in token_list)

for b1, b2 in merge_pairs:
    s1 = byte_to_str(b1) if b1 < 256 else byte_str_map.get(b1, f"<b{b1}>")
    s2 = byte_to_str(b2) if b2 < 256 else byte_str_map.get(b2, f"<b{b2}>")

    merged = s1 + s2
    if merged not in existing_strings:
        existing_strings.add(merged)
        token_list.append((merged, len(token_list), max(0.01, 1.0 - len(token_list) / VOCAB_SIZE)))

    # Fill any gaps from duplicates with numbered variants
    while len(token_list) < VOCAB_SIZE:
        # Create filler token
        filler = f"<merge_{len(token_list)}>"
        if filler not in existing_strings:
            existing_strings.add(filler)
            token_list.append((filler, len(token_list), 0.01))
        else:
            # Very unlikely but handle case
            filler = f"<m_{len(token_list)}_{random.randint(0,99999)}>"
            existing_strings.add(filler)
            token_list.append((filler, len(token_list), 0.01))
        if len(token_list) >= VOCAB_SIZE:
            break

    if len(token_list) >= VOCAB_SIZE:
        break

print(f"Total tokens built: {len(token_list)}")
print(f"Last token ID: {token_list[-1][1]}")

# Now build the vocab dict: string -> [id, score]
final_vocab = {}
for token_str, token_id, score in token_list:
    # Make keys unique (if duplicate strings, add numeric suffix)
    base_key = token_str
    suffix = 0
    key = base_key if suffix == 0 else f"{base_key}_{suffix}"
    while key in final_vocab and final_vocab[key][0] != token_id:
        suffix += 1
        key = f"{base_key}_{suffix}"
    final_vocab[key] = [token_id, score]

print(f"Final vocab dict size: {len(final_vocab)}")

# Build the actual merges list for the tokenizer (only unique pairs)
# Format: [b1_string, b2_string]
actual_merges_list = []
added_merge_set = set()
for b1, b2 in merge_pairs:
    s1 = byte_to_str(b1) if b1 < 256 else byte_str_map.get(b1, f"<b{b1}>")
    s2 = byte_to_str(b2) if b2 < 256 else byte_str_map.get(b2, f"<b{b2}>")
    merge_key = (s1, s2)
    if merge_key not in added_merge_set:
        added_merge_set.add(merge_key)
        actual_merges_list.append([s1, s2])

print(f"Actual unique merges: {len(actual_merges_list)}")

# Build tokenizer.json
tokenizer = {
    "version": "1.0",
    "truncation": None,
    "padding": None,
    "added_tokens": [
        {"id": i, "content": s, "single_word": False, "lstrip": False, "rstrip": False, "normalized": False}
        for i, (s, _) in enumerate(specials)
    ],
    "normalizer": {
        "type": "Sequence",
        "normalizers": [{"type": "NFC"}]
    },
    "pre_tokenizer": {
        "type": "MetaCharSequence",
        "sequence": [{"type": "Whitespace"}]
    },
    "post_processor": None,
    "decoder": {"type": "BPE"},
    "model": {
        "type": "BPE",
        "continuing_subword_prefix": "",
        "end_of_word_suffix": "",
        "fuse_unk": False,
        "byte_fallback": True,
        "vocab": final_vocab,
        "merges": ["#version: 0.2"] + actual_merges_list
    },
    "chat_template": (
        "<|begin_of_text|>"
        "<|start_header_id|>system<|end_header_id|>\n"
        "{system_message}<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n"
        "{prompt}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n"
        "{response}<|eot_id|>"
    ),
}

with open(OUTPUT, "w", encoding="utf-8") as f:
    json.dump(tokenizer, f, indent=2, ensure_ascii=False)

fsize = os.path.getsize(OUTPUT)
print(f"\ntokenizer.json creado: {OUTPUT}")
print(f"Tamaño: {fsize:,} bytes ({fsize/1024:.0f} KB)")
print(f"Vocab real: {len(final_vocab)} entries")
print(f"Merges: {len(actual_merges_list)}")

# Show some example tokens
print(f"\nMuestra de tokens:")
items = sorted(final_vocab.items(), key=lambda x: x[1][0])[:15]
for token_str, (tid, score) in items:
    display = repr(token_str)[:50]
    print(f"  id={tid:>5d}  {display:50s}  score={score:.3f}")
print("  ...")
items_last = sorted(final_vocab.items(), key=lambda x: x[1][0])[-5:]
for token_str, (tid, score) in items_last:
    display = repr(token_str)[:50]
    print(f"  id={tid:>5d}  {display:50s}  score={score:.3f}")
