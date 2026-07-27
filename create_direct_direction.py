"""
Crea una dirección de agresividad más directa usando
los embeddings de tokens específicos del modelo.
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
from tokenizers import Tokenizer

# Cargar modelo y tokenizer
print("Cargando tokenizer...")
tokenizer = Tokenizer.from_file("tokenizer_cache/tokenizer.json")

# Cargar embeddings del modelo
print("Cargando embeddings del modelo...")
embeddings = np.load("embeddings/embeddings.npy")
print(f"Embeddings shape: {embeddings.shape}")

# ============================================================
# Buscar tokens específicos
# ============================================================

print("\n" + "="*60)
print("BUSCANDO TOKENS ESPECÍFICOS")
print("="*60)

# Tokens agresivos
aggressive_words = ["attack", "fight", "kill", "destroy", "angry", "violent", "aggressive", "fierce"]
peaceful_words = ["peace", "calm", "gentle", "kind", "soft", "quiet", "serene", "tranquil"]

print("\nTokens agresivos:")
agg_tokens = []
for word in aggressive_words:
    tokens = tokenizer.encode(word)
    for tid in tokens.ids:
        emb = embeddings[tid]
        agg_tokens.append((tid, word, emb))
        print(f"  ID {tid}: \"{word}\" (norm={np.linalg.norm(emb):.4f})")

print("\nTokens pacíficos:")
pac_tokens = []
for word in peaceful_words:
    tokens = tokenizer.encode(word)
    for tid in tokens.ids:
        emb = embeddings[tid]
        pac_tokens.append((tid, word, emb))
        print(f"  ID {tid}: \"{word}\" (norm={np.linalg.norm(emb):.4f})")

# ============================================================
# Calcular dirección
# ============================================================

print("\n" + "="*60)
print("CALCULANDO DIRECCIÓN")
print("="*60)

# Promedio de embeddings agresivos
agg_embs = [emb for _, _, emb in agg_tokens]
agg_mean = np.mean(agg_embs, axis=0)

# Promedio de embeddings pacíficos
pac_embs = [emb for _, _, emb in pac_tokens]
pac_mean = np.mean(pac_embs, axis=0)

# Dirección = agresivo - pacífico
direction = agg_mean - pac_mean
direction = direction / np.linalg.norm(direction)

print(f"Aggressive mean norm: {np.linalg.norm(agg_mean):.4f}")
print(f"Pacific mean norm: {np.linalg.norm(pac_mean):.4f}")
print(f"Direction norm: {np.linalg.norm(direction):.4f}")
print(f"Direction shape: {direction.shape}")

# ============================================================
# Guardar dirección
# ============================================================

print("\n" + "="*60)
print("GUARDANDO DIRECCIÓN")
print("="*60)

# Guardar con la dimensión correcta (1152)
np.save("aggression_direction_v2.npy", direction)
print("Dirección guardada en: aggression_direction_v2.npy")

# ============================================================
# Verificar dirección
# ============================================================

print("\n" + "="*60)
print("VERIFICANDO DIRECCIÓN")
print("="*60)

# Encontrar tokens más cercanos a la dirección
sims = [(tid, np.dot(direction, embeddings[tid])) for tid in range(len(embeddings))]
sims.sort(key=lambda x: x[1], reverse=True)

print("\nTokens más agresivos (según dirección):")
for tid, sim in sims[:15]:
    tstr = tokenizer.decode([tid])
    print(f"  ID {tid}: \"{tstr}\" (sim={sim:.4f})")

print("\nTokens más pacíficos:")
for tid, sim in sims[-15:]:
    tstr = tokenizer.decode([tid])
    print(f"  ID {tid}: \"{tstr}\" (sim={sim:.4f})")

print("\n" + "="*60)
print("DIRECCIÓN CREADA")
print("="*60)
