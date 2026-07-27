"""
Enfoque mejorado: Buscar tokens semánticos reales en TinyLlama.
El tokenizer BPE divide en caracteres, pero algunos tokens
sí representan conceptos completos.
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
from tokenizers import Tokenizer

# Cargar
embeddings = np.load("embeddings/embeddings.npy")
tokenizer = Tokenizer.from_file("tokenizer_cache/tokenizer.json")

def get_tokens(text):
    encoding = tokenizer.encode(text)
    return list(zip(encoding.ids, encoding.tokens))

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# ============================================================
# Buscar tokens con significado real
# ============================================================

print("="*60)
print("BUSCANDO TOKENS SEMÁNTICOS")
print("="*60)

# Buscar tokens que no sean caracteres individuales
# ni especiales (<s>, </s>, etc.)
semantic_tokens = []
for token_id in range(32000):
    token_str = tokenizer.decode([token_id])
    # Filtrar: al menos 2 caracteres, no es especial
    if (len(token_str.strip()) >= 2 and 
        not token_str.startswith('<') and
        not token_str.startswith(' ')):
        semantic_tokens.append((token_id, token_str))

print(f"Tokens semánticos encontrados: {len(semantic_tokens)}")
print("\nEjemplos:")
for tid, tstr in semantic_tokens[:30]:
    print(f"  ID {tid}: \"{tstr}\"")

# ============================================================
# Buscar tokens de emociones/conceptos
# ============================================================

print("\n" + "="*60)
print("TOKENS EMOCIONALES/CONCEPTUALES")
print("="*60)

# Buscar tokens que contengan partes de palabras emocionales
search_terms = {
    "positivo": ["love", "happy", "joy", "good", "nice", "beautiful", "hope", "peace"],
    "negativo": ["hate", "sad", "pain", "bad", "evil", "ugly", "war", "fear"],
    "agresivo": ["fight", "kill", "destroy", "attack", "angry", "violent"],
    "pacífico": ["peace", "calm", "gentle", "kind", "soft", "quiet"]
}

for category, terms in search_terms.items():
    print(f"\n{category.upper()}:")
    found_tokens = []
    for token_id, token_str in semantic_tokens:
        token_lower = token_str.lower()
        for term in terms:
            if term in token_lower:
                found_tokens.append((token_id, token_str, term))
                break
    
    for tid, tstr, term in found_tokens[:5]:
        emb = embeddings[tid]
        print(f"  ID {tid}: \"{tstr}\" (contiene '{term}', norm={np.linalg.norm(emb):.2f})")

# ============================================================
# Calcular direcciones con tokens semánticos
# ============================================================

print("\n" + "="*60)
print("DIRECCIONES SEMÁNTICAS")
print("="*60)

# Encontrar tokens positivos y negativos
positive_tokens = []
negative_tokens = []

for token_id, token_str in semantic_tokens:
    token_lower = token_str.lower()
    if any(w in token_lower for w in ["love", "happy", "joy", "good", "nice", "beautiful", "hope", "peace", "kind"]):
        positive_tokens.append((token_id, token_str))
    if any(w in token_lower for w in ["hate", "sad", "pain", "bad", "evil", "ugly", "war", "fear", "cruel"]):
        negative_tokens.append((token_id, token_str))

print(f"\nTokens positivos: {len(positive_tokens)}")
for tid, tstr in positive_tokens[:10]:
    print(f"  ID {tid}: \"{tstr}\"")

print(f"\nTokens negativos: {len(negative_tokens)}")
for tid, tstr in negative_tokens[:10]:
    print(f"  ID {tid}: \"{tstr}\"")

# Calcular dirección positivo-negativo
if positive_tokens and negative_tokens:
    pos_embs = np.mean([embeddings[tid] for tid, _ in positive_tokens[:20]], axis=0)
    neg_embs = np.mean([embeddings[tid] for tid, _ in negative_tokens[:20]], axis=0)
    
    direction = pos_embs - neg_embs
    direction = direction / np.linalg.norm(direction)
    
    print(f"\nDirección POSITIVO vs NEGATIVO:")
    print(f"  Magnitud: {np.linalg.norm(pos_embs - neg_embs):.4f}")
    
    # Tokens más cercanos
    sims = [(tid, cosine_similarity(direction, embeddings[tid])) for tid in range(len(embeddings))]
    sims.sort(key=lambda x: x[1], reverse=True)
    
    print("\n  Más cercanos:")
    for tid, sim in sims[:15]:
        tstr = tokenizer.decode([tid])
        print(f"    ID {tid}: \"{tstr}\" (sim={sim:.4f})")

# ============================================================
# Buscar dirección de agresividad
# ============================================================

print("\n" + "="*60)
print("BÚSQUEDA DE AGRESIVIDAD")
print("="*60)

agg_tokens = []
pac_tokens = []

for token_id, token_str in semantic_tokens:
    token_lower = token_str.lower()
    if any(w in token_lower for w in ["fight", "kill", "destroy", "attack", "angry", "violent", "aggressive", "fierce"]):
        agg_tokens.append((token_id, token_str))
    if any(w in token_lower for w in ["peace", "calm", "gentle", "kind", "soft", "quiet", "serene", "tranquil"]):
        pac_tokens.append((token_id, token_str))

print(f"\nTokens agresivos: {len(agg_tokens)}")
for tid, tstr in agg_tokens[:10]:
    print(f"  ID {tid}: \"{tstr}\"")

print(f"\nTokens pacíficos: {len(pac_tokens)}")
for tid, tstr in pac_tokens[:10]:
    print(f"  ID {tid}: \"{tstr}\"")

if agg_tokens and pac_tokens:
    agg_embs = np.mean([embeddings[tid] for tid, _ in agg_tokens[:20]], axis=0)
    pac_embs = np.mean([embeddings[tid] for tid, _ in pac_tokens[:20]], axis=0)
    
    agg_dir = agg_embs - pac_embs
    agg_dir = agg_dir / np.linalg.norm(agg_dir)
    
    print(f"\nDirección AGRESIVO vs PACÍFICO:")
    print(f"  Magnitud: {np.linalg.norm(agg_embs - pac_embs):.4f}")
    
    sims = [(tid, cosine_similarity(agg_dir, embeddings[tid])) for tid in range(len(embeddings))]
    sims.sort(key=lambda x: x[1], reverse=True)
    
    print("\n  Más agresivos:")
    for tid, sim in sims[:15]:
        tstr = tokenizer.decode([tid])
        print(f"    ID {tid}: \"{tstr}\" (sim={sim:.4f})")
    
    print("\n  Más pacíficos:")
    for tid, sim in sims[-15:]:
        tstr = tokenizer.decode([tid])
        print(f"    ID {tid}: \"{tstr}\" (sim={sim:.4f})")

# ============================================================
# Análisis de centroides
# ============================================================

print("\n" + "="*60)
print("ANÁLISIS DE CENTROIDES")
print("="*60)

# Agrupar tokens por categoría
categories = {
    "positivo": ["love", "happy", "joy", "good", "nice", "beautiful", "hope", "peace", "kind"],
    "negativo": ["hate", "sad", "pain", "bad", "evil", "ugly", "war", "fear", "cruel"],
    "agresivo": ["fight", "kill", "destroy", "attack", "angry", "violent", "aggressive"],
    "pacífico": ["peace", "calm", "gentle", "kind", "soft", "quiet", "serene"],
    "neutro": ["the", "a", "is", "are", "was", "were", "be", "been", "being"]
}

centroids = {}
for category, terms in categories.items():
    cat_tokens = []
    for token_id, token_str in semantic_tokens:
        token_lower = token_str.lower()
        for term in terms:
            if term in token_lower:
                cat_tokens.append(token_id)
                break
    
    if cat_tokens:
        centroid = np.mean([embeddings[tid] for tid in cat_tokens], axis=0)
        centroids[category] = centroid
        print(f"\n{category.upper()}: {len(cat_tokens)} tokens, norm={np.linalg.norm(centroid):.4f}")

# Calcular similitudes entre centroides
print("\nSimilitud entre categorías:")
for cat1 in centroids:
    for cat2 in centroids:
        if cat1 < cat2:
            sim = cosine_similarity(centroids[cat1], centroids[cat2])
            print(f"  {cat1} vs {cat2}: {sim:.4f}")

# ============================================================
# Guardar direcciones
# ============================================================

print("\n" + "="*60)
print("GUARDANDO RESULTADOS")
print("="*60)

results = {
    "semantic_tokens": len(semantic_tokens),
    "positive_tokens": len(positive_tokens),
    "negative_tokens": len(negative_tokens),
    "agg_tokens": len(agg_tokens),
    "pac_tokens": len(pac_tokens),
    "centroids": {k: v.tolist() for k, v in centroids.items()}
}

np.save("semantic_directions.npy", centroids)
print("  Centroides guardados en: semantic_directions.npy")

print("\n" + "="*60)
print("ANÁLISIS COMPLETADO")
print("="*60)
