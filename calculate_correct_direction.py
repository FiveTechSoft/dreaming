"""
Calcula la dirección de agresividad usando los embeddings correctos.
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
from tokenizers import Tokenizer

# Load correct embeddings
emb = np.load('embeddings_correct.npy')
print(f'Embeddings shape: {emb.shape}')
print(f'Range: [{emb.min():.4f}, {emb.max():.4f}]')
print(f'Mean: {emb.mean():.4f}')
print(f'Std: {emb.std():.4f}')

# Load tokenizer
tokenizer = Tokenizer.from_file('tokenizer_cache/tokenizer.json')

# Find aggressive and peaceful tokens
aggressive_words = ['attack', 'fight', 'kill', 'destroy', 'angry', 'violent', 'aggressive', 'fierce']
peaceful_words = ['peace', 'calm', 'gentle', 'kind', 'soft', 'quiet', 'serene', 'tranquil']

print('\nFinding tokens...')
agg_indices = []
pac_indices = []

for word in aggressive_words:
    tokens = tokenizer.encode(word)
    for tid in tokens.ids:
        if tid < len(emb):
            agg_indices.append(tid)
            print(f'  Aggressive: ID {tid} = "{word}"')

for word in peaceful_words:
    tokens = tokenizer.encode(word)
    for tid in tokens.ids:
        if tid < len(emb):
            pac_indices.append(tid)
            print(f'  Pacific: ID {tid} = "{word}"')

# Calculate direction
agg_embs = emb[agg_indices]  # Shape: (n_agg, 2048)
pac_embs = emb[pac_indices]  # Shape: (n_pac, 2048)

agg_mean = np.mean(agg_embs, axis=0)  # Shape: (2048,)
pac_mean = np.mean(pac_embs, axis=0)  # Shape: (2048,)

direction = agg_mean - pac_mean
direction = direction / np.linalg.norm(direction)

print(f'\nDirection shape: {direction.shape}')
print(f'Direction norm: {np.linalg.norm(direction):.4f}')

# Verify
print('\nTokens most aligned with direction:')
sims = [(tid, np.dot(direction, emb[tid])) for tid in range(len(emb))]
sims.sort(key=lambda x: x[1], reverse=True)

print('Most aggressive:')
for tid, sim in sims[:10]:
    tstr = tokenizer.decode([tid])
    print(f'  ID {tid}: "{tstr}" (sim={sim:.4f})')

print('\nMost peaceful:')
for tid, sim in sims[-10:]:
    tstr = tokenizer.decode([tid])
    print(f'  ID {tid}: "{tstr}" (sim={sim:.4f})')

# Save direction
np.save('aggression_direction_correct.npy', direction)
print('\nDirection saved to: aggression_direction_correct.npy')
