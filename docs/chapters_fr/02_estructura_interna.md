# Chapitre 2 : La Structure Interne de TinyLlama

## Les 22 Niveaux (Couches)

TinyLlama possède 22 couches transformantes. Chaque couche est comme un niveau de traitement que l'information doit traverser.

```
Couche 0 :     Entrée → Détection de motifs simples
Couche 1 :     Syntaxe de base
Couches 2-5 :  Relations entre mots adjacents
Couches 6-12 : Concepts abstraits (les « couches d'idées pures »)
Couches 13-20 : Intégration globale
Couche 21 :    Sortie → Génération de tokens
```

## Les 9 Planètes par Niveau (Tenseurs)

Chaque couche possède 9 tenseurs qui travaillent ensemble :

### Tenseurs d'Attention (4 tenseurs, ~19% des paramètres)
- **Query (Q)** : Qu'est-ce que je recherche ?
- **Key (K)** : Qu'ai-je à offrir ?
- **Value (V)** : Quelle information transmets-je ?
- **Output (O)** : Comment intègre-t-out ?

### Tenseurs FFN (3 tenseurs, ~69% des paramètres)
- **Gate (G)** : Quelle information laisse-t-on passer ?
- **Up (U)** : Comment expansionne-t-on l'information ?
- **Down (D)** : Comment compresse-t-on l'information ?

### Tenseurs de Normalisation (2 tenseurs, ~0.01% des paramètres)
- **AttnNorm** : Stabilise l'attention
- **FFNNorm** : Stabilise le réseau feed-forward

## Le Flux d'Information

L'information circule ainsi :

```
Token → Embedding (2048 dimensions)
     → Couche 0 → Couche 1 → ... → Couche 21
     → Prédiction du token suivant
```

Chaque couche transforme la représentation de 2048 dimensions en une nouvelle représentation de 2048 dimensions. La forme est conservée ; le *contenu sémantique* évolue.

## Premier Regard sur les Données

Valeurs lues du GGUF de TinyLlama-1.1B (`llama.*` dans le header du modèle) :

### Paramètres par composant (approx.)
- **FFN** : ~69% (mémoire / connaissance pratique)
- **Attention** : ~19% (connexions entre tokens)
- **Embedding + LM Head** : ~12%
- **Layer Norms** : ~0.01%

### Dimension cachée (`embedding_length`) : 2048
### Nombre de couches (`block_count`) : 22
### Taille du vocabulaire : 32 000 tokens
### Contexte maximum : 2048 tokens
### Têtes d'attention : 32 Q / 4 KV (GQA)
### Dimension par tête : 64
### FFN intermédiaire (`feed_forward_length`) : 5632
### RoPE `freq_base` : 10 000

### Formes logiques des tenseurs (par couche)

```
attn_norm     [2048]
attn_q        [2048, 2048]     # 32 têtes × 64
attn_k        [256,  2048]     #  4 têtes × 64  (GQA)
attn_v        [256,  2048]
attn_output   [2048, 2048]
ffn_norm      [2048]
ffn_gate      [5632, 2048]
ffn_up        [5632, 2048]
ffn_down      [2048, 5632]
```

Plus les globaux :

```
token_embd.weight   [32000, 2048]
output_norm.weight  [2048]
output.weight       [32000, 2048]
```

> **Note sur Q4_0 :** sur disque, un GGUF quantifié affiche des formes « empaquetées » (par exemple `token_embd` comme `[32000, 1152]`). C'est la disposition des blocs de 4 bits, pas la géométrie du modèle. La dimension réelle du vecteur résiduel reste 2048.

## La Structure Hiérarchique

```
Embeddings (géométrie du vocabulaire)
    ↓
Couches 0-5 (syntaxe et voisins locaux)
    ↓
Couches 6-12 (signification plus abstraite)
    ↓
Couches 13-21 (intégration et décision)
    ↓
Sortie (logits → token suivant)
```

## Conclusion

La structure de TinyLlama est élégante et hiérarchique. Chaque composant a un rôle spécifique, et ensemble ils créent un système capable de traiter et de générer du langage.

Avec 22 couches, 9 tenseurs par couche et un résiduel de 2048 dimensions, le modèle est assez petit pour l'ouvrir complètement — et assez riche pour surprendre.

---

*Chapitre suivant : Notre Moteur d'Inférence en C*
