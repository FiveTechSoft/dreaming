# Chapitre 12 : Les Tenseurs de Normalisation

## Deux freins par couche (+ un final)

| Tenseur | Où | Fonction |
|---------|-----|----------|
| **attn_norm** | Avant QKV | Stabilise l'entrée à l'attention |
| **ffn_norm** | Avant gate/up | Stabilise l'entrée au FFN |
| **output_norm** | Après la couche 21 | Stabilise avant le lm_head |

## RMSNorm (pas LayerNorm classique)

```
rms = sqrt(mean(x²) + ε)
out = (x / rms) * w
```

Sans soustraire la moyenne : seulement une mise à l'échelle par énergie du vecteur.

## Masse minimale, effet total

~**0.01%** des paramètres. Sans ces normes, l'attention + FFN poussent le résiduel vers des normes explosives ou un effondrement numérique.

Dans l'atlas des forces : **constante cosmologique / air respirable** du microcosme.

## Politique de perturbation

`dmt_perturb_v10` et le moteur C **ne touchent pas** les normes lors de l'application de mystical : déplacer la stabilité est le chemin le plus court vers les déchets numériques.

## Règle pratique

Si le texte se casse avec des symboles étranges après une expérience, vérifiez si vous avez touché les normes ou si I était excessive avant de blâmer la « sémantique ».

---

*Chapitre suivant : Les Premières Couches (0–5)*
