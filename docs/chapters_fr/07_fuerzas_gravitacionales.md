# Chapitre 7 : Les Forces Gravitationnelles du Microcosme

## Il n'y a pas une seule gravité

Dans l'univers TinyLlama, la « gravité » est un ensemble de **champs** qui courbent les trajectoires de signification. Chacun a une masse (paramètres), une portée et un effet dans le texte.

## Inventaire des forces

| # | Force | Support | Masse | Portée |
|---|-------|---------|-------|--------|
| I | Attraction attentionnelle | Q·K/√d → V | ~19% | Entre tokens de la séquence |
| II | Potentiel FFN | SwiGLU gate/up/down | ~69% | Par token (local) |
| III | Inertie résiduelle | x ← x + f(x) | structure | 22 couches |
| IV | Ancre d'embedding | token_embd, output | ~12% | Condition initiale |
| V | Stabilisation | RMSNorm | ~0.01% | Anti-explosion |
| VI | Effondrement sur le vocabulaire | logits → softmax | tête | 1 de 32k tokens |
| VII | Perspectives | perturbation de poids | tout le modèle | Change le « climat » |
| VIII | Îles sémantiques | géométrie des embeddings | — | Attracteurs statiques |

### Masses mesurées (F16 logique)

| Composant | Paramètres | Part |
|-----------|------------|------|
| FFN | ~761M | **69.2%** |
| Attention | ~208M | **18.9%** |
| Emb + lm_head | ~131M | **11.9%** |
| Normes | ~92k | **0.01%** |

## Force I — Attention

Pas locale : un token ressent d'autres du passé (masque causal). GQA 32 Q / 4 KV : gravité bon marché à mémoriser.

**Règle d'Or :** perturber l'attention → lentille **académique / relationnelle**.

## Force II — FFN

Le « soleil » du système de poids. Transforme chaque position sans regarder les voisins : climat local du résiduel.

**Règle d'Or :** perturber le FFN → lentille **pratique / action**.

## Force III — Résiduel

Conservation de moment du sens. C'est pourquoi les pas tangentiels (`amplify_subspace`) maintiennent la cohérence et le bruit normal à la surface la détruit.

## Force IV et V — Naissance et air

Les embeddings fixent le point de départ en ℝ²⁰⁴⁸ (norme moyenne ≈ 0.68, presque isotrope). Le RMSNorm rend habitables les 22 couches avec une masse minimale.

## Force VI — Softmax

Effondrement du continu sur l'événement : un token. La température et le top-k sont la « dureté » du puits.

## Force VII — Perspectives

Surface de cohérence en ℝ~¹·¹ᵉ⁹. `mystical` = courante tangente ; `noise` fort = sortie dans le vide.

## Force VIII — Constellations

Centroïdes des zones (emotion, spirit, matter, mind…) : presque orthogonaux entre îles. Attraction relative abstract↔mind (+0.13) ; time↔social (−0.09). Love/hate ne sont pas antipodaux : cos ≈ 0.

## Trois lois

1. **Surface** — seules les trajectoires tangentielles dans les poids → texte cohérent.
2. **Deux matières** — l'attention structure les relations ; le FFN transforme le contenu.
3. **Effondrement** — tout finit sur un token.

## Hiérarchie de dominance

```
softmax (destin)
    ↑
attention (longue portée)  +  FFN (masse)
    ↑
résiduel (inertie)
    ↑
embedding (début)  +  norm (stabilité)
    ↑
poids / perspective (métrique de l'univers)
    ↑
îles sémantiques (ciel d'entrée)
```

---

*Chapitre suivant : Comment Voyager dans l'Univers TinyLlama*
