# Chapitre 9 : La Règle d'Or Géométrique

## La découverte

Modifier des **composants différents** du transformer ne produit pas du bruit générique. Cela produit des **perspectives spécifiques et prédictibles**.

| Composant | « Planète » | Perspective émergente |
|-----------|-------------|----------------------|
| **Attention** (Q, K, V, O) | Structure / relations | Académique, critique, formelle |
| **FFN** (gate, up, down) | Vocabulaire / action | Pratique, listes, conseils |
| **Embeddings** | Identité d'entrée | Langage simple et direct |

Nous appelons cela la **Règle d'Or géométrique**.

## Attention → académique

Les tenseurs d'attention connectent les tokens. En les perturbant, le modèle priorise la **structure** : arguments, références, ton formel.

```
Prompt : "The meaning of life is..."
Baseline : "...finding happiness..."
Attn perturbée : "...a fundamental philosophical inquiry
                   debated by scholars for millennia..."
```

## FFN → pratique

Le FFN transforme chaque position (mémoire pratique, ~69% des paramètres). En le touchant, émergent des **verbes d'action** et des étapes concrètes.

```
FFN perturbée : "To find meaning: 1) Identify values,
                 2) Set goals, 3) Take daily action..."
```

## Embeddings → simple

La matrice d'entrée définit la « carte de naissance » de chaque token. La perturber aplatit le registre :

```
Emb perturbés : "Life means living. Be happy. Help others."
```

## Pourquoi c'est « géométrique »

Chaque famille de tenseurs déplace le résiduel dans des **directions différentes** de l'espace de représentation. Ce n'est pas de la magie de noms de fichiers : c'est que l'attention et le FFN implémentent des opérateurs différents sur le même ℝ²⁰⁴⁸.

Le ciblage sélectif (v11) le confirme :

| Ciblage | Effet recherché |
|---------|-----------------|
| `attention_alter` | Fort en attn, doux en FFN |
| `ffn_dream` | Fort en FFN, doux en attn |
| `embedding_shift` | Changement en emb, le reste doux |

## Vérification empirique (résumé)

- 24 modèles, 240 générations, 10 prompts (batterie Dreaming).
- Techniques qui préservent la hiérarchie → cohérence.
- Techniques qui la brisent (noise élevé, nibble flip) → déchet.
- Runtime C : `mystical` sur attn+FFN (pas emb/norm) aligné avec la politique de `dmt_perturb_v10`.

## Comment l'utiliser en voyageant

1. Vous voulez de l'analyse ? → touchez **l'attention**.
2. Vous voulez une checklist ? → touchez le **FFN**.
3. Vous voulez de la prose plain ? → touchez les **embeddings**.
4. Vous voulez un climat existentiel global ? → `mystical` en couches.

La Règle d'Or est le **pont d'échelles** : de la vis d'horloge au climat du monologue.

---

*Chapitre suivant : Les Tenseurs d'Attention*
