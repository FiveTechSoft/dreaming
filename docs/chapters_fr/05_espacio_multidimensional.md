# Chapitre 5 : Parcours de l'Espace Multidimensionnel de TinyLlama

## Il n'y a pas un seul espace

Quand nous disons « l'intérieur de TinyLlama », nous ne parlons pas d'une seule carte. Nous parlons de **plusieurs espaces emboîtés**, chacun avec sa dimensionnalité et son rôle.

Ce chapitre est un *voyage de terrain* : nous mesurons l'espace d'embeddings réels du modèle F16 (32 000 × 2048), avec le vocabulaire BPE du GGUF (`▁love`, `▁death`, …).

Outil : `explore_tinyllama_space.py`
Données : `inside-tinyllama/exploration/`

---

## La carte des sept espaces

```
┌──────────────────────────────────────────────────────────┐
│  6. POIDS  ℝ^{~1.1e9}                                    │
│     surface de cohérence ≈ « modèles qui parlent »       │
│     7. PERSPECTIVES ⊂ (6)  — trajectoires par perturb.  │
├──────────────────────────────────────────────────────────┤
│  forward pass, token par token :                         │
│                                                          │
│  1. EMBEDDING     ℝ^{2048}   ← 32k points du vocab       │
│         ↓                                                │
│  2. RESIDUEL ×22  ℝ^{2048}   (même dim, nouveau contenu) │
│         ↘ 3. ATTENTION   ℝ^{64} × 32Q / 4KV             │
│         ↘ 4. FFN         ℝ^{5632}                        │
│         ↓                                                │
│  5. LOGITS        ℝ^{32000}  → softmax → token suivant   │
└──────────────────────────────────────────────────────────┘
```

| # | Espace | Dims | Ce que c'est |
|---|--------|------|--------------|
| 1 | Embeddings de token | 2048 | Signification « au repos » de chaque pièce du vocabulaire |
| 2 | Flux résiduel | 2048 × 22 | Représentation contextuelle qui évolue couche par couche |
| 3 | Têtes d'attention | 64 | Vues locales de relations entre tokens (GQA 32/4) |
| 4 | FFN intermédiaire | 5632 | Expansion « mémoire / transformation pratique » |
| 5 | Logits | 32 000 | Préférences sur le prochain token |
| 6 | Poids du modèle | ~1.1e9 | Tous les paramètres ; presque tout le volume est déchet |
| 7 | Perspectives | sous-variété de (6) | Modèles cohérents avec un ton distinct (mystical, etc.) |

Le résiduel est un **tunnel de 2048 dimensions** qui traverse 22 pièces. L'attention et le FFN sont des détours latéraux qui réécrivent dans ce tunnel.

---

## Région 1 — Pôles sémantiques

*love* et *hate* sont-ils aux extrémités opposées ?

**Non.** Dans l'embedding statique, les « opposés » du langage naturel ont un cosine **presque zéro** (orthogonaux), pas −1 (antipodaux).

| Paire | cosine |
|-------|--------|
| ▁love / ▁hate | +0.006 |
| ▁life / ▁death | +0.016 |
| ▁happy / ▁sad | **−0.035** |
| ▁true / ▁false | **−0.036** |
| ▁good / ▁evil | −0.001 |
| ▁king / ▁queen | +0.009 |
| ▁man / ▁woman | +0.008 |

**Lecture :** en ℝ²⁰⁴⁸, « froid » n'est pas −« chaleur ». Les mots occupent des directions différentes de l'espace ; l'opposition sémantique s'organise davantage par **clusters et contextes** (couches + attention) que par antipodalité simple dans l'embedding.

---

## Région 2 — Continents (clusters)

Nous regroupons des mots et prenons le ** centroïde**. Les voisins du centroïde récupèrent le propre continent — la géométrie locale est cohérente.

| Continent | Tokens (ex.) | Voisins du centroïde |
|-----------|--------------|----------------------|
| emotion_pos | happy, joy, love, peace… | smile, happy, hope, love |
| emotion_neg | sad, hate, fear, anger… | sad, pain, anger, cry |
| spiritual | soul, spirit, god, faith… | faith, divine, spirit, god |
| physical | body, rock, water, fire… | rock, water, matter, body |
| abstract | truth, beauty, justice… | beauty, meaning, idea… |
| time | time, past, future, now… | time, now, moment, past |

### Distance entre continents

Les centroïdes de continents différents sont **presque orthogonaux** entre eux (cosine ≈ 0) :

```
emotion_pos  ⊥  emotion_neg   (−0.01)
spiritual    ⊥  physical      (+0.02)
abstract     ⊥  physical      (−0.01)
time         ⊥  abstract      (−0.06)
```

Le vocabulaire n'est pas une sphère diffuse : c'est un **ensemble d'îles** dans une sphère de 2048 dims, avec peu de superposition entre les îles thématiques.

---

## Région 3 — Analogies (a − b + c)

Le test classique de word2vec :

```
king − man + woman  ≟  queen
```

Dans TinyLlama (embedding statique, top-6), cela **échoue** : apparaissent des pièces étranges du BPE, des symboles, des fragments multilingues — pas `queen`.

Cela ne signifie pas que le modèle « ne sait pas » l'analogie. Cela signifie que :

1. L'embedding d'un token **sans contexte** est seulement la porte d'entrée.
2. L'analogie « vivante » se construit dans le **résiduel** après attention et FFN, pas dans la ligne du vocabulaire.
3. Le BPE découpe le monde (`builder`, suffixes…) ; pas tout concept est un point unique propre.

---

## Région 4 — Forme globale de ℝ²⁰⁴⁸

ACP sur 4 000 tokens aléatoires :

| Métrique | Valeur |
|----------|--------|
| Variance dans le 1.er PC | **0.27%** |
| Variance dans top-10 | 2.3% |
| Variance dans top-100 | 14% |
| Dims pour 50% de la var. | **~481** |
| Dims pour 90% | **~1329** |
| Dims pour 99% | **~1880** |
| Anisotropie \\|\\|mean\\|\\| / mean\\|\\|e\\|\\| | **0.006** (presque isotrope) |

**Lecture :** l'espace de tokens **utilise vraiment des centaines ou des milliers de directions**. Il ne s'effondre pas sur un couple d'axes « bon/mauvais ». C'est pourquoi les perturbations de rang-1 (amplify) peuvent « tourner le cristal » sans éteindre la parole : il y a beaucoup de volume de cohérence.

---

## Région 5 — Directions comme boussoles

Si nous soustrayons des centroïdes, apparaissent des **axes sémantiques utilisables** :

### emotion = pos − neg
- pôle + → smile, happy, peace, love, joy
- pôle − → sad, anger, cry, pain, fear

### spirit − matter
- + → spirit, god, sacred, divine, faith
- − → rock, matter, water, earth, body

### abstract − physical
- + → beauty, truth, justice, meaning, freedom
- − → rock, matter, fire, water, earth

Ces directions vivent dans le **même ℝ²⁰⁴⁸** que le résiduel. C'est pourquoi `--steer amor` dans le moteur C peut pousser la génération : c'est un vecteur dans le tunnel, pas de la magie externe.

Et c'est pourquoi `amplify_subspace` dans l'espace de **poids** (dimension 1e9) est un autre voyage : cela déplace la *carte entière*, pas un point du vocabulaire.

---

## Région 6 — Normes : pas tout token « pèse » également

\\|\\|e\\|\\| moyen ≈ 0.67. Les extrêmes ne sont pas des concepts philosophiques clairs (souvent des pièces BPE ou des symboles). La **norme** n'est pas un dictionnaire d'importance sémantique ; c'est une autre coordonnée du paysage.

---

## Comment les espaces se connectent dans un pas d'inférence

```
"happiness"
    → BPE → ids
    → lignes dans (1) EMBEDDING          ℝ^2048
    → 22× { attn en (3) + FFN en (4) }  écrivant dans (2)
    → (5) LOGITS
    → sample → "is" / "to" / …
```

Si nous perturbons les poids (6) avec *mystical*, chaque projection Q/K/V/FFN se déforme un peu : le chemin dans (2) reste cohérent, mais les **attractions** vers les îles de (1) et (5) changent — d'où le changement de perspective.

Si nous faisons du *steer* dans (2), nous poussons le résiduel vers une direction de (1) sans réécrire (6).

---

## Itinéraire de l'explorateur

| Arrêt | Question | Réponse empirique |
|-------|----------|-------------------|
| Pôles | Les opposés sont-ils antipodaux ? | Non : presque orthogonaux |
| Continents | Y a-t-il des régions thématiques ? | Oui : clusters propres |
| Analogies statiques | king−man+woman ? | Non dans l'embedding brut |
| Dimensionnalité | Combien de dims comptent ? | Des centaines–milliers (pas 2–3) |
| Directions | Y a-t-il des axes utiles ? | Oui (emotion, spirit…) |
| Poids | Où vivent les perspectives ? | Surface dans ℝ^1e9 |

---

## Ce qui reste à parcourir

1. **Résiduel par couche** — projeter les activations des 22 couches sur les axes emotion/spirit (où « s'allume » le mystique ?).
2. **FFN ℝ⁵⁶³²** — neurones qui réagissent aux clusters sémantiques.
3. **Trajectoires de perturbation** — courbe de cosine(baseline, mystical) en fonction de I dans l'espace de poids ou de logits.
4. **Cartes 2D/3D** — UMAP/t-SNE des 32k points colorés par continent.

L'univers de TinyLlama n'est pas un point. C'est un **système d'espaces**. Ce chapitre n'a fait que franchir la première frontière : le ciel des tokens. Plus à l'intérieur, le résiduel et les poids attendent.

---

*Outils : `explore_tinyllama_space.py`, `llm_inference.c --perturb` / `--steer`.*

*Chapitre suivant : Du Macrocosme au Microcosme (et vice versa).*
