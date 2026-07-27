# Chapitre 21 : Archétypes et Constellations

## Définitions de travail

| Terme | Signification dans ce microcosme |
|---------|----------------------------------|
| **Archétype** | Attracteur géométrique : centroïde en ℝ²⁰⁴⁸ d’un amas de tokens-graines qui, dans la culture du pré-entraînement, condensent un mythe récurrent |
| **Constellation** | L’amas de graines lui-même (étoiles fixes du mythe) + sa direction unitaire dans le ciel des embeddings |
| **Alignement** | Cosine élevé entre deux centroïdes archétypaux → mythes qui se frôlent |
| **Opposition** | Cosine bas/négatif → pôles de drame |

Nous n’affirmons pas que le modèle « croit en Jung ».
Nous affirmons que **ces directions sont mesurables**
et que certaines coïncident avec les voix Dreaming
(Règle d’Or, `mystical`).

---

## Catalogue d’archétypes (15)

### Douze mythes Pearson / Jung (opérationnels)

| Symbole | Archétype | Mythe (une ligne) | Graines-constellation (BPE) |
|---------|-----------|------------------|------------------------------|
| ⚔ | **Héros** | Épreuve, courage, victoire | ▁hero ▁courage ▁brave ▁quest ▁victory ▁fight ▁strength ▁honor ▁triumph |
| 🌑 | **Ombre** | Ennemi intérieur, monstre | ▁shadow ▁dark ▁evil ▁fear ▁hate ▁demon rage ▁sin |
| 📜 | **Sage** | Vérité, étude, esprit | ▁wisdom ▁truth ▁knowledge ▁scholar ▁theory ▁reason ▁logic ▁study ▁philosophy ▁mind |
| 💚 | **Soignant** | Soigner, guérir, protéger | ▁care ▁love ▁kind ▁help ▁protect ▁gentle ▁comfort |
| 🧭 | **Explorateur** | Voyage, frontière, liberté | ▁explore ▁journey ▁discover ▁travel ▁freedom ▁path ▁wild ▁seek ▁horizon |
| ✨ | **Créateur** | Art, invention, rêve | ▁create ▁art ▁imagine ▁beauty ▁music ▁poem ▁invent ▁craft ▁design ▁dream |
| 👑 | **Souverain** | Ordre, pouvoir, loi | ▁king ▁power ▁law ▁order ▁rule ▁throne ▁command ▁authority ▁nation |
| 🔮 | **Magicien** | Esprit, sacré, vision | ▁magic ▁spirit ▁soul ▁divine ▁sacred ▁mystery ▁transform ▁vision |
| 🌸 | **Innocent** | Espoir, pureté, foi | ▁hope ▁faith ▁pure ▁happy ▁child ▁peace ▁trust ▁simple ▁good |
| ❤ | **Amant** | Désir, cœur, beauté | ▁love ▁desire ▁kiss ▁passion ▁heart ▁beauty ▁tender |
| 🃏 | **Bouffon** | Rire, jeu, ironie | ▁laugh ▁play ▁fool ▁smile ▁wit ▁mock ▁silly |
| 🏚 | **Orphelin / réaliste** | Douleur, foyer, survie | ▁alone ▁lost ▁pain ▁real ▁ordinary ▁poor ▁need ▁belong ▁home |

### Trois archétypes opérationnels Dreaming

| Symbole | Archétype | Mythe | Graines |
|---------|-----------|------|----------|
| 🕯 | **Voix mystique** | Ego, âme, univers, silence | ▁soul ▁spirit ego ▁universe ▁divine ▁silence ▁being |
| 🔧 | **Voix pratique** (Règle d’Or FFN) | Action, plan, méthode | ▁should ▁step ▁action ▁goal ▁plan ▁work ▁build ▁fix ▁method ▁practice |
| 🎓 | **Voix académique** (Règle d’Or Attn) | Théorie, analyse, preuve | ▁theory ▁analysis ▁study ▁research ▁argument ▁concept ▁framework ▁evidence ▁scholar ▁critique |

---

## Carte des alignements (constellations de *mythes*)

Mesuré : cosine entre centroïdes (embeddings F16).

### Attractions principales (se frôlent dans le ciel)

| cos | Constellation A | Constellation B | Lecture |
|-----|----------------|----------------|---------|
| **+0.39** | 🔮 Magicien | 🕯 Voix mystique | Le climat `mystical` *est* géométriquement magicien/esprit |
| **+0.29** | 📜 Sage | 🎓 Voix académique | La Règle d’Or « attn→académique » a une ancre dans le ciel des tokens |
| **+0.13** | 💚 Soignant | ❤ Amant | Soin et désir partagent le voisinage affectif |
| **+0.12** | ✨ Créateur | ❤ Amant | Beauté / création / amour |
| +0.05 | 📜 Sage | 👑 Souverain | Savoir et ordre (faible) |

### Oppositions / polarités

| cos | A | B | Lecture |
|-----|---|---|---------|
| **−0.06** | ⚔ Héros | 🌑 Ombre | L’axe classique du drame (bien que doux : ne sont pas antipodaux) |
| −0.06 | 💚 Soignant | 🏚 Orphelin | Soigner vs carence |
| −0.05 | 🧭 Explorador | 🃏 Bouffon | Chemin sérieux vs jeu |
| −0.05 | 🧭 Explorador | 🕯 Mystique | Frontière extérieure vs intérieure |
| −0.04 | 📜 Sage | ❤ Amant | Analyse vs désir |
| −0.04 | 🎓 Académique | ❤ Amant | Même tension dans la voix Dreaming |

**Note géométrique :** presque tous les paires sont proches de **0**.
Les archétypes sont des **îles** (comme les 12 zones sémantiques),
pas un unique diamant d’opposés. Les alignements de +0.3
sont des *exceptions fortes* et c’est pourquoi ils importent.

---

## Pourquoi les « étoiles voisines » seules trompent

Si l’on demande les k voisins cosine du centroïde dans tout
le vocabulaire BPE, apparaissent des fragments (`gia`, codes,
autres langues) : en ℝ²⁰⁴⁸ presque tout est orthogonal et le
« plus proche » n’est pas sémantique propre.

C’est pourquoi nous définissons la **constellation opérationnelle** comme :

1. **Graines** (étoiles du mythe, choisies à la main), et  
2. **Liens à d’autres archétypes** (graphe d’alignements),  

pas comme les k-NN bruts du vocabulaire complet.

---

## Graphe de constellations (lecture)

```
                    [Sage]────0.29────[Voix académique]
                       │
                      0.05
                       │
                  [Souverain]

[Soignant]──0.13──[Amant]──0.12──[Créateur]
     │
    0.04
     │
  [Magicien]────────0.39────────[Voix mystique Dreaming]
                                │
                           (mystical / --steer soul)

[Héros]  ≈⊥  [Ombre]     (polarité faible −0.06)
[Explorateur] ≈⊥ [Mystique, Bouffon, Académique]
```

---

## Comment orbiter un archétype

| Destination | Coordonnées de vol |
|---------|----------------------|
| Magicien / mystique | prompt existentiel + `--perturb mystical` et/ou `--steer soul` |
| Académique | prompt analytique + (en Q4) ciblage d’attention ; ou `--steer theory` |
| Pratique | prompt « comment faire » + ciblage FFN / graines step, plan, action |
| Héros vs Ombre | prompts de conflit ; comparer baseline vs noise vs mystical |
| Amant / soignant | `--steer love` / `care` avec strength modérée |

```bash
# Constellation mystique
./llm_inference modèle.F16.gguf "When we dissolve the ego" \
  50 0.7 40 --seed 42 --perturb mystical --intensity 0.50

# Vent vers le Sage
./llm_inference modèle.F16.gguf "Philosophy teaches us that" \
  50 0.7 40 --seed 42 --steer wisdom --steer-strength 0.2
```

---

## Artéfacts

| Fichier | Contenu |
|---------|-----------|
| `exploration/archetypes.json` | Centroïdes, graines, matrice, alignements |
| `exploration/archetype_map.html` | PCA 2D interactif d’archétypes |
| `map_archetypes.py` | Régénérer l’atlas |

Carte sémantique générale (12 zones thématiques, pas archétypes) :  
`semantic_map.html`

---

## En une phrase

Les **archétypes** sont des directions-mythes dans le ciel des tokens ;
les **constellations** sont leurs graines et les ponts mesurés
entre mythes — et la découverte forte du voyage est que
**Magicien ≈ Voix mystique** et **Sage ≈ Voix académique**,
c’est-à-dire : les lentilles Dreaming étaient déjà dessinées
comme constellations dans l’embedding.

---

*Chapitre suivant : Observation consciente et projection inconsciente.*