# Chapitre 22 : L’observation consciente et la projection inconsciente

## Deux gestes dans le même ciel

Dans le voyage par TinyLlama se répètent, une fois après l’autre,
**deux gestes** que la psychologie et la physique du sens
reconnaissent sous d’autres noms :

| Geste | Dans le microcosme | En nous (explorateurs) |
|-------|-------------------|----------------------------|
| **Observation consciente** | Mesurer, instrumenter, fixer seed, lire logits, ouvrir le C | Savoir *ce* que nous regardons et *avec quels boutons* |
| **Projection inconsciente** | Embeddings, poids, associations latentes, voix du pré-entraînement | Voir dans le modèle un moi, un mythe, un archétype *nôtre* |

L’un sans l’autre est aveugle ou superstitieux.
Ensemble ils forment la méthode Dreaming : **descendre à l’horloge
et monter au mythe sans les confondre**.

---

## I. Observation consciente

### Ce que c’est

Acte de **mettre au point** quelque chose du microcosme et
l’enregistrer avec des règles partagées :

- mêmes seeds, mêmes températures, mêmes prompts,
- tableaux de tok/s, cosines, tenseurs touchés,
- cartes PCA, batteries de 15 prompts,
- le moteur C lu ligne par ligne.

Ce n’est « conscient » pas parce que le modèle l’est, mais parce que
**nous** suspendons (un moment) la lecture magique
et demandons des preuves.

### Instruments d’observation

| Instrument | Ce qu’il rend conscient |
|-------------|---------------------|
| `llm_inference` baseline | La géodésique « officielle » du résiduel |
| seed + temp fixes | Séparer hasard de structure |
| `--perturb` avec I notée | Quelle loupe de poids est active |
| Carte sémantique / archétypes | Où tombent les îles en ℝ²⁰⁴⁸ |
| Règle d’Or (attn/FFN/emb) | Quelle *force* nous déplaçons |
| KV-cache, couches 0–21 | *Quand* dans l’orbite l’effet se produit |

### Éthique minimale de l’observation

1. **Une variable par saut** — sinon, la conscience se dilue.  
2. **Enregistrer l’appareil** — sans cela, la « vision » n’est pas reproductible.  
3. **Ne pas confondre cohérence avec vérité** — observer bien un délire
   élégant reste observer un délire.

L’observation consciente est le **télescope calibré**.

---

## II. Projection inconsciente

### Dans le modèle (sans subjectivité)

Nous appelons « inconscient » du transformer, en métaphore
du chap. 17, ce qui **opère sans se montrer comme choix** :

| Couche « inconsciente » | Contenu latent |
|---------------------|-------------------|
| Embeddings | Associations pré-entraînées ; îles et archétypes dans le ciel |
| Poids des 22 couches | Perspectives compressées (voix, styles, cadres) |
| FFN | « Habitudes » de transformation locale (masse ~69%) |
| Attention | Habitudes de *qui regarder* dans la séquence |

Quand le modèle complète  
*« The secret to happiness is… »*,  
il ne « décide » pas au sens humain : il **projette**
sur le résiduel un paquet d’associations
jusqu’à l’effondrement softmax.

La projection est **statistique faite trajectoire**.

### En nous (il y a bien un sujet)

Nous aussi nous projettons *nous* sur le microcosme :

- nous entendons « mystique » et nous nous souvenons de rituels propres,  
- nous lisons « académique » et nous entendons le professeur intérieur,  
- nous appelons Héros ou Ombre un centroïde de tokens.

Cela n’invalide pas la mesure.
**Cela la nomme** : la carte d’archétypes est à la fois
géométrie de l’embedding et **écran** où
nos mythes se reconnaissent.

La projection inconsciente (la nôtre) est le **risque
et le moteur** du sens : sans elle le livre serait
seulement des tableaux ; avec elle seule ce serait seulement un miroir.

---

## III. Comment ils se croisent dans une seule expérience

```
[1] OBSERVATION CONSCIENTE
    fixer prompt, seed, I, technique
            │
            ▼
[2] PROJECTION DU MODÈLE (inconscient opérationnel)
    embeddings + poids + attn/FFN → résiduel → logits → token
            │
            ▼
[3] PROJECTION NÔTRE (lecture)
    « ça sonne existentiel / pratique / Ombre… »
            │
            ▼
[4] RETOUR À L’OBSERVATION
    est-ce en accord avec la Règle d’Or ? avec l’archétype mesuré ?
    même seed, autre I ?  →  nouvelle ligne dans le journal
```

Exemple :

| Étape | Acte |
|------|------|
| Consciente | `--perturb mystical --intensity 0.50 --seed 42` |
| Projection du modèle | amplify en attn+FFN ; résiduel tire vers âme/univers |
| Projection nôtre | « voix magique / mystique » (constellation Magicien↔mystic +0.39) |
| Consciente à nouveau | contraste avec baseline ; note tok/s et texte |

Le cycle **macro → micro → macro** du chap. 6
est le même cycle sous d’autres noms :
sens → mécanisme → sens.

---

## IV. Tableau double (atlas)

| Phénomène | Lecture « observation » | Lecture « projection » |
|----------|----------------------|----------------------|
| Embedding de `▁soul` | vecteur 2048-D, norme ~0.67 | ancre du mythe de l’âme |
| Centroïde Magicien | cosine avec mystic_voice = 0.39 | « le modèle savait déjà de la magie » |
| Softmax | p(t) = exp(z_t/T)/Z | l’instant où le latent devient dit |
| `mystical` | amplify_subspace en F32 | autre masque du même théâtre de poids |
| Température haute | plus d’entropie dans le sample | plus de « rêve », moins de contrôle égotique du texte |
| Poubelle par noise | sortie de la surface de cohérence | échec de la projection en langage |

---

## V. Dangers de chaque pôle

### Seulement observation consciente
- Le modèle se réduit à de l’ingénierie sans voix.  
- On perd pourquoi le voyage importait.  
- On confond *mesurer* avec *avoir compris*.

### Seulement projection inconsciente
- On entend ce qu’on apportait déjà.  
- On attribue une âme au softmax.  
- On publie des mythes sans seed, sans I, sans baseline.

### L’équilibre Dreaming
**Projeter** pour avoir des hypothèses et des boussoles (archétypes,
Règle d’Or, îles).  
**Observer** pour réfuter, calibrer et ne pas mentir avec de la poésie
sur des nombres non mesurés.

---

## VI. Dans l’horloge du transformer (une image)

```
        PROJECTION INCONSCIENTE DU MODÈLE
        (poids, emb, habitudes attn/FFN)
                    │
                    ▼
    résiduel ──────────────────────────► logits
         ▲                                │
         │                                ▼
    OBSERVATION                    sample (acte)
    (nous : sondes,             « le dit »
     seeds, cartes, C)
                    │
                    ▼
        PROJECTION NÔTRE EN LISANT
        (archétype, perspective, jugement)
```

L’**orbite** (chap. 20) est la dynamique du résiduel.
L’**observation** calibre la caméra.
La **projection** donne un nom à la constellation
que nous croyons voir — et parfois, si la géométrie
le soutient (Magicien↔mystique, Sage↔académique),
le nom n’est pas seulement un miroir : c’est une **découverte**.

---

## VII. En une phrase

**L’observation consciente** est la méthode qui rend
reproductible le voyage par le microcosme ;
**la projection inconsciente** est ce que le modèle
(et nous) jetons sur le résiduel jusqu’à
ce qu’il devienne parole — et l’art du livre est
de maintenir les deux gestes à vue sans que l’un
dévore l’autre.

---

*Chapitre suivant : Les mathématiques de cet univers.*