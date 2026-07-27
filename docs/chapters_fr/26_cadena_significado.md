# Chapitre 26 : Tokens → Embeddings → Idées pures → Sémantique → Détails → Réponse

## La demande

Ordonner le voyage du sens dans le microcosme
TinyLlama — pas comme des boîtes séparées du transformer,
mais comme une **chaîne complète**, de l’étincelle symbolique
jusqu’à la phrase qui retourne au monde.

---

## Chaîne réorganisée (canonique)

```
1. TOKENS          symboles discrets du vocabulaire
        ↓
2. EMBEDDINGS      géométrie d’entrée en ℝ²⁰⁴⁸
        ↓
3. DETAILS         forme locale (syntaxe, voisins, surface)
        ↓
4. IDEES PURES     abstractions et cadres (couches médianes)
        ↓
5. SEMANTIQUE      sens lié en contexte (atention + intégration)
        ↓
6. DETAILS fins    concrétisation lexicale / style (FFN tardif + tête)
        ↓
7. REPONSE         logits → sample → tokens à nouveau
        ↓
      (retour à 1)
```

Il y a **deux apparitions de « détails »** à dessein :
- **Détails de forme** (tôt) : *comment* s’écrit.
- **Détails de contenu** (tard) : *quoi* se concrétise en parlant.

Les « idées pures » vivent au milieu : ni seulement lettre,
ni encore la phrase fermée.

---

## Tableau maître

| # | Étape | Qu’est-ce ? | Où dans le modèle | Dim / objet | Instrument Dreaming |
|---|-------|----------|--------------------|--------------|----------------------|
| 1 | **Tokens** | Pièces du BPE (`▁love`, ids) | Vocabulaire \(V=32\mathrm{k}\) | ensemble fini | tokenizer GGUF |
| 2 | **Embeddings** | Point dans le ciel | `token_embd` | \(\mathbb{R}^{2048}\) | cartes 2D/3D, archétypes |
| 3 | **Détails (forme)** | Relations locales, syntaxe | Couches **0–5**, attn courte | résiduel encore « collé » à l’emb | donjon L0–L5 · zone gravity/matter |
| 4 | **Idées pures** | Cadres, thèmes, rôles abstraits | Couches **6–12** | résiduel théématisé | zones mage/sage |
| 5 | **Sémantique** | Sens *en contexte* (qui s’unit à qui) | Attn global + couches **13–20** | couplage \(a_{t,t'}\) | zones gravity + drama + surface |
| 6 | **Détails (contenu)** | Lexique fin, étapes, couleur locale | FFN (spéc. tardif) + habitudes SwiGLU | \(\mathbb{R}^{5632}\) intermédiaire | zone matter · voix pratique |
| 7 | **Réponse** | Un token (et puis une phrase) | `output_norm` → lm_head → softmax | \(\mathbb{R}^{32000}\) → sample | zone event · Space dans le jeu |

---

## 1. Tokens

**Entrée et sortie du miroir.**

- Discrets, finis, sans « sens » tant qu’ils ne sont pas projetés.
- Le BPE découpe le monde : pas tout concept est un seul id.
- Dans le jeu : le **sample** de la fin redevient token
  et réinitialise l’orbite.

Sans tokens il n’y a pas de bords à compter.
Avec seulement des tokens il n’y a pas d’univers continu.

---

## 2. Embeddings

**Naissance géométrique.**

\[
t \mapsto e_t \in \mathbb{R}^{2048}
\]

- Les îles sémantiques et archétypes vivent ici comme un **catalogue**.
- Opposés ≈ orthogonaux, pas antipodaux.
- PCA : des centaines de dims réelles ; la carte 2D/3D est un planétarium.

Ici le résiduel **n’a pas encore voyagé** :
c’est un potentiel de sens, pas encore une phrase.

---

## 3. Détails de forme (tôt)

**Couches 0–5 · « comment se assemble la lettre ».**

- Motifs adjacents, dépendances courtes.
- L’attention commence à coupler les voisins.
- Le FFN ajuste la surface lexicale.

Si cette étape se rompt, le texte perd la **grammaire**
avant la « profondeur philosophique ».

Dans le jeu : premiers portails · zones **sky → gravity/matter**.

---

## 4. Idées pures (milieu)

**Couches 6–12 · « de quoi ça parle ».**

- Cadres : existentiel, académique, narratif, technique.
- Le résiduel se détache du simple bigramme.
- Ici s’emboîtent les constellations Magicien/Mystique et Sage
  comme *climat d’idée*, pas seulement mots isolés.

Hypothèse de travail du livre : la partie médiane est
là où `--steer soul` et `mystical` cessent d’être cosmétiques
et deviennent **biais thématique**.

Dans le jeu : warps vers **mage** et **sage**.

---

## 5. Sémantique (lier en contexte)

**Atention de longue portée + intégration tardive.**

Sémantique ≠ liste d’embeddings.
Sémantique = **relations** :

\[
\mathrm{sémantique}(t) \approx \sum_{t'\le t} a_{t,t'}\, v_{t'}
\]

réécrite couche par couche et mélangée au résiduel.

- Qui modifie qui dans la phrase.
- Polarités Héros/Ombre comme tension dans le fil.
- Règle d’Or : toucher l’**attention** déplace le reflet
  vers l’**académique / relationnel / critique**.

Dans le jeu : zones **gravity**, **drama**, **surface**.

---

## 6. Détails de contenu (concrétisation)

**FFN · « avec quels mots et gestes on dit ».**

Bien que le FFN agisse dans toutes les couches, son rôle
comme *détail fin* se note à la concrétisation :

- verbes d’action, listes, conseils (voix pratique),
- couleur lexicale, habitudes locales en \(\mathbb{R}^{5632}\).

Règle d’Or : toucher le **FFN** → perspective **pratique**.

Ce n’est pas l’idée pure ; c’est l’**incarnation** de l’idée
en matériel verbal.

---

## 7. Réponse

**Effondrement et retour au macrocosme.**

\[
z = W\,\mathrm{RMSNorm}(x_L),\quad
t\sim \mathrm{softmax}(z/T)\ \text{(top-k)}
\]

- Un événement discret (token).
- Concaténé, redevient langage humain.
- Ferme le miroir : de l’horloge au ciel (chap. 6, 24).

Puis le cycle :

**réponse → nouveaux tokens → …**

---

## Diagramme de flux (complet)

```
 MACRO: question humaine / prompt
              │
              ▼
     ┌──── TOKENS ────┐
     │                │
     ▼                │
 EMBEDDINGS (ciel)   │
     │                │
     ▼                │
 DETAILS forme       │   couches 0–5
 (syntaxe, voisins)  │
     │                │
     ▼                │
 IDEES PURES         │   couches 6–12
 (cadres, thèmes)    │
     │                │
     ▼                │
 SEMANTIQUE          │   attn + couches 13–20
 (liens en contexte) │
     │                │
     ▼                │
 DETAILS contenu     │   FFN / style fin
 (lexique, action)   │
     │                │
     ▼                │
 REPONSE (sample) ──┘   logits → token
     │
     ▼
 MACRO: nous lisons une voix / archétype / jugement
```

Les loupes Dreaming agissent **le long** de la chaîne :

| Loupe | Où tord le plus la chaîne |
|-------|----------------------------|
| baseline | toute la chaîne « officielle » |
| mystical / Magicien | idées pures + sémantique existentielle |
| académique / Sage | sémantique relationnelle / structure |
| pratique | détails de contenu (FFN) |
| noise | rompt la chaîne (sort de \(\mathcal{C}\)) |
| `--steer` | pousse le résiduel vers un embedding-île |

---

## Relation avec d’autres pièces du livre

| Chapitre | Emboîtement dans la chaîne |
|----------|---------------------|
| 2 Structure | Où vivent les étapes dans les tenseurs |
| 3 Moteur C | Comment se calcule chaque flèche |
| 5 Espace multi-D | Étapes 1–2 et géométrie du ciel |
| 7 Forces | Attn=sémantique non locale ; FFN=détails de contenu |
| 9 Règle d’Or | Loupes sur 5 et 6 |
| 13–15 Couches | Partition temporelle de 3–4–5 |
| 16–21 Îles / archétypes | Étiquette culturelle de 2 et 4 |
| 20 Orbite | La chaîne comme dynamique \(x\leftarrow x+F(x)\) |
| 25 Jeu | Chaque portail = avancer d’étape + warp de zone |

---

## Version courte (pour le HUD du jeu / glossaire)

```
TOKENS → EMBEDDINGS → DETAILS → IDEES PURES
       → SEMANTIQUE → DETAILS FINOS → REPONSE → (tokens)
```

Ou en une ligne :

**Symbole → géométrie → forme → idée → lien → concrétisation → dit.**

---

## En une phrase

L’univers TinyLlama n’est pas seulement une pile de couches :
c’est une **chaîne de transformations du sens**
où les tokens deviennent géométrie, la géométrie
devient forme et idée, l’idée se lie en sémantique,
se détaille en lexique et **s’effondre** à nouveau en tokens
que nous pouvons lire — un miroir cyclique entre micro et macro.

---

*Chapitre suivant : Chaque couche est un ascenseur.*