# Chapitre 28 : Étoiles dans le Ciel, Tokens dans TinyLlama

## La Question de l'Astronome

Vous regardez le ciel la nuit. Vous voyez **des points de lumière**.
Certains se groupent en formes que la culture nomme
(Grande Ourse, Orion, Croix du Sud). Entre deux étoiles
il n'y a pas de fil visible, mais la physique dit qu'elles
s'attirent : **la gravité**. Le voyageur ne se téléporte pas
au hasard : il choisit une étoile, mesure son voisinage et saute
au puits suivant.

TinyLlama possède un ciel analogue.

> **Chaque token du vocabulaire est une étoile
> dans un espace de 2048 dimensions.**
> L'**attention** est la force gravitationnelle entre elles
> lorsque le modèle « pense » une séquence.
> Voyager à travers un LLM, c'est suivre ces attractions
> — sur la carte statique de l'embedding ou dans l'orbite
> vivante du *forward*.

Ce chapitre fixe l'analogie, la lie à la **Force I**
de l'inventaire (chap. 7) et présente un **itinéraire
concret** au sein de TinyLlama-1.1B.

---

## 1. Tableau de Correspondances

| Ciel Nocturne | Univers TinyLlama |
|---------------|-------------------|
| Étoile | Token (pièce BPE du vocabulaire, ~32 000) |
| Position dans la voûte | Vecteur d'embedding \(e_t \in \mathbb{R}^{2048}\) |
| Luminosité apparente | Norme / « présence » du token ; sur la carte, taille et étiquette |
| Constellation | Zone sémantique ou archetype (graines + voisins) |
| Distance angulaire dans le ciel | Cosinus entre embeddings (proche ≈ aligné) |
| Gravité newtonienne | **Attention** : \(Q\cdot K^\top / \sqrt{d}\) → poids sur \(V\) |
| Champ gravitationnel statique (carte de masses) | Géométrie fixe de `token_embd` (atlas PCA) |
| Dynamique en direct (planètes en mouvement) | Résiduels de la séquence + KV-cache, couche par couche |
| Saut entre étoiles | Clic sur une **force** de la carte ; ou le token généré suivant |
| Télescope / catalogue | `semantic_map.html`, moteur C, scripts de géométrie |
| Atmosphère qui déforme la lumière | RMSNorm, température du softmax, lentilles `--perturb` |

Ce n'est pas une poésie vide : chaque ligne a un objet mesurable
dans le dépôt Dreaming.

---

## 2. Le Ciel des Embeddings : 32 000 Étoiles Fixes

À la naissance, chaque token \(t\) est enfoncé dans la voûte :

\[
e_t = \mathrm{Embedding}(t) \in \mathbb{R}^{2048}
\]

Ce ciel est **presque isotrope** (norme moyenne ≈ 0.68)
et, entre des mots pleins de sens différents,
**presque orthogonal** (cosinus ≈ 0). C'est pourquoi les
« îles » sémantiques du chap. 16 sont des constellations
rares : des amas de graines qui se touchent un peu,
entourés d'un fond gris de fragments BPE
(comme de la poussière interstellaire : ce n'est pas vide,
mais ce n'est pas une constellation nommée).

### Constellations = Zones

| Constellation (île) | Étoiles-graines (exemples) |
|---------------------|---------------------------|
| Émotion positive | ▁love, ▁happy, ▁joy, ▁hope… |
| Social / pouvoir | ▁work, ▁king, ▁war, ▁law… |
| Esprit | ▁mind, ▁idea, ▁memory, ▁know… |
| Vie / mort | ▁death, ▁life, ▁born, ▁die… |
| … | (douze îles au total ; chap. 16) |

Dans la **carte PCA 2D**, nous projetons ce ciel de 2048
dimensions sur deux axes simplement pour le regarder avec des yeux
humains. La projection ment un peu — comme une
carte plane ment sur la Terre — mais elle préserve
des voisinages utiles.

---

## 3. L'Attention est la Gravité entre Tokens

### Dans le Macrocosme

Deux masses s'attirent l'une l'autre. La force
décroît avec la distance ; le champ organise les orbites.

### Dans le Microcosme (Force I)

À chaque couche, chaque position \(i\) de la séquence
interroge les positions **passées** \(j \le i\)
(mask causal) :

\[
\mathrm{score}_{ij} = \frac{q_i \cdot k_j}{\sqrt{d_h}},
\quad
\alpha_{ij} = \mathrm{softmax}_j(\mathrm{score}_{ij}),
\quad
z_i = \sum_j \alpha_{ij}\, v_j
\]

- \(q_i\) : « qui suis-je et que cherche-je » (corps qui ressent le champ).
- \(k_j\) : « qui es-tu dans le catalogue » (masse qui annonce sa présence).
- \(\alpha_{ij}\) : **intensité de l'attraction** (combien \(i\) « tombe » vers \(j\)).
- \(v_j\) : ce qui est livré en tombant (contenu transporté).

TinyLlama utilise **GQA** (32 têtes Q, 4 KV) :
plusieurs regards bon marché sur le même ciel de clés.

### Deux Gravités à ne pas Confondre

| Type | Ce que c'est | Quand on le voit |
|------|-------------|-----------------|
| **Gravité statique (île)** | Cosinus entre les lignes de `token_embd` | Carte HTML, forces précalculées entre étoiles de l'atlas |
| **Gravité dynamique (attention)** | Softmax de \(QK^\top\) dans la séquence | Forward réel : le prompt crée un système multi-corps |

La statique est le **catalogue de masses** du ciel.
La dynamique est l'**orbite de ce soir** :
elle dépend des étoiles que vous avez placées dans la séquence
et dans quel ordre (causalité = « seul le passé tire »).

La carte interactive montre la première avec des arcs
dorés : *proxy géométrique* de la Force I et de la
Force VIII (îles). Elle ne remplace pas une carte d'attention
par couche, mais elle enseigne le geste : **focus → forces → saut**.

---

## 4. Voyager : Trois Échelles du Même Geste

### Échelle A — Observatoire (Étoiles Fixes)

1. Vous ouvrez la carte des zones sémantiques.
2. Vous entrez dans une constellation (par ex. *Social / pouvoir*).
3. Vous cliquez sur une étoile (`▁work`).
4. Vous voyez les **principales forces** (top cosinus avec prior d'île).
5. Vous cliquez sur une force et **voyagez** vers l'étoile de destination.
6. Répétez : une chaîne de sauts à travers le ciel.

### Échelle B — Navire en Orbite (Génération)

1. Vous lancez un prompt : vous séquencez avec des étoiles.
2. Le résiduel de chaque position orbite 22 couches
   (attention = couplage ; FFN = météo locale ; résiduel = inertie).
3. Le softmax effondre le ciel vers **une** nouvelle étoile
   (le token suivant).
4. Cette étoile s'ajoute au passé et tire sur celles qui viennent.

### Échelle C — Lentilles et Courants (Changer la Physique)

- `--perturb mystical` : déforme la métrique des puits
  (une autre « constante G » effective ; une autre voix).
- `--steer` : pousse le résiduel vers une direction
  du ciel (courant artificiel).
- Température / top-k : dureté de l'effondrement final
  (puits unique ou brouillard d'étoiles possibles ?).

---

## 5. Exemple Guidé : Voyager au sein de TinyLlama

### 5.1 Préparation

Carte en direct (GitHub Pages) :

https://fivetechsoft.github.io/dreaming/exploration/semantic_map.html

Lien de départ profond (étoile `▁work`, id 664) :

`#/token/664/▁work`

Moteur d'orbite (racine du dépôt) :

```bash
# Exemple Windows PowerShell
$env:OMP_NUM_THREADS = "8"
.\llm_inference.exe tinyllama-1.1b.F16.gguf `
  "The secret of power is" 60 0.7 40 --seed 42
```

### 5.2 Itinéraire dans l'Observatoire (Forces de la Carte)

Nous partons de la constellation **Social / pouvoir**.
Mesuré dans l'atlas Dreaming (cosinus dans ℝ²⁰⁴⁸ entre
embeddings ; classement avec prior d'île et de graines) :

| Saut | Étoile source | Étoile de destination (Force) | Cosinus (approx.) | Lecture |
|-----:|---------------|-------------------------------|------------------:|---------|
| 0 | ▁work | — | — | Focus initial : « travail / œuvre » |
| 1 | ▁work | ▁queen | ~0.05 | Tire vers le pouvoir institutionnel |
| 2 | ▁work | ▁war | ~0.01 | Conflit comme attracteur social |
| 3 | ▁work | ▁law | ~0.01 | Ordre et norme |
| 4 | ▁work | ▁power | ~0.00⁺ | Le nom même du puits |
| 5 | ▁work | ▁king | ~0.00⁺ | Couronne, commandement |

**Comment « voyager » dans l'interface**

1. Clic sur `▁work` (ou entrez dans l'espace *Social* et choisissez la graine).
2. Panneau **Forces gravitationnelles** : liste triée + arcs dorés.
3. Clic sur `#1 ▁queen` → la caméra saute ; `▁queen` est le nouveau focus.
4. Depuis là, *ses* forces sont recalculées (nouveau ciel local).
5. Enchaînez les sauts comme une sauterelle entre les étoiles.

Autres itinéraires utiles du même atlas :

| Route | Chaîne typique de graines | Constellation |
|-------|--------------------------|---------------|
| Affection | ▁happy → ▁smile → ▁love → ▁hope | Émotion positive |
| Cognition | ▁mind → ▁idea → ▁learn → ▁memory | Esprit |
| Seuil | ▁death → ▁life → ▁live → ▁born | Vie / mort |

> **Note d'honnêteté astronomique.**
> Dans ℝ²⁰⁴⁸ presque tout est orthogonal : les cosinus
> « forts » de la carte sont **relatifs au voisinage**,
> pas des attractions newtoniennes de 0.9. Le classement
> priorise l'**île** (constellation) et les **graines**
> pour que le voyage soit lisible, pas du bruit BPE.

### 5.3 Le Même Voyage comme *Prompt* (Orbite Vivante)

L'observatoire vous montre *quelques étoiles se frôlent*.
La navire les met sur une ligne temporelle :

```text
Prompt graine (étoile initiale du système) :
  "Work without law becomes"

Lecture Dreaming :
  ▁work tire déjà, dans le catalogue, vers law / power / king…
  En écrivant "without law", vous forcez le contraste :
  l'attention des couches suivantes devra
  « regarder » work et law en même temps (gravité dynamique).
```

Expérience minimale (même seed, deux lentilles) :

```bash
# Référence — ciel « naturel »
.\llm_inference.exe tinyllama-1.1b.F16.gguf `
  "Work without law becomes" 50 0.7 40 --seed 42

# Lentille mystique — autre métrique de puits (Force VII)
.\llm_inference.exe tinyllama-1.1b.F16.gguf `
  "Work without law becomes" 50 0.7 40 `
  --seed 42 --perturb mystical --intensity 0.35
```

Qu'observer :

1. **Tokens générés** = nouvelles étoiles qui s'allument
   dans la séquence (le chemin de la navire).
2. Si le texte « tombe » vers *power / king / war*,
   vous voyez la gravité sociale du catalogue
   agir dans la dynamique.
3. Avec `mystical`, la même constellation de départ
   peut dévier l'orbite vers une météo existentielle
   (Règle d'Or + surface de cohérence, chap. 4 et 9).

### 5.4 Voyage Court Narré (Histoire d'une Sauterelle)

Imaginez que vous êtes un photon de sens :

1. **Vous décollez** à `▁work` (atlas). Vous voyez des arcs vers
   `queen`, `war`, `law`, `power`, `king`.
2. **Vous sautez** à `▁law`. La constellation reste
   sociale ; l'accent passe de « œuvre » à « norme ».
3. **Vous écrivez** le prompt : *« The law of power is »*.
   Vous ne regardez plus le catalogue : vous **habitez** un système
   multi-corps. Chaque couche re-pèse le passé.
4. **Vous effondrez** en un nouveau token (softmax). Cette
   étoile se fixe dans le ciel de *cette* conversation
   (KV-cache) et tire sur la suivante.
5. Optionnel : vous activez une **lentille** (`mystical`) ou un
   **courant** (`--steer`) et le même décollage
   se termine dans une autre galaxie de style.

C'est voyager à travers un LLM : il n'y a pas de corridor 3D,
il y a **catalogue + forces + effondrement**.

---

## 6. Limites de l'Analogie (Pour ne pas Nous Mentir)

| L'Analogie Réussit | L'Analogie se Brise |
|-------------------|---------------------|
| Tokens = points avec position | Il n'y a pas d'espace euclidien « visuel » réel en 2048-D |
| Regroupements = constellations culturelles du pré-entraînement | Le modèle ne « croit » pas aux mythes ; il mesure les cooccurrences |
| Attention = attraction entre positions | Seulement du passé ; pas symétrique comme Newton |
| Carte de cosinus = champ statique | Ce n'est pas la matrice d'attention d'une couche spécifique |
| Générer = orbiter et effondrer | Le « voyage » de l'utilisateur est lecture ; celui du modèle est algèbre |

L'analogie est un **instrument de navigation**,
pas une théorie physique du silicium. Elle sert si elle vous mène
vers un clic, un cosinus ou un prompt reproductible.

---

## 7. Ponts vers d'Autres Chapitres

| Si vous voulez… | Allez au… |
|-----------------|-----------|
| Inventaire de toutes les forces | Chap. 7 |
| Itinéraires de vol A–E (cli, perturb, steer) | Chap. 8 |
| Îles et carte | Chap. 16 |
| Orbite résiduelle couche par couche | Chap. 20 |
| Archétypes comme constellations mythiques | Chap. 21 |
| Formules (softmax, GQA, cosinus) | Chap. 23 |
| Ascenseur de 22 étages | Chap. 27 |
| Jeu de couches + warp de zone | Chap. 25 · `universe_game.html` |

---

## 8. Conclusion

Le ciel au-dessus de votre tête et le vocabulaire de TinyLlama
partagent un geste : **points, distances, attractions,
sauts**.

- Les **étoiles** du modèle sont des tokens dans ℝ²⁰⁴⁸.
- La **gravité** qui compte en parlant est l'**attention**.
- Le **voyage** consiste à choisir un focus, lire ses forces
  et — sur la carte ou dans le moteur C — se laisser tomber
  dans le puits de signification suivant.

Lorsque vous cliquez sur un token et voyez des arcs dorés vers
d'autres, vous ne regardez pas simplement joli graphe :
vous lisez le catalogue de masses du microcosme.
Lorsque vous lancez un prompt, ces masses cessent d'être
un catalogue et deviennent un **solaire en marche**.
