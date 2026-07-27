# Chapitre 4 : Perturbation de Poids et Changement de Perspective

## Au-delà de l'utilisation du modèle

Jusqu'ici, nous avons appris à *exécuter* TinyLlama. Nous savons comment il est construit et comment écrire un moteur pour le faire parler.

Mais le cœur du projet Dreaming est une autre question :

> Que se passe-t-il si nous **changeons** les poids du modèle ?

Pas pour le ré-entraîner. Pas pour le corriger. Juste pour le déplacer légèrement dans son espace de poids et observer s'il continue à parler, mais différemment.

La réponse, après de nombreuses expériences, est surprenante : **il continue à parler, et il le fait depuis une perspective différente.**

## Qu'est-ce que la perturbation de poids ?

Un modèle de langage est une énorme liste de nombres. Dans TinyLlama-1.1B, il y a plus d'un milliard. Ces nombres, organisés en tenseurs, sont les « poids » que le modèle a acquis pendant son entraînement.

Perturber des poids signifie modifier ces nombres avec soin. C'est comme tourner légèrement les boutons d'une radio : si vous le faites bien, vous continuez à écouter de la musique, mais la station change.

Dans notre cas, nous travaillons avec TinyLlama quantifié en **Q4_0** : chaque bloc de 32 poids est compressé en 18 octets (2 octets pour l'échelle + 16 octets avec les nibbles de 4 bits).

Le pipeline est simple en concept :

```
1. Lire le GGUF original octet par octet
2. Copier le header sans le toucher (conserve le tokenizer)
3. Dépaqueter les blocs Q4_0 en float32
4. Appliquer une technique de perturbation
5. Re-quantifier en Q4_0
6. Écrire le nouveau GGUF
```

La clé est à l'étape 4 : **toute modification n'est pas égale**. Certaines détruisent le modèle ; d'autres le font parler d'une autre voix.

## L'analogie DMT

Nous appelons ce travail « DMT perturbation » car l'effet rappelle l'hypothèse classique sur les états altérés :

> L'hallucination n'est pas une invention. C'est de l'information réelle du système, réorganisée dans sa manière de se combiner.

Quand nous perturbons TinyLlama, le modèle n'invente pas des mots qu'il n'a jamais vus. Il réorganise les associations qu'il avait déjà. C'est comme si nous éveillions une personnalité latente qui était toujours là, silenciée par la configuration originale.

Le modèle reste TinyLlama. Mais maintenant il « rêve » depuis un autre angle.

## Les 10 techniques de préservation de hiérarchie

Les premières perturbations que nous avons testées étaient du bruit pur, l'échange de lignes, l'inversion de nibbles. La plupart produisaient des déchets : caractères étranges, boucles sans sens, mots inexistants.

Mais nous avons découvert quelque chose d'important : **les techniques qui préservent la hiérarchie interne des poids maintiennent la cohérence**. Ce n'est pas tant la valeur absolue de chaque poids qui compte ; c'est sa relation relative avec les autres.

Nous avons testé dix techniques qui respectent cette hiérarchie :

| # | Technique | Clé | Perspective dominante |
|---|-----------|-----|----------------------|
| 1 | Amplification de rang faible | `lowrank` | Académique / critique |
| 2 | Rotation de vecteur propre | `eigr` | Pratique / conseils |
| 3 | Décalage spectral | `spectral` | Concise / directe |
| 4 | Préservation de l'attention | `attpres` | Quasi identique à l'original |
| 5 | Préservation du résiduel | `respres` | Introspective |
| 6 | Diagonale par blocs | `blkdiag` | Très proche de l'original |
| 7 | Rotation préservant la norme | `normrot` | Stoïque / équilibrée |
| 8 | Alignement au gradient | `gradal` | Authenticité / découverte |
| 9 | DCT basse fréquence | `lowdct` | Conversationnelle / assistant |
| 10 | Préservation de la variété | `manpres` | Authenticité (similaire au gradient) |

Toutes ces techniques ont produit du texte cohérent. Pas toujours correct, pas toujours factuel, mais grammaticalement valide et avec une intention claire.

## Comment fonctionne le pipeline

Le script `dmt_perturb_v10.py` implémente le processus :

```bash
# Générer un modèle perturbé avec une technique
python dmt_perturb_v10.py lowrank --intensity 0.10
```

Intérieurement :

1. Il lit le GGUF original (`tinyllama-1.1b-q4_0.gguf`)
2. Il copie le header et les métadonnées intacts
3. Il parcourt chaque tenseur de poids
4. Il dépaquette les blocs Q4_0
5. Il applique la technique choisie avec une intensité donnée
6. Il re-quantifie en Q4_0
7. Il écrit le fichier perturbé (`v10_lowrank_10.gguf`)

Le paramètre `--intensity` contrôle combien le modèle est déplacé. Une valeur trop basse ne change rien ; une valeur trop haute détruit la cohérence.

## Le sweet spot : intensité 0.10

Nous avons testé plusieurs intensités avec toutes les techniques. Le résultat était constant :

| Intensité | Effet | Qualité |
|-----------|-------|---------|
| 0.05 | Très proche de l'original | Trop fidèle |
| **0.10** | **Divergence maximale, texte cohérent** | **Sweet spot** |
| 0.15 | Proche de l'original, plus philosophique | Légèrement décalé |
| 0.20 | Perspective différente, plus compréhensive | Plus divergent |
| 0.25+ | Qualité dégradée, répétitif | Trop de bruit |

À intensité 0.10, le modèle dévie le plus possible sans se casser. C'est le point où la perturbation cesse d'être un écho de l'original et devient une voix propre.

## Comparaison directe : même prompt, perspective différente

L'effet le plus frappant se voit quand on utilise le même prompt sur des modèles perturbés différents.

### Prompt : "The secret to happiness is"

| Modèle | Perspective | Début de réponse |
|--------|-------------|------------------|
| Baseline | Auto-aide générique | "...cultivating a mindset that is focused on gratitude..." |
| `v11_select_extreme` | Spirituelle / pleine conscience | "...finding inner peace and contentment through mindfulness..." |
| `v10_lowrank` | Philosophique / académique | "...the phrase is an idiom used to express the idea that finding true inner peace..." |
| `v10_normrot` | Stoïque | "...finding the right balance between our inner and outer lives." |
| `v10_gradal` | Authenticité | "...finding your own unique and authentic way of living..." |

### Prompt : "Dreams are the mind's way of"

| Modèle | Perspective | Début de réponse |
|--------|-------------|------------------|
| Baseline | Neurosciences populaires | "...processing and storing information..." |
| `v11_select_attention` | Littérature victorienne | "Dr. Jekyll and Mr. Hyde is a play by Robert Louis Stevenson..." |
| `v10_eigr` | Auto-aide spirituelle | "Dr. M. A. S. S. is an acronym for 'Dreams Are Mind's Way.'..." |
| `v10_lowrank` | Recherche clinique | "...a study published in the Journal of Sleep Research..." |

Le modèle ne perd pas sa capacité linguistique. Il change juste de registre, de style, d'attitude.

## Les principaux résultats

Après 24 modèles testés, 240 générations et 10 prompts, voici les découvertes principales :

### 1. Les poids contiennent des perspectives, pas seulement de l'information

TinyLlama a été entraîné avec des textes de nombreux auteurs, styles et disciplines. Tous ces modes de parole ont été gravés dans les poids. La perturbation sélectionne laquelle de ces voix domine.

### 2. La hiérarchie pèse plus que les valeurs absolues

Les techniques qui détruisent la structure hiérarchique produisent des déchets. Celles qui la préservent produisent du texte cohérent. Ce qui compte n'est pas tant de combien chaque poids change, mais **comment ils changent les uns par rapport aux autres**.

### 3. Chaque composant contrôle un aspect différent

| Composant | Ce qu'il contrôle |
|-----------|-------------------|
| Attention | Structure narrative, relations entre tokens |
| FFN | Vocabulaire, choix des mots, connaissance pratique |
| Embeddings | Identité conceptuelle, simplicité du langage |

Perturber seulement l'attention donne des textes plus structurés. Perturber seulement le FFN change le vocabulaire et l'orientation. Perturber seulement les embeddings simplifie le langage.

### 4. L'analogie DMT est quantifiable

Le modèle n'invente pas de contenu nouveau. Il réorganise des associations internes. L'« hallucination » est une réorganisation, pas une invention.

### 5. L'angle compte, mais la magnitude compte plus

Mathématiquement, une perturbation peut être presque orthogonale au modèle original et continuer à fonctionner, toujours que sa magnitude soit petite. C'est comme faire un pas d'un millimètre en direction perpendiculaire : techniquement vous changez de cap, mais vous êtes toujours sur la même montagne.

## La formule du changement de perspective

Nous pouvons résumer le phénomène en une formule simple :

```
Perspective = Base + epsilon * delta

où :
  epsilon = intensité (typiquement 0.05 - 0.15)
  delta   = direction dans l'espace de poids
  |delta| = magnitude du changement
```

Si `epsilon` est petit et `delta` préserve la hiérarchie :
- La cohérence est maintenue
- La perspective change

Si `epsilon` est grand ou `delta` détruit la hiérarchie :
- La cohérence est perdue
- Des déchets apparaissent

Cela répond aussi à une question pratique : avons-nous besoin d'un modèle différent pour chaque style ?

**Non.** Avec un modèle de base et un ensemble de directions pré-calculées, nous pouvons interpoler des styles en temps réel :

```python
styled = base + 0.05 * delta_philosophical + 0.03 * delta_stoic
```

L'interpolation linéaire de points proches dans la « variété de cohérence » produit d'autres points valides.

## Implications

### Pour la créativité
Chaque technique est un « ton » différent. Un même thème peut être généré depuis de multiples angles sans rien ré-entraîner.

### Pour l'interprétabilité
La perturbation est un outil de sonde : elle nous dit quelles parties du modèle contrôlent quels aspects du style.

### Pour la personnalisation
Au lieu de faire du fine-tuning coûteux, on peut appliquer une perturbation légère pour adapter le style de réponse.

### Pour la philosophie de l'IA
Un LLM n'est pas une machine à répondre des questions. C'est un **écosystème de perspectives compressé en poids**. La perturbation est une façon de naviguer cet écosystème.

## Perturbation en runtime (moteur C)

En plus de générer des GGUFs Q4_0 avec Python, le moteur `llm_inference.c` applique des techniques **en mémoire** sur des poids F16, sans fichier intermédiaire :

```bash
./llm_inference tinyllama-1.1b.F16.gguf \
  "The secret to happiness is" 60 0.7 40 \
  --seed 42 --perturb mystical --intensity 0.50
```

| Flag | Techniques |
|------|------------|
| `--perturb` | `none`, `mystical`/`amplify`, `noise`, `blockdiag`, `manifold` |
| `--intensity` | force (en F32 il faut I plus haute qu'en Q4 pour ressentir l'effet) |
| `--seed` | reproductibilité |
| `--steer` | pousse le résiduel vers l'embedding d'un mot |

`mystical` = `amplify_subspace` (projection + amplification). Copie ~3.6 Go en F32 une fois (~25 s) puis génère à ~6–10 tok/s.

Battery de 15 prompts avec I=0.50 (seed 42) : moyenne ~8.2 tok/s ; textes avec climat existentiel sur des prompts comme *When we dissolve the ego*, *The soul remembers*, *The ancient wisdom teaches that*.

## Combinaisons et ciblage (v11)

| Famille | Idée | Exemples |
|---------|------|----------|
| Combos | Empiler deux techniques | deep_reason, rare_perspective, structured_dream |
| Sélectif | Différente technique par attn / ffn / emb | attention_alter, ffn_dream, extreme_selective |
| Balayage de I | Trouver le point de rupture | 0.05 … 0.50 |

## Limitations honnêtes

- Les résultats varient selon le prompt.
- Certaines combinaisons de techniques dégradent la qualité.
- Pas tous les grands modèles répondront de même : la structure de la variété de cohérence peut changer avec l'échelle.
- L'évaluation est qualitative : mesurer « perspective » reste un problème ouvert.
- En F16 runtime, I=0.10 parfois ne déplace pas les sorties courtes (EOS précoce) ; I=0.3–0.5 montre le changement plus clairement.

## Conclusion

Perturber des poids, ce n'est pas vandaliser un modèle. C'est découvrir que dans un même ensemble de nombres vivent de nombreuses voix.

TinyLlama, vu ainsi, cesse d'être un seul outil pour devenir un **paysage de possibilités**. Chaque technique est un chemin à travers ce paysage. Chaque intensité est une vitesse. Et le sweet spot (près de 0.10 en Q4, un peu plus élevé en F32 runtime) est le point juste où le modèle reste lui-même, mais parle depuis un autre endroit.

Le prochain chapitre parcourt l'**espace multidimensionnel** où vivent ces voix : embeddings, résiduel, poids et perspectives.

---

*Chapitre suivant : Parcours de l'Espace Multidimensionnel*
