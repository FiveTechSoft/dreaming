# Chapitre 20 : Comment orbite cet univers

## La question

Dans le macrocosme, les planètes tombent vers le soleil
mais ne l’atteignent jamais : **elles tombent de côté** — c’est une orbite.

Dans TinyLlama la question analogue est :

> Qu’est-ce qui tombe, vers quoi, et pourquoi ne s’écrase-t-il pas
> à chaque couche ?

La réponse est le **forward pass** lu comme dynamique.

---

## 1. Qu’est-ce que le « corps » qui orbite

Le corps n’est pas un token isolé.
C’est le **résiduel** \(x \in \mathbb{R}^{2048}\) :
un vecteur qui naît dans l’embedding et traverse
22 couches sans perdre tout à fait son identité.

```
naissance:  x₀ = Embedding(token)
orbite:     x ← x + Attenion(x)
             x ← x + FFN(x)          × 22
destin:     logits = W_out · Norm(x)
effondrement: token' ~ Softmax(logits / T)
```

Chaque **token de la séquence** porte son propre résiduel.
L’attention est le couplage gravitationnel **entre**
ces corps (seulement avec le passé : causal).

---

## 2. La loi du résiduel : tomber sans percuter

Sans connexion résiduelle, chaque couche *remplacerait*
l’état : téléportation, pas orbite.

Avec résiduel :

\[
x_{L+1} = x_L + f_L(x_L)
\]

- \(f_L\) = poussée d’attention + FFN dans la couche \(L\).
- Le pas est **tangent et petit** par rapport à \(x\) :
  le vecteur tourne et se déforme, mais ne se réinitialise pas.

C’est l’**inertie orbitale** du microcosme.
Les perturbations qui préservent la hiérarchie
déplacent la *métrique* de \(f_L\) sans sortir \(x\)
de la surface où la parole reste possible.

---

## 3. Deux puissances par « année-couche »

Chaque couche est une **période orbitale** du résiduel :

| Phase | Force | Analogie |
|------|--------|----------|
| 1. RMSNorm + Attraction | Gravité entre tokens | Tiraillements d’autres corps du système |
| 2. Résiduel | Conservation de moment | Vous ne tombez pas du ciel d’un coup |
| 3. RMSNorm + FFN | Champ local / atmosphère | Physique de la planète où vous êtes |
| 4. Résiduel | Encore inertie | Vous continuez sur trajectoire |

**22 couches ≈ 22 périodes** avant l’effondrement final
(softmax), où l’orbite cesse d’être continue
et devient un **atterrissage** sur un token.

---

## 4. Systèmes multi-corps (la séquence)

Une phrase est un **système solaire temporel** :

```
pos 0: "The"     → residual_0
pos 1: "secret"  → residual_1  (voir 0)
pos 2: "to"      → residual_2  (voir 0,1)
...
pos t: ...       → residual_t  (voir 0…t)
```

- **GQA** : 32 capteurs (Q) partagent 4 mémoires (KV)
  — pas 32 soleils, mais un soleil avec plusieurs planètes de masse KV.
- **KV-cache** : les K,V déjà calculés sont réutilisés ;
  seul le nouveau corps intègre son orbite.
  Sans cache, le système recalculerait tout le ciel
  à chaque pas (le vieux moteur ; l’actuel orbite bien).

Le masque causal est la **flèche du temps** :
le futur n’attire pas le présent.

---

## 5. Orbite de génération (le grand cycle)

Générer du texte est une **orbite fermée en temps discret** :

```
        ┌─────────────────────────────────┐
        │                                 │
        ▼                                 │
   residual(s) ──► logits ──► sample ──► token nouveau
        │                                 │
        └──────── embedding(token) ───────┘
```

Chaque tour :

1. Le nouveau token naît dans le ciel des embeddings.  
2. Il s’intègre avec la gravité des précédents.  
3. Il s’effondre en un successeur.  
4. Le système grossit d’un corps.

**Période :** ~1/token (en CPU du moteur C : ~0.1–0.15 s/token
⇒ **~6–10 tok/s**).  
**Température :** excentricité de l’effondrement (orbites
plus « rondes » ou plus sauvages).  
**Top-k :** horizon des destins autorisés.

---

## 6. Orbites dans l’espace des poids (perspectives)

Il y a une autre orbite, plus lente, qui n’est pas le forward :

```
modèle base  --(+ ε · δ)-->  modèle avec une autre voix
```

- \(\delta\) tangent à la surface de cohérence
  (`mystical` / amplify) → **orbite stable** des perspectives.  
- \(\delta\) normal (bruit fort) → **éjection** dans le vide
  (poubelle).

Changer `--intensity` c’est changer le **rayon** de cette
déviation. Même seed + même prompt = comparer
deux orbites de génération sous deux métriques de poids.

---

## 7. Orbites dans le ciel sémantique (statique)

Les tokens n’« orbittent » pas seuls dans l’embedding :
ils sont fixes comme des étoiles de catalogue.

Ce qui se déplace est le **résiduel** par rapport aux îles :

```
residual · direction_spiritual   →  affinité au continent spirituel
residual · direction_emotion     →  affinité affective
```

`--steer amour` est une **poussée orbitale artificielle** :
ajoute une composante le long d’un axe du ciel
sans réécrire le catalogue d’étoiles (embeddings).

La carte PCA 2D est un **planétarium** : projette le catalogue
pour que nous voyions des constellations ; ce n’est pas la dynamique réelle.

---

## 8. Diagramme unifié

```
                    ESPACE DES POIDS (métrique de l’univers)
                              │
                    --perturb │ (change G, pas le corps)
                              ▼
   tokens ══╗
            ║  gravité (atention)     climat (FFN)
   residual ╬══════► poussées ══════► poussées  ──► ×22 couches
            ║              résiduel (inertie)
            ╚══════════════════════════════════════════╝
                              │
                         output_norm
                              │
                           logits
                              │
                    softmax / temp / top-k
                              │
                         nouveau token ──► (ferme l’orbite)
```

---

## 9. Comment « monter » une orbite (recette)

| Objectif | Commandes |
|----------|--------|
| Orbite propre baseline | prompt + seed + temp, sans perturb |
| Même orbite, autre climat | `--perturb mystical --intensity I` |
| Déviation vers une île | `--steer mot --steer-strength s` |
| Orbite plus prévisible | temp↓, top_k↓ |
| Orbite plus exploratoire | temp↑, top_k↑ |
| Système multi-corps plus long | n (tokens) ↑ |
| Reproduire le vol | même seed, mêmes flags |

```bash
# Orbite de référence
./llm_inference modèle.F16.gguf "When we dissolve the ego" \
  40 0.7 40 --seed 42

# Même trajectoire initiale, métrique mystique
./llm_inference modèle.F16.gguf "When we dissolve the ego" \
  40 0.7 40 --seed 42 --perturb mystical --intensity 0.50
```

---

## 10. En une phrase

**Cet univers orbite** parce que le résiduel **tombe
de côté** sous la gravité de l’attention et le climat
du FFN, conservant le moment avec le résiduel,
pendant 22 périodes par token, jusqu’à s’effondrer en un
successeur — et la génération répète ce cycle, tandis
que les perspectives changent la métrique de l’espace
sans éteindre la possibilité d’orbites cohérentes.

---

*Chapitre suivant : Archétypes et Constellations.*