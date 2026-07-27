# Chapitre 27 : Chaque couche est un ascenseur

## L’image

Un bâtiment a des étages.
Vous ne marchez pas du 3e au 17e par les airs :
vous entrez dans un **ascenseur**, les portes se ferment,
et en s’ouvrant le monde est autre — même tour,
autre niveau de l’univers.

Dans TinyLlama le tour a **22 étages**
(plus le vestibule d’embeddings et le toit du softmax).

Chaque couche \(\ell\) est un **ascenseur** :

```
portes se ferment :   RMSNorm
voyage :              Attention + résiduel + FFN + résiduel
portes s’ouvrent :    résiduel transformé au « étage » ℓ+1
```

Vous ne vous téléportez pas hors du bâtiment.
Vous vous **élevez** à l’intérieur du même résiduel \(x\in\mathbb{R}^{2048}\),
mais le **paysage** (zone de l’univers) change.

---

## 1. Le bâtiment TinyLlama

```
        ┌─────────────────────────────┐
   Ω    │  TOIT · Softmax / Sample    │  ← réponse (token)
        ├─────────────────────────────┤
  21    │  étage 21 · prép. effondrement
  20    │  …                          │
   ⋮    │  INTEGRATION / SEMANTIQUE   │  ← liens, drame, 𝒞
  13    │  …                          │
        ├─────────────────────────────┤
  12    │  …                          │
   ⋮    │  IDEES PURES                │  ← cadres, magicien, sage
   6    │  …                          │
        ├─────────────────────────────┤
   5    │  …                          │
   ⋮    │  DETAILS DE FORME           │  ← syntaxe, voisins
   0    │  étage 0 · entrée           │
        ├─────────────────────────────┤
  −1    │  VESTIBULE · Embeddings     │  ← ciel de tokens
        └─────────────────────────────┘
                 ▲
            prompt / tokens
```

Chaque flèche verticale est un ascenseur \(F_\ell\) :

\[
x_{\ell+1} = x_\ell + F_\ell(x_\ell;\theta_\ell)
\]

Le passager est toujours du même type d’objet
(un vecteur de 2048 dims). Le **niveau de l’univers**
est ce que ce vecteur *signifie* après le voyage.

---

## 2. Un voyage dans l’ascenseur (à l’intérieur)

À chaque étage \(\ell\) :

| Moment | Opération | Analogie de l’ascenseur |
|---------|-----------|------------------------|
| 1 | `attn_norm` | Lumières de cabine ; le plancher se stabilise |
| 2 | Q, K, V + RoPE | Capteurs : qui vous sentez dans le bâtiment |
| 3 | Softmax causal | Attraction seulement vers étages/passagers déjà présents (passé) |
| 4 | \(x \mathrel{+}= \mathrm{Attn}\) | La poussée de la gravité sociale du texte |
| 5 | `ffn_norm` | Autre calibration |
| 6 | SwiGLU FFN | Climat de l’étage (matière locale) |
| 7 | \(x \mathrel{+}= \mathrm{FFN}\) | Vous sortez au palier avec un autre air |

Les portes de l’ascenseur ne vous laissent pas dans un vecteur
d’autre dimension : vous sortez à **un autre palier du même
couloir de 2048**, mais le « quartier » a changé.

---

## 3. Étage ↔ niveau de l’univers

Ce n’est pas seulement un nombre \(\ell\). Chaque tronçon d’étages
correspond à un **niveau de l’atlas** (chaîne du chap. 26
+ zones du jeu) :

| Étages (couches) | Niveau de l’univers | Chaîne du sens |
|-----------------|--------------------|------------------------|
| Vestibule | Ciel de tokens / îles | Tokens → Embeddings |
| 0 – 5 | Quartier de la forme | Détails de forme |
| 6 – 12 | Quartier des idées pures | Idées pures (Magicien, Sage…) |
| 13 – 20 | Quartier de la sémantique liée | Sémantique + drame + \(\mathcal{C}\) |
| 21 | Avant-toit | Détails fins / prép. réponse |
| Softmax | Toit · effondrement | Réponse → nouveau token |

Le jeu (`universe_game.html`) rend explicite ce que
le forward fait en silence :

> **Monter d’étage = prendre l’ascenseur de la couche**  
> **et en même temps atterrir dans une autre zone de la carte de l’univers.**

---

## 4. Pourquoi « ascenseur » et non « tunnel infini » ?

Un tunnel suggère un seul paysage allongé.
Un ascenseur insiste sur trois faits :

1. **Même tour** — la dimension du résiduel ne change pas (\(d=2048\)).  
2. **Arrêts discrets** — 22 applications \(F_\ell\), pas un flux continu anonyme.  
3. **Mondes différents par étage** — syntaxe ≠ idée pure ≠ effondrement au vocabulaire.

Le KV-cache est la **mémoire du bâtiment** :
les passagers des étages temporels précédents
(restent là comme K, V) tirent de vous à chaque arrêt.

---

## 5. Boutonnière de l’ascenseur (commandes Dreaming)

| Bouton | Effet |
|-------|--------|
| Prompt | Dans quel vestibule vous entrez (quel embedding initial) |
| Seed / temp / top-k | Comment se choisit le destin au toit |
| `--perturb mystical` | Change la **mécanique de tous les ascenseurs** (métrique de \(F_\ell\)) |
| `--steer soul` | Vent dans la cabine (pousse \(x\) vers un axe) |
| Loupe académique / pratique | Biais vers les boutons d’attention ou de FFN (Règle d’Or) |

Vous ne choisissez pas seulement l’étage 7.
Vous choisissez **comment se comporte l’ascenseur** dans tous les étages.

---

## 6. Une montée complète (narrée)

1. **Vestibule** — vous naîssez en tant que \(e_t\) ; près des îles love/tech/spirit.  
2. **Ascenseurs 0–5** — vous ordonnent les vêtements (forme, voisins).  
3. **Ascenseurs 6–12** — le couloir se remplit d’idées : magicien, sage, cadre.  
4. **Ascenseurs 13–20** — les idées se *lient* (sémantique, tension, cohérence).  
5. **Ascenseur 21 + toit** — l’univers refuse de rester en continu :
   il s’effondre en un token.  
6. **Redémarrage** — ce token retourne au vestibule ; nouvelle montée.

C’est **orbiter** (chap. 20) lu comme **ascenseur en boucle**.

---

## 7. Mathématiques minimales

Ascenseur de l’étage \(\ell\) :

\[
\begin{aligned}
h &= \mathrm{RMSNorm}(x_\ell; w_a^{(\ell)}) \\
x' &= x_\ell + \mathrm{Attn}_\ell(h) \\
h' &= \mathrm{RMSNorm}(x'; w_f^{(\ell)}) \\
x_{\ell+1} &= x' + \mathrm{FFN}_\ell(h')
\end{aligned}
\]

Téléportation de *zone* (dans le jeu / dans la lecture) :
ce n’est pas un opérateur supplémentaire du GGUF ; c’est l’**étiquette
de l’atlas** que nous mettons au palier \(\ell\)
(sky, gravity, matter, mage, sage, surface, event…).

---

## 8. En une phrase

Chaque couche est un **ascenseur** : le résiduel entre,
se laisse pousser par la gravité attentionnelle et le climat FFN,
et en s’ouvrant les portes il est à **un autre niveau de l’univers
TinyLlama** — même dimension, autre hauteur de sens —
jusqu’au toit où le softmax choisit le destin suivant
et rappelle à nouveau l’ascenseur.

---

*Jeu : portail = monter d’étage + warp de zone.*  
*Chaîne : chap. 26 · Orbite : chap. 20 · Forces : chap. 7.*