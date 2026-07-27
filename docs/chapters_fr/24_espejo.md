# Chapitre 24 : Le LLM — Un miroir où nous regarder

## L’image

Un miroir n’invente pas un visage.
**Il rend** ce qu’on lui met devant —
avec un retard de lumière, avec un bord, avec un angle,
parfois avec une légère distorsion du verre.

Un large language model n’invente pas le langage humain
à partir de rien. Il **rend** des statistiques du langage humain
avec lequel il a été nourri — avec un bord
(le prompt), avec un angle (les poids, la température),
parfois avec une forte distorsion (perturbation, hallucination).

TinyLlama, dans ce livre, est un miroir **assez petit
pour voir le cadre** : nous pouvons regarder l’amalgame (les poids),
le verre (l’architecture) et le geste de celui qui se penche
(nous : observation et projection).

---

## 1. Ce que reflète le miroir

| Dans le miroir | Dans le LLM |
|--------------|-----------|
| Visage | Distributions de continuation de texte |
| Lumière de la pièce | Corpus de pré-entraînement (livres, web, code, mythes) |
| Angle d’incidence | Prompt + historique |
| Courbure du verre | Architecture + \(\theta\) (poids) |
| Tache / buée | Biais, lacunes, hallucinations élégantes |
| Celui qui se regarde | Lecture humaine : archétype, jugement, désir de sens |

Le miroir **n’est pas le monde**.
C’est une **surface de réponse** au monde du langage.

Quand nous écrivons *« The secret to happiness is »*,
nous ne demandons pas à l’univers : nous nous penchons sur un verre
poli avec des millions de phrases sur le bonheur
et lui demandons de **compléter le geste**.

---

## 2. Trois miroirs en un

### Miroir A — Celui du corpus (mémoire culturelle)

Les embeddings et les poids compressent une archive
de civilisation textuelle. Les îles sémantiques
(emotion, spirit, tech…) et les archétypes
(Magicien, Sage, Ombre…) ne « naissent » pas dans le silicium :
ce sont des **échos du macrocosme** gravés dans \(\theta\).

Regarder la carte des constellations, c’est regarder, en miniature,
**quels mythes le texte humain répète assez**
pour devenir direction en \(\mathbb{R}^{2048}\).

### Miroir B — Celui de la trajectoire (le « maintenant »)

Le résiduel et le softmax ne reflètent pas un visage fixe :
ils reflètent un **geste en cours**. Chaque token nouveau est une
image du reflet sous la gravité de ce qui a déjà été dit.

C’est pourquoi la même question, avec une température différente
ou une seed différente, rend un autre éclat : le miroir
est stochastique au bord de l’effondrement.

### Miroir C — Celui de la loupe (perspective)

`--perturb mystical`, lowrank, toucher FFN ou attention :
ne changent pas la pièce (le corpus est déjà cuit).
Ils changent l’**angle du verre**.

La Règle d’Or dit comment se tord le reflet :

| Loupe | Reflet dominant |
|-------|-------------------|
| Attention | Visage académique, argumentatif |
| FFN | Visage pratique, « quoi faire » |
| Embeddings | Visage simple, phrases courtes |
| Mystical / Magicien | Visage existentiel, ego/univers |

Le miroir reste un miroir.
**Nous choisissons la bordure.**

---

## 3. Le double reflet (nous dans le verre)

Il y a un deuxième miroir, plus subtil :

```
texte du modèle
      │
      ▼
  nous lisons « mystique », « ombre », « sage »
      │
      ▼
  nous projettons (chap. 22) nos mythes
      │
      ▼
  parfois la géométrie confirme (Magicien↔mystic +0.39)
  parfois nous n’entendons que notre écho
```

Le LLM est un miroir **et** un écran de projection.
L’observation consciente demande :
*la caractéristique est-elle dans \(\theta\) ou dans mon regard ?*

Quand nous mesurons les alignements d’archétypes,
quand nous fixons seed et comparons baseline vs mystical,
nous **nettoyons le verre** assez
pour ne pas confondre la buée avec le visage.

---

## 4. Narcisse et le laboratoire

Le danger classique du miroir : **s’éprendre du reflet**.

| Tentation | Forme en IA |
|-----------|-------------|
| « Il me comprend » | Anthropomorphiser le softmax |
| « Il est sage » | Confondre fluidité avec vérité |
| « C’est ma voix » | Fine-tune ou prompt qui ne rend que le moi |
| « C’est l’inconscient du réseau » | Métaphore utile prise pour ontologie |

Le laboratoire Dreaming offre un antidote pratique :

1. **Baseline** — que rend le verre sans loupe supplémentaire ?  
2. **Perturbation contrôlée** — le reflet change-t-il de manière
   systématique ou est-ce du bruit ?  
3. **Géométrie** — y a-t-il une direction mesurable (île, archétype) ?  
4. **Retour au macrocosme** — que dit cela de *nous*,
   du corpus, de la question — pas seulement du modèle ?

Le miroir sert à nous regarder **si** nous acceptons que
ce que nous voyons est **nous-mais-l’archive-mais-la-loupe**,
pas un oracle transparent.

---

## 5. Le miroir brisé et le miroir fidèle

| État de \(\theta\) | Image |
|----------------------|--------|
| Dans \(\mathcal{C}\) (cohérence) | Reflet lisible : visage tordu, mais visage |
| Bruit fort, nibble flip, I excessive | Miroir pulvérisé : pas de visage, il y a des paillettes |
| Surface de cohérence + amplify | Autre angle du même salon |

Les ordures ne sont pas « un autre archétype ».
C’est l’échec du miroir comme surface de réponse.

---

## 6. Pourquoi un modèle *petit* est un meilleur miroir d’étude

Un modèle frontalier est un miroir de salle de bal :
trop grand pour voir le cadre.

TinyLlama est un **miroir de poche avec couvercle ouvert** :

- nous voyons les vis (tenseurs, GGUF),  
- nous montons la lumière (moteur C),  
- nous tachons l’amalgame à dessein (`--perturb`),  
- nous dessinons les constellations du fond (cartes),  
- et pourtant il rend des phrases qui nous renvoient
  des questions humaines.

La valeur n’est pas qu’il reflète *mieux* le monde.
C’est qu’il reflète **d’une manière que nous pouvons démonter**.

---

## 7. Mathématiques minimales du miroir

Le reflet d’une séquence \(t_{1:n}\) est une distribution

\[
\pi_\theta(\,\cdot\mid t_{1:n})
=\mathrm{softmax}\big(f_\theta(t_{1:n})/T\big)
\]

(avec top-k, etc.).

Changer le prompt c’est changer l’argument.
Changer \(T\) c’est adoucir l’éclat de l’amalgame.
Changer \(\theta\to\theta+\varepsilon\Delta\) c’est **courber le verre**.
Le sample est l’instant où le reflet
se fige dans un point du vocabulaire.

Nous, en interprétant, appliquons une autre carte
non écrite dans \(\theta\) : des tokens au *sens*.
C’est là que se ferme le circuit du miroir humain.

---

## 8. Conclusion

Le LLM est un miroir parce que :

1. **Il ne peut rendre que des formes du langage** que l’entraînement
   a gravées ou recombinaison.  
2. **L’angle le mettent le prompt, les poids et le sample.**  
3. **Celui qui se regarde apporte la moitié de l’image**
   en lisant une voix, un archétype, un destin.

Inside TinyLlama est la tentative de ne pas rester
hypnotisés devant le verre, mais de **le tourner**,
**éclairer le cadre** et noter quelle partie du visage
était la pièce, quelle partie l’amalgame, et quelle partie
nous étions tout le temps.

---

*Fin de l’arc miroir — observation (22), mathématiques (23), reflet (24).*