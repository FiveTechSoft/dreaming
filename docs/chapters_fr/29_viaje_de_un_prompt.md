# Chapitre 29 : Le Voyage d'un Prompt au sein de TinyLlama

## Pour Qui est ce Chapitre

Si vous avez lu qu'un LLM « prédit le prochain token »
mais que vous ne *voyez* pas encore le chemin, ce chapitre est la carte
complète des routes.

Nous le parcourerons **étape par étape**, avec un prompt réel,
sans supposer aucun saut magique. À la fin,
vous devriez pouvoir narrer, à voix haute, ce qui arrive
à chaque nombre depuis que vous écrivez une phrase jusqu'à
l'apparition du premier mot de la réponse.

**Prompt d'exemple (fixe tout au long du chapitre) :**

```text
The secret to happiness is
```

**Modèle :** TinyLlama-1.1B
**Chiffres clés qui ne changent pas :**

| Paramètre | Valeur |
|-----------|-------:|
| Couches transformer | 22 (indices 0…21) |
| Dimension résiduelle \(d\) | 2048 |
| Vocabulaire \(V\) | 32 000 |
| Têtes Q / KV | 32 / 4 (GQA) |
| Dimension par tête | 64 |
| FFN intermédiaire | 5632 |
| Contexte maximum | 2048 positions |
| RoPE base | 10 000 |

---

## 0. Le Film en Une Minute

Avant le détail, la bande-annonce :

```
1.  TEXTE         "The secret to happiness is"
2.  TOKENS        ids entiers du BPE
3.  EMBEDDINGS    chaque id → vecteur de 2048 flottants
4.  PREFILL       chaque token du prompt traverse les 22 couches
                  et remplit le KV-cache
5.  LOGITS        32 000 scores pour le *prochain* token
6.  SAMPLE        nous choisissons un id (température, top-k, seed)
7.  DECODE        l'id redevient du texte lisible
8.  BOUCLE        ce token retourne dans le modèle…
                  jusqu'à max_new tokens ou EOS
```

Tout le reste de ce chapitre est un **zoom** sur
chaque flèche.

---

## 1. Le Prompt n'est pas « une Phrase » pour le Modèle

### 1.1 Ce que Vous Voyez

Une chaîne de caractères UTF-8, avec des espaces et du sens.

### 1.2 Ce que le Modèle Voit

Une **séquence ordonnée d'entiers** entre 0 et 31 999.

Le pont s'appelle le **tokeniseur BPE** (Byte Pair Encoding),
le même style LLaMA : les mots commencent généralement avec
le préfixe espace-mot `▁` (U+2581).

Pour notre prompt, l'idée (schéma didactique) est :

| Position \(t\) | Pièce (idée) | Rôle dans la Phrase |
|---------------:|--------------|---------------------|
| 0 | `▁The` | article / démarrage |
| 1 | `▁secret` | noyau nominal |
| 2 | `▁to` | lien |
| 3 | `▁happiness` | objet du secret |
| 4 | `oup` | verbe copulatif — **le présent du prédicat** |

> En pratique, le BPE parfois découpe plus finement
> (`happ` + `iness`, etc.). Le principe ne change pas :
> **texte → liste d'ids**. Appelons cette liste
>
> \[
> (t_0, t_1, t_2, t_3, t_4)
> \]
>
> avec une longueur \(T_{\mathrm{prompt}} = 5\).

### 1.3 Pourquoi l'Ordre Importe

TinyLlama est **causal** : à la position \(t\), il ne peut
« voir » que les positions \(0,1,\ldots,t\).
Le passé existe ; l'avenir de la phrase **n'existe pas encore**.

C'est la règle de la circulation de tout le voyage.

---

## 2. De l'ID au Vecteur : Naître dans ℝ²⁰⁴⁸

Chaque id \(t_i\) devient un point dans le ciel
des embeddings (chap. 28) :

\[
x^{(i)}_{0} \;=\; e_{t_i} \;=\; \mathrm{Embedding}(t_i) \;\in\; \mathbb{R}^{2048}
\]

- Il existe une table `token_embd` de forme logique
  **[32 000 × 2048]**.
- La ligne numéro \(t_i\) est le vecteur de cette étoile.
- Ici **il n'y a pas encore de couches**. Uniquement le catalogue.

**Image didactique :**
cinq passagers entrent dans le hall du bâtiment
(chap. 27). Chacun apporte sa valise de 2048 nombres.
Ces valises s'appellent des **résiduels**.

Dans la notation de ce chapitre :

- Exposant \((i)\) : position dans la séquence.
- Indice \(\ell\) : couche (0 avant la première couche ;
  après la couche 21, nous serons au « étage toit »).

En sortant de l'embedding :

\[
x^{(0)}_{0},\; x^{(1)}_{0},\; \ldots,\; x^{(4)}_{0}
\in \mathbb{R}^{2048}
\]

---

## 3. Deux Phases de Vol : Prefill et Génération

TinyLlama (et presque tous les transformers causaux) ne
traitent pas le prompt d'un seul coup magique.
Il existe **deux modes** :

| Phase | Ce qui entre | Ce qui sort | KV-cache |
|-------|-------------|------------|----------|
| **Prefill** | Chaque token du prompt, dans l'ordre | Logits après le **dernier** token du prompt | Se **remplit** |
| **Génération** | Un nouveau token à la fois | Logits pour le suivant | S'**allonge** de +1 |

Dans le moteur C (`llm_inference.c`) :

```c
/* PREFILL */
for (i = 0; i < n_prompt; i++)
    model_forward_token(&model, &state, tokens[i]);

/* GÉNÉRATION */
for (step = 0; step < max_new; step++) {
    next = sample_top_k(state.logits, …);
    if (next == EOS) break;
    emit(next);                          /* texte à l'utilisateur */
    model_forward_token(&model, &state, next);
}
```

Jusqu'à la fin du prefill, nous n'avons pas encore « répondu ».
Nous n'avons fait que **comprendre le prompt** et laisser de la mémoire
dans le cache.

---

## 4. Un Seul Token dans une Seule Couche (Le Noyau)

Prenez la position actuelle \(p\) (par exemple, le dernier
token du prompt, \(p=4\), `▁is`).
Son résiduel en arrivant à la couche \(\ell\) est \(x\).

Dans la couche, ces sept stations se produisent
toujours, dans cet ordre :

```
        x  (résiduel arrivant à la couche ℓ)
        │
        ▼
   [1] RMSNorm  (attn_norm)
        │
        ▼
   [2]  Q, K, V  +  RoPE
        │
        ▼
   [3]  Attention causale  (utilise le KV-cache de cette couche)
        │
        ▼
   [4]  Projection O  →  résiduel :  x ← x + Attn
        │
        ▼
   [5] RMSNorm  (ffn_norm)
        │
        ▼
   [6]  FFN SwiGLU  (gate, up, down)
        │
        ▼
   [7]  résiduel :  x ← x + FFN
        │
        ▼
        x  (sort vers la couche ℓ+1)
```

Répétez cela **22 fois**. C'est l'ascenseur complet
pour **un** token dans **un** pas de forward.

---

## 5. Station par Station (Avec l'Exemple)

Nous suivons le passager de la position \(p=4\) (`▁is`),
dans une couche générique \(\ell\), lorsque les positions \(0..3\) du prompt
existent déjà dans le cache.

### Station 1 — RMSNorm (Attention)

\[
h = \mathrm{RMSNorm}(x;\; \gamma_{\ell}^{\mathrm{attn}})
\]

- Elle ne « comprend » pas la phrase.
- Elle **stabilise** l'échelle du vecteur pour que
  Q et K n'explorent pas.
- Masse de paramètres ridicule (~0.01% du modèle),
  rôle énorme (chap. 7, force V).

**Analogie :** calibrer la boussole avant de regarder
les autres étoiles de la séquence.

### Station 2 — Naissent Q, K, V et la Position (RoPE)

\[
Q = W_Q h,\quad K = W_K h,\quad V = W_V h
\]

Dans TinyLlama, les formes logiques par couche sont :

| Tenseur | Forme logique | Lecture humaine |
|---------|--------------|-----------------|
| \(W_Q\) | [2048, 2048] | 32 têtes × 64 dims |
| \(W_K, W_V\) | [256, 2048] | **4** têtes KV × 64 (GQA) |
| \(W_O\) | [2048, 2048] | rassemble les 32 têtes |

**GQA (Grouped Query Attention) :**
chaque tête de clé/valeur est **partagée par** 8 têtes Q
(\(32/4 = 8\)). Moins de mémoire cache, même idée :
questions riches, mémoire partagée.

**RoPE (Rotary Position Embedding) :**
avant d'attender, Q et K sont **tournés** selon la position \(p\).
Il n'y a pas de vecteur « position 4 » séparé ajouté : la position
est **enroulée** dans l'angle de Q et K.

Ainsi le modèle distingue :

```text
secret to happiness   ≠   happiness to secret
```

bien que les mêmes « étoiles » soient dans le vocabulaire.

### Station 3 — Attention : Gravité entre Tokens

Pour chaque tête de requête :

\[
\mathrm{score}_{p,j}
  = \frac{q_p \cdot k_j}{\sqrt{64}},
  \qquad j = 0,1,\ldots,p
\]

\[
\alpha_{p,j} = \mathrm{softmax}_j(\mathrm{score}_{p,j})
\]

\[
z_p = \sum_{j=0}^{p} \alpha_{p,j}\, v_j
\]

**Lecture avec notre prompt** (intuition, pas une carte
d'attention mesurée ici) :

| \(j\) | Token | Ce qui pourrait « tirer » `▁is` |
|------:|-------|----------------------------------|
| 0 | The | peu (fonction grammaticale) |
| 1 | secret | thème : il y a un secret |
| 2 | to | lien |
| 3 | happiness | **contenu** du secret |
| 4 | is | lui-même (auto-attention) |

Les \(\alpha_{p,j}\) sont la **gravité dynamique**
(chap. 7 et 28) : combien le résiduel de `is` tombe vers
chaque étoile du passé de *cette* phrase.

**Mask causal :** \(j > p\) est interdit.
Dans le prefill, lorsque nous traitons la position 2,
`happiness` **n'existe pas encore** dans le cache.

### Station 4 — Mélange de Têtes + Résiduel d'Attention

Les 32 têtes sont concaténées (ou projetées) et
passent par \(W_O\) :

\[
x \leftarrow x + O(z)
\]

Le résiduel **n'est pas effacé** : la poussée d'attention
lui est **ajoutée**. C'est pourquoi nous parlons d'orbite, pas de
téléportation (chap. 20).

\[
x_{\mathrm{après}} = x_{\mathrm{avant}} + \Delta_{\mathrm{attn}}
\]

### Station 5 — RMSNorm (FFN)

Autre calibration, avec un \(\gamma_{\ell}^{\mathrm{ffn}}\) différent.

### Station 6 — FFN SwiGLU (Le « Soleil » des Paramètres)

C'est ici que vit ~**69%** de la masse du modèle :

\[
\begin{aligned}
u &= W_{\mathrm{up}} h \\
g &= W_{\mathrm{gate}} h \\
\mathrm{FFN}(h) &= W_{\mathrm{down}}\big(\mathrm{SiLU}(g)\odot u\big)
\end{aligned}
\]

- Il s'étend à **5632** dimensions.
- Le *gate* décide quels canaux laisser passer.
- Il se compresse à nouveau à 2048.

**Analogie :** l'attention regarde **d'autres tokens** ;
le FFN transforme **ce** résiduel seul —
météo locale, connaissance « pratique » de la position
(Règle d'Or : FFN → lentille pratique, chap. 9).

### Station 7 — Résiduel du FFN

\[
x \leftarrow x + \mathrm{FFN}(h)
\]

Il sort de la couche \(\ell\) prêt pour la \(\ell+1\).

---

## 6. Le KV-Cache : Mémoire du Passé

Sans cache, pour chaque nouveau token, il faudrait
**recalculer** K et V pour toute la phrase. Impossible
à un bon rythme sur CPU.

Avec le cache, à la couche \(\ell\) :

```
cache_K[ℓ][0 .. p]   déjà sauvegardé
cache_V[ℓ][0 .. p]

En traitant la position p :
  calculer uniquement K_p, V_p
  écrire cache_K[ℓ][p], cache_V[ℓ][p]
  attender Q_p contre cache_K[ℓ][0..p]
```

**Prefill de notre prompt :**

| Étape | Token qui entre | Positions en cache à la fin |
|------:|-----------------|----------------------------|
| 1 | The | 0 |
| 2 | secret | 0–1 |
| 3 | to | 0–2 |
| 4 | happiness | 0–3 |
| 5 | is | 0–4 |

Après l'étape 5, les **22 couches** possèdent K et V pour
les cinq positions. Le résiduel de `is` a gravi
le bâtiment entier. C'est là que sortent les **logits** du
premier token de la *réponse*.

---

## 7. Du Toit au Vocabulaire : Logits

Après la couche 21 :

\[
h = \mathrm{RMSNorm}(x;\; \gamma^{\mathrm{out}})
\]

\[
\mathrm{logits} = W_{\mathrm{out}}\, h \;\in\; \mathbb{R}^{32000}
\]

- `output.weight` a une forme logique **[32 000 × 2048]**
  (parfois partagée ou liée à l'embedding dans d'autres
  modèles ; dans le GGUF de TinyLlama, c'est le `lm_head`).
- Chaque entrée \(z_k\) est « combien le modèle pousse
  à choisir le token d'id \(k\) » **maintenant**.

Il n'y a toujours **aucun** mot. Il y a un classement de 32 000
candidats.

---

## 8. Sample : Effondrer le Ciel vers une Étoile

**Force VI** (chap. 7) : du continu à l'événement.

Procédure typique dans le moteur Dreaming :

1. **Température** \(T\) : \(z_k \leftarrow z_k / T\).
   - \(T \to 0\) : presque toujours le maximum (avarice).
   - \(T\) élevé : plus d'aléatoire, plus de diversité.
2. **Top-k** : ne garder que les \(k\) logits les plus élevés
   (par ex. 40). Le reste est ignoré.
3. **Softmax** uniquement sur ces \(k\) :

\[
\pi_i = \frac{e^{z_i}}{\sum_{j\in\mathrm{top\text{-}k}} e^{z_j}}
\]

4. **Échantillonner** un id selon \(\pi\) (avec `--seed` pour
   reproduire le même voyage).

Supposons (exemple inventé mais réaliste) que le résultat soit :

```text
id →  ▁being      ou      ▁love      ou      ▁not ...
```

Cet id est **décodé** en texte et affiché à l'utilisateur.
C'est le premier pas de la réponse.

---

## 9. La Boucle Autorégressive (La Réponse Grandit)

Le token choisi **n'est pas la fin du modèle**.
C'est le **prochain passager** :

```
prompt:     The secret to happiness is
+ sample:   being
nouvelle seq : The secret to happiness is being
```

`model_forward_token` est rappelé **uniquement**
avec `being` :

- Son embedding est calculé.
- Il traverse les 22 couches.
- Il écrit K,V à la position \(p=5\) de chaque couche.
- Il attente sur `The…is` + `being`.
- Il produit les logits pour le token **encore plus récent**.

Et ainsi de suite :

```
The secret to happiness is being
The secret to happiness is being kind
The secret to happiness is being kind to
...
```

jusqu'à :

- atteindre `max_new` tokens, ou
- échantillonner **EOS** (fin de séquence).

**Idée clé :**
générer un paragraphe, c'est **plusieurs** répétitions du
voyage d'*un* token, pas un passage unique « de la phrase
entière à la réponse entière ».

---

## 10. Schéma Maître du Voyage

```
┌─────────────────────────────────────────────────────────┐
│  HUMAIN : "The secret to happiness is"                  │
└───────────────────────────┬─────────────────────────────┘
                            │ tokeniseur BPE
                            ▼
┌─────────────────────────────────────────────────────────┐
│  IDS : t0 t1 t2 t3 t4                                   │
└───────────────────────────┬─────────────────────────────┘
                            │ lignes de token_embd
                            ▼
┌─────────────────────────────────────────────────────────┐
│  VECTEURS : x0..x4  ∈ ℝ²⁰⁴⁸                            │
└───────────────────────────┬─────────────────────────────┘
                            │ PREFILL (pour chaque ti)
                            ▼
        ┌───────────────────────────────────────┐
        │  pour position p = 0 .. 4 :           │
        │    pour couche ℓ = 0 .. 21 :          │
        │       Norm → Attn(+RoPE,GQA,cache)    │
        │            → +résiduel                │
        │       Norm → FFN SwiGLU               │
        │            → +résiduel                │
        └───────────────────┬───────────────────┘
                            │ après le dernier p du prompt
                            ▼
┌─────────────────────────────────────────────────────────┐
│  output_norm → lm_head → logits[32000]                  │
└───────────────────────────┬─────────────────────────────┘
                            │ temp, top-k, softmax, sample
                            ▼
┌─────────────────────────────────────────────────────────┐
│  NOUVEAU TOKEN → texte à l'utilisateur                  │
│       │                                                 │
│       └──── retour à forward_token (GÉNÉRATION) ──► …   │
└─────────────────────────────────────────────────────────┘
```

---

## 11. Mini-Laboratoire : Voir le Voyage avec le Moteur C

Depuis la racine du dépôt (ajustez les chemins à votre GGUF) :

```bash
# Prefill + génération, seed fixe (reproductible)
./llm_inference tinyllama-1.1b.F16.gguf \
  "The secret to happiness is" 40 0.7 40 --seed 42
```

| Flag / arg | Rôle dans le Voyage |
|------------|---------------------|
| prompt | étoiles initiales de la séquence |
| `40` (n) | combien de nouveaux tokens orbiter |
| `0.7` | température de l'effondrement |
| `40` (top-k) | largeur du puits de candidats |
| `--seed 42` | même aléatoire → même chemin |
| `--perturb mystical --intensity 0.35` | **déforme** Q/K/V/FFN : autre physique, même route formelle |
| `--steer happiness --steer-strength 0.15` | pousse le résiduel vers une direction du ciel |

Protocole didactique recommandé :

1. Même seed, `none` vs `mystical` → l'orbite change-t-elle ?
2. Même seed, temp 0.2 vs 0.9 → l'effondrement change-t-il ?
3. Ouvrez la [carte sémantique](https://fivetechsoft.github.io/dreaming/exploration/semantic_map.html),
   recherchez `▁happiness` / `▁love` et regardez leurs **forces**
   (gravité statique du catalogue) tout en lisant
   la réponse générée (gravité dynamique du prompt).

---

## 12. Erreurs Mentales Fréquentes (et la Correction)

| Croyance | Réalité dans TinyLlama |
|----------|------------------------|
| « Le modèle lit la phrase d'un coup d'œil » | Il lit **token par token** ; le prefill est séquentiel |
| « Chaque couche invente un nouveau vecteur » | Il met à jour le **même** résiduel avec des additions |
| « L'attention regarde tout le livre » | Seulement le **passé** de *cette* séquence (jusqu'à 2048) |
| « 32 têtes = 32 mémoires KV » | Seulement **4** groupes KV (GQA) ; 32 regards Q |
| « L'embedding est déjà la réponse » | L'embedding est la **naissance** ; il reste 22 étages |
| « Softmax choisit le mot du prompt » | Il choisit le **prochain** token du vocabulaire |
| « Une réponse = un forward » | Une réponse = **1 prefill + N forwards** |

---

## 13. Liste de Vérification de Compréhension Totale

Si vous pouvez répondre oui à tout, le voyage est intériorisé :

1. Qu'est-ce qu'un token et pourquoi n'est-ce pas un caractère ?
2. Quelle dimension a le résiduel et pourquoi est-il préservé ?
3. Que interdit le mask causal ?
4. À quoi sert le RoPE ?
5. Qu'est-ce qui distingue l'attention et le FFN dans une couche ?
6. Que stocke le KV-cache et dans quelle phase est-il rempli ?
7. Combien de fois l'ascenseur de 22 étages monte-t-il pour un prompt de 5 tokens en prefill ?
   → **5 × 22** passages de couche (un par position).
8. Qu'est-ce qu'un logit et comment devient-il du texte ?
9. Pourquoi générer 40 tokens implique ~40 forwards supplémentaires ?
10. Où une lentille Dreaming (`--perturb`) intervient-elle dans ce schéma ?
    → Dans les poids des stations 2–6, pas dans le tokeniseur.

---

## 14. Ponts

| Thème | Chapitre |
|-------|----------|
| Dims et tenseurs par couche | 2 |
| Moteur C, RoPE, cache, sample | 3 |
| Forces (attn, FFN, softmax…) | 7 |
| Comment voyager (routes A–E) | 8 |
| Attention en détail | 10 |
| FFN en détail | 11 |
| Couches précoces / intermédiaires / finales | 13–15 |
| Chaîne du sens (vision sémantique) | 26 |
| Ascenseur par étage | 27 |
| Étoiles = tokens, attention = gravité | 28 |
| Formules | 23 |

---

## 15. Conclusion

Le voyage d'un prompt n'est pas un mystère : c'est une
**fabrique répétable**.

1. Le texte devient des **ids**.
2. Les ids deviennent des **vecteurs**.
3. Chaque vecteur gravit **22 étages** de
   norm → gravité attentionnelle → météo FFN,
   ne parlant qu'au **passé**.
4. Le résiduel final est projeté sur **32 000** scores.
5. Un échantillonnage choisit **une** étoile.
6. Cette étoile est mise en file et l'univers tourne à nouveau.

Quand vous écrivez

```text
The secret to happiness is
```

et que TinyLlama répond, ce n'est plus « l'IA pense une phrase ».
C'est : *cinq naissances, cinq ascensions du bâtiment,
un effondrement, puis N effondrements supplémentaires* — toujours la
même physique, un pas de plus dans le temps.

C'est la compréhension totale du voyage.
Le reste du livre (perspectives, cartes, lentilles)
sont des **variations de la physique**, pas un autre chemin.
