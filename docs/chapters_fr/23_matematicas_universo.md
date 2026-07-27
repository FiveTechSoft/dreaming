# Chapitre 23 : Les mathématiques de cet univers  
### Au-delà de la structure du transformer

## Pourquoi ce chapitre

Les chapitres précédents nomment des *pièces* (couches, QKV, FFN)
et des *forces* (gravité attentionnelle, climat FFN, orbites).
Ici nous écrivons les **équations** qui les rendent précises :
pas un cours de deep learning, mais le **minimum formel**
du microcosme TinyLlama-1.1B tel que nous le mesurons
et le déplaçons dans Dreaming.

Conventions :

- Vecteurs colonnes, sauf indication contraire.
- \(\langle u,v\rangle = u^\top v\), \(\|u\|=\sqrt{\langle u,u\rangle}\).
- \(\mathrm{softmax}(z)_i = e^{z_i}/\sum_j e^{z_j}\).

---

## 1. Les espaces comme objets mathématiques

| Symbole | Espace | Dimension | Rôle |
|---------|---------|-----------|-----|
| \(\mathcal{V}\) | vocabulaire | \(V=32000\) | ensemble fini d’ids |
| \(E\) | embeddings | \(E\in\mathbb{R}^{V\times d}\), \(d=2048\) | lignes \(e_t\in\mathbb{R}^d\) |
| \(x_\ell^{(t)}\) | résiduel | \(\mathbb{R}^d\) | état en couche \(\ell\), position \(t\) |
| \(W_{q,\ell}\), etc. | poids de couche | matrices en \(\mathbb{R}^{\cdot\times\cdot}\) | opérateurs linéaires (+ non-linéarités) |
| \(\theta\) | poids totaux | \(\theta\in\mathbb{R}^N\), \(N\sim 1.1\cdot 10^9\) | « point-univers » du modèle |
| \(\mathcal{C}\subset\mathbb{R}^N\) | surface de cohérence | sous-ensemble (pas une sous-variété prouvée) | modèles qui génèrent du texte lisible |

Le forward est une composition

\[
f_\theta : \mathcal{V}^{T} \to \mathbb{R}^{V}
\]

qui à une séquence de tokens associe les logits du dernier
(ou de chaque position, selon le mode).

La génération est l’itération

\[
t_{n+1}\sim \pi\big(f_\theta(t_1,\ldots,t_n)\big),
\]

où \(\pi\) est un échantillonnage (température, top-k).

---

## 2. Géométrie du ciel de tokens

### 2.1 Similarité

Pour \(u,v\in\mathbb{R}^d\),

\[
\cos(u,v)=\frac{\langle u,v\rangle}{\|u\|\,\|v\|}\in[-1,1].
\]

**Fait empirique (TinyLlama) :** paires « opposées » du langage
(`love`/`hate`, `life`/`death`) ont \(\cos\approx 0\), pas \(-1\).
Dans ce ciel, l’antinomie sémantique **n’est pas** antipodalité.

### 2.2 Centroïdes et archétypes

Donné un ensemble de graines \(S=\{t_1,\ldots,t_m\}\subset\mathcal{V}\),

\[
c_S=\frac{1}{m}\sum_{i=1}^m e_{t_i},\qquad
\hat c_S=\frac{c_S}{\|c_S\|}.
\]

Un **archétype** (chap. 21) est un \(\hat c_S\) avec sémantique
culturelle fixée. Une **île sémantique** (chap. 16) est la même chose
avec un autre choix de \(S\).

**Alignement entre archétypes :**

\[
A(S,S')=\cos(\hat c_S,\hat c_{S'}).
\]

Mesuré : \(A(\mathrm{Magicien},\mathrm{Mystique})\approx 0.39\),
\(A(\mathrm{Sage},\mathrm{Académique})\approx 0.29\).

### 2.3 Direction de contraste

\[
\delta_{S|S'}=\frac{c_S-c_{S'}}{\|c_S-c_{S'}\|}.
\]

Ex. : émotion positive moins négative → pôle `smile/happy`
vs `sad/anger` en classant \(\cos(e_t,\delta)\).

### 2.4 PCA du ciel (structure globale)

Échantillon \(X\in\mathbb{R}^{n\times d}\) de lignes de \(E\) centrées.
SVD \(X=U\Sigma V^\top\). Fraction de variance dans les \(k\) premiers
composants :

\[
\mathrm{EVR}(k)=\frac{\sum_{i=1}^k \sigma_i^2}{\sum_{i=1}^{d}\sigma_i^2}.
\]

**Fait empirique :** \(\mathrm{EVR}(10)\approx 2.3\%\),
dims pour 50% / 90% de var. \(\approx 481\) / \(1329\).
Le ciel **utilise** des centaines de directions ; il ne s’effondre pas en 2D.
Les cartes HTML sont des projections

\[
\mathbb{R}^d\ni e \mapsto V_{:2}^\top (e-\bar e)\in\mathbb{R}^2,
\]

utiles et trompeuses à la fois (chap. 6).

### 2.5 Anisotropie

\[
\alpha=\frac{\|\bar e\|}{\frac{1}{V}\sum_t \|e_t\|},\quad
\bar e=\frac{1}{V}\sum_t e_t.
\]

Mesuré : \(\alpha\approx 0.006\) — presque isotrope en norme moyenne.
La structure est **directionnelle**, pas d'« étoiles beaucoup plus massives ».

---

## 3. Un pas de couche comme système dynamique

Soit \(x\in\mathbb{R}^d\) le résiduel à la position actuelle
(ommettant l’indice de position quand il est superflu).

### 3.1 RMSNorm

\[
\mathrm{RMSNorm}(x;w)=
\frac{x}{\sqrt{\frac{1}{d}\|x\|^2+\varepsilon}}\odot w,
\]

avec \(w\in\mathbb{R}^d\), \(\varepsilon>0\) (par ex. \(10^{-5}\)).

Il ne soustrait pas la moyenne (contrairement à LayerNorm). C’est une
**projection sur la sphère** (approximativement) suivie d’un
échelonnage par coordonnées.

### 3.2 Attention (tête \(h\), GQA)

\[
Q=x W_Q,\quad
K=x W_K,\quad
V=x W_V
\]

(en pratique : matrices partagées par couche ; K,V de
dimension \(n_{kv}\cdot d_h\) avec \(n_{kv}=4\), \(d_h=64\),
\(n_q=32\)).

Pour la tête \(h\), avec indice KV \(h_{kv}=\lfloor h\cdot n_{kv}/n_q\rfloor\),

\[
a_{h,t}
=\mathrm{softmax}_{t'\le t}
\Bigg(\frac{\langle q_h^{(t)}, k_{h_{kv}}^{(t')}\rangle}{\sqrt{d_h}}\Bigg),
\quad
o_h^{(t)}=\sum_{t'\le t} a_{h,t'}\, v_{h_{kv}}^{(t')}.
\]

Le masque \(t'\le t\) est la **causalité** : le futur
a un potentiel infini (probabilité 0).

Sortie multi-têtes :

\[
\mathrm{Attn}(x)=\mathrm{Concat}(o_1,\ldots,o_{n_q})\,W_O.
\]

**Interprétation :** \(a_{h,t'}\) est un **noyau de couplage**
(non symétrique, non invariant par translation) entre positions.
La « gravité » du chap. 7 est ce noyau normalisé.

### 3.3 SwiGLU (FFN)

\[
\mathrm{SiLU}(z)=z\cdot\sigma(z),\quad
\sigma(z)=\frac{1}{1+e^{-z}},
\]

\[
\mathrm{FFN}(x)=
W_d\Big(\mathrm{SiLU}(W_g x)\odot (W_u x)\Big),
\]

avec \(W_g,W_u\in\mathbb{R}^{d_{ff}\times d}\), \(d_{ff}=5632\),
\(W_d\in\mathbb{R}^{d\times d_{ff}}\).

C’est une application **point à point** dans la séquence : ne mélange pas \(t\).
C’est pourquoi c’est une « physique locale » face à l’attention non locale.

### 3.4 Bloc résiduel

\[
\begin{aligned}
x &\leftarrow x + \mathrm{Attn}(\mathrm{RMSNorm}(x;w_a)),\\
x &\leftarrow x + \mathrm{FFN}(\mathrm{RMSNorm}(x;w_f)).
\end{aligned}
\]

Forme abstraite d’un pas d’Euler dans un champ \(F_\ell\) :

\[
x_{\ell+1}=x_\ell + F_\ell(x_\ell;\theta_\ell).
\]

L’**orbite** du chap. 20 est la trajectoire
\(\{x_\ell\}_{\ell=0}^{22}\) en \(\mathbb{R}^d\).

### 3.5 Pourquoi le résiduel importe (mathématiques de la cohérence)

Si c’était \(x_{\ell+1}=F_\ell(x_\ell)\) sans skip,
de petits changements dans \(F_\ell\) se composent de façon
exponentiellement instable (produit de Jacobiens).

Avec skip, le Jacobien est

\[
D x_{\ell+1}= I + DF_\ell,
\]

et si \(\|DF_\ell\|\) est modérée, la dynamique ressemble à
une **perturbation de l’identité** : les trajectoires
restent proches d'une « ligne de base » sémantique.
C’est la version analytique de « tomber de côté ».

---

## 4. Du résiduel à l’événement : logits et échantillonnage

### 4.1 Tête de langage

\[
z = E_{\mathrm{out}}\,\mathrm{RMSNorm}(x_L)\in\mathbb{R}^V
\]

(dans GGUF : `output.weight` ; parfois lié à l’embedding).

### 4.2 Température

\[
\pi_T(t)=\mathrm{softmax}(z/T)_t
=\frac{e^{z_t/T}}{\sum_{j} e^{z_j/T}}.
\]

- \(T\to 0\) : masse en \(\arg\max z\) (orbite « circulaire », gloutonne).  
- \(T\to\infty\) : vers uniforme (entropie maximale).

Entropie \(H(\pi_T)=-\sum_t \pi_T(t)\log\pi_T(t)\) croît avec \(T\).

### 4.3 Top-k

Soit \(S_k\subset\mathcal{V}\) les \(k\) indices de plus grand \(z_t\).
On redéfinit

\[
\pi(t)\propto e^{z_t/T}\mathbf{1}_{t\in S_k}.
\]

C’est une **troncature du support** : l’horizon des événements
autorisés se réduit à \(k\) destins.

### 4.4 Génération comme chaîne de Markov

Conditionné à \(\theta\) et à la politique \(\pi\),

\[
\mathbb{P}(t_{1:N})=\prod_{n=0}^{N-1}
\pi\big(t_{n+1}\mid t_{1:n};\theta\big).
\]

Changer \(\theta\) (perturbation) change la famille de
chaînes ; pas seulement un sample isolé.

---

## 5. Espace des poids et surface de cohérence

### 5.1 Le modèle comme point

\[
\theta\in\mathbb{R}^N,\qquad N\approx 1.1\times 10^9.
\]

Presque tout le volume de \(\mathbb{R}^N\) produit des ordures.
Nous définissons de manière opérationnelle (pas topologique rigoureuse)

\[
\mathcal{C}=\{\theta : \text{le texte de }f_\theta\text{ est cohérent dans une batterie de prompts}\}.
\]

Dreaming étudie des **mouvements** \(\theta\mapsto\theta'\)
qui restent dans \(\mathcal{C}\) mais changent la
distribution de styles (perspectives).

### 5.2 amplify_subspace (mystical)

Sur un tenseur aplat \(w\in\mathbb{R}^m\)
(ou concaténation de tenseurs de couche) :

1. Échantillonner \(v\sim\mathcal{N}(0,I_m)\), normaliser \(\hat v=v/\|v\|\).  
2. Projection scalaire \(p=\langle w,\hat v\rangle\).  
3. Mettre à jour

\[
w' = w + \varepsilon\, p\, \hat v
= w + \varepsilon\, ( \hat v\hat v^\top ) w.
\]

C’est une **mise à jour rank-1** en direction aléatoire :

\[
w'=(I+\varepsilon\, \hat v\hat v^\top)w.
\]

Valeur propre \(1+\varepsilon\) dans la direction \(\hat v\),
\(1\) dans l’hyperplan orthogonal.

**Lecture :** on amplifie une composante ; on préserve
la structure dans le reste → candidat à un mouvement
**tangent** à \(\mathcal{C}\) si \(\varepsilon\) est modéré.

### 5.3 Bruit échelonné (contre-exemple)

\[
w'_i = w_i + \varepsilon\, \xi_i\, |w_i|,\quad \xi_i\sim\mathcal{N}(0,1).
\]

Rompt les corrélations entre coordonnées : poussée
**générique**, pas rank-1 alignée. Expérimentalement,
pour \(\varepsilon\) grand, \(\theta'\notin\mathcal{C}\).

### 5.4 Intensité et « sweet spot »

En Q4_0 (pipeline Python v10), \(\varepsilon\approx 0.10\)
a été le sweet spot de nombreuses techniques.
En F32 runtime du moteur C, parfois il faut
\(\varepsilon\in[0.3,0.5]\) pour voir une divergence de texte
dans des prompts courts (quantisation et EOS interagissent).

Il n’y a pas un unique \(\varepsilon^*\) universel : cela dépend
du format numérique et de la batterie.

### 5.5 Formule de perspective (opérationnelle)

\[
\theta_{\mathrm{persp}}
= \theta_0 + \sum_r \varepsilon_r\, \Delta_r,
\]

avec \(\|\varepsilon_r\Delta_r\|\) petits et chaque \(\Delta_r\)
préservant la hiérarchie (lowrank, amplify, normrot, …).
L’interpolation proche dans \(\mathcal{C}\) tend à
rester dans \(\mathcal{C}\) (hypothèse empirique du projet).

---

## 6. Steering : projection dans le résiduel

Donné un mot avec embedding \(e_\star\),

\[
\hat u=\frac{e_\star}{\|e_\star\|},\qquad
x \leftarrow x + \lambda\, \langle x,\hat u\rangle\, \hat u
= \big(I+\lambda\,\hat u\hat u^\top\big)x.
\]

Même algèbre rank-1 que amplify, mais appliquée à l’**état** \(x\), pas aux poids \(\theta\).

- \(\lambda\) : force du « vent ».  
- Ne réécrit pas l’univers ; dévie l’orbite en vol.

---

## 7. Quantisation (l’univers en entiers)

En Q4_0, blocs de 32 poids :

\[
w_i \approx s\cdot (q_i-8),\quad q_i\in\{0,\ldots,15\},
\]

\(s\) en float16 par bloc. La perturbation Python
opère en \(\mathbb{R}\) après déquant et revient en
\((s,q)\). Cela introduit une **erreur de projection**
sur un réseau discret : autre raison pour laquelle
\(\varepsilon\) « optimal » ne coïncide pas avec le F32 pur.

Le moteur C en F16 déquantise dans le matmul via table
de \(2^{16}\) entrées : isomorphisme exact half→float
dans chaque poids lu, sans requantiser en baseline.

---

## 8. Complexité d’un pas (orbite efficace)

Pour une séquence de longueur \(T\), un pas d’attention
naïf dans une couche est \(\mathcal{O}(T^2 d)\) si on recalcule
tout ; avec le **KV-cache** en générant le token \(T\) :

\[
\mathcal{O}(T\cdot d\cdot d_h\cdot n_q)
\quad\text{(scores nouveaux contre }T\text{ clés)}
\]

plus les matmuls \(\mathcal{O}(d\cdot d_{ff})\) du FFN.
C’est pourquoi le moteur actuel orbite à ~6–10 tok/s en CPU
et celui sans cache tombait à ~0.03 tok/s.

---

## 9. Dictionnaire : physique du livre ↔ formule

| Langage de l’atlas | Objet mathématique |
|--------------------|-------------------|
| Ciel d’étoiles | lignes de \(E\in\mathbb{R}^{V\times d}\) |
| Île / constellation | \(c_S\), graines \(S\) |
| Gravité entre tokens | \(a_{t,t'}=\mathrm{softmax}(QK^\top/\sqrt{d_h})\) |
| Climat local | \(\mathrm{FFN}:\mathbb{R}^d\to\mathbb{R}^d\) |
| Inertie orbitale | \(x\mapsto x+F(x)\) |
| Air respirable | \(\mathrm{RMSNorm}\) |
| Effondrement sur l’événement | \(\pi_T=\mathrm{softmax}(z/T)\) (top-k) |
| Surface habitable | \(\mathcal{C}\subset\mathbb{R}^N\) |
| Loupe mystique | \(w'=(I+\varepsilon\hat v\hat v^\top)w\) |
| Vent de steering | \(x'=(I+\lambda\hat u\hat u^\top)x\) |
| Carte du planétarium | PCA : \(e\mapsto V_{:2}^\top(e-\bar e)\) |

---

## 10. Ce que les mathématiques *n’affirment* pas encore

1. \(\mathcal{C}\) n’est pas démontré comme variété différentielle ;
   c’est une région **opérationnelle** définie par des tests.  
2. « Tangent à \(\mathcal{C}\) » est une métaphore géométrique
   soutenue par rank-1 / SVD / orthogonalité à des modes
   destructifs — pas un théorème de géométrie riemannienne
   de la loss.  
3. Les archétypes ne sont pas des facteurs latents uniques ;
   ce sont des **directions choisies** avec des graines humaines.  
4. La Règle d’Or est **empirique** dans TinyLlama ;
   elle ne se dérive pas ici d’un principe variationnel général.

L’honnêteté du microcosme inclut la frontière
de ce qui n’est pas démontré.

---

## 11. En une phrase

Au-delà du diagramme de boîtes du transformer, cet
univers est : **géométrie de \(E\) et de \(\theta\)**,
**dynamique résiduelle** \(x\leftarrow x+F_\ell(x)\),
**noyaux d’attention causaux**, **applications SwiGLU**,
**effondrement softmax**, et **mises à jour rank-1** qui
déplacent des orbites ou des poids sans (parfois) abandonner
la région où le langage reste possible.

---

*Chapitre suivant : Le LLM — Un miroir où nous regarder.*