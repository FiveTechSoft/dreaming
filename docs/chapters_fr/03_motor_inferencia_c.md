# Chapitre 3 : Notre Moteur d'Inférence en C pour TinyLlama

## Pourquoi écrire un moteur propriétaire

Pour étudier TinyLlama, il ne suffit pas de *l'utiliser*. Il faut *le voir fonctionner*.

La plupart des frameworks d'inférence cachent le parcours du token derrière des couches d'abstraction :

- PyTorch et CUDA (gigaoctets de dépendances)
- Runtimes en Python
- Kernels opaques sur GPU

Nous voulions le contraire :

1. **C lisible** — un seul fichier, opérations explicites
2. **Pas de GPU obligatoire** — tout en CPU
3. **Pas de magie** — chaque étape du transformer est une fonction
4. **Portable** — un binaire petit, reproductible

Le résultat est `llm_inference.c` : moteur d'inférence qui implémente la boucle du transformer depuis le fichier GGUF jusqu'au token suivant — et plus tard, **perturbations de poids en runtime** sans besoin d'un GGUF pré-cuit.

Ce n'est pas le moteur le plus rapide du monde. C'est le moteur que *nous comprenons ligne par ligne* et avec lequel nous pouvons *toucher les poids consciencieusement*.

## Ce qu'un moteur d'inférence doit faire

Rappelons le flux du chapitre précédent :

```
Texte → tokens → embeddings
     → 22 couches (attention + FFN)
     → logits → token suivant
     → (répéter)
```

En pratique, le moteur se divise en cinq pièces :

```
┌─────────────────────────────────────────┐
│            llm_inference.c              │
├─────────────────────────────────────────┤
│  1. Lecteur GGUF                        │
│     - magic, metadata, indices          │
│     - tenseurs F16 (et fallback tokenizer│
│       depuis Q4_0 si nécessaire)        │
├─────────────────────────────────────────┤
│  2. Tokenizer BPE                       │
│     - tokens + merges du propre GGUF    │
│     - texte ⇄ ids                       │
├─────────────────────────────────────────┤
│  3. Moteur Transformer + KV-cache       │
│     - embedding, RMSNorm, RoPE          │
│     - attention GQA, SwiGLU             │
│     - un token nouveau par étape        │
├─────────────────────────────────────────┤
│  4. Sampling                            │
│     - température, top-k                │
├─────────────────────────────────────────┤
│  5. Perturbation / steering (optionnel) │
│     - --perturb mystical, noise, …      │
│     - --steer <mot> (activations)       │
└─────────────────────────────────────────┘
```

| Bloc | Question |
|------|----------|
| GGUF | Où sont les poids sur le disque ? |
| Tokenizer | Comment le langage est-il converti en nombres ? |
| Transformer | Comment la représentation est-elle transformée ? |
| Sampling | Comment le mot suivant est-il choisi ? |
| Perturbation | Comment change-t-on la *perspective* du modèle ? |

## TinyLlama en chiffres (ceux que le moteur utilise)

| Paramètre | Valeur dans TinyLlama-1.1B |
|-----------|---------------------------|
| Couches (`block_count`) | 22 |
| Dimension cachée | 2048 |
| Têtes Q / KV (GQA) | 32 / 4 |
| Dimension par tête | 64 |
| FFN intermédiaire | 5632 |
| Vocabulaire | 32 000 |
| Contexte du modèle | 2048 |
| RoPE `freq_base` | 10 000 |
| Format de poids du moteur | GGUF **F16** |

Neuf tenseurs par couche (chap. 2) plus les globaux `token_embd`, `output_norm`, `output`. Le moteur n'invente pas l'architecture : il la *matérialise* en mémoire.

> **Note sur Q4_0 :** les expériences massives du projet utilisent également des GGUF quantifiés (~638 Mo) avec `llama-cli`. Notre C pur lit **F16** (~2.1 Go) pour ne pas réimplémenter tous les kernels de quantification. Le *forward* est le même transformer.

## Compiler et exécuter

```bash
# Recommandé (Windows/MinGW ou Linux)
gcc -O3 -march=native -ffast-math -fopenmp \
    -o llm_inference llm_inference.c -lm

# Variables utiles
export OMP_NUM_THREADS=8   # Linux/macOS
# PowerShell : $env:OMP_NUM_THREADS = "8"
```

### Utilisation de base (baseline)

```bash
./llm_inference tinyllama-1.1b.F16.gguf \
    "The secret to happiness is" \
    40 0.7 40 \
    --seed 42
```

Arguments positionnels : `modèle`, `prompt`, `n_tokens`, `température`, `top_k`.

### Perturbation mystique en runtime

```bash
./llm_inference tinyllama-1.1b.F16.gguf \
    "The secret to happiness is" \
    40 0.7 40 \
    --seed 42 \
    --perturb mystical \
    --intensity 0.50
```

Il n'est pas nécessaire d'avoir un fichier `DMT_*.gguf` préalable : la technique est appliquée **en mémoire** sur les poids fraîchement chargés.

### Autres drapeaux

| Flag | Effet |
|------|-------|
| `--perturb` / `-P` | `none`, `mystical`, `amplify`, `noise`, `blockdiag`, `manifold` |
| `--intensity` / `-I` | force de la perturbation (par ex. 0.10–0.50) |
| `--seed` | PRNG de la perturbation (+ sampling si fixé) |
| `--steer` | mot dont la direction d'embedding tire du résiduel |
| `--steer-strength` | intensité du steering (par ex. 0.15) |

Dépendances de *build* : `libm` et, pour le matmul parallèle, **OpenMP** (`-fopenmp`). Sans OpenMP, le moteur compile quand même, juste plus lentement.

## 1. Lire un GGUF en C

GGUF possède trois zones :

```
[ magic "GGUF" | version | n° tenseurs | n° KV ]
[ metadata clé-valeur ]
[ index des tenseurs → données alignées ]
```

Le moteur charge le fichier complet (sous Windows avec I/O 64 bits : le F16 dépasse 2 Go et `ftell` 32 bits ne suffit pas). Ensuite il indexe les tenseurs par nom :

```c
snprintf(name, sizeof(name), "blk.%d.attn_q.weight", l);
m->wq[l] = must_tensor_f16(&m->gguf, name);
/* wk, wv, wo, w1/w3/w2 (gate/up/down), normes… */
```

### F16 sur disque, float dans le matmul

Les poids de couche sont laissés en **F16** dans le buffer du fichier. Dans le produit matrice-vecteur, ils sont convertis avec une **table de 65 536 entrées** (une valeur pour chaque motif de half) :

```c
static float g_f16_table[65536];
/* init : g_f16_table[i] = decode_ieee_half(i); */
static inline float f16_to_f32(uint16_t h) {
    return g_f16_table[h];
}
```

Ainsi nous évitons de décoder des bits à chaque poids de la boucle chaude. Ce n'est que lorsque nous demandons `--perturb` que nous copions les matrices de couche en **F32 mutable** (~3.6 Go) : le delta de `amplify_subspace` est petit et un aller-retour en F16 l'effacerait.

### Tokenizer : BPE du propre GGUF

Nous lisons `tokenizer.ggml.tokens` et `tokenizer.ggml.merges` de la metadata, avec des tables hash pour les merges rapides.

Détail pratique : certains F16 de TinyLlama viennent avec un vocabulaire *tronqué* (peu de tokens dans le header). Si nous détectons cela, nous chargeons **seulement le tokenizer** depuis le Q4_0 frère (`tinyllama-1.1b.Q4_0.gguf`), sans toucher aux poids F16. Les ids correspondent à HuggingFace (par ex. `"Hello"` → `[1, 15043]`).

## 2. Les opérations du transformer

### RMSNorm

```c
/* x / rms(x) * w   — w vient en F16 */
static void rmsnorm(float *out, const float *x,
                    const uint16_t *w_f16, int n) {
    float ss = 0.f;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    float scale = 1.f / sqrtf(ss / (float)n + 1e-5f);
    for (int i = 0; i < n; i++)
        out[i] = x[i] * scale * f16_to_f32(w_f16[i]);
}
```

Peu de poids (`attn_norm`, `ffn_norm`), beaucoup de contrôle.

### Matmul (le coût réel)

```c
/* W [out, in] · x[in] → out[out]
 * OpenMP sur les lignes de sortie ; F16 via LUT */
static void matmul_f16(float *out, const float *x,
                       const uint16_t *W,
                       int in_dim, int out_dim) {
#pragma omp parallel for schedule(static)
    for (int j = 0; j < out_dim; j++) {
        const uint16_t *row = W + (size_t)j * in_dim;
        float sum = 0.f;
        for (int i = 0; i < in_dim; i++)
            sum += x[i] * f16_to_f32(row[i]);
        out[j] = sum;
    }
}
```

Avec `--perturb`, la même boucle utilise `matmul_f32` sur les copies déjà perturbées. Q, K, V, O, gate, up, down et le `lm_head` sont tous des matmuls.

### RoPE

Position sans embeddings absolus : chaque paire de dimensions de Q et K est tournée. Le moteur précalcule sin/cos par position jusqu'à `MAX_SEQ` pour ne pas appeler `sinf`/`cosf` dans le chemin chaud.

### Attention avec GQA + KV-cache

```
score(t) = (Q_pos · K_t) / sqrt(head_dim)
poids    = softmax_causal(scores)   # seulement 0..pos
sortie   = Σ poids[t] * V_t
```

TinyLlama : **32 têtes Q, 4 KV** → chaque tête KV dessert 8 têtes Q (`hkv = h / 8`).

À chaque étape de génération, nous ne calculons Q/K/V que du **token nouveau**, nous gardons K et V dans le cache `[couche][pos][kv_dim]` et nous attendons sur `0 .. pos`. C'est ce qui rend la CPU viable : sans KV-cache, nous recomputerions toute la séquence à chaque token.

### SwiGLU (FFN)

```
h' = Down( SiLU(Gate(x)) ⊙ Up(x) )
```

~69% des paramètres (chap. 2) : la « mémoire » pratique du modèle. Après l'attention, résiduel ; après le FFN, un autre résiduel.

## 3. Un pas forward (un token)

En langage humain, par couche et par *position actuelle* :

```
x = embedding(token)

pour L = 0 .. 21 :
    h = RMSNorm(x, attn_norm[L])
    Q,K,V = projections(h) ; RoPE(Q,K)
    stocker K,V en cache[L][pos]
    x = x + O( Attention(Q, cache_K, cache_V) )

    h = RMSNorm(x, ffn_norm[L])
    x = x + SwiGLU_FFN(h)

logits = output · RMSNorm(x, output_norm)
```

Dans le code, c'est `model_forward_token(...)`. La génération :

```c
/* prefill : chaque token du prompt remplit le cache */
for (i = 0; i < n_prompt; i++)
    model_forward_token(&model, &state, tokens[i]);

for (step = 0; step < max_new; step++) {
    next = sample_top_k(state.logits, …);
    if (next == EOS) break;
    emit(next);
    model_forward_token(&model, &state, next);
}
```

## 4. Sampling

32 000 logits → un choix :

1. mettre à l'échelle par température
2. garder top-k
3. softmax uniquement sur ces k
4. échantillonner

| Paramètre | Effet |
|-----------|-------|
| `temperature → 0` | presque greedy |
| `temperature` élevée | plus de diversité |
| `top_k` bas | vocabulaire étroit |

Pour comparer des perspectives, nous fixons `temp ≈ 0.7`, `top_k` et `--seed` lorsque nous voulons la reproductibilité.

## 5. Perturbation en runtime (le pont vers Dreaming)

Dreaming ne consiste pas seulement à générer du texte : c'est **modifier des poids** (ou des activations) et observer ce qui change.

### Techniques disponibles dans le moteur C

| `--perturb` | Mécanique | Notes |
|-------------|-----------|--------|
| `none` | pas de changement | baseline F16 |
| `mystical` / `amplify` | `amplify_subspace` : \\(w \leftarrow w + I\,(w\cdot v)\,v\\) | lentille philosophique / existentielle |
| `noise` | bruit ∝ \\|w\\| | à I élevé dégrade |
| `blockdiag` | amplifie les blocs 16×16 | parfois monotone / écho |
| `manifold` | bruit local ∝ std du bloc | I élevé peut effondrer |

Politique (comme dans `dmt_perturb_v10`) : ce sont les matrices d'**attention et FFN** des 22 couches qui sont touchées ; les normes, embeddings et `lm_head` ne sont pas touchés.

```bash
# 15 prompts avec I=0.50, seed=42 → ~8 tok/s en génération
./llm_inference modele.F16.gguf "When we dissolve the ego" \
    60 0.7 40 --seed 42 --perturb mystical --intensity 0.50
```

Exemple réel (même config) :

> *When we dissolve the ego, we dissolve the self.
> When we let go of the ego, we allow ourselves to
> become part of the universe…*

### Intensité

| I | Comportement typique |
|---|---------------------|
| 0.05–0.10 | presque baseline ; delta subtil |
| 0.30–0.50 | perspective plus claire (et parfois EOS différent) |
| noise / manifold élevés | risque de déchet |

Le setup mystique coûte ~**25 s** et ~**3.6 Go** de RAM F32 *une seule fois par processus*. Ensuite, la génération s'exécute au même ordre de grandeur que le baseline.

### Steering d'activations (`--steer`)

Complément à la perturbation de poids : on construit un vecteur direction à partir de l'embedding d'un mot et on pousse le résiduel dans cette direction pendant le forward. C'est une autre façon de *marquer* la génération sans réécrire le GGUF.

```bash
./llm_inference modele.F16.gguf "The world is" 40 0.7 40 \
    --steer amor --steer-strength 0.15
```

## Performance (ordres de grandeur)

Mesuré en CPU avec OpenMP (8 fils), TinyLlama F16 :

| Situation | Valeur typique |
|-----------|----------------|
| Génération baseline | **~6–10 tok/s** (wall, avec prefill) |
| Génération avec poids F32 post-perturb | **~6–10 tok/s** (parfois un peu plus rapide : pas dequant dans la boucle) |
| Prefill + peu de tokens (EOS précoce) | tok/s wall *bas* (le prefill pèse) |
| Chargement F16 + tokenizer | **~1 s** |
| Appliquer `mystical` (154 tenseurs → F32) | **~25 s** + ~3.6 Go |

Nous ne rivalisons pas avec les GPU ni les kernels Q4 de llama.cpp. Nous rivalisons avec l'*opacité* : ici chaque multiplication a un nom et une direction.

## Deux outils, un même modèle

| Outil | Rôle | Format |
|-------|------|--------|
| `llm_inference.c` | comprendre, enseigner, **perturber en runtime**, sondes | GGUF F16 |
| `llama-cli` | batteries massives, Q4_0, vitesse d'expérimentation | GGUF Q4_0 |

Les 240 générations de l'étude de perspectives se sont beaucoup appuyées sur `llama-cli` + GGUFs déjà perturbés. Le moteur en C ferme la boucle : nous pouvons **répéter l'idée de la perturbation sans fichier intermédiaire** et voir le forward de l'intérieur.

## Limitations honnêtes (mises à jour)

1. **Seulement F16 pour les poids** — pas de kernels Q4_0/Q6_K en C
2. **Pas de GPU** — CPU uniquement
3. **Pas de batching** — une séquence à la fois
4. **Contexte du moteur** limité (`MAX_SEQ`, par ex. 512) bien que le modèle admette 2048
5. **Perturbation F32 coûteuse en RAM** — ~3.6 Go supplémentaires
6. **Pas toutes les techniques v10/v11** sont en C (manquent lowrank, spectral, selective targeting, …)

Acceptable : l'objectif n'est pas de servir un chat à des millions d'utilisateurs. L'objectif est d'**ouvrir le crâne** d'un modèle de 1.1B et de pouvoir **tourner le cristal** de ses poids avec un drapeau.

## Comment cela se connecte au reste du livre

Jusqu'ici :

- *Qu'est-ce* que TinyLlama (chapitre 1)
- *Comment* il est organisé à l'intérieur (chapitre 2)
- *Avec quoi* nous l'exécutons, le mesurons et le **perturbons** (ce chapitre)

Le prochain chapitre entre de plein dans la **perturbation de poids et le changement de perspective** : analogie DMT, les techniques qui préservent la hiérarchie, les intensités, et pourquoi le modèle ne « se casse » pas mais *parle d'une autre voix*.

Derrière ces découvertes, il y a un moteur réel. Le nôtre est dans un fichier C, tourne en CPU, et ne lit plus seulement les poids : il sait aussi *les déplacer*.

---

*Chapitre suivant : Perturbation de Poids et Changement de Perspective*
