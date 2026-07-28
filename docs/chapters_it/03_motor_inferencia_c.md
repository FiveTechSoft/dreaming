# Capitolo 3: Il Nostro Motore di Inferenza in C per TinyLlama

## Perché scrivere un motore proprio

Per studiare TinyLlama non basta *usarlo*.
Bisogna *vederlo funzionare*.

La maggior parte dei framework di inferenza nasconde
il percorso del token dietro strati di astrazione:

- PyTorch e CUDA (gigabyte di dipendenze)
- Runtime in Python
- Kernel opachi su GPU

Noi volevamo il contrario:

1. **C leggibile** — un solo file, operazioni esplicite
2. **Senza GPU obbligatoria** — tutto su CPU
3. **Senza magia** — ogni passo del transformer è una funzione
4. **Portabile** — un binario piccolo, riproducibile

Il risultato è `llm_inference.c`: motore di
inferenza che implementa il ciclo del transformer
dal file GGUF al token successivo —
e, più tardi, **perturbazioni dei pesi in runtime**
senza bisogno di un GGUF pre-cotto.

Non è il motore più veloce del mondo.
È il motore che *capiamo riga per riga*
e con cui possiamo *toccare i pesi a fondo*.

## Cosa deve fare un motore di inferenza

Ricorda il flusso del capitolo precedente:

```
Testo → token → embedding
     → 22 layer (attenzione + FFN)
     → logits → token successivo
     → (ripeti)
```

Nella pratica il motore si divide in cinque parti:

```
┌─────────────────────────────────────────┐
│            llm_inference.c              │
├─────────────────────────────────────────┤
│  1. Lettore GGUF                        │
│     - magic, metadata, indici           │
│     - tensori F16 (e fallback tokenizer │
│       da Q4_0 se necessario)            │
├─────────────────────────────────────────┤
│  2. Tokenizer BPE                       │
│     - token + merge dallo stesso GGUF   │
│     - testo ⇄ id                        │
├─────────────────────────────────────────┤
│  3. Motore Transformer + KV-cache       │
│     - embedding, RMSNorm, RoPE          │
│     - attenzione GQA, SwiGLU            │
│     - un token nuovo per passo          │
├─────────────────────────────────────────┤
│  4. Sampling                            │
│     - temperatura, top-k                │
├─────────────────────────────────────────┤
│  5. Perturbazione / steering (opzionale)│
│     - --perturb mystical, noise, …      │
│     - --steer <parola> (attivazioni)    │
└─────────────────────────────────────────┘
```

| Blocco | Domanda |
|--------|---------|
| GGUF | Dove si trovano i pesi su disco? |
| Tokenizer | Come si converte il linguaggio in numeri? |
| Transformer | Come si trasforma la rappresentazione? |
| Sampling | Come si sceglie la parola successiva? |
| Perturbazione | Come cambiamo la *prospettiva* del modello? |

## TinyLlama in numeri (quelli che usa il motore)

| Parametro | Valore in TinyLlama-1.1B |
|-----------|--------------------------|
| Layer (`block_count`) | 22 |
| Dimensione nascosta | 2048 |
| Teste Q / KV (GQA) | 32 / 4 |
| Dimensione per testa | 64 |
| FFN intermedio | 5632 |
| Vocabolario | 32.000 |
| Contesto del modello | 2048 |
| RoPE `freq_base` | 10.000 |
| Formato dei pesi del motore | GGUF **F16** |

Nove tensori per layer (cap. 2) più quelli globali
`token_embd`, `output_norm`, `output`. Il motore non
inventa l'architettura: la *materializza* in memoria.

> **Nota su Q4_0:** gli esperimenti massicci del
> progetto usano anche GGUF quantizzati (~638 MB)
> con `llama-cli`. Il nostro C puro legge **F16**
> (~2,1 GB) per non reimplementare tutti i kernel
> di quantizzazione. Il *forward* è lo stesso transformer.

## Compilare ed eseguire

```bash
# Consigliato (Windows/MinGW o Linux)
gcc -O3 -march=native -ffast-math -fopenmp \
    -o llm_inference llm_inference.c -lm

# Variabili utili
export OMP_NUM_THREADS=8   # Linux/macOS
# PowerShell: $env:OMP_NUM_THREADS = "8"
```

### Uso base (baseline)

```bash
./llm_inference tinyllama-1.1b.F16.gguf \
    "The secret to happiness is" \
    40 0.7 40 \
    --seed 42
```

Argomenti posizionali: `modello`, `prompt`, `n_tokens`,
`temperatura`, `top_k`.

### Perturbazione mistica in runtime

```bash
./llm_inference tinyllama-1.1b.F16.gguf \
    "The secret to happiness is" \
    40 0.7 40 \
    --seed 42 \
    --perturb mystical \
    --intensity 0.50
```

Non serve un file `DMT_*.gguf` precedente:
la tecnica si applica **in memoria** sui pesi
appena caricati.

### Altre bandiere

| Flag | Effetto |
|------|---------|
| `--perturb` / `-P` | `none`, `mystical`, `amplify`, `noise`, `blockdiag`, `manifold` |
| `--intensity` / `-I` | intensità della perturbazione (es. 0,10–0,50) |
| `--seed` | PRNG della perturbazione (+ sampling se fissata) |
| `--steer` | parola la cui direzione di embedding tira il residuale |
| `--steer-strength` | intensità dello steering (es. 0,15) |

Dipendenze di *build*: `libm` e, per il matmul
parallelo, **OpenMP** (`-fopenmp`). Senza OpenMP il
motore compila ugualmente, solo più lentamente.

## 1. Leggere un GGUF in C

GGUF ha tre zone:

```
[ magic "GGUF" | versione | nº tensori | nº KV ]
[ metadata chiave-valore ]
[ indice dei tensori → dati allineati ]
```

Il motore carica l'intero file (su Windows con
I/O a 64 bit: l'A16 supera 2 GB e `ftell` a 32
bit non basta). Poi indica i tensori per nome:

```c
snprintf(name, sizeof(name), "blk.%d.attn_q.weight", l);
m->wq[l] = must_tensor_f16(&m->gguf, name);
/* wk, wv, wo, w1/w3/w2 (gate/up/down), norme… */
```

### F16 su disco, float nel matmul

I pesi del layer restano in **F16** nel buffer
del file. Nel prodotto matrice-vettore si
convertono con una **tabella da 65.536 voci**
(un valore per ogni schema di half):

```c
static float g_f16_table[65536];
/* init: g_f16_table[i] = decode_ieee_half(i); */
static inline float f16_to_f32(uint16_t h) {
    return g_f16_table[h];
}
```

Così evitiamo di decodificare bit a ogni peso nel ciclo
caldo. Solo quando chiediamo `--perturb` copiamo
le matrici del layer in **F32 mutabile** (~3,6 GB):
il delta di `amplify_subspace` è piccolo e un
round-trip in F16 lo cancellerebbe.

### Tokenizer: BPE dallo stesso GGUF

Leggiamo `tokenizer.ggml.tokens` e
`tokenizer.ggml.merges` dalla metadata, con tabelle
hash per merge rapidi.

Dettaglio pratico: alcuni F16 di TinyLlama portano
un vocabolario *troncato* (pochi token nell'header).
Se rileviamo ciò, carichiamo **solo il
tokenizer** dal Q4_0 fratello
(`tinyllama-1.1b.Q4_0.gguf`), senza toccare i pesi
F16. Gli id coincidono con HuggingFace
(es. `"Hello"` → `[1, 15043]`).

## 2. Le operazioni del transformer

### RMSNorm

```c
/* x / rms(x) * w   — w viene in F16 */
static void rmsnorm(float *out, const float *x,
                    const uint16_t *w_f16, int n) {
    float ss = 0.f;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    float scale = 1.f / sqrtf(ss / (float)n + 1e-5f);
    for (int i = 0; i < n; i++)
        out[i] = x[i] * scale * f16_to_f32(w_f16[i]);
}
```

Poco peso (`attn_norm`, `ffn_norm`), molto controllo.

### Matmul (il costo reale)

```c
/* W [out, in] · x[in] → out[out]
 * OpenMP sulle righe di uscita; F16 via LUT */
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

Con `--perturb`, lo stesso ciclo usa `matmul_f32`
sulle copie già perturbate. Q, K, V, O, gate,
up, down e `lm_head` sono tutti matmul.

### RoPE

Posizione senza embedding assoluti: ogni coppia di
dimensioni di Q e K viene ruotata. Il motore precalcola
sin/cos per posizione fino a `MAX_SEQ` per non chiamare
`sinf`/`cosf` nel percorso caldo.

### Attenzione con GQA + KV-cache

```
score(t) = (Q_pos · K_t) / sqrt(head_dim)
pesi     = softmax_causal(scores)   # solo 0..pos
uscita   = Σ pesi[t] * V_t
```

TinyLlama: **32 teste Q, 4 KV** → ogni testa KV
attende a 8 teste Q (`hkv = h / 8`).

A ogni passo di generazione calcoliamo solo Q/K/V
del **token nuovo**, salviamo K e V nella cache
`[layer][pos][kv_dim]` e attendiamo su
`0 .. pos`. Questo è ciò che rende fattibile la CPU:
senza KV-cache ricalcoleremmo tutta la sequenza
a ogni token.

### SwiGLU (FFN)

```
h' = Down( SiLU(Gate(x)) ⊙ Up(x) )
```

~69% dei parametri (cap. 2): la "memoria"
pratica del modello. Dopo l'attenzione, residuale;
dopo il FFN, altro residuale.

## 3. Un passo forward (un token)

In linguaggio umano, per layer e per *posizione attuale*:

```
x = embedding(token)

per L = 0 .. 21:
    h = RMSNorm(x, attn_norm[L])
    Q,K,V = proiezioni(h);  RoPE(Q,K)
    salva K,V in cache[L][pos]
    x = x + O( Attenzione(Q, cache_K, cache_V) )

    h = RMSNorm(x, ffn_norm[L])
    x = x + SwiGLU_FFN(h)

logits = output · RMSNorm(x, output_norm)
```

Nel codice questo è `model_forward_token(...)`.
La generazione:

```c
/* prefill: ogni token del prompt riempie la cache */
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

32.000 logits → una scelta:

1. scalare per temperatura  
2. tenere top-k  
3. softmax solo su quei k  
4. campionare  

| Parametro | Effetto |
|-----------|---------|
| `temperature → 0` | quasi greedy |
| `temperature` alta | più diversità |
| `top_k` basso | vocabolario ristretto |

Per confrontare prospettive fissiamo `temp ≈ 0,7`,
`top_k` e `--seed` quando vogliamo riproducibilità.

## 5. Perturbazione in runtime (il ponte a Dreaming)

Dreaming non è solo generare testo: è **modificare
i pesi** (o le attivazioni) e osservare cosa cambia.

### Tecniche disponibili nel motore C

| `--perturb` | Meccanica | Note |
|-------------|-----------|------|
| `none` | nessuna modifica | baseline F16 |
| `mystical` / `amplify` | `amplify_subspace`: \(w \leftarrow w + I\,(w\cdot v)\,v\) | lente filosofica / esistenziale |
| `noise` | rumore ∝ \|w\| | a I alto degrada |
| `blockdiag` | amplifica blocchi 16×16 | a volte monotono / eco |
| `manifold` | rumore locale ∝ std del blocco | I alto può collassare |

Politica (come in `dmt_perturb_v10`): si toccano le
matrici di **attenzione e FFN** dei 22 layer;
non si toccano norme, embedding né `lm_head`.

```bash
# 15 prompt con I=0,50, seed=42 → ~8 tok/s in generazione
./llm_inference modello.F16.gguf "When we dissolve the ego" \
    60 0.7 40 --seed 42 --perturb mystical --intensity 0.50
```

Esempio reale (stessa configurazione):

> *When we dissolve the ego, we dissolve the self.
> When we let go of the ego, we allow ourselves to
> become part of the universe…*

### Intensità

| I | Comportamento tipico |
|---|----------------------|
| 0,05–0,10 | quasi baseline; delta sottile |
| 0,30–0,50 | prospettiva più chiara (e a volte EOS diverso) |
| noise / manifold alti | rischio di rifiuti |

Il setup mistico costa ~**25 s** e ~**3,6 GB** di
RAM F32 *una volta per processo*. Dopo, la
generazione corre dello stesso ordine di grandezza
della baseline.

### Steering delle attivazioni (`--steer`)

Complemento alla perturbazione dei pesi: si costruisce
un vettore direzione dall'embedding di una
parola e si spinge il residuale in quella direzione
durante il forward. È un'altra forma di *segnare* la
generazione senza riscrivere il GGUF.

```bash
./llm_inference modello.F16.gguf "The world is" 40 0,7 40 \
    --steer amor --steer-strength 0,15
```

## Prestazioni (ordini di grandezza)

Misurato su CPU con OpenMP (8 thread), TinyLlama F16:

| Situazione | Valore tipico |
|------------|----------------|
| Generazione baseline | **~6–10 tok/s** (wall, con prefill) |
| Generazione con pesi F32 post-perturb | **~6–10 talvolta un po' più veloce: non dequant nel ciclo) |
| Prefill + pochi token (EOS anticipato) | tok/s wall *bassa* (il prefill pesa) |
| Caricamento F16 + tokenizer | **~1 s** |
| Applicare `mystical` (154 tensori → F32) | **~25 s** + ~3,6 GB |

Non competiamo con GPU né con kernel Q4 di
llama.cpp. Competiamo con l'*opacità*: qui ogni
moltiplicazione ha nome e direzione.

## Due strumenti, lo stesso modello

| Strumento | Ruolo | Formato |
|-----------|-------|---------|
| `llm_inference.c` | capire, insegnare, **perturbare in runtime**, sonde | GGUF F16 |
| `llama-cli` | batterie massive, Q4_0, velocità di sperimentazione | GGUF Q4_0 |

Le 240 generazioni dello studio di prospettive
si sono appese molto a `llama-cli` + GGUF già
perturbati. Il motore C chiude il cerchio:
possiamo **ripetere l'idea della perturbazione senza
file intermedio** e vedere il forward dall'interno.

## Limitazioni oneste (aggiornate)

1. **Solo F16 nei pesi** — senza kernel Q4_0/Q6_K in C
2. **Senza GPU** — solo CPU
3. **Senza batching** — una sequenza alla volta
4. **Contesto del motore** limitato (`MAX_SEQ`, es. 512)
   anche se il modello ammette 2048
5. **Perturbazione F32 costosa in RAM** — ~3,6 GB extra
6. **Non tutte le tecniche v10/v11** sono in C
   (mancano lowrank, spectral, selective targeting, …)

Accettabile: l'obiettivo non è servire una chat a
milioni di utenti. L'obiettivo è **aprire il
cranio** di un modello di 1,1B, e poter **girare il
cristallo** dei suoi pesi con una bandiera.

## Come si collega al resto del libro

Fino a qui:

- *Cos'*è TinyLlama (capitolo 1)
- *Come* è organizzato all'interno (capitolo 2)
- *Con cosa* lo eseguiamo, lo misuriamo e lo
  **perturbiamo** (questo capitolo)

Il prossimo capitolo entra nel vivo della
**perturbazione dei pesi e del cambiamento di prospettiva**:
analogia DMT, le tecniche che preservano la gerarchia,
intensità, e perché il modello non "si rompe"
ma *parla con un'altra voce*.

Sotto quei risultati c'è un motore reale.
Il nostro sta in un file C, gira su CPU,
e ormai non legge più solo i pesi: sa anche *muoverli*.

---

*Capitolo successivo: Perturbazione dei Pesi e Cambiamento di Prospettiva*
