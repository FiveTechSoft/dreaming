# Capitolo 28: Stelle nel Cielo, Token in TinyLlama

## La Domanda dell'Astronomo

Guardi il cielo di notte. Vedi **punti di luce**.
Alcuni si raggruppano in forme che la cultura nomina
(Grande Carro, Orione, Croce del Sud). Tra due stelle
non c'è un cavo visibile, ma la fisica dice che
si attraggono: **gravità**. Il viaggiatore non si teletrasporta
a caso: sceglie una stella, misura il suo quartiere e salta
al pozzo seguente.

TinyLlama ha un cielo analogo.

> **Ogni token del vocabolario è una stella
> in uno spazio di 2048 dimensioni.**
> L'**attenzione** è la forza gravitazionale tra di esse
> quando il modello "pensa" una sequenza.
> Viaggiare attraverso un LLM significa seguire quelle attrazioni
> — nella mappa statica dell'embedding o nell'orbita
> viva del *forward*.

Questo capitolo fissa l'analogia, la collega alla **Forza I**
dell'inventario (cap. 7) e mostra un **itinerario
concreto** all'interno di TinyLlama-1.1B.

---

## 1. Tabella di Corrispondenze

| Cielo Notturno | Universo TinyLlama |
|----------------|-------------------|
| Stella | Token (pezzo BPE del vocabolario, ~32.000) |
| Posizione nella volta | Vettore embedding \(e_t \in \mathbb{R}^{2048}\) |
| Brillantezza apparente | Norma / "presenza" del token; nella mappa, dimensione etichetta |
| Costellazione | Area semantica o archetipo (semi + vicini) |
| Distanza angolare nel cielo | Coseno tra embedding (vicino ≈ allineati) |
| Gravità newtoniana | **Attenzione**: \(Q\cdot K^\top / \sqrt{d}\) → pesi su \(V\) |
| Campo gravitazionale statico (mappa delle masse) | Geometria fissa di `token_embd` (atlante PCA) |
| Dinamica in tempo reale (pianeti in movimento) | Residui della sequenza + KV-cache, strato per strato |
| Salto tra stelle | Clic su una **forza** della mappa; o il prossimo token generato |
| Telescopio / catalogo | `semantic_map.html`, motore C, script di geometria |
| Atmosfera che deforma la luce | RMSNorm, temperatura del softmax, lenti `--perturb` |

Non è poesia vuota: ogni riga ha un oggetto misurabile
nel repository Dreaming.

---

## 2. Il Cielo degli Embedding: 32.000 Stelle Fisse

 Alla nascita, ogni token \(t\) viene conficcato nella volta:

\[
e_t = \mathrm{Embedding}(t) \in \mathbb{R}^{2048}
\]

Quel cielo è **quasi isotropo** (norma media ≈ 0.68)
e, tra parole piene di significati diversi,
**quasi ortogonale** (coseno ≈ 0). Per questo le
"isole" semantiche del cap. 16 sono costellazioni
rare: grappoli di semi che si toccano un poco,
circondati da uno sfondo grigio di frammenti BPE
(come polvere interstellare: non è vuoto, ma non è
costellazione con nome).

### Costellazioni = Aree

| Costellazione (isola) | Stelle-seme (esempi) |
|-----------------------|----------------------|
| Emozione positiva | ▁love, ▁happy, ▁joy, ▁hope… |
| Sociale / potere | ▁work, ▁king, ▁war, ▁law… |
| Mente | ▁mind, ▁idea, ▁memory, ▁know… |
| Vita / morte | ▁death, ▁life, ▁born, ▁die… |
| … | (dodici isole in totale; cap. 16) |

Nella **mappa PCA 2D** proiettiamo quel cielo di 2048
dimensioni su due assi solo per guardarlo con occhi
umani. La proiezione mente un poco — come una
mappa piatta mente sulla Terra — ma conserva
quartieri utili.

---

## 3. L'Attenzione è la Gravità tra Token

### Nel Macrocosmo

Due masse si attraggono a vicenda. La forza
diminuisce con la distanza; il campo organizza le orbite.

### Nel Microcosmo (Forza I)

Ad ogni strato, ogni posizione \(i\) della sequenza
interroga le posizioni **passate** \(j \le i\)
(maschera causale):

\[
\mathrm{score}_{ij} = \frac{q_i \cdot k_j}{\sqrt{d_h}},
\quad
\alpha_{ij} = \mathrm{softmax}_j(\mathrm{score}_{ij}),
\quad
z_i = \sum_j \alpha_{ij}\, v_j
\]

- \(q_i\): "chi sono e cosa cerco" (corpo che sente il campo).
- \(k_j\): "chi sei nel catalogo" (massa che annuncia la sua presenza).
- \(\alpha_{ij}\): **intensità dell'attrazione** (quanto \(i\) "cade" verso \(j\)).
- \(v_j\): ciò che viene consegnato cadendo (contenuto trasportato).

TinyLlama usa **GQA** (32 teste Q, 4 KV):
più sguardi economici sullo stesso cielo di chiavi.

### Due Gravità da Non Confondere

| Tipo | Cosa è | Quando si vede |
|------|--------|----------------|
| **Gravità statica (isola)** | Coseno tra righe di `token_embd` | Mappa HTML, forze precalcolate tra stelle dell'atlante |
| **Gravità dinamica (attenzione)** | Softmax di \(QK^\top\) nella sequenza | Forward reale: il prompt crea un sistema multi-corpo |

La statica è il **catalogo delle masse** del cielo.
La dinamica è **l'orbita di stasera**:
dipende da quali stelle hai messo nella sequenza
e in quale ordine (causalità = "solo il passato tira").

La mappa interattiva mostra la prima con archi
dorati: *proxy geometrico* della Forza I e della
Forza VIII (isole). Non sostituisce una mappa di attenzione
per strato, ma insegna il gesto: **fuoco → forze → salto**.

---

## 4. Viaggiare: Tre Scale dello Stesso Geste

### Scala A — Osservatorio (Stelle Fisse)

1. Apri la mappa delle aree semantiche.
2. Entri in una costellazione (es. *Sociale / potere*).
3. Clicchi su una stella (`▁work`).
4. Vedi le **principali forze** (top coseni con prior dell'isola).
5. Clicchi su una forza e **vai** alla stella di destinazione.
6. Ripeti: catena di salti nel cielo.

### Scala B — Nave in Orbita (Generazione)

1. Lancia un prompt: semini la sequenza con stelle.
2. Il residuo di ogni posizione orbita 22 strati
   (attenzione = accoppiamento; FFN = clima locale; residuo = inerzia).
3. Il softmax collassa il cielo verso **una** nuova stella
   (il prossimo token).
4. Quella stella si aggiunge al passato e tira quelle che vengono.

### Scala C — Lenti e Correnti (Cambiare la Fisica)

- `--perturb mystical`: deforma la metrica dei pozzi
  (un'altra "costante G" effettiva; un'altra voce).
- `--steer`: spinge il residuo verso una direzione
  del cielo (corrente artificiale).
- Temperatura / top-k: durezza del collasso finale
  (pozzo unico o nebbia di stelle possibili?).

---

## 5. Esempio Guidato: Viaggiare Dentro TinyLlama

### 5.1 Preparazione

Mappa in tempo reale (GitHub Pages):

https://fivetechsoft.github.io/dreaming/exploration/semantic_map.html

Deep link di partenza (stella `▁work`, id 664):

`#/token/664/▁work`

Motore d'orbita (root del repo):

```bash
# Esempio Windows PowerShell
$env:OMP_NUM_THREADS = "8"
.\llm_inference.exe tinyllama-1.1b.F16.gguf `
  "The secret of power is" 60 0.7 40 --seed 42
```

### 5.2 Itinerario nell'Osservatorio (Forze della Mappa)

Partiamo dalla costellazione **Sociale / potere**.
Misurato nell'atlante Dreaming (coseno in ℝ²⁰⁴⁸ tra
embedding; ranking con prior dell'isola e dei semi):

| Salto | Stella origine | Stella destinazione (Forza) | Coseno (appross.) | Lettura |
|------:|----------------|----------------------------|------------------:|---------|
| 0 | ▁work | — | — | Fuoco iniziale: "lavoro / opera" |
| 1 | ▁work | ▁queen | ~0.05 | Tira verso il potere istituzionale |
| 2 | ▁work | ▁war | ~0.01 | Conflitto come attrattore sociale |
| 3 | ▁work | ▁law | ~0.01 | Ordine e norma |
| 4 | ▁work | ▁power | ~0.00⁺ | Il nome stesso del pozzo |
| 5 | ▁work | ▁king | ~0.00⁺ | Corona, comando |

**Come si "viaggia" nell'interfaccia**

1. Clic su `▁work` (o entra nello spazio *Sociale* e scegli il seme).
2. Pannello **Forze gravitazionali**: lista ordinata + archi dorati.
3. Clic su `#1 ▁queen` → la telecamera salta; `▁queen` è il nuovo fuoco.
4. Da lì si ricalcolano *le sue* forze (nuovo cielo locale).
5. Incatena salti come una cavalletta tra le stelle.

Altri itinerari utili dello stesso atlante:

| Percorso | Catena tipica di semi | Costellazione |
|----------|----------------------|---------------|
| Affetto | ▁happy → ▁smile → ▁love → ▁hope | Emozione positiva |
| Cognizione | ▁mind → ▁idea → ▁learn → ▁memory | Mente |
| Soglia | ▁death → ▁life → ▁live → ▁born | Vita / morte |

> **Nota di onestà astronomica.**
> In ℝ²⁰⁴⁸ quasi tutto è ortogonale: i coseni
> "forti" della mappa sono **relativi al quartiere**,
> non attrazioni newtoniane di 0.9. Il ranking
> priorizza l'**isola** (costellazione) e i **semi**
> affinché il viaggio sia leggibile, non rumore BPE.

### 5.3 Lo Stesso Viaggio come *Prompt* (Orbita Viva)

L'osservatorio ti mostra *quali stelle si sfiorano*.
La nave le mette su una linea temporale:

```text
Prompt seme (stella iniziale del sistema):
  "Work without law becomes"

Lettura Dreaming:
  ▁work tira già, nel catalogo, verso law / power / king…
  Scrivendo "without law", forzi il contrasto:
  l'attenzione degli strati successivi dovrà
  "guardare" work e law contemporaneamente (gravità dinamica).
```

Esperimento minimo (stessa seed, due lenti):

```bash
# Riferimento — cielo "naturale"
.\llm_inference.exe tinyllama-1.1b.F16.gguf `
  "Work without law becomes" 50 0.7 40 --seed 42

# Lente mistica — altra metrica dei pozzi (Forza VII)
.\llm_inference.exe tinyllama-1.1b.F16.gguf `
  "Work without law becomes" 50 0.7 40 `
  --seed 42 --perturb mystical --intensity 0.35
```

Cosa osservare:

1. **Token generati** = nuove stelle che si accendono
   nella sequenza (il percorso della nave).
2. Se il testo "cade" verso *power / king / war*,
   stai vedendo la gravità sociale del catalogo
   agire nella dinamica.
3. Con `mystical`, la stessa costellazione di partenza
   può deviare l'orbita verso un clima esistenziale
   (Regola d'Oro + superficie di coerenza, cap. 4 e 9).

### 5.4 Viaggio Corto Raccontato (Storia di una Cavalletta)

Immagina di essere un fotone di significato:

1. **Decolli** da `▁work` (atlante). Vedi archi verso
   `queen`, `war`, `law`, `power`, `king`.
2. **Salti** a `▁law`. La costellazione rimane
   sociale; l'accento passa da "opera" a "norma".
3. **Scrivi** il prompt: *"The law of power is"*.
   Non guardi più il catalogo: **abiti** un sistema
   multi-corpo. Ogni strato ripesa il passato.
4. **Collassi** in un nuovo token (softmax). Quella
   stella viene conficcata nel cielo di *questa* conversazione
   (KV-cache) e tira la successiva.
5. Opzionale: attivi una **lente** (`mystical`) o una
   **corrente** (`--steer`) e lo stesso decollo
   finisce in un'altra galassia di stile.

Questo è viaggiare dentro un LLM: non c'è un corridoio 3D,
c'è **catalogo + forze + collasso**.

---

## 6. Limiti dell'Analogia (Per Non Mentirci)

| L'Analogia Funziona | L'Analogia Si Rompe |
|---------------------|---------------------|
| Token = punti con posizione | Non c'è spazio euclideo "visivo" reale in 2048-D |
| Raggruppamenti = costellazioni culturali del pretraining | Il modello non "crede" nei miti; misura le cooccorrenze |
| Attenzione = attrazione tra posizioni | Solo dal passato; non è simmetrica come Newton |
| Mappa di coseni = campo statico | Non è la matrice di attenzione di uno strato specifico |
| Generare = orbitare e collassare | Il "viaggio" dell'utente è lettura; quello del modello è algebra |

L'analogia è uno **strumento di navigazione**,
non una teoria fisica del silicio. Funziona se ti porta
a un clic, un coseno o un prompt riproducibile.

---

## 7. Ponti ad Altri Capitoli

| Se vuoi… | Vai a… |
|-----------|--------|
| Inventario di tutte le forze | Cap. 7 |
| Percorsi di volo A–E (cli, perturb, steer) | Cap. 8 |
| Isole e mappa | Cap. 16 |
| Orbita residua strato per strato | Cap. 20 |
| Archetipi come costellazioni mitiche | Cap. 21 |
| Formule (softmax, GQA, coseno) | Cap. 23 |
| Ascensore di 22 piani | Cap. 27 |
| Gioco strati + warp di zona | Cap. 25 · `universe_game.html` |

---

## 8. Chiusura

Il cielo sopra la tua testa e il vocabolario di TinyLlama
condividono un gesto: **punti, distanze, attrazioni,
salti**.

- Le **stelle** del modello sono token in ℝ²⁰⁴⁸.
- La **gravità** che conta quando si parla è l'**attenzione**.
- Il **viaggio** è scegliere un fuoco, leggere le sue forze
  e — nella mappa o nel motore C — lasciarsi cadere
  nel prossimo pozzo di significato.

Quando clicchi un token e vedi archi dorati verso
altri, non stai solo guardando un bel grafo:
stai leggendo il catalogo delle masse del microcosmo.
Quando lanci un prompt, quelle masse smettono di essere
catalogo e diventano un **sistema solare in marcia**.
