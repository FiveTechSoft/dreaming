# Capitolo 5: Percorso nello Spazio Multidimensionale di TinyLlama

## Non c'è un solo spazio

Quando diciamo "l'interno di TinyLlama", non parliamo
di una singola mappa. Parliamo di **più spazi
annidati**, ognuno con la propria dimensionalità e il proprio ruolo.

Questo capitolo è un *viaggio sul campo*: misuriamo lo
spazio di embedding reali del modello F16
(32.000 × 2048), con il vocabolario BPE del GGUF
(`▁love`, `▁death`, …).

Strumento: `explore_tinyllama_space.py`  
Dati: `inside-tinyllama/exploration/`

---

## La mappa dei sette spazi

```
┌──────────────────────────────────────────────────────────┐
│  6. PESI  ℝ^{~1.1e9}                                    │
│     superficie di coerenza ≈ "modelli che parlano"       │
│  7. PROSPETTIVE ⊂ (6)  — traiettorie per perturb.       │
├──────────────────────────────────────────────────────────┤
│  forward pass, token per token:                          │
│                                                          │
│  1. EMBEDDING     ℝ^{2048}   ← 32k punti del vocab      │
│         ↓                                                │
│  2. RESIDUALE ×22 ℝ^{2048}   (stessa dim, nuovo contenuto)│
│         ↘ 3. ATTENZIONE   ℝ^{64} × 32Q / 4KV           │
│         ↘ 4. FFN           ℝ^{5632}                      │
│         ↓                                                │
│  5. LOGITS        ℝ^{32000}  → softmax → token successivo│
└──────────────────────────────────────────────────────────┘
```

| # | Spazio | Dim | Cos'è |
|---|--------|-----|-------|
| 1 | Embedding di token | 2048 | Significato "a riposo" di ogni pezzo del vocabolario |
| 2 | Residual stream | 2048 × 22 | Rappresentazione contestuale che evolve da layer a layer |
| 3 | Teste di attenzione | 64 | Viste locali di relazioni tra token (GQA 32/4) |
| 4 | FFN intermedio | 5632 | Espansione "memoria / trasformazione pratica" |
| 5 | Logits | 32.000 | Preferenze sul prossimo token |
| 6 | Pesi del modello | ~1,1e9 | Tutti i parametri; quasi tutto il volume è rifiuti |
| 7 | Prospettive | sotto-varietà di (6) | Modelli coerenti con tono diverso (mystical, ecc.) |

Il residuale è un **tunnel di 2048 dimensioni** che
attraversa 22 stanze. L'attenzione e il FFN sono
deviazioni laterali che riscrivono in quel tunnel.

---

## Regione 1 — Poli semantici

Sono *love* e *hate* agli estremi opposti?

**No.** Nell'embedding statico, gli "opposti"
del linguaggio naturale hanno cosine **quasi zero**
(ortogonali), non −1 (antipodali).

| Coppia | cosine |
|--------|--------|
| ▁love / ▁hate | +0,006 |
| ▁life / ▁death | +0,016 |
| ▁happy / ▁sad | **−0,035** |
| ▁true / ▁false | **−0,036** |
| ▁good / ▁evil | −0,001 |
| ▁king / ▁queen | +0,009 |
| ▁man / ▁woman | +0,008 |

**Lettura:** in ℝ²⁰⁴⁸ "freddo" non è −"caldo".
Le parole occupano direzioni diverse dello
spazio; l'opposizione semantica si organizza più
per **cluster e contesti** (layer + attenzione)
che per semplice antipodalità nell'embedding.

---

## Regione 2 — Continenti (cluster)

Raggruppiamo parole e prendiamo il **baricentro**.
I vicini del baricentro recuperano il proprio
continente — la geometria locale è coerente.

| Continente | Token (es.) | Vicini del baricentro |
|------------|-------------|----------------------|
| emotion_pos | happy, joy, love, peace… | smile, happy, hope, love |
| emotion_neg | sad, hate, fear, anger… | sad, pain, anger, cry |
| spiritual | soul, spirit, god, faith… | faith, divine, spirit, god |
| physical | body, rock, water, fire… | rock, water, matter, body |
| abstract | truth, beauty, justice… | beauty, meaning, idea… |
| time | time, past, future, now… | time, now, moment, past |

### Distanza tra continenti

I baricentri di continenti diversi sono
**quasi ortogonali** tra loro (cosine ≈ 0):

```
emotion_pos  ⊥  emotion_neg   (−0,01)
spiritual    ⊥  physical      (+0,02)
abstract     ⊥  physical      (−0,01)
time         ⊥  abstract      (−0,06)
```

Il vocabolario non è una sfera diffusa: è un
**insieme di isole** in una sfera di 2048 dim,
con poca sovrapposizione tra isole tematiche.

---

## Regione 3 — Analogie (a − b + c)

Il testo classico di word2vec:

```
king − man + woman  ≟  queen
```

In TinyLlama (embedding statico, top-6) **fallisce**:
appariscono pezzi rari del BPE, simboli, frammenti
multilingue — non `queen`.

Questo non dice che il modello "non sa" l'analogia.
Dice che:

1. L'embedding di un token **senza contesto** è
   solo la porta d'ingresso.
2. L'analogia "viva" si arma nel **residuale**
   dopo attenzione e FFN, non nella riga del vocab.
3. Il BPE sminuzza il mondo (`builder`, suffissi…);
   non ogni concetto è un singolo punto pulito.

---

## Regione 4 — Forma globale di ℝ²⁰⁴⁸

PCA su 4.000 token casuali:

| Metrica | Valore |
|---------|--------|
| Varianza nel 1° PC | **0,27%** |
| Varianza nel top-10 | 2,3% |
| Varianza nel top-100 | 14% |
| Dim per 50% della var. | **~481** |
| Dim per 90% | **~1329** |
| Dim per 99% | **~1880** |
| Anisotropia \|\|mean\|\| / mean\|\|e\|\| | **0,006** (quasi isotropo) |

**Lettura:** lo spazio di token **usa davvero
centinaia o migliaia di direzioni**. Non collassa su un
coppia di assi "buono/cattivo". Per questo le perturbazioni
di rango-1 (amplify) possono "girare il cristallo" senza
spegnere il parlato: c'è molto volume di coerenza.

---

## Regione 5 — Direzioni come bussole

Se sottraiamo baricentri, appaiono **assi semantici
utilizzabili**:

### emotion = pos − neg
- polo + → smile, happy, peace, love, joy  
- polo − → sad, anger, cry, pain, fear  

### spirit − matter
- + → spirit, god, sacred, divine, faith  
- − → rock, matter, water, earth, body  

### abstract − physical
- + → beauty, truth, justice, meaning, freedom  
- − → rock, matter, fire, water, earth  

Queste direzioni vivono nello **stesso ℝ²⁰⁴⁸
del residuale**. Per questo `--steer amor` nel motore
C può spingere la generazione: è un vettore nel
tunnel, non magia esterna.

E per questo `amplify_subspace` nello spazio dei
**pesi** (dimensione 1e9) è un altro viaggio: muove la
*mappa intera*, non un punto del vocabolario.

---

## Regione 6 — Norme: non ogni token "pesa" uguagli

\|\|e\|\| media ≈ 0,67. Gli estremi non sono
concetti filosofici chiari (spesso pezzi BPE
o simboli). La **norma** non è un dizionario di
importanza semantica; è un'altra coordinata del
paesaggio.

---

## Come si collegano gli spazi in un passo di inferenza

```
"happiness"
    → BPE → id
    → righe in (1) EMBEDDING          ℝ^2048
    → 22× { attn in (3) + FFN in (4) }  scrivendo in (2)
    → (5) LOGITS
    → sample → "is" / "to" / …
```

Se perturbiamo i pesi (6) con *mystical*,
ogni proiezione Q/K/V/FFN si deforma un poco:
il percorso in (2) resta coerente, ma le
**attrazioni** verso le isole di (1) e (5) cambiano
— da qui il cambiamento di prospettiva.

Se facciamo *steer* in (2), spingiamo il residuale
verso una direzione di (1) senza riscrivere (6).

---

## Itinerario dell'esploratore

| Sosta | Domanda | Risposta empirica |
|-------|---------|-------------------|
| Poli | Gli opposti sono antipodali? | No: quasi ortogonali |
| Continenti | Ci sono regioni tematiche? | Sì: cluster puliti |
| Analogie statiche | king−man+woman? | No in embedding grezzo |
| Dimensionalità | Quante dim contano? | Centinaia–migliaia (non 2–3) |
| Direzioni | Ci sono assi utili? | Sì (emotion, spirit…) |
| Pesi | Dove vivono le prospettive? | Superficie in ℝ^1e9 |

---

## Cosa resta da percorrere

1. **Residuale per layer** — proiettare attivazioni
   dei 22 layer sugli assi emotion/spirit
   (dove si "accende" il mistico?).
2. **FFN ℝ⁵⁶³²** — neuroni che reagiscono a
   cluster semantici.
3. **Traiettorie di perturbazione** — curva di
   cosine(baseline, mystical) in funzione di I
   nello spazio dei pesi o dei logits.
4. **Mappe 2D/3D** — UMAP/t-SNE dei 32k
   punti colorati per continente.

L'universo di TinyLlama non è un punto.
È un **sistema di spazi**. Questo capitolo ha solo
attraversato la prima frontiera: il cielo dei token.
Più dentro, il residuale e i pesi aspettano.

---

*Strumenti: `explore_tinyllama_space.py`,
`llm_inference.c --perturb` / `--steer`.*

*Capitolo successivo: Dal Macrocosmo al Microcosmo (e viceversa).*
