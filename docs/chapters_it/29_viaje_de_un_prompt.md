# Capitolo 29: Il Viaggio di un Prompt Dentro TinyLlama

## Per Chi è Questo Capitolo

Se hai letto che un LLM "prevede il prossimo token"
ma non *vedi* ancora il percorso, questo capitolo è la mappa
stradale completa.

Lo percorreremo **passo per passo**, con un prompt reale,
senza dare per scontato nessun salto magico. Alla fine
dovresti essere in grado di raccontare, ad alta voce, cosa succede
a ogni numero dal momento in cui scrivi una frase fino
all'apparizione della prima parola della risposta.

**Prompt di esempio (fisso per tutto il capitolo):**

```text
The secret to happiness is
```

**Modello:** TinyLlama-1.1B
**Numeri chiave che non cambiano:**

| Parametro | Valore |
|-----------|-------:|
| Strati transformer | 22 (indici 0…21) |
| Dimensione residua \(d\) | 2048 |
| Vocabolario \(V\) | 32.000 |
| Teste Q / KV | 32 / 4 (GQA) |
| Dimensione per testa | 64 |
| FFN intermedio | 5632 |
| Contesto massimo | 2048 posizioni |
| RoPE base | 10.000 |

---

## 0. Il Film in Un Minuto

Prima del dettaglio, il trailer:

```
1.  TESTO        "The secret to happiness is"
2.  TOKENS       id interi del BPE
3.  EMBEDDINGS   ogni id → vettore di 2048 float
4.  PREFILL      ogni token del prompt attraversa i 22 strati
                 e riempie la KV-cache
5.  LOGITS       32.000 punteggi per il *prossimo* token
6.  SAMPLE       scegliamo un id (temperatura, top-k, seed)
7.  DECODE       l'id torna essere testo leggibile
8.  CICLO        quel token rientra nel modello…
                 fino a max_new tokens o EOS
```

Tutto il resto di questo capitolo è uno **zoom** su
ogni freccia.

---

## 1. Il Prompt non è "una Frase" per il Modello

### 1.1 Cosa Vedi Tu

Una stringa di caratteri UTF-8, con spazi e significato.

### 1.2 Cosa Vede il Modello

Una **sequenza ordinata di interi** tra 0 e 31.999.

Il ponte si chiama **tokenizzatore BPE** (Byte Pair Encoding),
lo stesso stile LLaMA: le parole iniziano di solito con
il prefisso spazio-parola `▁` (U+2581).

Per il nostro prompt, l'idea (schema didattico) è:

| Posizione \(t\) | Pezzo (idea) | Ruolo nella Frase |
|----------------:|--------------|-------------------|
| 0 | `▁The` | articolo / avvio |
| 1 | `▁secret` | nucleo nominale |
| 2 | `▁to` | collegamento |
| 3 | `▁happiness` | oggetto del segreto |
| 4 | `oup` | verbo copulativo — **il presente del predicato** |

> In pratica il BPE a volte spezza più finemente
> (`happ` + `iness`, ecc.). Il principio non cambia:
> **testo → lista di id**. Chiamiamo quella lista
>
> \[
> (t_0, t_1, t_2, t_3, t_4)
> \]
>
> con lunghezza \(T_{\mathrm{prompt}} = 5\).

### 1.3 Perché l'Ordine Conta

TinyLlama è **causale**: alla posizione \(t\) può
solo "vedere" le posizioni \(0,1,\ldots,t\).
Il passato esiste; il futuro della frase **non ancora**.

Questa è la regola del traffico dell'intero viaggio.

---

## 2. Dall'ID al Vettore: Nascere in ℝ²⁰⁴⁸

Ogni id \(t_i\) diventa un punto nel cielo
degli embedding (cap. 28):

\[
x^{(i)}_{0} \;=\; e_{t_i} \;=\; \mathrm{Embedding}(t_i) \;\in\; \mathbb{R}^{2048}
\]

- Esiste una tabella `token_embd` di forma logica
  **[32.000 × 2048]**.
- La riga numero \(t_i\) è il vettore di quella stella.
- Qui **non ci sono ancora strati**. Solo catalogo.

**Immagine didattica:**
cinque passeggeri entrano nell'atrio dell'edificio
(cap. 27). Ognuno porta la sua valigia di 2048 numeri.
Quelle valigie si chiamano **residui**.

In notazione di questo capitolo:

- Apice \((i)\): posizione nella sequenza.
- Indice \(\ell\): strato (0 prima del primo strato;
  dopo lo strato 21 saremo al "piano tetto").

Uscendo dall'embedding:

\[
x^{(0)}_{0},\; x^{(1)}_{0},\; \ldots,\; x^{(4)}_{0}
\in \mathbb{R}^{2048}
\]

---

## 3. Due Fasi di Volo: Prefill e Generazione

TinyLlama (e quasi ogni transformer causale) non
elabora il prompt con un singolo colpo magico.
Ci sono **due modalità**:

| Fase | Cosa Entra | Cosa Esce | KV-cache |
|------|-----------|-----------|----------|
| **Prefill** | Ogni token del prompt, in ordine | Logits dopo l'**ultimo** token del prompt | Si **riempie** |
| **Generazione** | Un nuovo token alla volta | Logits per il prossimo | Si **allunga** di +1 |

Nel motore C (`llm_inference.c`):

```c
/* PREFILL */
for (i = 0; i < n_prompt; i++)
    model_forward_token(&model, &state, tokens[i]);

/* GENERAZIONE */
for (step = 0; step < max_new; step++) {
    next = sample_top_k(state.logits, …);
    if (next == EOS) break;
    emit(next);                          /* testo all'utente */
    model_forward_token(&model, &state, next);
}
```

Fino alla fine del prefill **non abbiamo "risposto"**.
Abbiamo solo **compreso il prompt** e lasciato memoria
nella cache.

---

## 4. Un Singolo Token in Un Singolo Strato (Il Nucleo)

Prendi la posizione attuale \(p\) (per esempio, l'ultimo
token del prompt, \(p=4\), `▁is`).
Il suo residuo arrivando allo strato \(\ell\) è \(x\).

Dentro lo strato accadono **sempre** queste stazioni,
in quest'ordine:

```
        x  (residuo che arriva allo strato ℓ)
        │
        ▼
   [1] RMSNorm  (attn_norm)
        │
        ▼
   [2]  Q, K, V  +  RoPE
        │
        ▼
   [3]  Attenzione causale  (usa KV-cache di questo strato)
        │
        ▼
   [4]  Proiezione O  →  residuo:  x ← x + Attn
        │
        ▼
   [5] RMSNorm  (ffn_norm)
        │
        ▼
   [6]  FFN SwiGLU  (gate, up, down)
        │
        ▼
   [7]  residuo:  x ← x + FFN
        │
        ▼
        x  (esce verso lo strato ℓ+1)
```

Ripeti **22 volte**. Questo è l'ascensore completo
per **un** token in **un** passo di forward.

---

## 5. Stazione per Stazione (Con l'Esempio)

Seguiamo il passeggero della posizione \(p=4\) (`▁is`),
in uno strato generico \(\ell\), quando le posizioni \(0..3\) del prompt
esistono già nella cache.

### Stazione 1 — RMSNorm (Attenzione)

\[
h = \mathrm{RMSNorm}(x;\; \gamma_{\ell}^{\mathrm{attn}})
\]

- Non "capisce" la frase.
- **Stabilizza** la scala del vettore affinché
  Q e K non esplodano.
- Massa di parametri ridicola (~0.01% del modello),
  ruolo enorme (cap. 7, forza V).

**Analogia:** calibrare la bussola prima di guardare
le altre stelle della sequenza.

### Stazione 2 — Nascono Q, K, V e la Posizione (RoPE)

\[
Q = W_Q h,\quad K = W_K h,\quad V = W_V h
\]

In TinyLlama le forme logiche per strato sono:

| Tensore | Forma Logica | Lettura Umana |
|---------|--------------|---------------|
| \(W_Q\) | [2048, 2048] | 32 teste × 64 dim |
| \(W_K, W_V\) | [256, 2048] | **4** teste KV × 64 (GQA) |
| \(W_O\) | [2048, 2048] | riunisce le 32 teste |

**GQA (Grouped Query Attention):**
ogni testa chiave/valore è **condivisa da** 8 teste Q
(\(32/4 = 8\)). Meno memoria cache, stessa idea:
domande ricche, memoria condivisa.

**RoPE (Rotary Position Embedding):**
prima di attendere, Q e K vengono **ruotati** secondo la posizione \(p\).
Non c'è un vettore separato "posizione 4": la posizione
è **avvolta** nell'angolo di Q e K.

Così il modello distingue:

```text
secret to happiness   ≠   happiness to secret
```

anche se le stesse "stelle" sono nel vocabolario.

### Stazione 3 — Attenzione: Gravità tra Token

Per ogni testa di query:

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

**Lettura con il nostro prompt** (intuizione, non una mappa
di attenzione misurata qui):

| \(j\) | Token | Cosa potrebbe "tirare" `▁is` |
|------:|-------|-------------------------------|
| 0 | The | poco (funzione grammaticale) |
| 1 | secret | tema: c'è un segreto |
| 2 | to | collegamento |
| 3 | happiness | **contenuto** del segreto |
| 4 | is | se stesso (auto-attenzione) |

Gli \(\alpha_{p,j}\) sono la **gravità dinamica**
(cap. 7 e 28): quanto il residuo di `is` cade verso
ogni stella del passato di *questa* frase.

**Maschera causale:** \(j > p\) è proibito.
Nel prefill, quando elaboriamo la posizione 2,
`happiness` **non esiste ancora** nella cache.

### Stazione 4 — Miscuglio di Teste + Residuo di Attenzione

Le 32 teste vengono concatenate (o proiettate) e
passano per \(W_O\):

\[
x \leftarrow x + O(z)
\]

Il residuo **non viene cancellato**: la spinta dell'attenzione
viene **addizionata**. Per questo parliamo di orbita, non di
teletrasporto (cap. 20).

\[
x_{\mathrm{dopo}} = x_{\mathrm{prima}} + \Delta_{\mathrm{attn}}
\]

### Stazione 5 — RMSNorm (FFN)

Un'altra calibrazione, con un altro \(\gamma_{\ell}^{\mathrm{ffn}}\).

### Stazione 6 — FFN SwiGLU (Il "Sole" dei Parametri)

Qui vive ~**69%** della massa del modello:

\[
\begin{aligned}
u &= W_{\mathrm{up}} h \\
g &= W_{\mathrm{gate}} h \\
\mathrm{FFN}(h) &= W_{\mathrm{down}}\big(\mathrm{SiLU}(g)\odot u\big)
\end{aligned}
\]

- Si espande a **5632** dimensioni.
- Il *gate* decide quali canali far passare.
- Si comprime di nuovo a 2048.

**Analogia:** l'attenzione guarda **altri token**;
il FFN trasforma **questo** residuo da solo —
clima locale, conoscenza "pratica" della posizione
(Regola d'Oro: FFN → lente pratica, cap. 9).

### Stazione 7 — Residuo del FFN

\[
x \leftarrow x + \mathrm{FFN}(h)
\]

Esce dallo strato \(\ell\) pronto per la \(\ell+1\).

---

## 6. La KV-Cache: Memoria del Passato

Senza cache, per ogni nuovo token bisognerebbe
**ricalcolare** K e V per l'intera frase. Impossibile
su CPU a buon ritmo.

Con cache, nello strato \(\ell\):

```
cache_K[ℓ][0 .. p]   già salvato
cache_V[ℓ][0 .. p]

Elaborando la posizione p:
  calcolare solo K_p, V_p
  scrivere cache_K[ℓ][p], cache_V[ℓ][p]
  attendere Q_p contro cache_K[ℓ][0..p]
```

**Prefill del nostro prompt:**

| Passo | Token che Entra | Posizioni in Cache alla Fine |
|------:|-----------------|-----------------------------|
| 1 | The | 0 |
| 2 | secret | 0–1 |
| 3 | to | 0–2 |
| 4 | happiness | 0–3 |
| 5 | is | 0–4 |

Dopo il passo 5, tutti i **22 strati** hanno K e V per
le cinque posizioni. Il residuo di `is` ha scalato
l'intero edificio. Da lì escono i **logits** del
primo token della *risposta*.

---

## 7. Dal Tetto al Vocabolario: Logits

Dopo lo strato 21:

\[
h = \mathrm{RMSNorm}(x;\; \gamma^{\mathrm{out}})
\]

\[
\mathrm{logits} = W_{\mathrm{out}}\, h \;\in\; \mathbb{R}^{32000}
\]

- `output.weight` ha forma logica **[32.000 × 2048]**
  (a volte condivisa o legata all'embedding in altri
  modelli; nel GGUF di TinyLlama è il `lm_head`).
- Ogni voce \(z_k\) è "quanto il modello spinge
  a scegliere il token con id \(k\)" **adesso**.

C'è ancora **nessuna** parola. C'è una classifica di 32.000
candidati.

---

## 8. Sample: Collassare il Cielo verso una Stella

**Forza VI** (cap. 7): dal continuo all'evento.

Procedura tipica nel motore Dreaming:

1. **Temperatura** \(T\): \(z_k \leftarrow z_k / T\).
   - \(T \to 0\): quasi sempre il massimo (avidità).
   - \(T\) alto: più caso, più diversità.
2. **Top-k**: tenere solo i \(k\) logits più alti
   (es. 40). Il resto viene ignorato.
3. **Softmax** solo su quei \(k\):

\[
\pi_i = \frac{e^{z_i}}{\sum_{j\in\mathrm{top\text{-}k}} e^{z_j}}
\]

4. **Campionare** un id secondo \(\pi\) (con `--seed` per
   riprodurre lo stesso viaggio).

Supponiamo (esempio inventato ma realistico) che l'output sia:

```text
id →  ▁being      oppure      ▁love      oppure      ▁not ...
```

Quel id viene **decodificato** in testo e mostrato all'utente.
Questo è il primo passo della risposta.

---

## 9. Il Ciclo AutorGRESSIVO (La Risposta Cresce)

Il token scelto **non è la fine del modello**.
È il **prossimo passeggero**:

```
prompt:     The secret to happiness is
+ sample:   being
nuova seq:  The secret to happiness is being
```

`model_forward_token` viene chiamato di nuovo **solo**
con `being`:

- Viene calcolato il suo embedding.
- Attraversa i 22 strati.
- Scrive K,V alla posizione \(p=5\) di ogni strato.
- Attende `The…is` + `being`.
- Produce logits per il token **ancora più nuovo**.

E così via:

```
The secret to happiness is being
The secret to happiness is being kind
The secret to happiness is being kind to
...
```

fino a:

- raggiungere `max_new` token, oppure
- campionare **EOS** (fine della sequenza).

**Idea chiave:**
generare un paragrafo sono **molte** ripetizioni del
viaggio di *un* token, non un singolo passaggio "dalla frase
completa alla risposta completa".

---

## 10. Schema Maestro del Viaggio

```
┌─────────────────────────────────────────────────────────┐
│  UMANO: "The secret to happiness is"                    │
└───────────────────────────┬─────────────────────────────┘
                            │ tokenizzatore BPE
                            ▼
┌─────────────────────────────────────────────────────────┐
│  IDS:  t0 t1 t2 t3 t4                                   │
└───────────────────────────┬─────────────────────────────┘
                            │ righe di token_embd
                            ▼
┌─────────────────────────────────────────────────────────┐
│  VETTORI:  x0..x4  ∈ ℝ²⁰⁴⁸                             │
└───────────────────────────┬─────────────────────────────┘
                            │ PREFILL (per ogni ti)
                            ▼
        ┌───────────────────────────────────────┐
        │  per posizione p = 0 .. 4:            │
        │    per strato ℓ = 0 .. 21:            │
        │       Norm → Attn(+RoPE,GQA,cache)    │
        │            → +residuo                  │
        │       Norm → FFN SwiGLU               │
        │            → +residuo                  │
        └───────────────────┬───────────────────┘
                            │ dopo ultimo p del prompt
                            ▼
┌─────────────────────────────────────────────────────────┐
│  output_norm → lm_head → logits[32000]                  │
└───────────────────────────┬─────────────────────────────┘
                            │ temp, top-k, softmax, sample
                            ▼
┌─────────────────────────────────────────────────────────┐
│  NUOVO TOKEN  →  testo all'utente                       │
│       │                                                 │
│       └──── torna a forward_token (GENERAZIONE) ──► …   │
└─────────────────────────────────────────────────────────┘
```

---

## 11. Mini-Laboratorio: Vedere il Viaggio con il Motore C

Dalla root del repo (regola i percorsi al tuo GGUF):

```bash
# Prefill + generazione, seed fisso (riproducibile)
./llm_inference tinyllama-1.1b.F16.gguf \
  "The secret to happiness is" 40 0.7 40 --seed 42
```

| Flag / arg | Ruolo nel Viaggio |
|------------|-------------------|
| prompt | stelle iniziali della sequenza |
| `40` (n) | quanti nuovi token orbitare |
| `0.7` | temperatura del collasso |
| `40` (top-k) | larghezza del pozzo dei candidati |
| `--seed 42` | stesso caso → stesso percorso |
| `--perturb mystical --intensity 0.35` | **deforma** Q/K/V/FFN: altra fisica, stesso percorso formale |
| `--steer happiness --steer-strength 0.15` | spinge il residuo verso una direzione del cielo |

Protocollo didattico raccomandato:

1. Stessa seed, `none` vs `mystical` → l'orbita cambia?
2. Stessa seed, temp 0.2 vs 0.9 → il collasso cambia?
3. Apri la [mappa semantica](https://fivetechsoft.github.io/dreaming/exploration/semantic_map.html),
   cerca `▁happiness` / `▁love` e guarda le sue **forze**
   (gravità statica del catalogo) mentre leggi
   la risposta generata (gravità dinamica del prompt).

---

## 12. Errori Mentali Frequenti (E la Correzione)

| Credenza | Realtà in TinyLlama |
|----------|---------------------|
| "Il modello legge la frase in un colpo d'occhio" | Legge **token per token**; il prefill è sequenziale |
| "Ogni strato inventa un nuovo vettore" | Aggiorna lo **stesso** residuo con somme |
| "L'attenzione guarda tutto il libro" | Solo il **passato** di *questa* sequenza (fino a 2048) |
| "32 teste = 32 memorie KV" | Solo **4** gruppi KV (GQA); 32 sguardi Q |
| "L'embedding è già la risposta" | L'embedding è la **nascita**; mancano 22 piani |
| "Softmax sceglie la parola del prompt" | Sceglie il **prossimo** token del vocabolario |
| "Una risposta = un forward" | Una risposta = **1 prefill + N forwards** |

---

## 13. Checklist di Comprensione Totale

Se puoi rispondere sì a tutto, il viaggio è interiorizzato:

1. Cos'è un token e perché non è un carattere?
2. Che dimensione ha il residuo e perché viene conservato?
3. Cosa proibisce la maschera causale?
4. A cosa serve RoPE?
5. Cosa distingue attenzione e FFN in uno strato?
6. Cosa memorizza la KV-cache e in quale fase si riempie?
7. Quante volte l'ascensore di 22 piani sale per un prompt di 5 token nel prefill?
   → **5 × 22** passaggi di strato (uno per posizione).
8. Cos'è un logit e come diventa testo?
9. Perché generare 40 token implica ~40 forwards extra?
10. Dove entra una lente Dreaming (`--perturb`) in questo disegno?
    → Nei pesi delle stazioni 2–6, non nel tokenizzatore.

---

## 14. Ponti

| Tema | Capitolo |
|------|----------|
| Dim e tensori per strato | 2 |
| Motore C, RoPE, cache, sample | 3 |
| Forze (attn, FFN, softmax…) | 7 |
| Come viaggiare (percorsi A–E) | 8 |
| Attenzione in dettaglio | 10 |
| FFN in dettaglio | 11 |
| Strati precoci / medi / finali | 13–15 |
| Catena del significato (visione semantica) | 26 |
| Ascensore per piano | 27 |
| Stelle = token, attenzione = gravità | 28 |
| Formule | 23 |

---

## 15. Chiusura

Il viaggio di un prompt non è un mistero: è una
**fabbrica ripetibile**.

1. Il testo diventa **id**.
2. Gli id diventano **vettori**.
3. Ogni vettore scala **22 piani** di
   norm → gravità attentiva → clima FFN,
   parlando solo con il **passato**.
4. L'ultimo residuo viene proiettato su **32.000** punteggi.
5. Un campionamento sceglie **una** stella.
6. Quella stella viene accodata e l'universo gira di nuovo.

Quando scrivi

```text
The secret to happiness is
```

e TinyLlama risponde, non è più "l'IA pensa una frase".
È: *cinque nascite, cinque salite dell'edificio,
un collasso, e poi N collassi in più* — sempre la
stessa fisica, un passo in più nel tempo.

Questa è la comprensione totale del viaggio.
Il resto del libro (prospettive, mappe, lenti)
sono **variazioni della fisica**, non un altro percorso.
