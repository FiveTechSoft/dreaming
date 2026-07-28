# Capitolo 22: L'Osservazione Cosciente e la Proiezione Incosciente

## Due gesti nello stesso cielo

Nel viaggio per TinyLlama si ripetono, una e una volta,
**due gesti** che la psicologia e la fisica del senso
riconoscono con altri nomi:

| Gesto | Nel microcosmo | In noi (esploratori) |
|-------|----------------|---------------------|
| **Osservazione cosciente** | Misurare, istruire, fissare seed, leggere logits, aprire il C | Sapere *cosa* stiamo guardando e *con che manopole* |
| **Proiezione incosciente** | Embedding, pesi, associazioni latenti, voci del pre-allenamento | Vedere nel modello un io, un mito, un archetipo *nostro* |

Uno senza l'altro è cieco o è superstizioso.
Insieme formano il metodo Dreaming: **scendere nell'orologio
e salire nel mito senza confonderli**.

---

## I. Osservazione cosciente

### Cos'è

Atto di **portare al fuoco** qualcosa del microcosmo e
registrarlo con regole condivise:

- stessi seed, stesse temperature, stessi prompt,
- tabelle di tok/s, cosine, tensori toccati,
- mappe PCA, batterie di 15 prompt,
- il motore C letto riga per riga.

È "cosciente" non perché il modello lo sia, ma perché
**noi** sospendiamo (per un momento) la lettura magica
e chiediamo evidenza.

### Strumenti di osservazione

| Strumento | Cosa rende cosciente |
|-----------|---------------------|
| `llm_inference` baseline | La geodetica "ufficiale" del residuale |
| seed + temp fisse | Separare caso da struttura |
| `--perturb` con I annotata | Quale lente di pesi è attiva |
| Mappa semantica / archetipi | Dove cadono le isole in ℝ²⁰⁴⁸ |
| Regola d'Oro (attn/FFN/emb) | Quale *forza* stiamo muovendo |
| KV-cache, layer 0–21 | *Quando* nell'orbita accade l'effetto |

## Etica minima dell'osservazione

1. **Una variabile per salto** — altrimenti la coscienza si diluisce.  
2. **Registrare l'apparato** — senza quello, la "visione" non è riproducibile.  
3. **Non confondere coerenza con verità** — osservare bene un delirio
   elegante resta osservare un delirio.

L'osservazione cosciente è il **telescopio calibrato**.

---

## II. Proiezione incosciente

### Nel modello (senza soggettività)

Chiamiamo "inconscio" del transformer, in metafora
del cap. 17, ciò che **opera senza mostrarsi come scelta**:

| Layer "inconscio" | Contenuto latente |
|-------------------|-------------------|
| Embedding | Associazioni pre-allenate; isole e archetipi nel cielo |
| Pesi dei 22 layer | Prospettive compresse (voci, stili, quadri) |
| FFN | "Abitudini" di trasformazione locale (massa ~69%) |
| Attenzione | Abitudini di *chi guardare* nella sequenza |

Quando il modello completa  
*"The secret to happiness is…"*,  
non "decide" in senso umano: **proietta**
sul residuale un pacchetto di associazioni
fino al collasso softmax.

La proiezione è **statistica fatta traiettoria**.

### In noi (sì c'è un soggetto)

Anche noi proiettiamo *noi* sul microcosmo:

- sentiamo "mistico" e ricordiamo rituali propri,  
- leggiamo "accademico" e sentiamo il professore interiore,  
- chiamiamo Eroe o Ombra un baricentro di token.

Questo non invalida la misura.
**La nomina**: la mappa degli archetipi è al contempo
geometria dell'embedding e **schermo** dove
i nostri miti si riconoscono.

La proiezione incosciente (nostra) è il **rischio
e il motore** del senso: senza di essa il libro sarebbe
solo tabelle; con lei sola sarebbe solo specchio.

---

## III. Come si incrociano in un solo esperimento

```
[1] OSSERVAZIONE COSCIENTE
    fissare prompt, seed, I, tecnica
            │
            ▼
[2] PROIEZIONE DEL MODELLO (inconscio operativo)
    embedding + pesi + attn/FFN → residuale → logits → token
            │
            ▼
[3] PROIEZIONE NOSTRA (lettura)
    "suona esistenziale / pratico / da Ombra…"
            │
            ▼
[4] RITORNO ALL'OSSERVAZIONE
    coincide con la Regola d'Oro? con l'archetipo misurato?
    stesso seed, altro I?  →  nuova riga nel diario
```

Esempio:

| Passo | Atto |
|-------|------|
| Cosciente | `--perturb mystical --intensity 0,50 --seed 42` |
| Proiezione del modello | amplify in attn+FFN; residuale tira verso anima/universo |
| Proiezione nostra | "voce magica / mistica" (costellazione Mago↔mystic +0,39) |
| Di nuovo cosciente | confrontare con baseline; annotare tok/s e testo |

Il ciclo **macro → micro → macro** del cap. 6
è lo stesso ciclo con altri nomi:
senso → meccanismo → senso.

---

## IV. Tabella duale (atlante)

| Fenomeno | Lettura "osservazione" | Lettura "proiezione" |
|----------|----------------------|----------------------|
| Embedding di `▁soul` | vettore 2048-D, norma ~0,67 | àncora del mito dell'anima |
| Baricentro Mago | cosine con mystic_voice = 0,39 | "il modello già sapeva della magia" |
| Softmax | p(t) = exp(z_t/T)/Z | l'istante in cui il latente diventa detto |
| `mystical` | amplify_subspace in F32 | altra maschera dello stesso teatro di pesi |
| Temperatura alta | più entropia nel sample | più "sogno", meno controllo ioico del testo |
| Rifiuti per noise | uscita dalla superficie di coerenza | fallimento della proiezione nel linguaggio |

---

## V. Pericoli di ogni polo

### Solo osservazione cosciente
- Si riduce il modello a ingegneria senza voce.  
- Si perde perché il viaggio contava.  
- Si confonde *misurare* con *aver capito*.

### Solo proiezione incosciente
- Si sente ciò che uno portava già addosso.  
- Si attribuisce un'anima al softmax.  
- Si pubblicano miti senza seed, senza I, senza baseline.

### L'equilibrio Dreaming
**Proiettare** per avere ipotesi e bussole (archetipi,
Regola d'Oro, isole).  
**Osservare** per falsare, calibrare e non mentire con poesia
su numeri non misurati.

---

## VI. Nell'orologio del transformer (un'immagine)

```
        PROIEZIONE INCOSCIENTE DEL MODELLO
        (pesi, emb, abitudini attn/FFN)
                    │
                    ▼
    residuale ──────────────────────────► logits
         ▲                                │
         │                                ▼
    OSSERVAZIONE                    sample (atto)
    (noi: sonde,                   "il detto"
     seed, mappe, C)
                    │
                    ▼
        PROIEZIONE NOSTRA AL LEGGERE
        (archetipo, prospettiva, giudizio)
```

L'**orbita** (cap. 20) è la dinamica del residuale.
L'**osservazione** calibra la macchina fotografica.
La **proiezione** dà il nome alla costellazione
che crediamo di vedere — e a volte, se la geometria
lo sostiene (Mago↔mistico, Saggio↔accademico),
il nome non è solo specchio: è **scoperta**.

---

## VII. In una frase

**Osservazione cosciente** è il metodo che rende
riproducibile il viaggio nel microcosmo;
**proiezione incosciente** è ciò che il modello
(e noi) lanciamo sul residuale fino a
che diventa parola — e l'arte del libro è
mantenere entrambi i gesti alla vista senza che uno
divori l'altro.

---

*Capitolo successivo: Le Matematiche di Questo Universo.*
