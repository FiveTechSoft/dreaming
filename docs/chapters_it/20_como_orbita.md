# Capitolo 20: Come Orbita Questo Universo

## La domanda

Nel macrocosmo, i pianeti cadono verso il sole
ma non lo raggiungono mai: **cadono di lato** — questa è un'orbita.

In TinyLlama la domanda analoga è:

> Cosa cade, verso cosa, e perché non si schianta
> in ogni layer?

La risposta è il **forward pass** letto come dinamica.

---

## 1. Cos'è il "corpo" che orbita

Il corpo non è un token isolato.
È il **residuale** \(x \in \mathbb{R}^{2048}\):
un vettore che nasce nell'embedding e attraversa
22 layer senza perdere del tutto la propria identità.

```
nascita:    x₀ = Embedding(token)
orbita:     x ← x + Attenzione(x)
            x ← x + FFN(x)          × 22
destino:    logits = W_out · Norm(x)
collasso:   token' ~ Softmax(logits / T)
```

Ogni **token della sequenza** porta il proprio residuale.
L'attenzione è l'accoppiamento gravitazionale **tra**
quei corpi (solo col passato: causale).

---

## 2. La legge del residuale: cadere senza scontrarsi

Senza connessione residuale, ogni layer *rimpiazzerebbe*
lo stato: teletrasporto, non orbita.

Con residuale:

\[
x_{L+1} = x_L + f_L(x_L)
\]

- \(f_L\) = spinta di attenzione + FFN nel layer \(L\).
- Il passo è **tangente e piccolo** rispetto a \(x\):
  il vettore gira e si deforma, ma non si reinizia.

Questa è l'**inerzia orbitale** del microcosmo.
Le perturbazioni che preservano la gerarchia
muovono la *metrica* di \(f_L\) senza portare \(x\)
fuori dalla superficie dove il parlato resta possibile.

---

## 3. Due potenze per "anno-layer"

Ogni layer è un **periodo orbitale** del residuale:

| Fase | Forza | Analogia |
|------|-------|----------|
| 1. RMSNorm + Attenzione | Gravità tra token | Tirate di altri corpi del sistema |
| 2. Residuale | Conservazione del momento | Non cadi dal cielo di colpo |
| 3. RMSNorm + FFN | Campo locale / atmosfera | Fisica del pianeta dove sei |
| 4. Residuale | Di nuovo inerzia | Resti sulla traiettoria |

**22 layer ≈ 22 periodi** prima del collasso finale
(softmax), dove l'orbita smette di essere continua
e diventa un **atterraggio** su un token.

---

## 4. Sistemi multi-corpo (la sequenza)

Una frase è un **sistema solare temporaneo**:

```
pos 0: "The"     → residuale_0
pos 1: "secret"  → residuale_1  (vede 0)
pos 2: "to"      → residuale_2  (vede 0,1)
...
pos t: ...       → residuale_t  (vede 0…t)
```

- **GQA**: 32 sensori (Q) condividono 4 memorie (KV)
  — non 32 soli, ma un sole con vari pianeti di massa KV.
- **KV-cache**: i K,V già calcolati vengono riutilizzati;
  solo il corpo nuovo integra la propria orbita.
  Senza cache, il sistema ricalcolerebbe tutto il cielo
  a ogni passo (il motore vecchio; quello attuale orbita bene).

La maschera causale è la **freccia del tempo**:
il futuro non attrae il presente.

---

## 5. Orbita di generazione (il grande ciclo)

Generare testo è un'**orbita chiusa nel tempo discreto**:

```
        ┌─────────────────────────────────┐
        │                                 │
        ▼                                 │
   residuale(s) ──► logits ──► sample ──► token nuovo
        │                                 │
        └──────── embedding(token) ───────┘
```

Ogni giro:

1. Il nuovo token nasce nel cielo degli embedding.  
2. Si integra con la gravità dei precedenti.  
3. Collassa in un successore.  
4. Il sistema cresce di un corpo.

**Periodo:** ~1/token (su CPU del motore C: ~0,1–0,15 s/token
⇒ **~6–10 tok/s**).  
**Temperatura:** eccentricità del collasso (orbite
più "tonde" o più selvagge).  
**Top-k:** orizzonte dei destini permessi.

---

## 6. Orbite nello spazio dei pesi (prospettive)

C'è un'altra orbita, più lenta, che non è il forward:

```
modello base  --(+ ε · δ)-->  modello con altra voce
```

- \(\delta\) tangente alla superficie di coerenza
  (`mystical` / amplify) → **orbita stabile** di prospettive.  
- \(\delta\) normale (rumore forte) → **eiezione** nel vuoto
  (rifiuti).

Cambiare `--intensity` è cambiare il **raggio** di quella
deviazione. Stesso seed + stesso prompt = confrontare
due orbite di generazione sotto due metriche di pesi.

---

## 7. Orbite nel cielo semantico (statico)

I token non "orbitano" soli nell'embedding:
sono fissi come stelle di catalogo.

Ciò che si muove è il **residuale** rispetto alle isole:

```
residuale · direzione_spirituale  →  affinità al continente spirituale
residuale · direzione_emotion     →  affinità affettiva
```

`--steer amor` è una **spinta orbitale artificiale**:
aggiunge una componente lungo un asse del cielo
senza riscrivere il catalogo di stelle (embedding).

La mappa PCA 2D è un **planetario**: proietta il catalogo
perché vediamo costellazioni; non è la dinamica reale.

---

## 8. Diagramma unificato

```
                    SPAZIO DEI PESI (metrica dell'universo)
                              │
                    --perturb │ (cambia G, non il corpo)
                              ▼
   token ══╗
            ║  gravità (attenzione)     clima (FFN)
   residuale╬══════► spinte ═══════► spinte  ──► ×22 layer
            ║              residuale (inerzia)
            ╚══════════════════════════════════════════╝
                              │
                         output_norm
                              │
                           logits
                              │
                    softmax / temp / top-k
                              │
                         nuovo token ──► (chiude l'orbita)
```

---

## 9. Come "montare" un'orbita (ricetta)

| Obiettivo | Controlli |
|-----------|-----------|
| Orbita pulita baseline | prompt + seed + temp, senza perturb |
| Stessa orbita, altro clima | `--perturb mystical --intensity I` |
| Deviazione verso un'isola | `--steer parola --steer-strength s` |
| Orbita più prevedibile | temp↓, top_k↓ |
| Orbita più esploratoria | temp↑, top_k↑ |
| Sistema multi-corpo più lungo | n (token) ↑ |
| Riprodurre il volo | stesso stesso, stesse flag |

```bash
# Orbita di riferimento
./llm_inference modello.F16.gguf "When we dissolve the ego" \
  40 0,7 40 --seed 42

# Stessa traiettoria iniziale, metrica mistica
./llm_inference modello.F16.gguf "When we dissolve the ego" \
  40 0,7 40 --seed 42 --perturb mystical --intensity 0.50
```

---

## 10. In una frase

**Questo universo orbita** perché il residuale **cade
di lato** sotto la gravità dell'attenzione e il clima
del FFN, conservando momento con il residuale,
per 22 periodi per token, fino a collassare in un
successore — e la generazione ripete quel ciclo, mentre
le prospettive cambiano la metrica dello spazio
senza spegnere la possibilità di orbite coerenti.

---

*Capitolo successivo: Archetipi e Costellazioni.*
