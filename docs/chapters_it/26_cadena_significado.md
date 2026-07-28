# Capitolo 26: Token → Embedding → Idee pure → Semantica → Dettagli → Risposta

## L'ordine

Ordinare il viaggio del significato nel microcosmo
TinyLlama — non come scatole separate del transformer,
ma come **catena completa**, dalla scintilla simbolica
alla frase che torna al mondo.

---

## Catena riorganizzata (canonica)

```
1. TOKEN         simboli discreti del vocabolario
        ↓
2. EMBEDDING     geometria d'ingresso in ℝ²⁰⁴⁸
        ↓
3. DETTAGLI      forma locale (sintassi, vicini, superficie)
        ↓
4. IDEE PURE     astrazioni e quadri (layer medi)
        ↓
5. SEMANTICA     significato legato in contesto (attenzione + integrazione)
        ↓
6. DETTAGLI fini concrezione lessicale / stile (FFN tardivo + testa)
        ↓
7. RISPOSTA      logits → sample → di nuovo token
        ↓
      (torna a 1)
```

Ci sono **due comparse di "dettagli"** a ragione:
- **Dettagli di forma** (anticipi): *come* si scrive.
- **Dettagli di contenuto** (tardivi): *cosa* si concreta nel parlato.

Le "idee pure" vivono nel mezzo: né solo lettera,
né ancora la frase chiusa.

---

## Tabella maestra

| # | Fase | Cos'è? | Dove nel modello | Dim / oggetto | Strumento Dreaming |
|---|------|--------|------------------|---------------|-------------------|
| 1 | **Token** | Pezzi del BPE (`▁love`, id) | Vocabolario \(V=32\mathrm{k}\) | insieme finito | tokenizer GGUF |
| 2 | **Embedding** | Punto nel cielo | `token_embd` | \(\mathbb{R}^{2048}\) | mappe 2D/3D, archetipi |
| 3 | **Dettagli (forma)** | Relazioni locali, sintassi | Layer **0–5**, attn corta | residuale ancora "attaccato" all'emb | dungeon L0–L5 · zona gravity/matter |
| 4 | **Idee pure** | Quadri, temi, ruoli astratti | Layer **6–12** | residuale tematizzato | zone mage/sage |
| 5 | **Semantica** | Significato *in contesto* (chi si unisce a chi) | Attn globale + layer **13–20** | accoppiamento \(a_{t,t'}\) | zone gravity + drama + surface |
| 6 | **Dettagli (contenuto)** | Lessico fine, passi, colore locale | FFN (sp. tardivo) + abitudini SwiGLU | \(\mathbb{R}^{5632}\) intermedio | zona matter · voce pratica |
| 7 | **Risposta** | Un token (e poi una frase) | `output_norm` → lm_head → softmax | \(\mathbb{R}^{32000}\) → sample | zona event · Space nel gioco |

---

## 1. Token

**Ingresso e uscita dello specchio.**

- Discreti, finiti, senza "senso" finché non si proiettano.
- Il BPE sminuzza il mondo: non ogni concetto è un singolo id.
- Nel gioco: il **sample** finale torna ad essere token
  e riavvia l'orbita.

Senza token non ci sono bordi da raccontare.
Con solo token non c'è universo continuo.

---

## 2. Embedding

**Nascita geometrica.**

\[
t \mapsto e_t \in \mathbb{R}^{2048}
\]

- Isole semantiche e archetipi vivono qui come **catalogo**.
- Opposits ≈ ortogonali, non antipodali.
- PCA: centinaia di dim reali; la mappa 2D/3D è un planetario.

Qui il residuale **non ha ancora viaggiato**:
è potenziale di senso, non ancora frase.

---

## 3. Dettagli di forma (anticipi)

**Layer 0–5 · "come si unisce la lettera".**

- Schemi adiacenti, dipendenze corte.
- L'attenzione inizia ad accoppiare i vicini.
- Il FFN regola la superficie lessicale.

Se si rompe questa fase, il testo perde **grammatica**
prima della "profondità filosofica".

Nel gioco: primi portali · zone **sky → gravity/matter**.

---

## 4. Idee pure (mezzo)

**Layer 6–12 · "di cosa si tratta".**

- Quadri: esistenziale, accademico, narrativo, tecnico.
- Il residuale si distacca dal puro bigramma.
- Lì si incastrano le costellazioni Mago/Mistico e Saggio
  come *climi di idea*, non solo parole separate.

Ipotesi di lavoro del libro: il tratto medio è
dove `--steer soul` e `mystical` smettono di essere cosmetici
e diventano **bias tematico**.

Nel gioco: warp verso **mage** e **sage**.

---

## 5. Semantica (legare in contesto)

**Attenzione a lungo raggio + integrazione tardiva.**

Semantica ≠ lista di embedding.
Semantica = **relazioni**:

\[
\mathrm{semantica}(t) \approx \sum_{t'\le t} a_{t,t'}\, v_{t'}
\]

riscritta da layer a layer e mescolata nel residuale.

- Chi modifica chi nella frase.
- Polarità Eroe/Ombra come tensione nel filo.
- Regola d'Oro: toccare **attenzione** muove il riflesso
  verso l'**accademico / relazionale / critico**.

Nel gioco: zone **gravity**, **drama**, **surface**.

---

## 6. Dettagli di contenuto (concrezione)

**FFN · "con quali parole e gesti si dice".**

Anche se il FFN agisce in tutti i layer, il suo ruolo
come *dettaglio fine* si nota nel concretare:

- verbi d'azione, liste, consigli (voce pratica),
- colore lessicale, abitudini locali in \(\mathbb{R}^{5632}\).

Regola d'Oro: toccare **FFN** → prospettiva **pratica**.

Non è l'idea pura; è l'**incarnazione** dell'idea
in materiale verbale.

---

## 7. Risposta

**Collasso e ritorno al macrocosmo.**

\[
z = W\,\mathrm{RMSNorm}(x_L),\quad
t\sim \mathrm{softmax}(z/T)\ \text{(top-k)}
\]

- Un evento discreto (token).
- Concatenato, torna ad essere linguaggio umano.
- Chiude lo specchio: dall'orologio al cielo (cap. 6, 24).

Poi il ciclo:

**risposta → nuovi token → …**

---

## Diagramma di flusso (completo)

```
 MACRO: domanda umana / prompt
              │
              ▼
     ┌──── TOKEN ────┐
     │                │
     ▼                │
 EMBEDDING (cielo)    │
     │                │
     ▼                │
 DETTAGLI forma       │   layer 0–5
 (sintassi, vicini)   │
     │                │
     ▼                │
 IDEE PURE            │   layer 6–12
 (quadri, temi)       │
     │                │
     ▼                │
 SEMANTICA            │   attn + layer 13–20
 (legami in contesto) │
     │                │
     ▼                │
 DETTAGLI contenuto   │   FFN / stile fine
 (lessico, azione)    │
     │                │
     ▼                │
 RISPOSTA (sample) ──┘   logits → token
     │
     ▼
 MACRO: leggiamo una voce / archetipo / giudizio
```

Le lenti Dreaming agiscono **lungo** la catena:

| Lente | Dove piega di più la catena |
|-------|----------------------------|
| baseline | tutta la catena "ufficiale" |
| mystical / Mago | idee pure + semantica esistenziale |
| accademica / Saggio | semantica relazionale / struttura |
| pratica | dettagli di contenuto (FFN) |
| noise | rompe la catena (esce da \(\mathcal{C}\)) |
| `--steer` | spinge il residuale verso un embedding-isola |

---

## Relazione con altre parti del libro

| Capitolo | Incastratura nella catena |
|----------|--------------------------|
| 2 Struttura | Dove vivono le fasi nei tensori |
| 3 Motore C | Come si calcola ogni freccia |
| 5 Spazio multi-D | Fasi 1–2 e geometria del cielo |
| 7 Forze | Attn=semantica non locale; FFN=dettagli di contenuto |
| 9 Regola d'Oro | Lenti su 5 e 6 |
| 13–15 Layer | Partizione temporale di 3–4–5 |
| 16–21 Isole / archetipi | Etichettatura culturale di 2 e 4 |
| 20 Orbita | La catena come dinamica \(x\leftarrow x+F(x)\) |
| 25 Gioco | Ogni portale = avanzare fase + warp di zona |

---

## Versione corta (per l'HUD del gioco / glossario)

```
TOKEN → EMBEDDING → DETTAGLI → IDEE PURE
       → SEMANTICA → DETTAGLI FINI → RISPOSTA → (token)
```

O in una riga:

**Simbolo → geometria → forma → idea → legame → concrezione → detto.**

---

## In una frase

L'universo TinyLlama non è solo uno stack di layer:
è una **catena di trasformazioni del senso**
dove i token diventano geometria, la geometria
diventa forma e idea, l'idea si lega nella semantica,
si dettaglia nel lessico e **collassa** di nuovo in token
che possiamo leggere — uno specchio ciclico tra micro e macro.

---

*Capitolo successivo: Ogni Layer è un Ascensore.*
