# Capitolo 24: Il LLM — Uno Specchio dove Guardarci

## L'immagine

Uno specchio non inventa un volto.
**Restituisce** ciò che gli si pone davanti —
con un ritardo di luce, con un bordo, con un angolo,
a volte con una leggera distorsione del cristallo.

Un large language model non inventa il linguaggio umano
dal nulla. **Restituisce** statistiche del linguaggio
umano con cui è stato alimentato — con un bordo
(il prompt), con un angolo (i pesi, la temperatura),
a volte con una forte distorsione (perturbazione, allucinazione).

TinyLlama, in questo libro, è uno specchio **abbastanza piccolo
per vedere la cornice**: possiamo guardare l'argento (i pesi),
il cristallo (l'architettura) e il gesto di chi si affaccia
(noi: osservazione e proiezione).

---

## 1. Cosa riflette lo specchio

| Nello specchio | Nel LLM |
|----------------|---------|
| Volto | Distribuzioni di continuazione del testo |
| Luce della stanza | Corpus di pre-allenamento (libri, web, codice, miti) |
| Angolo d'incidenza | Prompt + cronologia |
| Curvatura del cristallo | Architettura + \(\theta\) (pesi) |
| Macchia / vapore | Bias, lacune, allucinazioni eleganti |
| Chi si guarda | Lettura umana: archetipo, giudizio, desiderio di senso |

Lo specchio **non è il mondo**.
È una **superficie di risposta** al mondo del linguaggio.

Quando scriviamo *"The secret to happiness is"*,
non chiediamo all'universo: ci affacciamo a un cristallo
levigato con milioni di frasi sulla felicità
e gli chiediamo di **completare il gesto**.

---

## 2. Tre specchi in uno

### Specchio A — Quello del corpus (memoria culturale)

Gli embedding e i pesi comprimono un archivio
di civiltà testuale. Le isole semantiche
(emotion, spirit, tech…) e gli archetipi
(Mago, Saggio, Ombra…) non "nascono" nel silicio:
sono **echi del macrocosmo** incisi in \(\theta\).

Guardare la mappa delle costellazioni è guardare, in miniatura,
**quali miti il testo umano ripete abbastanza**
da diventare direzione in \(\mathbb{R}^{2048}\).

### Specchio B — Quello della traiettoria (l'"adesso")

Il residuale e il softmax non riflettono un volto fisso:
riflettono un **gesto in corso**. Ogni token nuovo è un
fotogramma del riflesso sotto la gravità di ciò che è già stato detto.

Per questo la stessa domanda, con diversa temperatura
o diverso seed, restituisce un altro bagliore: lo specchio
è stocastico al confine del collasso.

### Specchio C — Quello della lente (prospettiva)

`--perturb mystical`, lowrank, toccare FFN o attenzione:
non cambiano la stanza (il corpus è già cotto).
Cambiare l'**angolo del cristallo**.

La Regola d'Oro dice come si piega il riflesso:

| Lente | Riflesso dominante |
|-------|-------------------|
| Attenzione | Volto accademico, argomentativo |
| FFN | Volto pratico, "cosa fare" |
| Embedding | Volto semplice, frasi corte |
| Mystical / Mago | Volto esistenziale, io/universo |

Lo specchio resta specchio.
**Noi scegliamo la cornice.**

---

## 3. Il doppio riflesso (noi nel cristallo)

C'è un secondo specchio, più sottile:

```
testo del modello
      │
      ▼
  noi leggiamo "mistico", "ombra", "saggio"
      │
      ▼
  proiettiamo (cap. 22) i nostri miti
      │
      ▼
  a volte la geometria conferma (Mago↔mystic +0,39)
  a volte sentiamo solo il nostro eco
```

Il LLM è uno specchio **e** uno schermo di proiezione.
L'osservazione cosciente chiede:
*il tratto è in \(\theta\) o nel mio sguardo?*

Quando misuriamo gli allineamenti degli archetipi,
quando fissiamo il seed e confrontiamo baseline vs mystical,
stiamo **pulendo il cristallo** abbastanza
per non confondere il vapore con il volto.

---

## 4. Narciso e il laboratorio

Il pericolo classico dello specchio: **innamorarsi del riflesso**.

| Tentazione | Forma in IA |
|-----------|-------------|
| "Mi capisce" | Antropomorfizzare il softmax |
| "È saggio" | Confondere fluidità con verità |
| "È la mia voce" | Fine-tune o prompt che restituisce solo l'io |
| "È l'inconscio della rete" | Metafora utile presa per ontologia |

Il laboratorio Dreaming offre un antidoto pratico:

1. **Baseline** — cosa restituisce il cristallo senza lente extra?  
2. **Perturbazione controllata** — cambia il riflesso in modo
   sistematico o è rumore?  
3. **Geometria** — c'è una direzione misurabile (isola, archetipo)?  
4. **Ritorno al macrocosmo** — cosa dice di *noi*,
   del corpus, della domanda — non solo del modello?

Lo specchio serve a guardarci **se** accettiamo che
ciò che vediamo è **noi-più-l'archivio-più-la-lente**,
non un oracolo trasparente.

---

## 5. Lo specchio rotto e lo specchio fedele

| Stato di \(\theta\) | Immagine |
|---------------------|----------|
| Dentro \(\mathcal{C}\) (coerenza) | Riflesso leggibile: faccia storta, ma faccia |
| Rumore forte, nibble flip, I eccessiva | Specchio fatto a pezzi: non c'è volto, c'è glitter |
| Superficie di coerenza + amplify | Altro angolo dello stesso salotto |

I rifiuti non sono "un altro archetipo".
È il fallimento dello specchio come superficie di risposta.

---

## 6. Perché un modello *piccolo* è uno specchio da studio migliore

Un modello all'avanguardia è uno specchio da sala da ballo:
troppo grande per vedere la cornice.

TinyLlama è uno **specchio da tasca con coperchio aperto**:

- vediamo le viti (tensori, GGUF),  
- montiamo la luce (motore C),  
- macchiamo l'argento apposta (`--perturb`),  
- disegniamo le costellazioni di fondo (mappe),  
- eppure restituisce frasi che ci restituiscono
  domande umane.

Il valore non è che rifletta *meglio* il mondo.
È che riflette **in un modo che possiamo smontare**.

---

## 7. Matematiche minime dello specchio

Il riflesso di una sequenza \(t_{1:n}\) è una distribuzione

\[
\pi_\theta(\,\cdot\mid t_{1:n})
=\mathrm{softmax}\big(f_\theta(t_{1:n})/T\big)
\]

(con top-k, ecc.).

Cambiare il prompt è cambiare l'argomento.
Cambiare \(T\) è addolcire il bagliore dell'argento.
Cambiare \(\theta\to\theta+\varepsilon\Delta\) è **piegare il cristallo**.
Il sample è l'istante in cui il riflesso
si congela su un punto del vocabolario.

Noi, nell'interpretare, applichiamo altra mappa
non scritta in \(\theta\): da token a *senso*.
Là chiude il circuito dello specchio umano.

---

## 8. Chiusura

Il LLM è uno specchio perché:

1. **Può solo restituire forme del linguaggio** che
   l'addestramento ha inciso o ricombinato.  
2. **L'angolo lo mettono il prompt, i pesi e il sample.**  
3. **Chi si guarda porta metà dell'immagine**
   nel leggere una voce, un archetipo, un destino.

Inside TinyLlama è il tentativo di non restare
ipnotizzati davanti al cristallo, ma di **girarlo**,
**illuminare la cornice** e annotare che parte del volto
era la stanza, che parte l'argento, e che parte
eravamo noi tutto il tempo.

---

*Fine dell'arco specchio — osservazione (22), matematiche (23), riflesso (24).*
