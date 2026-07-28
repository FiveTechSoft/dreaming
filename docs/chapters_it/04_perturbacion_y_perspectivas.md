# Capitolo 4: Perturbazione dei Pesi e Cambiamento di Prospettiva

## Oltre usare il modello

Finora abbiamo imparato ad *eseguire* TinyLlama.
Sappiamo come è costruito e come scrivere un motore
che lo faccia parlare.

Ma il cuore del progetto Dreaming è un'altra domanda:

> Cosa succede se **cambiamo** i pesi del modello?

Non per riallenarlo. Non per correggerlo.
Solo per muoverlo leggermente nel suo spazio dei pesi
e osservare se continua a parlare, ma in modo diverso.

La risposta, dopo molti esperimenti, è sorprendente:
**continua a parlare, e lo fa da una prospettiva diversa.**

## Cos'è la perturbazione dei pesi?

Un modello di linguaggio è un'enorme lista di numeri.
In TinyLlama-1.1B ci sono più di un miliardo.
Quei numeri, organizzati in tensori, sono i "pesi"
che il modello ha acquisito durante il suo addestramento.

Perturbare i pesi significa modificare quei numeri in modo
curato. È come ruotare leggermente le manopole di una radio:
se lo fai bene, continui ad ascoltare musica, ma cambia
la stazione.

Nel nostro caso lavoriamo con TinyLlama quantizzato in **Q4_0**:
ogni blocco di 32 pesi viene compresso in 18 byte
(2 byte per la scala + 16 byte con i nibble da 4 bit).

La pipeline è semplice in concetto:

```
1. Leggere l'GGUF originale byte per byte
2. Copiare l'header senza toccarlo (conserva il tokenizzatore)
3. Disimpacchettare i blocchi Q4_0 a float32
4. Applicare una tecnica di perturbazione
5. Quantizzare di nuovo a Q4_0
6. Scrivere il nuovo GGUF
```

La chiave è il passo 4: **non tutta la modifica è uguale**.
Alcune distruggono il modello; altre lo fanno parlare
con un'altra voce.

## L'analogia DMT

Chiamiamo questo lavoro "DMT perturbation" perché l'effetto
ricorda l'ipotesi classica sugli stati alterati:

> L'allucinazione non è invenzione. È informazione reale del sistema,
> riorganizzata nella sua forma di combinarsi.

Quando perturbiamo TinyLlama, il modello non inventa parole
che non ha mai visto. Riorganizza le associazioni che già aveva.
È come se risvegliassimo una personalità latente che è sempre
stata lì, silenziata dalla configurazione originale.

Il modello resta TinyLlama. Ma ora "sogna"
da un altro angolo.

## Le 10 tecniche di preservazione della gerarchia

Le prime perturbazioni che abbiamo provato erano rumore puro,
scambio di righe, inversione di nibble. La maggior parte
produceva rifiuti: caratteri strani, cicli senza senso,
parole che non esistono.

Ma abbiamo scoperto qualcosa di importante: **le tecniche che preservano
la gerarchia interna dei pesi mantengono la coerenza**.
Non importa tanto il valore assoluto di ogni peso; importa
la sua relazione relativa con gli altri.

Abbiamo provato dieci tecniche che rispettano quella gerarchia:

| # | Tecnica | Chiave | Prospettiva dominante |
|---|---------|--------|----------------------|
| 1 | Low-rank amplification | `lowrank` | Accademica / critica |
| 2 | Eigenvector rotation | `eigr` | Pratica / consigli |
| 3 | Spectral shift | `spectral` | Concisa / diretta |
| 4 | Attention-preserving | `attpres` | Quasi identica all'originale |
| 5 | Residual-preserving | `respres` | Introspectiva |
| 6 | Block-diagonal | `blkdiag` | Molto vicina all'originale |
| 7 | Norm-preserving rotation | `normrot` | Stoica / equilibrata |
| 8 | Gradient-aligned | `gradal` | Autenticità / scoperta |
| 9 | Low-frequency DCT | `lowdct` | Conversazionale / assistente |
| 10 | Manifold-preserving | `manpres` | Autenticità (simile a gradient) |

Tutte queste tecniche hanno prodotto testo coerente.
Non sempre corretto, non sempre fattuale, ma grammaticalmente
valido e con un'intenzione chiara.

## Come funziona la pipeline

Lo script `dmt_perturb_v10.py` implementa il processo:

```bash
# Generare un modello perturbato con una tecnica
python dmt_perturb_v10.py lowrank --intensity 0.10
```

Internalmente:

1. Legge l'GGUF originale (`tinyllama-1.1b-q4_0.gguf`)
2. Copia l'header e i metadata intatti
3. Percorre ogni tensore di pesi
4. Disimpacchetta i blocchi Q4_0
5. Applica la tecnica scelta con una data intensità
6. Quantizza di nuovo a Q4_0
7. Scrive il file perturbato (`v10_lowrank_10.gguf`)

Il parametro `--intensity` controlla quanto si muove il modello.
Un valore troppo basso non cambia nulla; uno troppo alto
distrugge la coerenza.

## Il sweet spot: intensità 0,10

Abbiamo testato molteplici intensità con tutte le tecniche.
Il risultato è stato costante:

| Intensità | Effetto | Qualità |
|-----------|---------|---------|
| 0,05 | Molto vicino all'originale | Troppo fedele |
| **0,10** | **Massima divergenza, testo coerente** | **Sweet spot** |
| 0,15 | Vicino all'originale, più filosofico | Leggermente spostato |
| 0,20 | Prospettiva diversa, più comprensiva | Più divergente |
| 0,25+ | Qualità degradata, ripetitivo | Troppo rumore |

A intensità 0,10 il modello si scosta il più possibile
senza rompersi. È il punto in cui la perturbazione smette di essere
un eco dell'originale e diventa una voce propria.

## Confronto diretto: stesso prompt, diversa prospettiva

L'effetto più vistoso si vede quando si usa lo stesso prompt
in modelli perturbati diversi.

### Prompt: "The secret to happiness is"

| Modello | Prospettiva | Inizio della risposta |
|---------|-------------|----------------------|
| Baseline | Self-help generica | "...cultivating a mindset that is focused on gratitude..." |
| `v11_select_extreme` | Spirituale / mindfulness | "...finding inner peace and contentment through mindfulness..." |
| `v10_lowrank` | Filosofica / accademica | "...the phrase is an idiom used to express the idea that finding true inner peace..." |
| `v10_normrot` | Stoica | "...finding the right balance between our inner and outer lives." |
| `v10_gradal` | Autenticità | "...finding your own unique and authentic way of living..." |

### Prompt: "Dreams are the mind's way of"

| Modello | Prospettiva | Inizio della risposta |
|---------|-------------|----------------------|
| Baseline | Neuroscienze popolari | "...processing and storing information..." |
| `v11_select_attention` | Letteratura vittoriana | "Dr. Jekyll and Mr. Hyde is a play by Robert Louis Stevenson..." |
| `v10_eigr` | Self-help spirituale | "Dr. M. A. S. S. is an acronym for 'Dreams Are Mind's Way.'..." |
| `v10_lowrank` | Ricerca clinica | "...a study published in the Journal of Sleep Research..." |

Il modello non perde capacità linguistica. Cambia solo
registro, stile, atteggiamento.

## I principali risultati

Dopo 25 modelli testati, 240 generazioni e 10 prompt,
questi sono i risultati principali:

### 1. I pesi contengono prospettive, non solo informazioni

TinyLlama è stato addestrato con testi di molti autori,
stili e discipline. Tutti quei modi di parlare sono stati
incisi nei pesi. La perturbazione seleziona quale
di quelle voci domina.

### 2. La gerarchia pesa più dei valori assoluti

Le tecniche che distruggono la struttura gerarchica
generano rifiuti. Quelle che la preservano generano testo coerente.
L'importante non è quanto cambia ogni peso, ma
**come cambiano unos rispetto agli altri**.

### 3. Ogni componente controlla un aspetto diverso

| Componente | Cosa controlla |
|------------|----------------|
| Attenzione | Struttura narrativa, relazioni tra token |
| FFN | Vocabolario, scelta delle parole, conoscenza pratica |
| Embedding | Identità concettuale, semplicità del linguaggio |

Perturbare solo l'attenzione dà testi più strutturati.
Perturbare solo il FFN cambia il vocabolario e l'approccio.
Perturbare solo gli embedding semplifica il linguaggio.

### 4. L'analogia DMT è quantificabile

Il modello non inventa contenuti nuovi. Riorganizza
associazioni interne. L'"allucinazione" è riorganizzazione,
non invenzione.

### 5. L'angolo importa, ma la grandezza importa di più

Matematicamente, una perturbazione può essere quasi ortogonale
al modello originale e continuare a funzionare, purché la sua
grandezza sia piccola. È come fare un passo di un millimetro
in direzione perpendicolare: tecnicamente cambi rotta,
maresti sulla stessa montagna.

## La formula del cambiamento di prospettiva

Possiamo riassumere il fenomeno in una formula semplice:

```
Prospettiva = Base + epsilon * delta

dove:
  epsilon = intensità (tipicamente 0,05 - 0,15)
  delta   = direzione nello spazio dei pesi
  |delta| = grandezza del cambiamento
```

Se `epsilon` è piccolo e `delta` preserva la gerarchia:
- La coerenza si mantiene
- La prospettiva cambia

Se `epsilon` è grande o `delta` distrugge la gerarchia:
- La coerenza si perde
- Appaiono rifiuti

Questo risponde anche a una domanda pratica:
abbiamo bisogno di un modello diverso per ogni stile?

**No.** Con un modello base e un insieme di direzioni
precalcolate possiamo interpolare stili in tempo reale:

```python
styled = base + 0.05 * delta_philosophical + 0,03 * delta_stoic
```

L'interpolazione lineare di punti vicini nella "varietà
di coerenza" produce altri punti validi.

## Implicazioni

### Per la creatività
Ogni tecnica è un "tono" diverso. Uno stesso tema può
essere generato da molteplici angoli senza addestrare nulla di nuovo.

### Per l'interpretabilità
La perturbazione è uno strumento di sonda: ci dice
quali parti del modello controllano quali aspetti dello stile.

### Per la personalizzazione
Invece di fare fine-tuning costoso, si può applicare
una perturbazione leggera per adattare lo stile di risposta.

### Per la filosofia dell'IA
Un LLM non è una macchina per rispondere domande.
È un **ecosistema di prospettive compresso nei pesi**.
La perturbazione è un modo di navigare quell'ecosistema.

## Perturbazione in runtime (motore C)

Oltre a generare GGUF Q4_0 con Python, il motore
`llm_inference.c` applica tecniche **in memoria**
sui pesi F16, senza file intermedio:

```bash
./llm_inference tinyllama-1.1b.F16.gguf \
  "The secret to happiness is" 60 0.7 40 \
  --seed 42 --perturb mystical --intensity 0.50
```

| Flag | Tecniche |
|------|----------|
| `--perturb` | `none`, `mystical`/`amplify`, `noise`, `blockdiag`, `manifold` |
| `--intensity` | forza (in F32 serve I più alta che in Q4 per notare l'effetto) |
| `--seed` | riproducibilità |
| `--steer` | spinge il residuale verso l'embedding di una parola |

`mystical` = `amplify_subspace` (proiezione + amplificazione).
Copia ~3,6 GB in F32 una volta (~25 s) e poi genera a ~6–10 tok/s.

Batteria di 15 prompt con I=0,50 (seed 42): media ~8,2 tok/s;
testi con clima esistenziale in prompt come
*When we dissolve the ego*, *The soul remembers*,
*The ancient wisdom teaches that*.

## Combinazioni e targeting (v11)

| Famiglia | Idea | Esempi |
|----------|------|--------|
| Combos | Impilare due tecniche | deep_reason, rare_perspective, structured_dream |
| Selective | Tecnica diversa per attn / ffn / emb | attention_alter, ffn_dream, extreme_selective |
| Sweep di I | Cercare il punto di rottura | 0,05 … 0,50 |

## Limitazioni oneste

- I risultati variano in base al prompt.
- Alcune combinazioni di tecniche degradano la qualità.
- Non tutto modello grande risponderà ugualmente: la struttura
  della varietà di coerenza può cambiare con la scala.
- La valutazione è qualitativa: misurare "prospettiva"
  resta un problema aperto.
- In F16 runtime, I=0,10 a volte non muove uscite corte
  (EOS anticipato); I=0,3–0,5 mostra il cambiamento con più chiarezza.

## Conclusione

Perturbare i pesi non è vandalizzare un modello.
È scoprire che dentro uno stesso insieme di numeri
vivono molte voci.

TinyLlama, visto così, smette di essere un unico strumento
per diventare un **paesaggio di possibilità**.
Ogni tecnica è un sentiero in quel paesaggio. Ogni intensità
è una velocità. E il sweet spot (vicino a 0,10 in Q4,
un po' maggiore in F32 runtime) è il punto giusto in cui il modello
resta se stesso, ma parla da un altro posto.

Il prossimo capitolo percorre lo **spazio multidimensionale**
dove vivono quelle voci: embedding, residuale, pesi e prospettive.

---

*Capitolo successivo: Percorso nello Spazio Multidimensionale*
