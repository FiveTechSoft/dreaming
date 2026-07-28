# Capitolo 1: Cos'è TinyLlama? Perché TinyLlama?

## La storia di TinyLlama

TinyLlama è un modello di linguaggio **aperto** di
**~1,1 miliardi di parametri** (1,1B), con l'architettura
e il tokenizzatore della famiglia **Llama 2**,
ridotti a una dimensione che sta in un laboratorio
modesto: **22 layer**, residuale di **2048** dimensioni,
vocabolario di **32.000** token.

Non è nato in un big-tech closed lab. È nato come
**progetto aperto** del gruppo **StatNLP** della
**Singapore University of Technology and Design (SUTD)**:
pre-allenare un "Llama piccolo" su un corpus massiccio
(dell'ordine del **miliardo di token**, ~3 epoche nel
rapporto tecnico) e pubblicare **codice + checkpoint**.

Per Dreaming è il **microcosmo** ideale: abbastanza
piccolo da aprirlo intero, abbastanza ricco per
avere voce, geometria e prospettive diverse.

## Il team dietro il modello

### Autori del rapporto tecnico

Il paper *TinyLlama: An Open-Source Small Language Model*
([arXiv:2401.02385](https://arxiv.org/abs/2401.02385))
firma:

| Autore | Ruolo nel racconto pubblico del progetto |
|--------|------------------------------------------|
| **Peiyuan Zhang** | Co-leader dello sforzo aperto; repo / ingegneria di pre-allenamento |
| **Guangtao Zeng** | Co-leader; contatto di invio su arXiv (v1/v2) |
| **Tianduo Wang** | Coautore (StatNLP / SUTD) |
| **Wei Lu** | Coautore; facoltà, StatNLP Research Group (SUTD) |

Affiliazione istituzionale (paper e GitHub):

> **StatNLP Research Group**  
> **Singapore University of Technology and Design (SUTD)**

Contatti che figurano nel rapporto (dominio SUTD):
`peiyuan_zhang`, `tianduo_wang`, `luwei` @ sutd.edu.sg;
`guangtao_zeng` @ mymail.sutd.edu.sg.

Nel repository ufficiale il progetto si presenta come
contributo di quei quattro nomi, con Zhang e Zeng
in primo piano come promotori dell'*open endeavor*.

### Linea del tempo (fatti pubblici)

| Data | Passaggio chiave |
|------|-----------------|
| **2023-09-01** | Inizio pubblico del pre-allenamento (annuncio nel [GitHub del progetto](https://github.com/jzhang38/TinyLlama): *training has started*). |
| **Autunno–inverno 2023** | Addestramento su larga scala; note intermedie, checkpoint e correzioni di curva/bug (es. aggiornamenti di dic. 2023 nel repo). |
| **2024-01-04** | **Pubblicazione del paper su arXiv** (v1): modello 1.1B, dati, efficienza (FlashAttention, stack tipo Lit-GPT) e risultati vs. altri open di dimensioni comparabili. |
| **2024-01-05** | Ecosistema immediato: il paper appare come evidenziato del giorno in elenchi della comunità (es. Hugging Face Papers). |
| **2024 (primi mesi)** | Diffusione di checkpoint su **Hugging Face** (base / passaggi intermedi / varianti **Chat**, es. `TinyLlama/TinyLlama-1.1B-Chat-v1.0`) e adozione in demo, finetune e motori locali (llama.cpp, GGUF, ecc.). |
| **2024-06-04** | **Revisione v2** del rapporto su arXiv (aggiornamento del technical report). |

### Cosa hanno costruito (in una frase tecnica)

Un **SLM (Small Language Model)** open-source:

- architettura **Llama 2** a scala 1,1B,
- pre-allenamento massiccio su misture di dati aperti,
- codice e pesi **riutilizzabili**,
- obiettivo esplicito: dimostrare che un modello *piccolo*
  ben pre-allenato **compete** con altri aperti di
  dimensione simile e serve come base per ricerca
  e deployment leggero.

Licenza e cultura: il progetto si colloca nell'onda
**open weights** successiva a LLaMA/Llama 2: non è un
API chiuso, è un artefatto che si può scaricare,
quantizzare e dissezionare — esattamente ciò che fa
Dreaming in questo libro.

### Reazioni e ricezione

**Nella comunità di ricerca e open-source**
(gennaio 2024 in poi):

1. **Entusiasmo per il "piccolo ben fatto".**  
   Dopo anni di narrazione "solo scala gigante",
   TinyLlama ha rafforzato l'interesse per gli **SLM**:
   stanno in GPU da consumatore / CPU con quantizzazione,
   eppure generano testo utile.

2. **Credibilità per l'apertura totale.**  
   Paper + GitHub + checkpoint (non solo un blog post)
   hanno permesso di riprodurre, fine-tunare e *fare fork*.
   Questo spiega la sua rapida comparsa in tutorial,
   collezioni HF e backend tipo llama.cpp.

3. **Confronto con pari di ~1B.**  
   Il rapporto sostiene che **supera** vari modelli
   aperti di dimensioni comparabili in task *downstream*.
   La ricezione non è stata "un altro toy model", ma
   "baseline seria da 1B".

4. **Impatto bibliografico e d'uso.**  
   L'arXiv accumula un volume alto di citazioni per un
   technical report di SLM (ordine di **centinaia /
   ~mila+** secondo contatori pubblici dopo ~2 anni),
   segnale che è diventato **riferimento di citazione** quando
   qualcuno ha bisogno di un Llama-like piccolo e aperto.

5. ** sfumatura / dibattito sano.**  
   Parte della comunità ha discusso il costo del pre-allenamento
   di un 1,1B (contraddice Chinchilla?). Il team stesso
   ha risposto nel FAQ del repo: il valore non è solo
   il punto ottimale teorico di calcolo, ma **un artefatto
   aperto**, ben addestrato, per la comunità.
   Ci sono stati anche ritardi e note oneste sulle curve
   di training e sugli schedule: più "lab aperto" che marketing
   opaco.

6. **Ecosistema Dreaming / questo libro.**  
   Per noi la reazione rilevante è pratica:
   esiste un GGUF F16/Q4_0 stabile, un vocabolario BPE
   LLaMA, e una dimensione che permette motore C, mappe di
   embedding e perturbazioni senza cluster. Senza il
   team StatNLP e la loro decisione di **aprire** il modello,
   questo diario non avrebbe microcosmo.

### Collegamenti canonici

| Risorsa | URL |
|---------|-----|
| Paper (arXiv) | https://arxiv.org/abs/2401.02385 |
| Codice e diario del pre-allenamento | https://github.com/jzhang38/TinyLlama |
| Modelli su Hugging Face (famiglia TinyLlama) | https://huggingface.co/TinyLlama |

### Citazione (BibTeX del progetto)

```bibtex
@misc{zhang2024tinyllama,
  title={TinyLlama: An Open-Source Small Language Model},
  author={Peiyuan Zhang and Guangtao Zeng and Tianduo Wang and Wei Lu},
  year={2024},
  eprint={2401.02385},
  archivePrefix={arXiv},
  primaryClass={cs.CL}
}
```

## Perché è speciale

TinyLlama è speciale perché:

1. **Dimensione gestibile**: Con ~1,1B parametri, possiamo
   studiare tutta la sua struttura senza perdersi nella complessità
   di un modello di decine di miliardi.

2. **Risultati interessanti**: Nonostante sia piccolo,
   produce testo coerente e utile; il paper lo colloca al
   di sopra di vari open-source di dimensioni comparabili.

3. **Trasparenza**: pesi e codice pubblici → possiamo
   visualizzare, perturbare e misurare ogni componente
   (motore C, mappe, lenti Dreaming).

4. **Stirpe Llama 2**: stessa "grammatica" di layer (RoPE,
   GQA, SwiGLU…) dei grandi, su scala da laboratorio.

## La nostra motivazione per studiarlo

Studiamo TinyLlama perché ci permette di:

- Capire come funzionano i modelli di linguaggio
  **con un forward leggibile da punta a punta** (cap. 29)
- Esplorare la geometria dello spazio di significato
- Scoprire come le modifiche dei pesi
  possono cambiare prospettive
- Imparare sulla struttura interna dei transformer
- Apprezzare, nella pratica, il lavoro di un team
  accademico che ha puntato sull'**aperto**

## Conclusione

TinyLlama non è solo "un modello piccolo". È il
risultato di un team concreto (Zhang, Zeng, Wang, Lu;
StatNLP / SUTD), di un pre-allenamento partito a
**settembre 2023** e di un paper uscito alla luce
il **4 gennaio 2024** — e di una comunità che lo
ha adottato come SLM di riferimento.

Nei prossimi capitoli esploriamo la sua struttura
interna e i suoi segreti come **microcosmo** misurabile.

---

*Capitolo successivo: La Struttura Interna di TinyLlama*
