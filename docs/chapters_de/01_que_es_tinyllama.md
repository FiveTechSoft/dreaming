# Kapitel 1: Was ist TinyLlama? Warum TinyLlama?

## Die Geschichte von TinyLlama

TinyLlama ist ein **offenes** Sprachmodell mit
**etwa 1,1 Milliarden Parametern** (1.1B), mit der
Architektur und dem Tokenizer der **Llama 2**-Familie,
reduziert auf eine Größe, die in einem bescheidenen
Labor Platz hat: **22 Schichten**, Residual von **2048**
Dimensionen, Vokabular von **32.000** Tokens.

Es wurde nicht in einem geschlossenen Big-Tech-Labor geboren.
Es entstand als **Open-Source-Projekt** der Gruppe **StatNLP**
der **Singapore University of Technology and Design (SUTD)**:
ein „kleines Llama" auf einem massiven Corpus vortrainieren
(Größenordnung von **Billionen Tokens**, ~3 Epochen im
technischen Bericht) und **Code + Checkpoints** veröffentlichen.

Für Dreaming ist es der ideale **Mikrokosmos**: klein genug,
um vollständig geöffnet zu werden, reich genug, um
Stimme, Geometrie und verschiedene Perspektiven zu haben.

## Das Team hinter dem Modell

### Autoren des technischen Berichts

Das Paper *TinyLlama: An Open-Source Small Language Model*
([arXiv:2401.02385](https://arxiv.org/abs/2401.02385))
wurde unterzeichnet von:

| Autor | Rolle in der öffentlichen Erzählung des Projekts |
|-------|--------------------------------------------------|
| **Peiyuan Zhang** | Co-Leiter der Open-Source-Initiative; Repo / Vetrainings-Engineering |
| **Guangtao Zeng** | Co-Leiter; arXiv-Einreichungskontakt (v1/v2) |
| **Tianduo Wang** | Co-Autor (StatNLP / SUTD) |
| **Wei Lu** | Co-Autor; Fakultät, StatNLP Research Group (SUTD) |

Institutionelle Zugehörigkeit (Paper und GitHub):

> **StatNLP Research Group**  
> **Singapore University of Technology and Design (SUTD)**

Kontaktadressen im Bericht (SUTD-Domäne):
`peiyuan_zhang`, `tianduo_wang`, `luwei` @ sutd.edu.sg;
`guangtao_zeng` @ mymail.sutd.edu.sg.

Im offiziellen Repository wird das Projekt als
Beitrag dieser vier Namen vorgestellt, wobei Zhang und Zeng
als Förderer des *open endeavor* hervorgehoben werden.

### Zeitstrahl (öffentliche Fakten)

| Datum | Meilenstein |
|-------|-------------|
| **2023-09-01** | Öffentlicher Beginn des Vetrainings (Ankündigung im [GitHub des Projekts](https://github.com/jzhang38/TinyLlama): *training has started*). |
| **Herbst–Winter 2023** | Training in großem Maßstab; Zwischennotizen, Checkpoints und Kurven-/Bug-Anpassungen (z. B. Dezember-Updates 2023 im Repo). |
| **2024-01-04** | **Veröffentlichung des Papers auf arXiv** (v1): Modell 1.1B, Daten, Effizienz (FlashAttention, Lit-GPT-ähnlicher Stack) und Ergebnisse im Vergleich zu anderen Open-Source-Modellen vergleichbarer Größe. |
| **2024-01-05** | Unmittelbares Ökosystem: Das Paper erscheint als Tageshighlight in Community-Listen (z. B. Hugging Face Papers). |
| **2024 (erste Monate)** | Verbreitung von Checkpoints auf **Hugging Face** (Basis / Zwischenschritte / **Chat**-Varianten, z. B. `TinyLlama/TinyLlama-1.1B-Chat-v1.0`) und Übernahme in Demos, Finetunes und lokale Engines (llama.cpp, GGUF usw.). |
| **2024-06-04** | **Revision v2** des Berichts auf arXiv (Aktualisierung des technischen Reports). |

### Was sie gebaut haben (in einem technischen Satz)

Ein **SLM (Small Language Model)** als Open Source:

- **Llama 2**-Architektur im Maßstab 1.1B,
- massives Vetraining auf offenen Datensätzen,
- **wiederverwendbarer** Code und Gewichte,
- explizites Ziel: zu zeigen, dass ein *kleines*, gut
  vorgetrainiertes Modell **konkurrenzfähig** mit anderen
  Open-Source-Modellen vergleichbarer Größe ist und als Basis
  für Forschung und leichtes Deployment dient.

Lizenz und Kultur: Das Projekt fällt in die Welle
**offener Gewichte** nach LLaMA/Llama 2: Es ist keine
geschlossene API, sondern ein Artefakt, das heruntergeladen,
quantisiert und seziert werden kann — genau das, was
Dreaming in diesem Buch tut.

### Reaktionen und Rezeption

**In der Forschungs- und Open-Source-Community**
(ab Januar 2024):

1. **Begeisterung über das „kleine, gut gemachte".**  
   Nach Jahren der Erzählung „nur riesige Skalierung"
   stärkte TinyLlama das Interesse an **SLMs**:
   Sie passen auf Verbraucher-GPUs / CPUs mit Quantisierung
   und erzeugen dennoch nützlichen Text.

2. **Glaubwürdigkeit durch totale Offenheit.**  
   Paper + GitHub + Checkpoints (nicht nur ein Blogpost)
   ermöglichten Reproduktion, Finetuning und *Forking*.
   Das erklärt sein schnelles Erscheinen in Tutorials,
   HF-Sammlungen und llama.cpp-Backends.

3. **Vergleich mit Gleichaltrigen um ~1B.**  
   Der Bericht behauptet, dass es mehrere Open-Source-Modelle
   vergleichbarer Größe bei *Downstream*-Aufgaben **übertrifft**.
   Die Rezeption war nicht „noch ein Toy-Modell", sondern
   „ernstzunehmende 1B-Baseline".

4. **Bibliografischer und Nutzungseffekt.**  
   arXiv akkumuliert ein hohes Zitationsvolumen für einen
   technischen SLM-Bericht (Größenordnung **hunderte /
   ~tausend+** nach öffentlichen Zählern nach ~2 Jahren),
   ein Zeichen, dass es zur **Referenzzitation** wurde,
   wenn jemand ein kleines, offenes Llama-ähnliches Modell benötigt.

5. **Nuance / gesunde Debatte.**  
   Ein Teil der Community diskutierte die Kosten des Vetrainings
   eines 1.1B-Modells (widerspricht das Chinchilla?). Das eigene
   Team antwortete im FAQ des Repos: Der Wert liege nicht nur
   im theoretischen optimalen Berechnungspunkt, sondern in einem
   **offenen Artefakt**, gut trainiert, für die Community.
   Es gab auch Verzögerungen und ehrliche Notizen zu Trainingskurven
   und Schedules: eher „offenes Labor" als trübes Marketing.

6. **Dreaming-Ökosystem / dieses Buch.**  
   Für uns ist die relevante Reaktion praktischer Art:
   Es gibt ein stabiles GGUF F16/Q4_0, ein BPE-Vokabular
   LLaMA und eine Größe, die C-Engine, Embedding-Karten und
   Perturbationen ohne Cluster ermöglicht. Ohne das StatNLP-Team
   und seine Entscheidung, das Modell zu **öffentlichen**, hätte
   dieses Logbuch keinen Mikrokosmos.

### Kanonische Verknüpfungen

| Ressource | URL |
|-----------|-----|
| Paper (arXiv) | https://arxiv.org/abs/2401.02385 |
| Code und Vetraining-Logbuch | https://github.com/jzhang38/TinyLlama |
| Modelle auf Hugging Face (TinyLlama-Familie) | https://huggingface.co/TinyLlama |

### Zitat (BibTeX des Projekts)

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

## Warum es besonders ist

TinyLlama ist besonders, weil:

1. **Handhabbare Größe**: Mit ~1.1B Parametern können wir
   die gesamte Struktur studieren, ohne in der Komplexität
   eines Modells mit zig Milliarden unterzugehen.

2. **Interessante Ergebnisse**: Obwohl es klein ist,
   erzeugt es kohärenten und nützlichen Text; das Paper
   ordnet es über mehreren Open-Source-Modellen vergleichbarer Größe.

3. **Transparenz**: Öffentliche Gewichte und Code → wir können
   jede Komponente visualisieren, perturbieren und messen
   (C-Engine, Karten, Dreaming-Linsen).

4. **Llama 2-Abstammung**: Gleiche „Grammatik" der Schichten (RoPE,
   GQA, SwiGLU…) wie die Großen, im Labormaßstab.

## Unsere Motivation, es zu studieren

Wir studieren TinyLlama, weil es uns ermöglicht:

- Zu verstehen, wie Sprachmodelle funktionieren
  **mit einem von Anfang bis Ende lesbaren Forward** (Kap. 29)
- Die Geometrie des Bedeutungsraums zu erkunden
- Zu entdecken, wie Gewichtsmodifikationen
  Perspektiven ändern können
- Über die interne Struktur von Transformatoren zu lernen
- In der Praxis die Arbeit eines akademischen Teams zu würdigen,
  das auf **Offenheit** gesetzt hat

## Fazit

TinyLlama ist nicht nur „ein kleines Modell". Es ist das
Ergebnis eines konkreten Teams (Zhang, Zeng, Wang, Lu;
StatNLP / SUTD), eines Vetrainings, das im
**September 2023** begann, und eines Papers, das am
**4. Januar 2024** ans Licht kam — und einer Community, die es
als Referenz-SLM übernahm.

In den nächsten Kapiteln erforschen wir seine interne
Struktur und seine Geheimnisse als messbarer **Mikrokosmos**.

---

*Nächstes Kapitel: Die interne Struktur von TinyLlama*