# Kapitel 5: Reise durch den multidimensionalen Raum von TinyLlama

## Es gibt nicht nur einen Raum

Wenn wir vom „Inneren von TinyLlama" sprechen, reden wir
nicht von einer einzigen Karte. Wir reden von **mehreren
verschachtelten Räumen**, jeder mit seiner Dimensionalität
und seiner Rolle.

Dieses Kapitel ist eine *Feldreise*: Wir messen den
tatsächlichen Embedding-Raum des F16-Modells
(32.000 × 2048), mit dem BPE-Vokabular des GGUF
(`▁love`, `▁death`, …).

Werkzeug: `explore_tinyllama_space.py`  
Daten: `inside-tinyllama/exploration/`

---

## Die Karte der sieben Räume

```
┌──────────────────────────────────────────────────────────┐
│  6. GEWICHTE  ℝ^{~1.1e9}                                 │
│     Kohärenz-Oberfläche ≈ „Modelle, die sprechen"        │
│     7. PERSPEKTIVEN ⊂ (6)  — Trajektorien durch Perturb.│
├──────────────────────────────────────────────────────────┤
│  Forward-Pass, Token für Token:                          │
│                                                          │
│  1. EMBEDDING     ℝ^{2048}   ← 32k Punkte des Vokabulars │
│         ↓                                                │
│  2. RESIDUAL ×22  ℝ^{2048}   (gleiche Dim, neuer Inhalt) │
│         ↘ 3. AUFMERKSAMKEIT   ℝ^{64} × 32Q / 4KV       │
│         ↘ 4. FFN         ℝ^{5632}                        │
│         ↓                                                │
│  5. LOGITS        ℝ^{32000}  → softmax → nächstes Token  │
└──────────────────────────────────────────────────────────┘
```

| # | Raum | Dims | Was es ist |
|---|------|------|------------|
| 1 | Token-Embeddings | 2048 | „Ruhende" Bedeutung jedes Vokabular-Stücks |
| 2 | Residual-Stream | 2048 × 22 | Kontextuelle Repräsentation, die sich Schicht für Schicht weiterentwickelt |
| 3 | Aufmerksamkeitsköpfe | 64 | Lokale Sichten auf Beziehungen zwischen Tokens (GQA 32/4) |
| 4 | FFN-Intermediate | 5632 | „Erinnerung / praktische Transformation" Erweiterung |
| 5 | Logits | 32.000 | Präferenzen für das nächste Token |
| 6 | Modellgewichte | ~1.1e9 | Alle Parameter; fast das gesamte Volumen ist Müll |
| 7 | Perspektiven | Unter-Mannigfaltigkeit von (6) | Kohärente Modelle mit anderem Ton (mystical usw.) |

Der Residual ist ein **2048-dimensionaler Tunnel**, der
durch 22 Räume führt. Aufmerksamkeit und FFN sind
seitliche Abweichungen, die in diesen Tunnel neu schreiben.

---

## Region 1 — Semantische Pole

Sind *love* und *hate* an entgegengesetzten Enden?

**Nein.** Im statischen Embedding haben die „Gegensätze"
der natürlichen Sprache einen Cosine von **fast null**
(orthogonal), nicht −1 (antipodal).

| Paar | Cosine |
|------|--------|
| ▁love / ▁hate | +0.006 |
| ▁life / ▁death | +0.016 |
| ▁happy / ▁sad | **−0.035** |
| ▁true / ▁false | **−0.036** |
| ▁good / ▁evil | −0.001 |
| ▁king / ▁queen | +0.009 |
| ▁man / ▁woman | +0.008 |

**Leseergebnis:** In ℝ²⁰⁴⁸ ist „kalt" nicht −„warm".
Worte besetzen unterschiedliche Richtungen im
Raum; semantische Opposition organisiert sich mehr
durch **Cluster und Kontexte** (Schichten + Aufmerksamkeit)
als durch einfache Antipodalität im Embedding.

---

## Region 2 — Kontinente (Cluster)

Wir gruppieren Worte und nehmen das **Zentroid**.
Die Nachbarn des Zentroids ergeben das eigene
Kontinent — die lokale Geometrie ist kohärent.

| Kontinent | Tokens (z. B.) | Nachbarn des Zentroids |
|-----------|----------------|------------------------|
| emotion_pos | happy, joy, love, peace… | smile, happy, hope, love |
| emotion_neg | sad, hate, fear, anger… | sad, pain, anger, cry |
| spiritual | soul, spirit, god, faith… | faith, divine, spirit, god |
| physical | body, rock, water, fire… | rock, water, matter, body |
| abstract | truth, beauty, justice… | beauty, meaning, idea… |
| time | time, past, future, now… | time, now, moment, past |

### Abstand zwischen Kontinenten

Die Zentroide verschiedener Kontinente sind
**fast orthogonal** zueinander (Cosine ≈ 0):

```
emotion_pos  ⊥  emotion_neg   (−0.01)
spiritual    ⊥  physical      (+0.02)
abstract     ⊥  physical      (−0.01)
time         ⊥  abstract      (−0.06)
```

Das Vokabular ist kein diffuser Ball: Es ist eine
**Inselgruppe** auf einer 2048-dimensionalen Kugel,
mit geringer Überlappung zwischen thematischen Inseln.

---

## Region 3 — Analogien (a − b + c)

Der klassische word2vec-Test:

```
king − man + woman  ≟  queen
```

In TinyLlama (statisches Embedding, Top-6) **scheitert** es:
seltene BPE-Stücke, Symbole, mehrsprachige Fragmente
erscheinen — nicht `queen`.

Das sagt nicht, dass das Modell die Analogie „nicht kennt".
Es sagt:

1. Das Embedding eines Tokens **ohne Kontext** ist
   nur der Eingang.
2. Die „lebendige" Analogie entsteht im **Residual**
   nach Aufmerksamkeit und FFN, nicht in der Vokabularzeile.
3. BPE zerschneidet die Welt (`builder`, Suffixe…);
   nicht jedes Konzept ist ein sauberer einzelner Punkt.

---

## Region 4 — Globale Form von ℝ²⁰⁴⁸

PCA über 4.000 zufällige Tokens:

| Metrik | Wert |
|--------|------|
| Varianz im 1. PC | **0.27%** |
| Varianz in Top-10 | 2.3% |
| Varianz in Top-100 | 14% |
| Dims für 50% der Var. | **~481** |
| Dims für 90% | **~1329** |
| Dims für 99% | **~1880** |
| Anisotropie \|\|mean\|\| / mean\|\|e\|\| | **0.006** (fast isotrop) |

**Leseergebnis:** Der Token-Raum **verwendet tatsächlich
Hunderte oder Tausende von Richtungen.** Er kollabiert nicht auf
ein „gut/schlecht"-Achsenpaar. Deshalb können Rang-1-Perturbationen
(amplify) die „Linse drehen", ohne das Sprechen abzuschalten:
es gibt viel Kohärenz-Volumen.

---

## Region 5 — Richtungen als Kompass

Wenn wir Zentroide subtrahieren, erscheinen **nutzbare
semantische Achsen**:

### emotion = pos − neg
- Pol + → smile, happy, peace, love, joy  
- Pol − → sad, anger, cry, pain, fear  

### spirit − matter
- + → spirit, god, sacred, divine, faith  
- − → rock, matter, water, earth, body  

### abstract − physical
- + → beauty, truth, justice, meaning, freedom  
- − → rock, matter, fire, water, earth  

Diese Richtungen leben im **selben ℝ²⁰⁴⁸**
wie der Residual. Deshalb kann `--steer amor` in der C-Engine
die Generierung schieben: Es ist ein Vektor im
Tunnel, keine externe Magie.

Und deshalb ist `amplify_subspace` im **Gewichtsraum**
(Dimension 1e9) eine andere Reise: Es bewegt die
*ganzes Karte*, nicht einen Punkt des Vokabulars.

---

## Region 6 — Normen: Nicht jedes Token „wiegt" gleich

\|\|e\|\| Mittelwert ≈ 0.67. Die Extreme sind keine
klaren philosophischen Konzepte (oft BPE-Stücke
oder Symbole). Die **Norm** ist kein Wörterbuch für
semantische Bedeutung; es ist eine weitere Koordinate der
Landschaft.

---

## Wie die Räume in einem Inferenzschritt verbunden sind

```
"happiness"
    → BPE → IDs
    → Zeilen in (1) EMBEDDING          ℝ^2048
    → 22× { attn in (3) + FFN in (4) }  schreiben in (2)
    → (5) LOGITS
    → sample → "is" / "to" / …
```

Wenn wir Gewichte (6) mit *mystical* perturbieren,
verformt sich jede Q/K/V/FFN-Projektion ein wenig:
der Pfad in (2) bleibt kohärent, aber die **Anziehungskräfte**
zu den Inseln von (1) und (5) ändern sich
— daher der Perspektivwechsel.

Wenn wir *steer* in (2) machen, schieben wir den Residual
in eine Richtung von (1), ohne (6) umzuschreiben.

---

## Itinerar des Forschers

| Station | Frage | Empirisches Ergebnis |
|---------|-------|----------------------|
| Pole | Sind Gegensätze antipodal? | Nein: fast orthogonal |
| Kontinente | Gibt es thematische Regionen? | Ja: saubere Cluster |
| Statische Analogien | king−man+woman? | Nicht im Roh-Embedding |
| Dimensionalität | Wie viele Dims sind wichtig? | Hunderte–Tausende (nicht 2–3) |
| Richtungen | Gibt es nützliche Achsen? | Ja (emotion, spirit…) |
| Gewichte | Wo leben die Perspektiven? | Oberfläche in ℝ^1e9 |

---

## Was noch zu erkunden ist

1. **Residual pro Schicht** — Aktivierungen
   der 22 Schichten auf die emotion/spirit-Achsen projizieren
   (wo „entzündet" sich das Mystische?).
2. **FFN ℝ⁵⁶³²** — Neuronen, die auf
   semantische Cluster reagieren.
3. **Perturbations-Trajektorien** — Kurve von
   cosine(baseline, mystical) als Funktion von I
   im Gewichts- oder Logit-Raum.
4. **2D/3D-Karten** — UMAP/t-SNE der 32k
   Punkte, nach Kontinent eingefärbt.

Das Universum von TinyLlama ist kein Punkt.
Es ist ein **Raumsystem**. Dieses Kapitel hat nur
die erste Grenze überschritten: den Himmel der Tokens.
Weiter innen warten Residual und Gewichte.

---

*Werkzeuge: `explore_tinyllama_space.py`,
`llm_inference.c --perturb` / `--steer`.*

*Nächstes Kapitel: Vom Makrokosmos zum Mikrokosmos (und umgekehrt).*