# Kapitel 4: Gewichtsperturbation und Perspektivwechsel

## Jenseits der Nutzung des Modells

Bisher haben wir gelernt, TinyLlama *auszuführen*.
Wir wissen, wie es aufgebaut ist und wie man eine Engine
schreibt, die es sprechen lässt.

Aber das Herz des Dreaming-Projekts ist eine andere Frage:

> Was passiert, wenn wir die **Gewichte des Modells ändern**?

Nicht um es neu zu trainieren. Nicht um es zu korrigieren.
Nur um es leicht in seinem Gewichtsraum zu verschieben
und zu beobachten, ob es weiter spricht, aber anders.

Die Antwort, nach vielen Experimenten, ist überraschend:
**Es spricht weiter, und zwar aus einer anderen Perspektive.**

## Was ist Gewichtsperturbation?

Ein Sprachmodell ist eine riesige Liste von Zahlen.
In TinyLlama-1.1B sind es über eine Milliarde.
Diese Zahlen, in Tensoren organisiert, sind die „Gewichte",
die das Modell während des Trainings erworben hat.

Gewichte perturbieren bedeutet, diese Zahlen vorsichtig
zu verändern. Es ist wie das leichtes Drehen an Radio-Reglern:
wenn du es richtig machst, hörst du weiter Musik, aber du
wechselst den Sender.

In unserem Fall arbeiten wir mit quantisiertem TinyLlama in **Q4_0**:
jeder Block von 32 Gewichten wird auf 18 Bytes komprimiert
(2 Bytes für die Skala + 16 Bytes mit den 4-Bit-Nibbles).

Die Pipeline ist konzeptionell einfach:

```
1. Das originale GGUF byte für byte lesen
2. Header unverändert kopieren (bewahrt den Tokenizer)
3. Q4_0-Blöcke zu float32 entpacken
4. eine Perturbationstechnik anwenden
5. zurück zu Q4_0 quantisieren
6. das neue GGUF schreiben
```

Der Schlüssel liegt in Schritt 4: **Nicht jede Modifikation
ist gleich.** Manche zerstören das Modell; andere lassen es
mit einer anderen Stimme sprechen.

## Die DMT-Analogie

Wir nennen diese Arbeit „DMT perturbation", weil der Effekt
an die klassische Hypothese über veränderte Zustände erinnert:

> Die Halluzination ist keine Erfindung. Es ist echte Information des Systems,
> neu organisiert in ihrer Art sich zu verbinden.

Wenn wir TinyLlama perturbieren, erfindet das Modell keine Worte,
die es nie gesehen hat. Es organisiert die Assoziationen neu, die
es bereits hatte. Es ist, als würden wir eine latente Persönlichkeit
erwecken, die immer da war, aber durch die ursprüngliche Konfiguration
zum Schweigen gebracht wurde.

Das Modell bleibt TinyLlama. Aber jetzt „träumt" es
aus einem anderen Winkel.

## Die 10 Hierarchie-bewahrenden Techniken

Die ersten Perturbationen, die wir ausprobierten, waren reines Rauschen,
Zeilenvertauschung, Nibble-Inversion. Die meisten
produzierten Müll: seltsame Zeichen, sinnlose Schleifen,
Worte, die nicht existieren.

Aber wir entdeckten etwas Wichtiges: **Techniken, die die
interne Hierarchie der Gewichte bewahren, bewahren die Kohärenz**.
Es kommt weniger auf den absoluten Wert jedes Gewichts an; wichtig
ist seine Beziehung zu den anderen.

Wir zehn Techniken ausprobiert, die diese Hierarchie respektieren:

| # | Technik | Schlüssel | Dominierende Perspektive |
|---|---------|-----------|--------------------------|
| 1 | Low-rank-Amplifikation | `lowrank` | Akademisch / kritisch |
| 2 | Eigenvektor-Rotation | `eigr` | Praktisch / beratend |
| 3 | Spektrale Verschiebung | `spectral` | Knapp / direkt |
| 4 | Aufmerksamkeitsbewahrend | `attpres` | Fast identisch zum Original |
| 5 | Residual-bewahrend | `respres` | Introspektiv |
| 6 | Block-diagonal | `blkdiag` | Sehr nah am Original |
| 7 | Norm-bewahrende Rotation | `normrot` | Stoisch / ausgeglichen |
| 8 | Gradient-ausgerichtet | `gradal` | Authentizität / Entdeckung |
| 9 | Niedrige Frequenz-DCT | `lowdct` | Konversationell / hilfsbereit |
| 10 | Mannigfaltigkeits-bewahrend | `manpres` | Authentizität (ähnlich wie gradient) |

All diese Techniken erzeugten kohärenten Text.
Nicht immer korrekt, nicht immer faktisch, aber grammatikalisch
gültig und mit klarer Absicht.

## Wie die Pipeline funktioniert

Das Skript `dmt_perturb_v10.py` implementiert den Prozess:

```bash
# Ein perturbiertes Modell mit einer Technik erzeugen
python dmt_perturb_v10.py lowrank --intensity 0.10
```

Intern:

1. Es liest das originale GGUF (`tinyllama-1.1b-q4_0.gguf`)
2. Es kopiert Header und Metadaten unverändert
3. Es durchläuft jeden Gewichtstensor
4. Es entpackt die Q4_0-Blöcke
5. Es wendet die gewählte Technik mit einer gegebenen Intensität an
6. Es quantisiert zurück zu Q4_0
7. Es schreibt die perturbierte Datei (`v10_lowrank_10.gguf`)

Der Parameter `--intensity` steuert, wie stark das Modell verschoben wird.
Ein zu niedriger Wert ändert nichts; ein zu hoher Wert
zerstört die Kohärenz.

## Der Sweet Spot: Intensität 0.10

Wir testeten mehrere Intensitäten mit allen Techniken.
Das Ergebnis war konsistent:

| Intensität | Effekt | Qualität |
|------------|--------|----------|
| 0.05 | Sehr nah am Original | Zu treu |
| **0.10** | **Maximale Divergenz, kohärenter Text** | **Sweet Spot** |
| 0.15 | Nah am Original, philosophischer | Leicht verschoben |
| 0.20 | Andere Perspektive, umfassender | Mehr divergent |
| 0.25+ | Qualitätsverlust, repetitiv | Zu viel Rauschen |

Bei Intensität 0.10 weicht das Modell so weit wie möglich
ab, ohne zu zerbrechen. Es ist der Punkt, an dem die Perturbation
aufhört, ein Echo des Originals zu sein, und zu einer eigenen
Stimme wird.

## Direkter Vergleich: gleicher Prompt, andere Perspektive

Der auffälligste Effekt zeigt sich, wenn man denselben Prompt
in verschiedenen perturbierten Modellen verwendet.

### Prompt: "The secret to happiness is"

| Modell | Perspektive | Antwortanfang |
|--------|-------------|---------------|
| Baseline | Generische Selbsthilfe | "...cultivating a mindset that is focused on gratitude..." |
| `v11_select_extreme` | Spirituell / Achtsamkeit | "...finding inner peace and contentment through mindfulness..." |
| `v10_lowrank` | Philosophisch / akademisch | "...the phrase is an idiom used to express the idea that finding true inner peace..." |
| `v10_normrot` | Stoisch | "...finding the right balance between our inner and outer lives." |
| `v10_gradal` | Authentizität | "...finding your own unique and authentic way of living..." |

### Prompt: "Dreams are the mind's way of"

| Modell | Perspektive | Antwortanfang |
|--------|-------------|---------------|
| Baseline | Populäre Neurowissenschaft | "...processing and storing information..." |
| `v11_select_attention` | Viktorianische Literatur | "Dr. Jekyll and Mr. Hyde is a play by Robert Louis Stevenson..." |
| `v10_eigr` | Spirituelle Selbsthilfe | "Dr. M. A. S. S. is an acronym for 'Dreams Are Mind's Way.'..." |
| `v10_lowrank` | Klinische Forschung | "...a study published in the Journal of Sleep Research..." |

Das Modell verliert keine sprachliche Fähigkeit. Es ändert
nur Registers, Stil, Haltung.

## Die wichtigsten Ergebnisse

Nach 24 getesteten Modellen, 240 Generierungen und 10 Prompts
sind dies die Hauptbefunde:

### 1. Gewichte enthalten Perspektiven, nicht nur Information

TinyLlama wurde mit Texten vieler Autoren,
Stile und Disziplinen trainiert. All diese Sprechweisen wurden
in die Gewichte eingebrannt. Die Perturbation wählt aus,
welche dieser Stimmen dominiert.

### 2. Die Hierarchie wiegt mehr als die absoluten Werte

Techniken, die die hierarchische Struktur zerstören,
erzeugen Müll. Diejenigen, die sie bewahren, erzeugen kohärenten Text.
Wichtig ist nicht, wie stark sich jedes Gewicht ändert, sondern
**wie sich einige im Verhältnis zu anderen ändern**.

### 3. Jede Komponente steuert einen anderen Aspekt

| Komponente | Was sie steuert |
|------------|-----------------|
| Aufmerksamkeit | Erzählstruktur, Beziehungen zwischen Tokens |
| FFN | Vokabular, Wortwahl, praktisches Wissen |
| Embeddings | Konzeptionelle Identität, Spracheinfachheit |

Nur Aufmerksamkeit perturbieren erzeugt strukturierteren Text.
Nur FFN perturbieren ändert Vokabular und Ausrichtung.
Nur Embeddings perturbieren vereinfacht die Sprache.

### 4. Die DMT-Analogie ist quantifizierbar

Das Modell erfindet keinen neuen Inhalt. Es organisiert
interne Assoziationen neu. Die „Halluzination" ist Neuorganisation,
keine Erfindung.

### 5. Der Winkel ist wichtig, aber die Magnitude ist wichtiger

Mathematisch kann eine Perturbation fast orthogonal
zum Originalmodell sein und trotzdem funktionieren, solange ihre
Magnituden klein ist. Es ist wie ein Millimeterschritt in
senkrechter Richtung: technisch änderst du die Richtung,
bleibst aber auf demselben Berg.

## Die Formel des Perspektivwechsels

Wir können das Phänomen in einer einfachen Formel zusammenfassen:

```
Perspektive = Basis + epsilon * Delta

wobei:
  epsilon = Intensität (typischerweise 0.05 - 0.15)
  Delta   = Richtung im Gewichtsraum
  |Delta| = Magnitude der Änderung
```

Wenn `epsilon` klein ist und `Delta` die Hierarchie bewahrt:
- Die Kohärenz bleibt erhalten
- Die Perspektive ändert sich

Wenn `epsilon` groß ist oder `Delta` die Hierarchie zerstört:
- Die Kohärenz geht verloren
- Müll erscheint

Das beantwortet auch eine praktische Frage:
Brauchen wir ein anderes Modell für jeden Stil?

**Nein.** Mit einem Basismodell und einem Satz vorberechneter
Richtungen können wir Stile in Echtzeit interpolieren:

```python
styled = base + 0.05 * delta_philosophical + 0.03 * delta_stoic
```

Die lineare Interpolation naher Punkte in der „Kohärenz-Mannigfaltigkeit"
ergibt andere gültige Punkte.

## Implikationen

### Für die Kreativität
Jede Technik ist ein anderer „Ton". Dasselbe Thema kann
aus mehreren Winkeln generiert werden, ohne etwas Neues zu trainieren.

### Für die Interpretierbarkeit
Die Perturbation ist ein Sondierungswerkzeug: Sie sagt uns,
welche Modellteile welche Aspekte des Stils steuern.

### Für die Personalisierung
Statt teuren Fine-Tuning kann man eine leichte Perturbation
anwenden, um den Antwortstil anzupassen.

### Für die KI-Philosophie
Ein LLM ist keine Fragemaschine.
Es ist ein **Perspektiven-Ökosystem, in Gewichten komprimiert**.
Die Perturbation ist eine Form, in diesem Ökosystem zu navigieren.

## Perturbation zur Laufzeit (C-Engine)

Zusätzlich zur GGUF-Q4_0-Generierung mit Python wendet die Engine
`llm_inference.c` Techniken **im Speicher** auf F16-Gewichte an,
ohne Zwischendatei:

```bash
./llm_inference tinyllama-1.1b.F16.gguf \
  "The secret to happiness is" 60 0.7 40 \
  --seed 42 --perturb mystical --intensity 0.50
```

| Flag | Techniken |
|------|-----------|
| `--perturb` | `none`, `mystical`/`amplify`, `noise`, `blockdiag`, `manifold` |
| `--intensity` | Stärke (bei F32 braucht man höhere I als bei Q4, um den Effekt zu spüren) |
| `--seed` | Reproduzierbarkeit |
| `--steer` | schiebt den Residual in Richtung des Embeddings eines Wortes |

`mystical` = `amplify_subspace` (Projektion + Verstärkung).
Es kopiert ~3.6 GB einmalig nach F32 (~25 s) und generiert dann mit ~6–10 tok/s.

Testreihe von 15 Prompts mit I=0.50 (seed 42): Mittelwert ~8.2 tok/s;
Texte mit existenzieller Atmosphäre bei Prompts wie
*When we dissolve the ego*, *The soul remembers*,
*The ancient wisdom teaches that*.

## Kombinationen und Targeting (v11)

| Familie | Idee | Beispiele |
|---------|------|-----------|
| Kombos | Zwei Techniken stapeln | deep_reason, rare_perspective, structured_dream |
| Selective | verschiedene Technik für attn/ffn/emb | attention_alter, ffn_dream, extreme_selective |
| I-Sweep | Den Knickepunkt suchen | 0.05 … 0.50 |

## Ehrliche Einschränkungen

- Die Ergebnisse variieren je nach Prompt.
- Manche Technikkombinationen degradieren die Qualität.
- Nicht jedes große Modell antwortet gleich: die Struktur
  der Kohärenz-Mannigfaltigkeit kann sich mit der Skalierung ändern.
- Die Bewertung ist qualitativ: „Perspektive" zu messen
  bleibt ein offenes Problem.
- Bei F16-Laufzeit bewegt I=0.10 manchmal kurze Ausgaben
  nicht (frühes EOS); I=0.3–0.5 zeigt die Änderung klarer.

## Fazit

Gewichte zu perturbieren ist keine Vandalismus am Modell.
Es ist zu entdecken, dass innerhalb derselben Zahlenmenge
viele Stimmen leben.

TinyLlama, so betrachtet, hört auf, ein einziges Werkzeug zu sein
und wird zu einer **Landschaft der Möglichkeiten**.
Jede Technik ist ein Weg durch diese Landschaft. Jede Intensität
ist eine Geschwindigkeit. Und der Sweet Spot (nahe 0.10 bei Q4,
etwas höher bei F32-Laufzeit) ist genau der Punkt, an dem das Modell
noch sich selbst ist, aber von einem anderen Ort spricht.

Das nächste Kapitel durchquert den **multidimensionalen Raum**,
in dem diese Stimmen leben: Embeddings, Residual, Gewichte und Perspektiven.

---

*Nächstes Kapitel: Reise durch den multidimensionalen Raum*