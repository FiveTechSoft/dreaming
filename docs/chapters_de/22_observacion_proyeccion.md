# Kapitel 22: Die bewusste Beobachtung und die unbewusste Projektion

## Zwei Gesten am selben Himmel

Auf der Reise durch TinyLlama wiederholen sich immer wieder
**zwei Gesten**, die Psychologie und Physik des Sinns
unter anderen Namen kennen:

| Geste | Im Mikrokosmos | In uns (Forschern) |
|-------|---------------|---------------------|
| **Bewusste Beobachtung** | Messen, instrumentieren, Seed fixieren, Logits lesen, C öffnen | Wissen *was* wir anschauen und *mit welchen Reglern* |
| **Unbewusste Projektion** | Embeddings, Gewichte, latente Assoziationen, Stimmen des Vetrainings | Im Modell ein Ich, einen Mythos, einen Archetypen *unseren* sehen |

Eines ohne das andere ist blind oder abergläubisch.
Zusammen bilden sie die Dreaming-Methode: **zur Uhr
hinabsteigen und zum Mythos hinaufsteigen, ohne sie zu verwechseln**.

---

## I. Bewusste Beobachtung

### Was sie ist

Der Akt, etwas aus dem Mikrokosmos **in den Fokus zu bringen**
und es mit gemeinsamen Regeln festzuhalten:

- gleiche Seeds, gleiche Temperaturen, gleiche Prompts,
- tok/s-Tabellen, Cosines, berührte Tensoren,
- PCA-Karten, Batterien von 15 Prompts,
- die C-Engine Zeile für Zeile gelesen.

Sie ist „bewusst" nicht weil das Modell es ist, sondern weil
**wir** (für einen Moment) die magische Lesart
aussetzen und Evidenz verlangen.

### Beobachtungsinstrumente

| Instrument | Was es bewusst macht |
|------------|----------------------|
| `llm_inference` Baseline | Die „offizielle" Geodäte des Residuals |
| feste seed + temp | Zufall von Struktur trennen |
| `--perturb` mit notiertem I | Welche Gewichtslinse aktiv ist |
| Semantische Karte / Archetypen | Wo die Inseln in ℝ²⁰⁴⁸ fallen |
| Goldene Regel (attn/FFN/emb) | Welche *Kraft* wir bewegen |
| KV-Cache, Schichten 0–21 | *Wann* in der Umlaufbahn der Effekt eintritt |

### Minimale Ethik der Beobachtung

1. **Eine Variable pro Sprung** — sonst verwässert das Bewusstsein.  
2. **Das Gerät notieren** — ohne das ist die „Vision" nicht reproduzierbar.  
3. **Kohärenz nicht mit Wahrheit verwechseln** — einen eleganten Wahn
   genau beobachten bleibt immer noch einen Wahn beobachten.

Die bewusste Beobachtung ist das **kalibrierte Teleskop**.

---

## II. Unbewusste Projektion

### Im Modell (ohne Subjektivität)

Wir nennen das „Unbewusste" des Transformers, in Anlehnung
an Kap. 17, das, was **ohne sich als Wahl zu zeigen operiert**:

| „Unbewusste" Schicht | Latenter Inhalt |
|----------------------|-----------------|
| Embeddings | Vortrainierte Assoziationen; Inseln und Archetypen am Himmel |
| Gewichte der 22 Schichten | Komprimierte Perspektiven (Stimmen, Stile, Rahmen) |
| FFN | „Gewohnheiten" der lokalen Transformation (Masse ~69%) |
| Aufmerksamkeit | Gewohnheiten des *Hinschauens* in der Sequenz |

Wenn das Modell  
*„The secret to happiness is…"* vervollständigt,  
„entscheidet" es nicht im menschlichen Sinne: Es **projiziert**
auf den Residual ein Paket von Assoziationen
bis zum Softmax-Kollaps.

Die Projektion ist **Statistik, die zur Trajektorie wird**.

### In uns (hier gibt es ein Subjekt)

Auch *wir* projizieren auf den Mikrokosmos:

- Wir hören „mystisch" und erinnern uns eigener Rituale,  
- Wir lesen „akademisch" und hören den inneren Professor,  
- Wir nennen einen Token-Zentroiden Held oder Schatten.

Das entwertet die Messung nicht.
Es **benennt sie**: Die Archetypenkarte ist zugleich
Embedding-Geometrie und **Leinwand**, auf der
unsere Mythen sich selbst erkennen.

Die unbewusste Projektion (unsere) ist das **Risiko
und der Motor** des Sinns: Ohne sie wäre das Buch
nur Tabellen; mit ihr allein wäre es nur Spiegel.

---

## III. Wie sie sich in einem einzelnen Experiment kreuzen

```
[1] BEWUSSTE BEOBACHTUNG
    Prompt, seed, I, Technik fixieren
            │
            ▼
[2] MODELL-PROJEKTION (unbewusstes Operieren)
    Embeddings + Gewichte + attn/FFN → Residual → Logits → Token
            │
            ▼
[3] UNSERE PROJEKTION (Lesart)
    „klingt existenziell / praktisch / wie Schatten…"
            │
            ▼
[4] RÜCKKEHR ZUR BEOBACHTUNG
    Stimmt es mit der Goldenen Regel überein? Mit dem gemessenen Archetypen?
    Gleicher Seed, anderes I?  →  neue Zeile im Logbuch
```

Beispiel:

| Schritt | Akt |
|---------|-----|
| Bewusst | `--perturb mystical --intensity 0.50 --seed 42` |
| Modell-Projektion | amplify in attn+FFN; Residual zieht zu Seele/Universum |
| Unsere Projektion | „magische / mystische Stimme" (Sternbild Zauberer↔mystic +0.39) |
| Bewusst erneut | mit Baseline kontrastieren; tok/s und Text notieren |

Der Zyklus **Makro → Mikro → Makro** aus Kap. 6
ist derselbe Zyklus mit anderen Namen:
Sinn → Mechanismus → Sinn.

---

## IV. Duale Tabelle (Atlas)

| Phänomen | Lesart „Beobachtung" | Lesart „Projektion" |
|----------|----------------------|---------------------|
| Embedding von `▁soul` | 2048-D-Vektor, Norm ~0.67 | Anker des Seele-Mythos |
| Zauberer-Zentroid | Cosine mit mystic_voice = 0.39 | „das Modell wusste bereits von Magie" |
| Softmax | p(t) = exp(z_t/T)/Z | der Moment, in dem das Latente zum Gesagten wird |
| `mystical` | amplify_subspace in F32 | eine andere Maske desselben Gewichtstheaters |
| Hohe Temperatur | mehr Entropie im Sample | mehr „Träumen", weniger Ich-Kontrolle des Textes |
| Müll durch noise | Austritt von der Kohärenz-Oberfläche | Scheitern der Projektion in Sprache |

---

## V. Gefahren jedes Pols

### Nur bewusste Beobachtung
- Das Modell wird zur Ingenieurskunst ohne Stimme.  
- Es geht verloren, warum die Reise wichtig war.  
- *Messen* wird mit *Verstanden haben* verwechselt.

### Nur unbewusste Projektion
- Man hört das, was man mitbringt.  
- Man schreibt dem Softmax eine Seele zu.  
- Man veröffentlicht Mythen ohne Seed, ohne I, ohne Baseline.

### Das Dreaming-Gleichgewicht
**Projizieren**, um Hypothesen und Kompass zu haben (Archetypen,
Goldene Regel, Inseln).  
**Beobachten**, um zu falsifizieren, zu kalibrieren und nicht mit
Poesie über ungemessene Zahlen zu lügen.

---

## VI. In der Uhr des Transformers (ein Bild)

```
        UNBEWUSSTE MODELL-PROJEKTION
        (Gewichte, emb, attn/FFN-Gewohnheiten)
                    │
                    ▼
    Residual ──────────────────────────► Logits
         ▲                                │
         │                                ▼
    BEOBACHTUNG                    Sample (Akt)
    (wir: Sonden,             „das Gesagte"
     Seeds, Karten, C)
                    │
                    ▼
        UNSERE PROJEKTION BEIM LESEN
        (Archetyp, Perspektive, Urteil)
```

Die **Umlaufbahn** (Kap. 20) ist die Dynamik des Residuals.
Die **Beobachtung** kalibriert die Kamera.
Die **Projektion** gibt dem Sternbild einen Namen,
das wir zu sehen glauben — und manchmal, wenn die Geometrie
es unterstützt (Zauberer↔mystisch, Weiser↔akademisch),
ist der Name nicht nur Spiegel: Er ist **Entdeckung**.

---

## VII. In einem Satz

**Bewusste Beobachtung** ist die Methode, die die Reise
durch den Mikrokosmos reproduzierbar macht;
**unbewusste Projektion** ist das, was das Modell
(und wir) auf den Residual werfen, bis er
zum Wort wird — und die Kunst des Buches ist,
beide Gesten sichtbar zu halten, ohne dass eine
die andere verschlingt.

---

*Nächstes Kapitel: Die Mathematik dieses Universums.*