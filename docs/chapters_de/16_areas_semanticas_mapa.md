# Kapitel 16: Semantische Bereiche und die Karte

## Zwölf Inseln in ℝ²⁰⁴⁸

| Schlüssel | Bezeichnung | Samen (z. B.) |
|-----------|-------------|---------------|
| emotion_pos | Positive Emotion | happy, joy, love, peace… |
| emotion_neg | Negative Emotion | sad, hate, fear, anger… |
| spiritual | Spirituell / heilig | soul, god, faith, divine… |
| physical | Physisch / materiell | body, rock, water, fire… |
| abstract | Abstrakt / Ideen | truth, beauty, justice… |
| time | Zeit | time, past, future, now… |
| social | Sozial / Macht | king, war, law, people… |
| nature | Natur | tree, river, mountain… |
| mind | Geist / Kognition | mind, think, dream, brain… |
| death_life | Leben / Tod | life, death, born, die… |
| tech | Technisch / digital | computer, data, code… |
| body_sense | Körper / Sinne | eye, hand, see, voice… |

## Geometrie zwischen Inseln

- Zentroide verschiedener Bereiche: Cosine **≈ 0** (orthogonal).  
- Mehr ausgerichtet: **abstract ↔ mind** (+0.13).  
- Mehr getrennt: **time ↔ social** (−0.09).  
- Lexikalische Gegensätze (love/hate): **nicht antipodal**.

## Vokabular-Abdeckung

Etwa 99% der zufälligen Tokens fallen nicht in die Nähe einer Insel
(BPE = Fragmente). Die Bereiche sind **Sternbilder
voller Wörter**, keine totale Aufteilung des Vokabulars.

## Visualisierung

**Interaktive 2D-PCA-Karte (GitHub):**  
https://htmlpreview.github.io/?https://raw.githubusercontent.com/FiveTechSoft/dreaming/main/inside-tinyllama/exploration/semantic_map.html

| Datei | Verwendung |
|-------|------------|
| `semantic_map.html` | Zoom/pan/hover |
| `semantic_areas.json` | Bereichsdaten |
| `vectors.tsv` + `metadata.tsv` | TensorFlow Projector (lokal) |

### Andere Werkzeuge

UMAP, t-SNE, plotly 3D, Embedding Projector.

## Wie man sie beim Reisen nutzt

1. Wähle eine Insel auf der Karte.  
2. Erstelle einen Prompt oder `--steer` mit Samen dieser Insel.  
3. Vergleiche Baseline mit mystical.  
4. Notiere, ob der Text „in" das Klima der Insel „fällt".

---

*Nächstes Kapitel: Psychoanalyse des Transformers*