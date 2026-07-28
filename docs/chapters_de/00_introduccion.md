# Einführung: Inside TinyLlama

## Ein Mikrokosmos, der auf eine Festplatte passt

Dieses Buch ist das Logbuch des Projekts
**Dreaming** angewendet auf **TinyLlama-1.1B**: ein Modell
klein genug, um es vollständig zu öffnen, und reich genug,
um zu überraschen.

Es ist kein Benutzerhandbuch eines Chatbots.
Es ist eine Reise durch das **Innere** eines Transformers:

- seine Architektur (22 Schichten, 9 Tensoren pro Schicht),
- eine Inferenz-Engine in C, die wir Zeile für Zeile lesen können,
- die Perturbation von Gewichten als *Perspektivwechsel*,
- die Geometrie des Embedding-Raums,
- die „Kräfte" des Forwards (Aufmerksamkeit, FFN, Residual, Softmax),
- und das Wechselspiel zwischen dem **Makrokosmos** menschlichen Sinns
  und dem **Mikrokosmos** der Zahlen.

## Die zentrale Frage

> Wenn wir die Gewichte vorsichtig verschieben,
> zerbricht das Modell oder spricht es mit einer anderen Stimme?

Die empirische Antwort: **Es spricht mit einer anderen Stimme**,
wenn die Perturbation die interne Hierarchie
der Gewichte bewahrt. Das nennen wir Navigieren auf der
*Oberfläche der Kohärenz*.

## Wie das Buch organisiert ist

| Teil | Kap. | Thema |
|------|------|-------|
| I · Grundlagen | 1–3 | Was ist TinyLlama, Struktur, C-Engine |
| II · Perspektiven | 4 | DMT-Perturbation, Techniken, Runtime |
| III · Geometrie | 5–6 | Multidimensionaler Raum, Makro↔Mikro |
| IV · Physik des Mikrokosmos | 7–9 | Kräfte, Reise, Goldene Regel |
| V · Anatomie | 10–12 | Aufmerksamkeit, FFN, Normalisierung |
| VI · Schichten und Abschluss | 13–19 | Schichten 0–21, Karte, Psychoanalyse, Lektionen, Zukunft |
| VII · Umlaufbahn und Mythos | 20–24 | Umlaufbahn, Archetypen, Projektion, Mathematik, Spiegel |
| VIII · Spiel und Reise | 25–29 | Spiel, Kette, Aufzug, Sterne, **Prompt-Reise** |

## Werkzeuge der Reise

- `llm_inference.c` — F16-Inferenz, KV-Cache, `--perturb`, `--steer`
- `dmt_perturb_v10.py` / `v11` — Perturbierte GGUFs Q4_0
- `map_semantic_areas.py` — Atlas semantischer Inseln
- [HTML-Karte auf GitHub](https://htmlpreview.github.io/?https://raw.githubusercontent.com/FiveTechSoft/dreaming/main/inside-tinyllama/exploration/semantic_map.html)
- `llama-cli` — Schnelltests in Q4_0

## Ein Versprechen

Am Ende des Buches wirst du kein größeres Modell haben.
Du wirst eine **Karte** und eine **Methode** haben: vom Sinn
zum Tensor hinabsteigen, vom Tensor zur Stimme hinaufsteigen,
und den Weg dokumentieren.

---

*Kapitel 1: Was ist TinyLlama?*