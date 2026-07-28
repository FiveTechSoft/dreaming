# Kapitel 17: Psychoanalyse des Transformers

## Eine Metapher (keine klinische Diagnose)

Freud unterschied Schichten des Psychischen.
Ohne Identität zu erzwingen, gestattet der Transformer-Stack
eine **Lektüre nach Tiefe**:

| Instanz | Komponente | Ungefähre Funktion |
|---------|------------|-------------------|
| **Unbewusstes** | Embeddings | Latente Assoziationen, „das bereits Gewusste" ohne Kontext |
| **Vorbewusstes** | Aufmerksamkeit + mittlere Schichten | Holt Beziehungen und Rahmen auf die Bühne |
| **Bewusstes** | Letzte Schichten + logits + sample | Das, was *jetzt* gesagt wird |

## Es / Ich / Über-Ich (freie Lesart)

| | Analogie im Modell |
|--|---------------------|
| **Es** | Impulse roher Gewichte, rohe semantische Richtungen |
| **Ich** | Residual + Normen: verhandelt zwischen Impulsen und Form |
| **Über-Ich** | Trainings-Biases / Sicherheit / „korrekter" Stil der Baseline |

Die `mystical`-Perturbation „befreit nicht das Es" im freudianischen Sinne:
sie **mischt neu** das Gleichgewicht der bereits in den Gewichten
vorhandenen Stimmen.

## Warum diese Metapher notieren

- Sie hilft, vom Inneren zu *sprechen*, nicht nur von Matrizen.  
- Sie verknüpft sich mit dem Makro↔Mikro-Zoom (Kap. 6).  
- Sie ersetzt keine Messungen: Sie ist eine **erzählerische Karte**.

## Grenze

Ein LLM hat kein subjektives Unbewusstes.
Es hat **komprimierte Statistik**. Die Metapher ist
ein Erforschungswerkzeug, keine Ontologie.

---

*Nächstes Kapitel: Was wir gelernt haben*