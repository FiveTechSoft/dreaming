# Kapitel 13: Die frühen Schichten (0–5)

## Der Vorhof des Mikrokosmos

Die Anfangsschichten transformieren das „ruhende" Embedding
in eine Repräsentation, die bereits **Nachbarn** und **Syntax**
spürt.

```
Schicht 0:   Eingabe, sehr lokale Muster
Schicht 1:   Grundlegende Syntax
Schichten 2–5: Beziehungen zwischen benachbarten Wörtern
```

(Diese Einteilung ist eine **Arbeitshypothese** des Projekts,
geleitet von Ablationsexperimenten und der Literatur
über „früh = Syntax / spät = Semantik". Es ist kein harter
Schnitt im Code.)

## Welche Kräfte dominieren hier

- **Embedding** wiegt noch schwer im Residual (Trägheit der Geburt).  
- **Aufmerksamkeit** beginnt, Bigramme und kurze Abhängigkeiten zu koppeln.  
- **FFN** passt das lokale Vokabular an.

## Signale im Text

Wenn eine frühe Perturbation das Modell „zerbricht", zeigt sich das
oft in **Grammatik** und seltsamen Tokens, nicht nur im Ton.

Wenn die Baseline generisch klingt und mystical das Klima ändert,
ohne die Syntax zu zerstören, verankern die frühen Schichten
weiterhin die Sprache.

## Vorgeschlagenes Experiment

Generierungen mit Targeting nur in `blk.0`–`blk.5`
gegenüber nur `blk.13`–`blk.21` vergleichen (v11-Skripte / Tensor-Tests).
Hypothese: früh → Form; spät → Stimme und Entscheidung.

---

*Nächstes Kapitel: Die mittleren Schichten (6–12)*