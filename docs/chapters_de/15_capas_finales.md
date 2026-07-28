# Kapitel 15: Die letzten Schichten (13–21)

## Integration und Entscheidung

```
Schichten 13–20: globale Integration
Schicht 21:     letzte Transformation vor output_norm
danach:         lm_head → logits → sample
```

Hier bereitet sich der Residual auf den **Kollaps**
zum Vokabular vor: Kraft VI des Atlas (softmax).

## Was am Ende auf dem Spiel steht

- Mischung der in der Mitte aufgebauten Themen.  
- Feine Stilpräferenzen (formal vs. schlicht).  
- Nähe zu Tokens für Abschluss (`</s>`) — deshalb
  stimmen Baseline und mystical manchmal bei **sehr
  kurzen** Ausgaben mit demselben Seed überein (gleiche EOS-Grube).

## Experiment der mystischen Testreihe

Mit I=0.50 und 60 maximalen Tokens füllten mehrere Prompts
das Längenbudget; andere brachen bei 2–8 Tokens ab.
Die letzten Schichten + Sampling entscheiden **wann aufhören**
genauso wie **was sagen**.

## Praktische Regel

Um Perspektiven zu vergleiche, verwende hohes `n` und betrachte
den **Körper** des Textes, nicht nur den ersten Satz,
wenn das Modell sich hetzt zum EOS.

---

*Nächstes Kapitel: Semantische Bereiche und die Karte*