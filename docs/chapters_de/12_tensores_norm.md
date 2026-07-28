# Kapitel 12: Die Normalisierungstensoren

## Zwei Bremsen pro Schicht (+ eine abschließende)

| Tensor | Wo | Funktion |
|--------|-----|----------|
| **attn_norm** | Vor QKV | Stabilisiert Eingang zur Aufmerksamkeit |
| **ffn_norm** | Vor gate/up | Stabilisiert Eingang zum FFN |
| **output_norm** | Nach Schicht 21 | Stabilisiert vor dem lm_head |

## RMSNorm (kein klassisches LayerNorm)

```
rms = sqrt(mean(x²) + ε)
out = (x / rms) * w
```

Ohne Mittelwert abzuziehen: skaliert nur nach Energie des Vektors.

## Minimale Masse, volle Wirkung

Etwa **0.01%** der Parameter. Ohne diese Normen
schieben Aufmerksamkeit + FFN den Residual in explosive
Normen oder numerischen Kollaps.

Im Kraftatlas: **kosmologische Konstante /
atembare Luft** des Mikrokosmos.

## Perturbationspolitik

`dmt_perturb_v10` und die C-Engine **berühren Normen nicht**,
wenn sie mystical anwenden: Die Stabilität zu verschieben ist
der kürzeste Weg zum numerischen Müll.

## Praktische Regel

Wenn der Text nach einem Experiment mit seltsamen Symbolen
zerbricht, prüfe zuerst, ob du Normen oder übermäßiges I
berührt hast, bevor du die „Semantik" beschuldigst.

---

*Nächstes Kapitel: Die frühen Schichten (0–5)*