# Kapitel 7: Die Schwerkraftkräfte des Mikrokosmos

## Es gibt nicht nur eine Schwerkraft

Im TinyLlama-Universum ist die „Schwerkraft" ein Set von
**Feldern**, die Bedeutungs-Trajektorien verbiegen.
Jedes hat Masse (Parameter), Reichweite und Wirkung
im Text.

## Kraftinventar

| # | Kraft | Träger | Masse | Reichweite |
|---|-------|--------|-------|------------|
| I | Aufmerksamkeitsanziehung | Q·K/√d → V | ~19% | Zwischen Tokens der Sequenz |
| II | FFN-Potenzial | SwiGLU gate/up/down | ~69% | Pro Token (lokal) |
| III | Residual-Trägheit | x ← x + f(x) | Struktur | 22 Schichten |
| IV | Embedding-Anker | token_embd, output | ~12% | Anfangsbedingung |
| V | Stabilisierung | RMSNorm | ~0.01% | Anti-Explosion |
| VI | Kollaps zum Vokabular | logits → softmax | Kopf | 1 von 32k Tokens |
| VII | Perspektiven | Gewichtsperturbation | gesamtes Modell | Ändert das „Klima" |
| VIII | Semantische Inseln | Embedding-Geometrie | — | Statische Attraktoren |

### Gemessene Massen (logisches F16)

| Komponente | Parameter | Anteil |
|------------|-----------|--------|
| FFN | ~761M | **69.2%** |
| Aufmerksamkeit | ~208M | **18.9%** |
| Emb + lm_head | ~131M | **11.9%** |
| Normen | ~92k | **0.01%** |

## Kraft I — Aufmerksamkeit

Nicht lokal: Ein Token spürt andere der Vergangenheit (kausale Maske).
GQA 32 Q / 4 KV: Schwerkraft, die wenig Speicher kostet.

**Goldene Regel:** Aufmerksamkeit perturbieren → Linse **akademisch / relational**.

## Kraft II — FFN

Die „Sonne" des Gewichtssystems. Transformiert jede Position
ohne Nachbarn zu betrachten: lokales Klima des Residuals.

**Goldene Regel:** FFN perturbieren → Linse **praktisch / Handlung**.

## Kraft III — Residual

Erhaltung des Bedeutungs-Impulses. Deshalb halten tangentiale
Schritte (`amplify_subspace`) die Kohärenz und normales
Rauschen an der Oberfläche zerstört sie.

## Kraft IV und V — Geburt und Luft

Embeddings fixieren den Ausgangspunkt in ℝ²⁰⁴⁸
(mittlere Norm ≈ 0.68, fast isotrop).
RMSNorm macht die 22 Schichten mit minimaler Masse bewohnbar.

## Kraft VI — Softmax

Kollaps des Kontinuierlichen zum Ereignis: ein Token.
Temperatur und top-k sind die „Härte" der Grube.

## Kraft VII — Perspektiven

Kohärenz-Oberfläche in ℝ~¹·¹ᵉ⁹.
`mystical` = Tangentialstrom; starkes `noise` = Austritt ins Leere.

## Kraft VIII — Sternbilder

Zentroide der Bereiche (emotion, spirit, matter, mind…):
fast orthogonal zwischen Inseln. Relative Anziehung
abstract↔mind (+0.13); time↔social (−0.09).
Love/hate sind nicht antipodal: cos ≈ 0.

## Drei Gesetze

1. **Oberfläche** — nur tangentiale Trajektorien in Gewichten → kohärenter Text.  
2. **Zwei Materien** — Aufmerksamkeit strukturiert Beziehungen; FFN transformiert Inhalt.  
3. **Kollaps** — alles endet in einem Token.

## Dominanzhierarchie

```
softmax (Schicksal)
    ↑
Aufmerksamkeit (Fernwirkung)  +  FFN (Masse)
    ↑
Residual (Trägheit)
    ↑
Embedding (Anfang)  +  Norm (Stabilität)
    ↑
Gewichte / Perspektive (Metrik des Universums)
    ↑
Semantische Inseln (Eingangshimmel)
```

---

*Nächstes Kapitel: Wie man durch das TinyLlama-Universum reist*