# Kapitel 10: Die Aufmerksamkeitstensoren

## Vier Planeten pro Schicht

In jeder der 22 Schichten:

| Tensor | Frage | Logische Form (TinyLlama) |
|--------|-------|---------------------------|
| **Q** (attn_q) | Wonach suche ich? | [2048, 2048] |
| **K** (attn_k) | Was biete ich an? | [256, 2048] (4×64) |
| **V** (attn_v) | Was übertrage ich? | [256, 2048] |
| **O** (attn_output) | Wie integriere ich? | [2048, 2048] |

Plus `attn_norm` (RMSNorm vor dem Block).

## GQA: 32 Augen, 4 Erinnerungen

TinyLlama hat nicht 32 unabhängige K und 32 V.
Es hat **32 Q-Köpfe** und **4 gemeinsame KV**
(jeder KV bedient 8 Q). Weniger Cache-Speicher,
dasselbe Multi-Head-Konzept.

## Die Formel

```
scores = (Q Kᵀ) / √64
weights = softmax_causal(scores)
out = weights V
out = O · out
x = x + out          # residual
```

In der C-Engine: Nur das neue Token berechnet Q/K/V;
K und V werden im **KV-Cache** gespeichert.

## Rolle im Universum

- **Kraft der Fernwirkung** zwischen Tokens.  
- **Goldene Regel:** Aufmerksamkeit berühren → akademische Perspektive.  
- Im Kraftatlas: ~19% der Masse, maximale *Reichweite*.

## Was beim Experimentieren zu beobachten ist

- Zitiert, strukturiert, „argumentiert" der Text?  
- Ändert sich mehr die *Beziehung* zwischen Ideen als das einzelne Vokabular?  
→ Signal, dass das Aufmerksamkeitsfeld das Klima dominiert.

---

*Nächstes Kapitel: Die FFN-Tensoren*