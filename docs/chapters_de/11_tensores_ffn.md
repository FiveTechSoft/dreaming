# Kapitel 11: Die FFN-Tensoren

## Drei Planeten der gewöhnlichen Materie

| Tensor | Rolle | Logische Form |
|--------|-------|---------------|
| **Gate** (ffn_gate) | SiLU-Gatter | [5632, 2048] |
| **Up** (ffn_up) | Erweiterung | [5632, 2048] |
| **Down** (ffn_down) | Kompression | [2048, 5632] |

Plus `ffn_norm` vor dem Block.

## SwiGLU

```
h' = Down( SiLU(Gate(x)) ⊙ Up(x) )
x  = x + h'
```

Zwischendimension **5632**: Der Residual wird in
einen breiteren Raum erweitert und kehrt zu 2048 zurück.

## Dominierende Masse

Etwa **69%** der Modellparameter leben hier.
Wenn die Aufmerksamkeit Schwerkraft zwischen Planeten ist, ist das FFN
die **innere Physik** jedes einzelnen.

## Goldene Regel

FFN perturbieren → **praktische** Perspektive:
Schritte, Ratschläge, Handlungsverben, „wie macht man das".

Selective `ffn_dream` (v11): Starke Kreativität im FFN,
sanft bei Aufmerksamkeit → „träumerisches aber umsetzbares" Klima.

## Was zu beobachten ist

- Numerierte Listen, Imperative, Tipps?  
- Weniger „wer mit wem interagiert" und mehr „was zu tun ist"?  
→ FFN-Feld auf dem Fahrersitz.

---

*Nächstes Kapitel: Die Normalisierungstensoren*