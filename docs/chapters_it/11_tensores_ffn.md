# Capitolo 11: I Tensori FFN

## Tre pianeti della materia ordinaria

| Tensore | Ruolo | Forma logica |
|---------|-------|--------------|
| **Gate** (ffn_gate) | Combinatore SiLU | [5632, 2048] |
| **Up** (ffn_up) | Espansione | [5632, 2048] |
| **Down** (ffn_down) | Compressione | [2048, 5632] |

Più `ffn_norm` prima del blocco.

## SwiGLU

```
h' = Down( SiLU(Gate(x)) ⊙ Up(x) )
x  = x + h'
```

Dimensione intermedia **5632**: il residuale si espande
in uno spazio più ampio e torna a 2048.

## Massa dominante

~**69%** dei parametri del modello vivono qui.
Se l'attenzione è gravità tra pianeti, il FFN è
la **fisica interna** di ognuno.

## Regola d'Oro

Perturbare FFN → prospettiva **pratica**:
passi, consigli, verbi d'azione, "come fare".

Selective `ffn_dream` (v11): creative forte in FFN,
dolce in attenzione → clima "sognatore ma azionabile".

## Cosa osservare

- Liste numerate, imperativi, suggerimenti?  
- Meno "chi si relaziona con chi" e più "cosa fare"?  
→ Campo FFN sul sedile del guidatore.

---

*Capitolo successivo: I Tensori di Normalizzazione*
