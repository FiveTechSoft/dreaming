# Capitolo 10: I Tensori di Attenzione

## Quattro pianeti per layer

In ognuno dei 22 layer:

| Tensore | Domanda | Forma logica (TinyLlama) |
|---------|---------|--------------------------|
| **Q** (attn_q) | Cosa cerco? | [2048, 2048] |
| **K** (attn_k) | Cosa offro? | [256, 2048] (4×64) |
| **V** (attn_v) | Cosa trasmetto? | [256, 2048] |
| **O** (attn_output) | Come integro? | [2048, 2048] |

Più `attn_norm` (RMSNorm prima del blocco).

## GQA: 32 occhi, 4 memorie

TinyLlama non ha 32 K e 32 V indipendenti.
Ha **32 teste Q** e **4 KV** condivise
(ogni KV serve 8 Q). Meno memoria di cache,
stessa idea multi-testa.

## La formula

```
scores = (Q Kᵀ) / √64
pesi   = softmax_causal(scores)
out    = pesi V
out    = O · out
x      = x + out          # residuale
```

Nel motore C: solo il token nuovo calcola Q/K/K;
K e V si salvano nella **KV-cache**.

## Ruolo nell'universo

- **Forza a lungo raggio** tra token.  
- **Regola d'Oro:** toccare attenzione → prospettiva accademica.  
- Nell'atlante delle forze: ~19% della massa, massima *portata*.

## Cosa osservare sperimentando

- Il testo cita, struttura, "argomenta"?  
- Cambia di più la *relazione* tra idee che il lessico libero?  
→ Segnale che il campo atencionale domina il clima.

---

*Capitolo successivo: I Tensori FFN*
