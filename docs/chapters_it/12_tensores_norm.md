# Capitolo 12: I Tensori di Normalizzazione

## Due freni per layer (+ uno finale)

| Tensore | Dove | Funzione |
|---------|------|----------|
| **attn_norm** | Prima di QKV | Stabilizza l'ingresso all'attenzione |
| **ffn_norm** | Prima di gate/up | Stabilizza l'ingresso al FFN |
| **output_norm** | Dopo il layer 21 | Stabilizza prima di lm_head |

## RMSNorm (non LayerNorm classico)

```
rms = sqrt(mean(x²) + ε)
out = (x / rms) * w
```

Senza sottrarre la media: scala solo per energia del vettore.

## Massa minima, effetto totale

~**0,01%** dei parametri. Senza queste norme,
attenzione + FFN spingono il residuale a norme
esplosive o a collasso numerico.

Nell'atlante delle forze: **costante cosmologica /
aria respirabile** del microcosmo.

## Politica di perturbazione

`dmt_perturb_v10` e il motore C **non toccano** le norme
quando applicano mystical: muovere la stabilità è la
strada più breve ai rifiuti numerici.

## Regola pratica

Se il testo si rompe con simboli strani dopo un
esperimento, verifica se hai toccato norme o I eccessiva
prima di incolpare la "semantica".

---

*Capitolo successivo: I Primi Layer (0–5)*
