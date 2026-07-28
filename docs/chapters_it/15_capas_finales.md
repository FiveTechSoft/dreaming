# Capitolo 15: Gli Ultimi Layer (13–21)

## Integrazione e decisione

```
Layer 13–20: integrazione globale
Layer 21:    ultima trasformazione prima di output_norm
poi:         lm_head → logits → sample
```

Qui il residuale si prepara al **collasso**
nel vocabolario: la forza VI dell'atlante (softmax).

## Cosa si gioca alla fine

- Miscela di temi armati nel mezzo.  
- Preferenze fini di stile (formale vs semplice).  
- Prossimità a token di chiusura (`</s>`) — per questo
  a volte baseline e mystical coincidono in uscite
  **molto corte** con lo stesso seed (stessa fossa di EOS).

## Esperimento della batteria mistica

Con I=0,50 e 60 token max, vari prompt hanno riempito
il budget di lunghezza; altri hanno tagliato a 2–8 token.
I layer finali + sample decidono **quando fermarsi**
tanto quanto **cosa dire**.

## Regola pratica

Per confrontare prospettive, usa `n` alto e guarda
il **corpo** del testo, non solo la prima frase
se il modello si affretta all'EOS.

---

*Capitolo successivo: Aree Semantiche e la Mappa*
