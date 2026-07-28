# Capitolo 7: Le Forze Gravitazionali del Microcosmo

## Non c'è una sola gravità

Nell'universo TinyLlama la "gravità" è un insieme
di **campi** che piegano le traiettorie di significato.
Ognuno ha massa (parametri), portata ed effetto
nel testo.

## Inventario delle forze

| # | Forza | Supporto | Massa | Portata |
|---|-------|----------|-------|---------|
| I | Attrazione atencionale | Q·K/√d → V | ~19% | Tra token della sequenza |
| II | Potenziale FFN | SwiGLU gate/up/down | ~69% | Per token (locale) |
| III | Inerzia residuale | x ← x + f(x) | struttura | 22 layer |
| IV | Ancora di embedding | token_embd, output | ~12% | Condizione iniziale |
| V | Stabilizzazione | RMSNorm | ~0,01% | Anti-esplosione |
| VI | Collasso al vocabolario | logits → softmax | testa | 1 di 32k token |
| VII | Prospettive | perturbazione dei pesi | tutto il modello | Cambia il "clima" |
| VIII | Isole semantiche | geometria degli embedding | — | Attrattori statici |

### Masse misurate (F16 logico)

| Componente | Parametri | Quota |
|------------|-----------|-------|
| FFN | ~761M | **69,2%** |
| Attenzione | ~208M | **18,9%** |
| Emb + lm_head | ~131M | **11,9%** |
| Norme | ~92k | **0,01%** |

## Forza I — Attenzione

Non locale: un token percepisce altri del passato (maschera causale).
GQA 32 Q / 4 KV: gravità economica da memorizzare.

**Regola d'Oro:** perturbare attenzione → lente **accademica / relazionale**.

## Forza II — FFN

Il "sole" del sistema di pesi. Trasforma ogni posizione
senza guardare i vicini: clima locale del residuale.

**Regola d'Oro:** perturbare FFN → lente **pratica / azione**.

## Forza III — Residuale

Conservazione del momento del significato. Per questo i
passi tangenti (`amplify_subspace`) mantengono coerenza
e il rumore normale alla superficie la distrugge.

## Forza IV e V — Nascita e aria

Gli embedding fissano il punto di partenza in ℝ²⁰⁴⁸
(norma media ≈ 0,68, quasi isotropo).
RMSNorm rende abitabili i 22 layer con massa minima.

## Forza VI — Softmax

Collasso del continuo all'evento: un token.
Temperatura e top-k sono la "durezza" della fossa.

## Forza VII — Prospettive

Superficie di coerenza in ℝ~1,1e9.
`mystical` = corrente tangente; `noise` forte = uscita nel vuoto.

## Forza VIII — Costellazioni

Baricentri di aree (emotion, spirit, matter, mind…):
quasi ortogonali tra le isole. Attrazione relativa
abstract↔mind (+0,13); time↔social (−0,09).
Love/hate non sono antipodali: cos ≈ 0.

## Tre leggi

1. **Superficie** — solo traiettorie tangenti nei pesi → testo coerente.  
2. **Due materie** — attenzione struttura le relazioni; FFN trasforma il contenuto.  
3. **Collasso** — tutto finisce in un token.

## Gerarchia di dominanza

```
softmax (destino)
    ↑
attenzione (lungo raggio)  +  FFN (massa)
    ↑
residuale (inerzia)
    ↑
embedding (inizio)  +  norm (stabilità)
    ↑
pesi / prospettiva (metrica dell'universo)
    ↑
isole semantiche (cielo d'ingresso)
```

---

*Capitolo successivo: Come Viaggiare nell'Universo TinyLlama*
