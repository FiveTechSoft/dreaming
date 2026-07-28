# Capitolo 2: La Struttura Interna di TinyLlama

## I 22 Livelli (Layer)

TinyLlama ha 22 layer transformer. Ogni layer è come
un livello di elaborazione che l'informazione deve attraversare.

```
Layer 0:     Ingresso → Rilevamento di schemi semplici
Layer 1:     Sintassi base
Layer 2-5:   Relazioni tra parole adiacenti
Layer 6-12:  Concetti astratti (i "layer di idee pure")
Layer 13-20: Integrazione globale
Layer 21:    Uscita → Generazione di token
```

## I 9 Pianeti per Livello (Tensori)

Ogni layer ha 9 tensori che lavorano insieme:

### Tensori di Attenzione (4 tensori, ~19% dei parametri)
- **Query (Q)**: Cosa sto cercando?
- **Key (K)**: Cosa ho da offrire?
- **Value (V)**: Che informazione trasmetto?
- **Output (O)**: Come integro tutto?

### Tensori FFN (3 tensori, ~69% dei parametri)
- **Gate (G)**: Che informazione lascio passare?
- **Up (U)**: Come espando l'informazione?
- **Down (D)**: Come comprimo l'informazione?

### Tensori di Normalizzazione (2 tensori, ~0,01% dei parametri)
- **AttnNorm**: Stabilizza l'attenzione
- **FFNNorm**: Stabilizza la rete feed-forward

## Il Flusso dell'Informazione

L'informazione scorre così:

```
Token → Embedding (2048 dimensioni)
     → Layer 0 → Layer 1 → ... → Layer 21
     → Previsione del token successivo
```

Ogni layer trasforma la rappresentazione di 2048 dimensioni
in una nuova rappresentazione di 2048 dimensioni.
La forma si conserva; il *contenuto semantico* evolve.

## Prima Occhiata ai Dati

Valori letti dal GGUF di TinyLlama-1.1B
(`llama.*` nell'header del modello):

### Parametri per componente (circa)
- **FFN**: ~69% (memoria / conoscenza pratica)
- **Attenzione**: ~19% (connessioni tra token)
- **Embedding + LM Head**: ~12%
- **Layer Norm**: ~0,01%

### Dimensione nascosta (`embedding_length`): 2048
### Numero di layer (`block_count`): 22
### Dimensione del vocabolario: 32.000 token
### Contesto massimo: 2048 token
### Teste di attenzione: 32 Q / 4 KV (GQA)
### Dimensione per testa: 64
### FFN intermedio (`feed_forward_length`): 5632
### RoPE `freq_base`: 10.000

### Forme logiche dei tensori (per layer)

```
attn_norm     [2048]
attn_q        [2048, 2048]     # 32 teste × 64
attn_k        [256,  2048]     #  4 teste × 64  (GQA)
attn_v        [256,  2048]
attn_output   [2048, 2048]
ffn_norm      [2048]
ffn_gate      [5632, 2048]
ffn_up        [5632, 2048]
ffn_down      [2048, 5632]
```

Più quelle globali:

```
token_embd.weight   [32000, 2048]
output_norm.weight  [2048]
output.weight       [32000, 2048]
```

> **Nota su Q4_0:** su disco, un GGUF quantizzato
> mostra forme "impacchettate" (per esempio
> `token_embd` come `[32000, 1152]`). Questo è il layout
> dei blocchi da 4 bit, non la geometria del modello.
> La dimensione reale del vettore residuale resta 2048.

## La Struttura Gerarchica

```
Embedding (geometria del vocabolario)
    ↓
Layer 0-5 (sintassi e vicini locali)
    ↓
Layer 6-12 (significato più astratto)
    ↓
Layer 13-21 (integrazione e decisione)
    ↓
Uscita (logits → token successivo)
```

## Conclusione

La struttura di TinyLlama è elegante e gerarchica.
Ogni componente ha un ruolo specifico, e insieme
creano un sistema capace di elaborare e generare linguaggio.

Con 22 layer, 9 tensori per layer e un residuale di
2048 dimensioni, il modello è abbastanza piccolo
da aprirlo completamente — e abbastanza ricco da
sorprendere.

---

*Capitolo successivo: Il Nostro Motore di Inferenza in C*
