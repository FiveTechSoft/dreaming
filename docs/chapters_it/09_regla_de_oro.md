# Capitolo 9: La Regola d'Oro Geometrica

## La scoperta

Modificare **componenti diverse** del transformer
non produce rumore generico. Produce **prospettive
specifiche e prevedibili**.

| Componente | "Pianeta" | Prospettiva emergente |
|------------|-----------|----------------------|
| **Attenzione** (Q, K, V, O) | Struttura / relazioni | Accademica, critica, formale |
| **FFN** (gate, up, down) | Vocabolario / azione | Pratica, liste, consigli |
| **Embedding** | Identità d'ingresso | Linguaggio semplice e diretto |

Chiamiamo questo la **Regola d'Oro geometrica**.

## Attenzione → accademico

I tensori di attenzione connettono i token.
Perturbandoli, il modello dà priorità alla **struttura**:
argomenti, riferimenti, tono formale.

```
Prompt: "The meaning of life is..."
Baseline:  "...finding happiness..."
Attn perturbata: "...a fundamental philosophical inquiry
                   debated by scholars for millennia..."
```

## FFN → pratico

Il FFN trasforma ogni posizione (memoria pratica,
~69% dei parametri). Tocandolo, emergono **verbi
d'azione** e passi concreti.

```
FFN perturbata: "To find meaning: 1) Identify values,
                 2) Set goals, 3) Take daily action..."
```

## Embedding → semplice

La matrice d'ingresso definisce la "mappa di nascita"
di ogni token. Perturbarla appiattisce il registro:

```
Emb perturbati: "Life means living. Be happy. Help others."
```

## Perché è "geometrica"

Ogni famiglia di tensori muove il residuale in
**direzioni diverse** dello spazio di rappresentazione.
Non è magia di nomi di file: è che attenzione
e FFN implementano operatori diversi sullo stesso ℝ²⁰⁴⁸.

Selective targeting (v11) lo conferma:

| Targeting | Effetto cercato |
|-----------|----------------|
| `attention_alter` | Amplifica forte in attn, dolce in FFN |
| `ffn_dream` | Creative forte in FFN, dolce in attn |
| `embedding_shift` | Cambio in emb, resto dolce |

## Verifica empirica (riassunto)

- 25 modelli, 240 generazioni, 10 prompt (batteria Dreaming).  
- Tecniche che preservano gerarchia → coerenza.  
- Tecniche che la rompono (rumore alto, nibble flip) → rifiuti.  
- Runtime C: `mystical` su attn+FFN (non emb/norm) si allinea
  con la politica di `dmt_perturb_v10`.

## Come usarla nel viaggio

1. Vuoi analisi? → guarda / tocca **attenzione**.  
2. Vuoi checklist? → guarda / tocca **FFN**.  
3. Vuoi prosa semplice? → guarda / tocca **embedding**.  
4. Vuoi clima esistenziale globale? → `mystical` nei layer.

La Regola d'Oro è il **ponte di scale**:
dalla vite dell'orologio al clima del monologo.

---

*Capitolo successivo: I Tensori di Attenzione*
