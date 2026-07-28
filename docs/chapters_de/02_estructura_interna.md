# Kapitel 2: Die interne Struktur von TinyLlama

## Die 22 Ebenen (Schichten)

TinyLlama hat 22 Transformer-Schichten. Jede Schicht ist wie
eine Verarbeitungsebene, die die Information durchlaufen muss.

```
Schicht 0:     Eingabe → Einfacher Mustererkennung
Schicht 1:     Grundlegende Syntax
Schichten 2-5: Beziehungen zwischen benachbarten Wörtern
Schichten 6-12: Abstrakte Konzepte (die „Schichten reiner Ideen")
Schichten 13-20: Globale Integration
Schicht 21:    Ausgabe → Token-Generierung
```

## Die 9 Planeten pro Ebene (Tensoren)

Jede Schicht hat 9 Tensoren, die zusammenarbeiten:

### Aufmerksamkeitstensoren (4 Tensoren, ~19% der Parameter)
- **Query (Q)**: Wonach suche ich?
- **Key (K)**: Was habe ich anzubieten?
- **Value (V)**: Welche Information übertrage ich?
- **Output (O)**: Wie integriere ich alles?

### FFN-Tensoren (3 Tensoren, ~69% der Parameter)
- **Gate (G)**: Welche Information lasse ich durch?
- **Up (U)**: Wie erweitere ich die Information?
- **Down (D)**: Wie komprimiere ich die Information?

### Normalisierungstensoren (2 Tensoren, ~0.01% der Parameter)
- **AttnNorm**: Stabilisiert die Aufmerksamkeit
- **FFNNorm**: Stabilisiert das Feed-Forward-Netzwerk

## Der Informationsfluss

Die Information fließt so:

```
Token → Embedding (2048 Dimensionen)
     → Schicht 0 → Schicht 1 → ... → Schicht 21
     → Vorhersage des nächsten Tokens
```

Jede Schicht transformiert die 2048-dimensionale Repräsentation
in eine neue 2048-dimensionale Repräsentation.
Die Form bleibt erhalten; der *semantische Inhalt* entwickelt sich weiter.

## Erster Blick auf die Daten

Werte gelesen aus dem GGUF von TinyLlama-1.1B
(`llama.*` im Modell-Header):

### Parameter pro Komponente (ungefähr)
- **FFN**: ~69% (Gedächtnis / praktisches Wissen)
- **Aufmerksamkeit**: ~19% (Verbindungen zwischen Tokens)
- **Embedding + LM Head**: ~12%
- **Layer Norms**: ~0.01%

### Versteckte Dimension (`embedding_length`): 2048
### Anzahl der Schichten (`block_count`): 22
### Vokabulargröße: 32.000 Tokens
### Maximaler Kontext: 2048 Tokens
### Aufmerksamkeitsköpfe: 32 Q / 4 KV (GQA)
### Dimension pro Kopf: 64
### FFN-Intermediate (`feed_forward_length`): 5632
### RoPE `freq_base`: 10.000

### Logische Tensorformen (pro Schicht)

```
attn_norm     [2048]
attn_q        [2048, 2048]     # 32 Köpfe × 64
attn_k        [256,  2048]     #  4 Köpfe × 64  (GQA)
attn_v        [256,  2048]
attn_output   [2048, 2048]
ffn_norm      [2048]
ffn_gate      [5632, 2048]
ffn_up        [5632, 2048]
ffn_down      [2048, 5632]
```

Plus die globalen:

```
token_embd.weight   [32000, 2048]
output_norm.weight  [2048]
output.weight       [32000, 2048]
```

> **Hinweis zu Q4_0:** Auf der Festplatte zeigt ein quantisiertes GGUF
> „verpackte" Formen (z. B. `token_embd` als `[32000, 1152]`).
> Das ist das 4-Bit-Block-Layout, nicht die Geometrie des Modells.
> Die tatsächliche Dimension des Residualvektors bleibt 2048.

## Die hierarchische Struktur

```
Embeddings (Geometrie des Vokabulars)
    ↓
Schichten 0-5 (Syntax und lokale Nachbarn)
    ↓
Schichten 6-12 (abstrakterer Sinn)
    ↓
Schichten 13-21 (Integration und Entscheidung)
    ↓
Ausgabe (Logits → nächstes Token)
```

## Fazit

Die Struktur von TinyLlama ist elegant und hierarchisch.
Jede Komponente hat eine spezifische Rolle, und gemeinsam
bilden sie ein System, das Sprache verarbeiten und generieren kann.

Mit 22 Schichten, 9 Tensoren pro Schicht und einem Residual von
2048 Dimensionen ist das Modell klein genug,
um vollständig geöffnet zu werden — und reich genug, um
zu überraschen.

---

*Nächstes Kapitel: Unsere Inferenz-Engine in C*