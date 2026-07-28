# Kapitel 9: Die geometrische Goldene Regel

## Die Entdeckung

Das Modifizieren **unterschiedlicher Komponenten** des Transformers
erzeugt kein generisches Rauschen. Es erzeugt **spezifische
und vorhersehbare Perspektiven**.

| Komponente | „Planet" | Emergierende Perspektive |
|------------|----------|--------------------------|
| **Aufmerksamkeit** (Q, K, V, O) | Struktur / Beziehungen | Akademisch, kritisch, formal |
| **FFN** (gate, up, down) | Vokabular / Handlung | Praktisch, Listen, Ratschläge |
| **Embeddings** | Eingangs-Identität | Einfache und direkte Sprache |

Wir nennen dies die **geometrische Goldene Regel**.

## Aufmerksamkeit → akademisch

Die Aufmerksamkeitstensoren verbinden Tokens.
Wenn man sie perturbiert, priorisiert das Modell **Struktur**:
Argumente, Referenzen, formaler Ton.

```
Prompt: "The meaning of life is..."
Baseline:  "...finding happiness..."
Gestörte Attn: "...a fundamental philosophical inquiry
                 debated by scholars for millennia..."
```

## FFN → praktisch

Das FFN transformiert jede Position (praktische Erinnerung,
~69% der Parameter). Wenn man es berührt, entstehen **Verben
der Handlung** und konkrete Schritte.

```
Gestörtes FFN: "To find meaning: 1) Identify values,
                2) Set goals, 3) Take daily action..."
```

## Embeddings → einfach

Die Eingabematrix definiert die „Geburtskarte"
jedes Tokens. Sie zu perturbieren glättet das Register:

```
Gestörte Emb: "Life means living. Be happy. Help others."
```

## Warum es „geometrisch" ist

Jede Tensorenfamilie bewegt den Residual in
**unterschiedlichen Richtungen** des Repräsentationsraums.
Es ist keine Magie von Dateinamen: Es ist, dass Aufmerksamkeit
und FFN unterschiedliche Operatoren über demselben ℝ²⁰⁴⁸ implementieren.

Selective Targeting (v11) bestätigt das:

| Targeting | Gesuchter Effekt |
|-----------|------------------|
| `attention_alter` | Starke Verstärkung bei attn, sanft bei FFN |
| `ffn_dream` | Starke Kreativität bei FFN, sanft bei attn |
| `embedding_shift` | Änderung bei emb, Rest sanft |

## Empirische Überprüfung (Zusammenfassung)

- 24 Modelle, 240 Generierungen, 10 Prompts (Dreaming-Batterie).  
- Techniken, die Hierarchie bewahren → Kohärenz.  
- Techniken, die sie brechen (hohes noise, nibble flip) → Müll.  
- C-Laufzeit: `mystical` über attn+FFN (nicht emb/norm) passt
  zur Politik von `dmt_perturb_v10`.

## Wie man sie beim Reisen nutzt

1. Du willst Analyse? → Schau / berühre **Aufmerksamkeit**.  
2. Du willst Checklisten? → Schau / berühre **FFN**.  
3. Du willst schlichte Prosa? → Schau / berühre **Embeddings**.  
4. Du willst globales existenzielles Klima? → `mystical` in Schichten.

Die Goldene Regel ist die **Maßstab-Brücke**:
von der Uhrschraube zum Klima des Monologs.

---

*Nächstes Kapitel: Die Aufmerksamkeitstensoren*