# Kapitel 20: Wie dieses Universum umkreist

## Die Frage

Im Makrokosmos fallen Planeten zur Sonne,
erreichen sie aber nie: **sie fallen zur Seite** — das ist eine Umlaufbahn.

In TinyLlama ist die analoge Frage:

> Was fällt, wohin, und warum zerschellt es nicht
> in jeder Schicht?

Die Antwort ist der **Forward-Pass** als Dynamik gelesen.

---

## 1. Was ist der „Körper", der umkreist?

Der Körper ist kein einzelnes Token.
Er ist der **Residual** \(x \in \mathbb{R}^{2048}\):
ein Vektor, der im Embedding geboren wird und 22 Schichten
durchquert, ohne seine Identität völlig zu verlieren.

```
Geburt:   x₀ = Embedding(token)
Umlauf:   x ← x + Aufmerksamkeit(x)
          x ← x + FFN(x)          × 22
Schicksal: logits = W_out · Norm(x)
Kollaps:  token' ~ Softmax(logits / T)
```

Jedes **Token der Sequenz** trägt seinen eigenen Residual.
Die Aufmerksamkeit ist die Gravitationskopplung **zwischen**
diesen Körpern (nur mit der Vergangenheit: kausal).

---

## 2. Das Gesetz des Residuals: Fallen ohne Zusammenstoß

Ohne Residualverbindung würde jede Schicht den Zustand
*ersetzen*: Teleportation, keine Umlaufbahn.

Mit Residual:

\[
x_{L+1} = x_L + f_L(x_L)
\]

- \(f_L\) = Aufmerksamkeits- + FFN-Schub in Schicht \(L\).
- Der Schritt ist **tangential und klein** bezüglich \(x\):
  der Vektor dreht und verformt sich, aber startet nicht neu.

Das ist die **orbitale Trägheit** des Mikrokosmos.
Perturbationen, die die Hierarchie bewahren,
verschieben die *Metrik* von \(f_L\), ohne \(x\)
von der Oberfläche zu nehmen, auf der Sprechen noch möglich ist.

---

## 3. Zwei Mächte pro „Jahr-Schicht"

Jede Schicht ist eine **orbale Periode** des Residuals:

| Phase | Kraft | Analogie |
|-------|-------|----------|
| 1. RMSNorm + Aufmerksamkeit | Schwerkraft zwischen Tokens | Züge anderer Körper des Systems |
| 2. Residual | Impulserhaltung | Du fällst nicht mit einem Schlag vom Himmel |
| 3. RMSNorm + FFN | Lokales Feld / Atmosphäre | Physik des Planeten, auf dem du bist |
| 4. Residual | Wieder Trägheit | Du bist weiterhin auf der Bahn |

**22 Schichten ≈ 22 Perioden** vor dem endgültigen Kollaps
(softmax), wo die Umlaufbahn aufhört, kontinuierlich zu sein
und zu einer **Landung** auf einem Token wird.

---

## 4. Multikörper-Systeme (die Sequenz)

Ein Satz ist ein **zeitliches Sonnensystem**:

```
Pos 0: "The"     → residual_0
Pos 1: "secret"  → residual_1  (sieht zu 0)
Pos 2: "to"      → residual_2  (sieht zu 0,1)
...
Pos t: ...       → residual_t  (sieht zu 0…t)
```

- **GQA**: 32 Sensoren (Q) teilen sich 4 Erinnerungen (KV)
  — keine 32 Sonnen, sondern eine Sonne mit mehreren Planeten der KV-Masse.
- **KV-Cache**: bereits berechnete K,V werden wiederverwendet;
  nur der neue Körper integriert seine Umlaufbahn.
  Ohne Cache würde das System bei jedem Schritt
  den gesamten Himmel neu berechnen (alte Engine; die aktuelle
  kreist gut).

Die kausale Maske ist der **Zeitpfeil**:
die Zukunft zieht die Gegenwart nicht an.

---

## 5. Generierungsumlaufbahn (der große Kreislauf)

Text generieren ist eine **geschlossene Umlaufbahn in diskreter Zeit**:

```
        ┌─────────────────────────────────┐
        │                                 │
        ▼                                 │
   residual(s) ──► logits ──► sample ──► neues Token
        │                                 │
        └──────── embedding(token) ───────┘
```

Jede Runde:

1. Das neue Token wird am Embedding-Himmel geboren.  
2. Es integriert sich mit der Schwerkraft der vorherigen.  
3. Es kollabiert zu einem Nachfolger.  
4. Das System wächst um einen Körper.

**Periode:** ~1/Token (in der C-Engine CPU: ~0.1–0.15 s/Token
⇒ **~6–10 tok/s**).  
**Temperatur:** Exzentrizität des Kollaps („rundere" oder wildere
Umlaufbahnen).  
**Top-k:** Horizont erlaubter Schicksale.

---

## 6. Umlaufbahnen im Gewichtsraum (Perspektiven)

Es gibt eine weitere, langsamere Umlaufbahn, die nicht der Forward ist:

```
Basis-Modell  --(+ ε · δ)-->  Modell mit anderer Stimme
```

- \(\delta\) tangential zur Kohärenz-Oberfläche
  (`mystical` / amplify) → **stabile Umlaufbahn** der Perspektiven.  
- \(\delta\) normal (starkes Rauschen) → **Ausstoßung** ins Leere
  (Müll).

`--intensity` zu ändern bedeutet, den **Radius** dieser
Abweichung zu ändern. Gleicher Seed + gleicher Prompt = Vergleich
zweier Generierungsumlaufbahnen unter zwei Gewichtsmetriken.

---

## 7. Umlaufbahnen am semantischen Himmel (statisch)

Tokens „kreisen" nicht allein im Embedding:
sie sind fixiert wie Katalogsterne.

Was sich bewegt, ist der **Residual** bezüglich der Inseln:

```
Residual · spirituelle_Richtung   →  Affinität zum spirituellen Kontinent
Residual · emotionale_Richtung    →  Affinität emotional
```

`--steer amor` ist ein **künstlicher orbitaler Schub**:
er fügt eine Komponente entlang einer Himmelsachse hinzu,
ohne den Sternenkatalog (Embeddings) umzuschreiben.

Die 2D-PCA-Karte ist ein **Planetarium**: Es projiziert den Katalog,
damit wir Sternbilder sehen; es ist nicht die tatsächliche Dynamik.

---

## 8. Einheitliches Diagramm

```
                    GEWICHTSRAUM (Metrik des Universums)
                              │
                    --perturb │ (ändert G, nicht den Körper)
                              ▼
   tokens ══╗
            ║  Schwerkraft (Aufmerksamkeit)    Klima (FFN)
   residual ╬══════► Stöße ══════► Stöße  ──► ×22 Schichten
            ║              Residual (Trägheit)
            ╚══════════════════════════════════════════╝
                              │
                         output_norm
                              │
                           logits
                              │
                    softmax / temp / top-k
                              │
                         neues Token ──► (schließt die Umlaufbahn)
```

---

## 9. Wie man eine Umlaufbahn „reitet" (Rezept)

| Ziel | Stellschrauben |
|------|----------------|
| Saubere Baseline-Umlaufbahn | Prompt + seed + temp, ohne Perturbation |
| Gleiche Umlaufbahn, anderes Klima | `--perturb mystical --intensity I` |
| Abweichung zu einer Insel | `--steer Wort --steer-strength s` |
| Vorhersagbarere Umlaufbahn | temp↓, top_k↓ |
| Explorativere Umlaufbahn | temp↑, top_k↑ |
| Längeres Multikörper-System | n (Tokens) ↑ |
| Den Flug reproduzieren | gleicher Seed, gleiche Flags |

```bash
# Referenz-Umlaufbahn
./llm_inference modell.F16.gguf "When we dissolve the ego" \
  40 0.7 40 --seed 42

# Gleiche Anfangsbahn, mystische Metrik
./llm_inference modell.F16.gguf "When we dissolve the ego" \
  40 0.7 40 --seed 42 --perturb mystical --intensity 0.50
```

---

## 10. In einem Satz

**Dieses Universum kreist**, weil der Residual unter
der Schwerkraft der Aufmerksamkeit und des FFN-Klimas
**zur Seite fällt**, Impuls mit dem Residual bewahrend,
22 Perioden pro Token lang, bis er in einen
Nachfolger kollabiert — und die Generierung diesen Kreislauf
wiederholt, während Perspektiven die Metrik des Raums ändern,
ohne die Möglichkeit kohärenter Umlaufbahnen auszulöschen.

---

*Nächstes Kapitel: Archetypen und Sternbilder.*