# Kapitel 26: Tokens → Embeddings → Reine Ideen → Semantik → Details → Antwort

## Der Auftrag

Den Weg der Bedeutung im Mikrokosmos
TinyLlama ordnen — nicht als lose Boxen des Transformers,
sondern als **vollständige Kette**, vom symbolischen Funken
bis zum Satz, der in die Welt zurückkehrt.

---

## Neugeordnete Kette (kanonisch)

```
1. TOKENS          diskrete Symbole des Vokabulars
        ↓
2. EMBEDDINGS      Eingabegeometrie in ℝ²⁰⁴⁸
        ↓
3. DETAILS         lokale Form (Syntax, Nachbarn, Oberfläche)
        ↓
4. REINE IDEEN     Abstraktionen und Rahmen (mittlere Schichten)
        ↓
5. SEMANTIK        im Kontext gebundene Bedeutung (Aufmerksamkeit + Integration)
        ↓
6. FEINE DETAILS   lexikalische Konkretisierung / Stil (spätes FFN + Kopf)
        ↓
7. ANTWORT         Logits → Sample → wieder Tokens
        ↓
      (kehrt zu 1 zurück)
```

Das **zweimalige Erscheinen von „Details"** ist Absicht:
- **Form-Details** (früh): *Wie* geschrieben wird.
- **Inhalts-Details** (spät): *Was* beim Sprechen konkret wird.

Die „reinen Ideen" leben in der Mitte: weder nur Buchstaben,
noch schon der geschlossene Satz.

---

## Master-Tabelle

| # | Stufe | Was ist es? | Wo im Modell | Dim / Objekt | Dreaming-Werkzeug |
|---|-------|-------------|--------------|--------------|-------------------|
| 1 | **Tokens** | BPE-Stücke (`▁love`, IDs) | Vokabular \(V=32\mathrm{k}\) | endliche Menge | GGUF-Tokenizer |
| 2 | **Embeddings** | Punkt am Himmel | `token_embd` | \(\mathbb{R}^{2048}\) | 2D/3D-Karten, Archetypen |
| 3 | **Details (Form)** | Lokale Beziehungen, Syntax | Schichten **0–5**, kurze attn | Residual noch „am emb klebt" | Dungeon L0–L5 · Zone gravity/matter |
| 4 | **Reine Ideen** | Rahmen, Themen, abstrakte Rollen | Schichten **6–12** | thematisierter Residual | Zonen mage/sage |
| 5 | **Semántik** | Bedeutung *im Kontext* (wer verbindet sich mit wem) | Globale attn + Schichten **13–20** | Kopplung \(a_{t,t'}\) | Zonen gravity + drama + surface |
| 6 | **Details (Inhalt)** | Feines Lexikon, Schritte, lokale Farbe | FFN (v. a. spät) + SwiGLU-Gewohnheiten | \(\mathbb{R}^{5632}\) Intermediate | Zone matter · praktische Stimme |
| 7 | **Antwort** | Ein Token (und dann ein Satz) | `output_norm` → lm_head → softmax | \(\mathbb{R}^{32000}\) → Sample | Zone event · Space im Spiel |

---

## 1. Tokens

**Ein- und Ausgang des Spiegels.**

- Diskret, endlich, ohne „Sinn" bis zur Projektion.
- BPE zerschneidet die Welt: nicht jedes Konzept ist eine einzige ID.
- Im Spiel: Das **Sample** am Ende wird wieder Token
  und startet die Umlaufbahn neu.

Ohne Tokens gibt es keine Kanten zu erzählen.
Nur mit Tokens gibt es kein kontinuierliches Universum.

---

## 2. Embeddings

**Geometrische Geburt.**

\[
t \mapsto e_t \in \mathbb{R}^{2048}
\]

- Semantische Inseln und Archetypen leben hier als **Katalog**.
- Gegensätze ≈ orthogonal, nicht antipodal.
- PCA: Hunderte tatsächliche Dims; die 2D/3D-Karte ist ein Planetarium.

Hier ist der Residual **noch nicht gereist**:
Er ist Sinn-Potenzial, noch kein Satz.

---

## 3. Form-Details (früh)

**Schichten 0–5 · „wie der Buchstabe zusammentrifft".**

- Nachbarschaftsmuster, kurze Abhängigkeiten.
- Die Aufmerksamkeit beginnt, Nachbarn zu koppeln.
- Das FFN passt die lexikalische Oberfläche an.

Wenn diese Phase zerbricht, verliert der Text **Grammatik**
bevor er „philosophische Tiefe" verliert.

Im Spiel: erste Portale · Zonen **sky → gravity/matter**.

---

## 4. Reine Ideen (Mitte)

**Schichten 6–12 · „worum es geht".**

- Rahmen: existenziell, akademisch, erzählerisch, technisch.
- Der Residual löst sich vom reinen Bigramm.
- Hier fügen sich die Sternbilder Zauberer/Mystiker und Weiser
  als *Ideenklimate* ein, nicht als einzelne Worte.

Arbeitshypothese des Buches: Der mittlere Bereich ist
der Ort, an dem `--steer soul` und `mystical` aufhören,
kosmetisch zu sein, und zu einem **thematischen Bias** werden.

Im Spiel: Warps zu **mage** und **sage**.

---

## 5. Semantik (im Kontext binden)

**Fernwirkungs-Attn + späte Integration.**

Semantik ≠ Embedding-Liste.
Semantik = **Beziehungen**:

\[
\mathrm{Semantik}(t) \approx \sum_{t'\le t} a_{t,t'}\, v_{t'}
\]

Schicht für Schicht neu geschrieben und mit dem Residual vermischt.

- Wer verändert wen im Satz.
- Polaritäten Held/Schatten als Spannung im Faden.
- Goldene Regel: **Aufmerksamkeit** berühren verschiebt den Reflex
  in Richtung **akademisch / relational / kritisch**.

Im Spiel: Zonen **gravity**, **drama**, **surface**.

---

## 6. Inhalts-Details (Konkretisierung)

**FFN · „mit welchen Worten und Gesten es gesagt wird".**

Obwohl das FFN in allen Schichten wirkt, zeigt sich seine Rolle
als *feines Detail* bei der Konkretisierung:

- Handlungsverben, Listen, Ratschläge (praktische Stimme),
- lexikalische Farbe, lokale Gewohnheiten in \(\mathbb{R}^{5632}\).

Goldene Regel: **FFN** berühren → **praktische** Perspektive.

Es ist nicht die reine Idee; es ist die **Inkarnation** der Idee
in verbalem Material.

---

## 7. Antwort

**Kollaps und Rückkehr zum Makrokosmos.**

\[
z = W\,\mathrm{RMSNorm}(x_L),\quad
t\sim \mathrm{softmax}(z/T)\ \text{(top-k)}
\]

- Ein diskretes Ereignis (Token).
- Concateniert wird es wieder menschliche Sprache.
- Der Spiegel schließt sich: von der Uhr zum Himmel (Kap. 6, 24).

Dann der Zyklus:

**Antwort → neue Tokens → …**

---

## Flussdiagramm (vollständig)

```
 MAKRO: menschliche Frage / Prompt
              │
              ▼
     ┌──── TOKENS ────┐
     │                │
     ▼                │
 EMBEDDINGS (Himmel)  │
     │                │
     ▼                │
 DETAILS Form         │   Schichten 0–5
 (Syntax, Nachbarn)   │
     │                │
     ▼                │
 REINE IDEEN          │   Schichten 6–12
 (Rahmen, Themen)     │
     │                │
     ▼                │
 SEMANTIK             │   attn + Schichten 13–20
 (Bindungen im Kontext)│
     │                │
     ▼                │
 DETAILS Inhalt       │   FFN / Feinstil
 (Lexikon, Handlung)  │
     │                │
     ▼                │
 ANTWORT (Sample) ──┘   Logits → Token
     │
     ▼
 MAKRO: wir lesen eine Stimme / einen Archetyp / ein Urteil
```

Die Dreaming-Linsen wirken **entlang der gesamten Kette**:

| Linse | Wo sie die Kette am stärksten krümmt |
|-------|--------------------------------------|
| baseline | die gesamte „offizielle" Kette |
| mystical / Zauberer | reine Ideen + existenzielle Semantik |
| akademisch / Weiser | relationale Semantik / Struktur |
| praktisch | Inhalts-Details (FFN) |
| noise | bricht die Kette (verlässt \(\mathcal{C}\)) |
| `--steer` | schiebt Residual in Richtung einer Embedding-Insel |

---

## Beziehung zu anderen Teilen des Buches

| Kapitel | Einbindung in die Kette |
|---------|------------------------|
| 2 Struktur | Wo die Stufen in Tensoren leben |
| 3 C-Engine | Wie jeder Pfeil berechnet wird |
| 5 Multidim-Raum | Stufen 1–2 und Himmelsgeometrie |
| 7 Kräfte | attn=lokale Semantik; FFN=Inhalts-Details |
| 9 Goldene Regel | Linsen über 5 und 6 |
| 13–15 Schichten | Zeitliche Aufteilung von 3–4–5 |
| 16–21 Inseln / Archetypen | Kulturelle Etikettierung von 2 und 4 |
| 20 Umlaufbahn | Die Kette als Dynamik \(x\leftarrow x+F(x)\) |
| 25 Spiel | Jedes Portal = Stufe + Zonenwarp |

---

## Kurzfassung (für das Spiel-HUD / Glossar)

```
TOKENS → EMBEDDINGS → DETAILS → REINE IDEEN
       → SEMANTIK → FEINE DETAILS → ANTWORT → (Tokens)
```

Oder in einer Zeile:

**Symbol → Geometrie → Form → Idee → Bindung → Konkretisierung → Gesagtes.**

---

## In einem Satz

Das TinyLlama-Universum ist nicht nur ein Schichten-Stack:
Es ist eine **Kette der Sinntransformationen**,
in der Tokens zu Geometrie werden, Geometrie
zu Form und Idee, Idee in Semantik gebunden wird,
sich in Lexikon detailiert und **wieder zu Tokens kollabiert**,
die wir lesen können — ein zyklischer Spiegel zwischen Mikro und Makro.

---

*Nächstes Kapitel: Jede Schicht ist ein Aufzug.*