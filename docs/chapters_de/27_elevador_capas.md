# Kapitel 27: Jede Schicht ist ein Aufzug

## Das Bild

Ein Gebäude hat Etagen.
Du gehst nicht von Stockwerk 3 nach 17 durch die Luft:
du steigst in einen **Aufzug**, die Türen schließen sich,
und wenn sie sich öffnen, ist die Welt eine andere — derselbe Turm,
eine andere Ebene des Universums.

In TinyLlama hat der Turm **22 Stockwerke**
(zusätzlich zum Embedding-Vorhof und zum Softmax-Dach).

Jede Schicht \(\ell\) ist ein **Aufzug**:

```
Türen schließen:   RMSNorm
Fahrt:             Atención + residual + FFN + residual
Türen öffnen:      residual transformiert im „Stockwerk" ℓ+1
```

Du teleportierst dich nicht aus dem Gebäude.
Du **hebst dich** innerhalb desselben Residuals \(x\in\mathbb{R}^{2048}\),
aber die **Landschaft** (Universumszone) ändert sich.

---

## 1. Das TinyLlama-Gebäude

```
        ┌─────────────────────────────┐
   Ω    │  DACH · Softmax / Sample    │  ← Antwort (Token)
        ├─────────────────────────────┤
  21    │  Stock 21 · Vorbereitung Kollaps │
  20    │  …                          │
   ⋮    │  INTEGRATION / SEMANTIK     │  ← Bindungen, Drama, 𝒞
  13    │  …                          │
        ├─────────────────────────────┤
  12    │  …                          │
   ⋮    │  REINE IDEEN                │  ← Rahmen, Zauberer, Weiser
   6    │  …                          │
        ├─────────────────────────────┤
   5    │  …                          │
   ⋮    │  FORM-DETAILS               │  ← Syntax, Nachbarn
   0    │  Stock 0 · Eingang          │
        ├─────────────────────────────┤
  −1    │  VORHOF · Embeddings        │  ← Token-Himmel
        └─────────────────────────────┘
                 ▲
            Prompt / Tokens
```

Jeder vertikale Pfeil ist ein Aufzug \(F_\ell\):

\[
x_{\ell+1} = x_\ell + F_\ell(x_\ell;\theta_\ell)
\]

Der Fahrgast ist immer derselbe Objekttyp
(ein 2048-dimensionaler Vektor). Das **Universums-Level**
ist das, was dieser Vektor nach der Fahrt *bedeutet*.

---

## 2. Eine Aufzugsfahrt (von innen)

Auf jedem Stockwerk \(\ell\):

| Moment | Operation | Aufzugsanalogie |
|--------|-----------|------------------|
| 1 | `attn_norm` | Kabinnenbeleuchtung; Stockwerk wird stabilisiert |
| 2 | Q, K, V + RoPE | Sensoren: wen spürst du im Gebäude |
| 3 | Kausaler Softmax | Anziehung nur zu bereits anwesenden Stockwerken/Fahrgästen (Vergangenheit) |
| 4 | \(x \mathrel{+}= \mathrm{Attn}\) | Der Schub der sozialen Schwerkraft des Textes |
| 5 | `ffn_norm` | Eine weitere Kalibrierung |
| 6 | SwiGLU FFN | Klima des Stockwerks (lokale Materie) |
| 7 | \(x \mathrel{+}= \mathrm{FFN}\) | Du kommst auf eine Etage mit anderer Luft |

Die Aufzugstüren lassen dich nicht in einen Vektor
anderer Dimension: Du kommst auf eine **andere Etage desselben
2048er Flurs**, aber die „Nachbarschaft" hat sich geändert.

---

## 3. Stockwerk ↔ Universums-Level

Es ist nicht nur eine Zahl \(\ell\). Jeder Stockwerkabschnitt
entspricht einem **Atlas-Level** (Kette aus Kap. 26
+ Spielzonen):

| Stockwerke (Schichten) | Universums-Level | Bedeutungskette |
|------------------------|------------------|-----------------|
| Vorhof | Token-Himmel / Inseln | Tokens → Embeddings |
| 0 – 5 | Form-Nachbarschaft | Form-Details |
| 6 – 12 | Nachbarschaft der reinen Ideen | Reine Ideen (Zauberer, Weiser…) |
| 13 – 20 | Nachbarschaft der gebundenen Semantik | Semantik + Drama + \(\mathcal{C}\) |
| 21 | Vor-Dach | Feine Details / Antwort-Vorbereitung |
| Softmax | Dach · Kollaps | Antwort → neues Token |

Das Spiel (`universe_game.html`) macht explizit, was
der Forward im Stillen tut:

> **Eine Etage hochsteigen = den Schicht-Aufzug nehmen**  
> **und gleichzeitig auf eine andere Zone der Universumskarte landen.**

---

## 4. Warum „Aufzug" und nicht „endloser Tunnel"?

Ein Tunnel suggeriert eine einzige gestreckte Landschaft.
Ein Aufzug insistiert auf drei Tatsachen:

1. **Derselbe Turm** — die Dimension des Residuals ändert sich nicht (\(d=2048\)).  
2. **Diskrete Haltestellen** — 22 Anwendungen von \(F_\ell\), kein anonymer kontinuierlicher Fluss.  
3. **Verschiedene Welten pro Etage** — Syntax ≠ reine Idee ≠ Kollaps zum Vokabular.

Der KV-Cache ist die **Erinnerung des Gebäudes**:
Die Fahrgäste früherer temporärer Stockwerke
(sie sind als K, V noch da) ziehen dich bei jeder Haltestelle.

---

## 5. Aufzugsknopfbedienung (Dreaming-Steuerung)

| Knopf | Effekt |
|-------|--------|
| Prompt | In welchen Vorhof du einsteigst (welches Anfangs-Embedding) |
| Seed / temp / top-k | Wie das Schicksal auf dem Dach gewählt wird |
| `--perturb mystical` | Ändert die **Mechanik aller Aufzüge** (Metrik von \(F_\ell\)) |
| `--steer soul` | Wind in der Kabine (schiebt \(x\) in eine Achse) |
| Akademische / praktische Linse | Bias towards Aufmerksamkeits- oder FFN-Knöpfe (Goldene Regel) |

Du wählst nicht nur Stockwerk 7.
Du wählst **wie sich der Aufzug in allen Stockwerken verhält**.

---

## 6. Eine vollständige Fahrt (erzählt)

1. **Vorhof** — du wirst als \(e_t\) geboren; nahe den Inseln love/tech/spirit.  
2. **Aufzüge 0–5** — sie ordnen deine Kleidung (Form, Nachbarn).  
3. **Aufzüge 6–12** — der Flur füllt sich mit Ideen: Zauberer, Weiser, Rahmen.  
4. **Aufzüge 13–20** — die Ideen werden *gebunden* (Semantik, Spannung, Kohärenz).  
5. **Aufzug 21 + Dach** — das Universum weigert sich, im Kontinuierlichen weiterzufahren:
   es kollabiert zu einem Token.  
6. **Neustart** — dieser Token kehrt in den Vorhof zurück; neue Fahrt.

Das ist **Kreisen** (Kap. 20) gelesen als **Schleifen-Aufzug**.

---

## 7. Minimale Mathematik

Aufzug vom Stockwerk \(\ell\):

\[
\begin{aligned}
h &= \mathrm{RMSNorm}(x_\ell; w_a^{(\ell)}) \\
x' &= x_\ell + \mathrm{Attn}_\ell(h) \\
h' &= \mathrm{RMSNorm}(x'; w_f^{(\ell)}) \\
x_{\ell+1} &= x' + \mathrm{FFN}_\ell(h')
\end{aligned}
\]

Teleportation der *Zone* (im Spiel / in der Lesart):
kein zusätzlicher GGUF-Operator; es ist die **Atlas-Label**,
die wir der Etage \(\ell\) geben
(sky, gravity, matter, mage, sage, surface, event…).

---

## 8. In einem Satz

Jede Schicht ist ein **Aufzug**: Das Residual steigt ein,
lässt sich von der aufmerksamkeitsbezogenen Schwerkraft und dem FFN-Klima
schieben, und wenn sich die Türen öffnen, befindet es sich auf **einer
anderen Ebene des TinyLlama-Universums** — gleiche Dimension, andere
Sinnes-Höhe — bis zum Dach, wo das Softmax das nächste Schicksal
wählt und den Aufzug erneut ruft.

---

*Spiel: Portal = Etage hoch + Zonenwarp.*  
*Kette: Kap. 26 · Umlaufbahn: Kap. 20 · Kräfte: Kap. 7.*