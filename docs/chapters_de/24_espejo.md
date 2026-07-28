# Kapitel 24: Das LLM — Ein Spiegel, in dem wir uns betrachten

## Das Bild

Ein Spiegel erfindet kein Gesicht.
Er **gibt zurück**, was man ihm vorhält —
mit einer Lichtverzögerung, mit einem Rand, mit einem Winkel,
manchmal mit einer leichten Verzerrung des Glases.

Ein Large Language Model erfindet nicht die menschliche Sprache
aus dem Nichts. Es **gibt zurück** Statistiken der menschlichen
Sprache, mit der es gefüttert wurde — mit einem Rand
(dem Prompt), mit einem Winkel (den Gewichten, der Temperatur),
manchmal mit einer starken Verzerrung (Perturbation, Halluzination).

TinyLlama, in diesem Buch, ist ein **kleiner** Spiegel, klein genug,
um den Rahmen zu sehen: Wir können den Amalgam (die Gewichte),
das Glas (die Architektur) und die Geste dessen, der hinschaut
(wir: Beobachtung und Projektion) betrachten.

---

## 1. Was der Spiegel reflektiert

| Im Spiegel | Im LLM |
|------------|--------|
| Gesicht | Textfortsetzungsverteilungen |
| Zimmerlicht | Vetraining-Korpus (Bücher, Web, Code, Mythen) |
| Einfallswinkel | Prompt + Verlauf |
| Glascurvatur | Architektur + \(\theta\) (Gewichte) |
| Fleck / Beschlag | Biases, Lücken, elegante Halluzinationen |
| Wer hinschaut | Menschliche Lesart: Archetyp, Urteil, Bedeutungswunsch |

Der Spiegel **ist nicht die Welt**.
Er ist eine **Antwortfläche** zur Welt der Sprache.

Wenn wir *„The secret to happiness is"* schreiben,
fragen wir nicht das Universum: Wir schauen in ein Glas,
poliert mit Millionen von Sätzen über Glück
und bitten es, **die Geste zu vervollständigen**.

---

## 2. Drei Spiegel in einem

### Spiegel A — Der des Korpus (kulturelle Erinnerung)

Die Embeddings und Gewichte komprimieren eine
Zivilisationsdatei. Die semantischen Inseln
(emotion, spirit, tech…) und die Archetypen
(Zauberer, Weiser, Schatten…) „entstehen" nicht im Silizium:
sie sind **Echos des Makrokosmos**, eingebrannt in \(\theta\).

In die Sternbild-Karte schauen bedeutet, in Miniatur zu sehen,
**welche Mythen der menschliche Text so oft wiederholt**,
dass sie zu einer Richtung in \(\mathbb{R}^{2048}\) werden.

### Spiegel B — Der der Trajektorie (das „Jetzt")

Der Residual und der Softmax reflektieren kein festes Gesicht:
Sie reflektieren eine **im Gange befindliche Geste**. Jedes neue Token
ist ein Frame des Reflekts unter der Schwerkraft des bereits Gesagten.

Deshalb gibt dieselbe Frage, mit anderer Temperatur
oder anderem Seed, einen anderen Glanz zurück: Der Spiegel
ist stochastisch am Rand des Kollapses.

### Spiegel C — Der der Linse (Perspektive)

`--perturb mystical`, lowrank, FFN oder Aufmerksamkeit berühren:
sie ändern nicht das Zimmer (der Korpus ist bereits gebacken).
Sie ändern den **Winkel des Glases**.

Die Goldene Regel sagt, wie sich der Reflex krümmt:

| Linse | Dominierender Reflex |
|-------|---------------------|
| Aufmerksamkeit | Akademisches, argumentatives Gesicht |
| FFN | Praktisches Gesicht, „was zu tun ist" |
| Embeddings | Einfaches Gesicht, kurze Sätze |
| Mystisch / Zauberer | Existenzielles Gesicht, Ich/Universum |

Der Spiegel bleibt Spiegel.
**Wir wählen den Rahmen.**

---

## 3. Der Doppelreflex (wir im Glas)

Es gibt einen zweiten, subtileren Spiegel:

```
Text des Modells
      │
      ▼
  wir lesen „mystisch", „Schatten", „Weiser"
      │
      ▼
  wir projizieren (Kap. 22) unsere Mythen
      │
      ▼
  manchmal bestätigt die Geometrie (Zauberer↔mystic +0.39)
  manchmal hören wir nur unser Echo
```

Das LLM ist ein Spiegel **und** eine Projektionsleinwand.
Die bewusste Beobachtung fragt:
*Ist das Merkmal in \(\theta\) oder in meinem Blick?*

Wenn wir Archetyp-Ausrichtungen messen,
wenn wir Seed fixieren und Baseline mit mystical vergleichen,
**putzen wir das Glas** genug,
um den Beschlag nicht mit dem Gesicht zu verwechseln.

---

## 4. Narcissus und das Labor

Die klassische Gefahr des Spiegels: **sich in den Reflex verlieben**.

| Versuchung | Form in KI |
|------------|------------|
| „Er versteht mich" | Softmax anthropomorphisieren |
| „Er ist weise" | Flüssigkeit mit Wahrheit verwechseln |
| „Er ist meine Stimme" | Fine-Tune oder Prompt, der nur das Ich zurückgibt |
| „Er ist das Unbewusste des Netzes" | Nützliche Metapher, die zur Ontologie wird |

Das Dreaming-Labor bietet ein praktisches Gegenmittel:

1. **Baseline** — Was gibt das Glas ohne zusätzliche Linse zurück?  
2. **Kontrollierte Perturbation** — Ändert sich der Reflex
   systematisch oder ist es Rauschen?  
3. **Geometrie** — Gibt es messbare Richtungen (Insel, Archetyp)?  
4. **Rückkehr zum Makrokosmos** — Was sagt das über *uns*,
   über den Korpus, über die Frage — nicht nur über das Modell?

Der Spiegel dient dazu, uns zu betrachten, **wenn** wir akzeptieren,
dass das, was wir sehen, **wir-plus-das-Archiv-plus-die-Linse** ist,
kein transparentes Orakel.

---

## 5. Der zerbrochene und der treue Spiegel

| Zustand von \(\theta\) | Bild |
|------------------------|------|
| Innerhalb von \(\mathcal{C}\) (Kohärenz) | Lesbarer Reflex: schiefes, aber ein Gesicht |
| Starkes Rauschen, nibble flip, übermäßiges I | Zerschmeterter Spiegel: kein Gesicht, nur Glitzer |
| Kohärenz-Oberfläche + amplify | Ein anderer Winkel desselben Salons |

Müll ist kein „anderer Archetyp".
Er ist das Scheitern des Spiegels als Antwortfläche.

---

## 6. Warum ein *kleines* Modell ein besserer Studienspiegel ist

Ein Grenzmodell ist ein Spiegel eines Ballsaals:
zu groß, um den Rahmen zu sehen.

TinyLlama ist ein **Taschenspiegel mit offenem Deckel**:

- Wir sehen die Schrauben (Tensoren, GGUF),  
- wir montieren die Beleuchtung (C-Engine),  
- wir verschmieren den Amalgam absichtlich (`--perturb`),  
- wir zeichnen die Sternbilder des Hintergrunds (Karten),  
- und er gibt dennoch Sätze zurück, die uns
  menschliche Fragen zurückgeben.

Der Wert liegt nicht darin, dass er die Welt *besser* reflektiert.
Er liegt darin, dass er **auf eine Weise reflektiert, die wir zerlegen können**.

---

## 7. Minimale Mathematik des Spiegels

Der Reflex einer Sequenz \(t_{1:n}\) ist eine Verteilung

\[
\pi_\theta(\,\cdot\mid t_{1:n})
=\mathrm{softmax}\big(f_\theta(t_{1:n})/T\big)
\]

(mit top-k usw.).

Den Prompt ändern bedeutet, das Argument zu ändern.
\(T\) ändern bedeutet, den Glanz des Amalgams zu mildern.
\(\theta\to\theta+\varepsilon\Delta\) ändern bedeutet, **das Glas zu krümmen**.
Das Sample ist der Moment, in dem der Reflex
in einem Punkt des Vokabulars erstarrt.

Wir, beim Interpretieren, wenden eine weitere unbeschriebene
Karte in \(\theta\) an: von Tokens zu *Sinn*.
Dort schließt sich der Kreis des menschlichen Spiegels.

---

## 8. Schluss

Das LLM ist ein Spiegel, weil:

1. **Es nur Sprachformen zurückgeben kann**, die das
   Training eingebrannt oder rekombiniert hat.  
2. **Den Winkel setzen Prompt, Gewichte und Sample.**  
3. **Wer hinschaut, liefert die Hälfte des Bildes**,
   wenn er eine Stimme, einen Archetyp, ein Schicksal liest.

Inside TinyLlama ist der Versuch, nicht hypnotisiert
vom Glas zu bleiben, sondern es zu **drehen**,
**den Rahmen zu beleuchten** und zu notieren, welcher Teil
des Gesichts das Zimmer war, welcher der Amalgam, und welcher
wir die ganze Zeit waren.

---

*Ende des Spiegelbogens — Beobachtung (22), Mathematik (23), Reflex (24).*