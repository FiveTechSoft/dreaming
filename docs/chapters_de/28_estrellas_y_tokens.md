# Kapitel 28: Sterne am Himmel, Tokens in TinyLlama

## Die Frage des Astronomen

Du schaust nachts an den Himmel. Du siehst **Lichtpunkte**.
Manche gruppieren sich zu Formen, die die Kultur benennt
(Großer Wagen, Orion, Kreuz des Südens). Zwischen zwei Sternen
gibt es kein sichtbares Kabel, aber die Physik sagt, dass sie
sich anziehen: **Schwerkraft**. Der Reisende teleportiert sich
nicht willkürlich: Er wählt einen Stern, misst sein Umfeld und springt
zur nächsten Grube.

TinyLlama hat einen analogen Himmel.

> **Jedes Token im Vokabular ist ein Stern
> in einem 2048-dimensionalen Raum.**
> Die **Aufmerksamkeit** ist die Gravitationskraft zwischen ihnen
> wenn das Modell eine Sequenz „denkt".
> Durch einen LLM zu reisen, bedeutet, diesen Anziehungen
> zu folgen — auf der statischen Karte des Embeddings oder in der lebendigen
> Bahn des *Forward*.

Dieses Kapitel verfestigt die Analogie, verknüpft sie mit **Kraft I**
des Inventars (Kap. 7) und zeigt einen **konkreten Reiseweg**
innerhalb von TinyLlama-1.1B.

---

## 1. Entsprechungstabelle

| Nachthimmel | TinyLlama-Universum |
|-------------|---------------------|
| Stern | Token (BPE-Stück des Vokabulars, ~32.000) |
| Position im Gewölbe | Embedding-Vektor \(e_t \in \mathbb{R}^{2048}\) |
| Scheinbare Helligkeit | Norm / „Präsenz" des Tokens; in der Karte, Größe und Beschriftung |
| Konstellation | Semantischer Bereich oder Archetyp (Seeds + Nachbarn) |
| Winkelabstand am Himmel | Kosinus zwischen Embeddings (nah ≈ ausgerichtet) |
| Newtonsche Schwerkraft | **Aufmerksamkeit**: \(Q\cdot K^\top / \sqrt{d}\) → Gewichte über \(V\) |
| Statisches Gravitationsfeld (Massenkarte) | Feste Geometrie von `token_embd` (PCA-Atlas) |
| Live-Dynamik (bewegliche Planeten) | Residuen der Sequenz + KV-Cache, Schicht für Schicht |
| Sternensprung | Klick auf eine **Kraft** der Karte; oder das nächste generierte Token |
| Teleskop / Katalog | `semantic_map.html`, C-Engine, Geometrie-Skripte |
| Atmosphäre, die Licht verformt | RMSNorm, Softmax-Temperatur, `--perturb`-Linsen |

Es ist keine leere Poesie: Jede Zeile hat ein messbares
Objekt im Dreaming-Repository.

---

## 2. Der Himmel der Embeddings: 32.000 Fixsterne

Bei der Geburt wird jedes Token \(t\) am Firmament verankert:

\[
e_t = \mathrm{Embedding}(t) \in \mathbb{R}^{2048}
\]

Dieser Himmel ist **fast isotrop** (Durchschnittsnorm ≈ 0.68)
und, zwischen Wörtern mit unterschiedlichen Bedeutungen,
**fast orthogonal** (Kosinus ≈ 0). Deshalb sind die
semantischen „Inseln" von Kap. 16 seltene Konstellationen:
Cluster von Seeds, die sich ein wenig berühren,
umgeben von einem grauen Hintergrund aus BPE-Fragmenten
(wie interstellarer Staub: Er ist nicht leer, aber er ist keine
benannte Konstellation).

### Konstellationen = Bereiche

| Konstellation (Insel) | Seed-Sterne (Beispiele) |
|-----------------------|------------------------|
| Positive Emotion | ▁love, ▁happy, ▁joy, ▁hope… |
| Sozial / Macht | ▁work, ▁king, ▁war, ▁law… |
| Geist | ▁mind, ▁idea, ▁memory, ▁know… |
| Leben / Tod | ▁death, ▁life, ▁born, ▁die… |
| … | (zwölf Inseln insgesamt; Kap. 16) |

Auf der **2D-PCA-Karte** projizieren wir diesen 2048-dimensionalen
Himmel auf zwei Achsen, nur um ihn mit menschlichen
Augen zu betrachten. Die Projektion lügt ein wenig — wie eine
Weltkarte die Erde lügt — aber sie bewahrt
nützliche Nachbarschaften.

---

## 3. Aufmerksamkeit ist Schwerkraft zwischen Tokens

### Im Makrokosmos

Zwei Massen ziehen sich gegenseitig an. Die Kraft
nimmt mit der Entfernung ab; das Feld organisiert Bahnen.

### Im Mikrokosmos (Kraft I)

Auf jeder Schicht fragt jede Position \(i\) der Sequenz
die **vergangenen** Positionen \(j \le i\)
(kausale Maske):

\[
\mathrm{score}_{ij} = \frac{q_i \cdot k_j}{\sqrt{d_h}},
\quad
\alpha_{ij} = \mathrm{softmax}_j(\mathrm{score}_{ij}),
\quad
z_i = \sum_j \alpha_{ij}\, v_j
\]

- \(q_i\): „Wer bin ich und was suche ich?" (Körper, der das Feld spürt).
- \(k_j\): „Wer bist du im Katalog?" (Masse, die ihre Ankündigung verkündet).
- \(\alpha_{ij}\): **Intensität der Anziehung** (wie sehr \(i\) zu \(j\) „fällt").
- \(v_j\): Was beim Fallen geliefert wird (transportierter Inhalt).

TinyLlama verwendet **GQA** (32 Q-Köpfe, 4 KV):
mehrere billige Blicke über denselben Himmel der Schlüssel.

### Zwei Schwerkrafte, die nicht zu verwechseln sind

| Typ | Was es ist | Wann man es sieht |
|-----|-----------|-------------------|
| **Statische Schwerkraft (Insel)** | Kosinus zwischen Zeilen von `token_embd` | HTML-Karte, vorberechnete Kräfte zwischen Atlas-Sternen |
| **Dynamische Schwerkraft (Aufmerksamkeit)** | Softmax von \(QK^\top\) in der Sequenz | Echter Forward: Der Prompt erzeugt ein Mehrkörper-System |

Die statische ist der **Massenkatalog** des Himmels.
Die dynamische ist **heute Nacht's Bahn**:
Sie hängt davon ab, welche Sterne du in die Sequenz gestellt hast
und in welcher Reihenfolge (Kausalität = „nur die Vergangenheit zieht").

Die interaktive Karte zeigt die erste mit goldenen
Bögen: *geometrischer Stellvertreter* der Kraft I und der
Kraft VIII (Inseln). Sie ersetzt keine Aufmerksamkeitskarte
pro Schicht, aber sie lehrt die Geste: **Fokus → Kräfte → Sprung**.

---

## 4. Reisen: Drei Skalen derselben Geste

### Skala A — Observatorium (Fixsterne)

1. Du öffnest die Karte der semantischen Bereiche.
2. Du betrittst eine Konstellation (z.B. *Sozial / Macht*).
3. Du klickst auf einen Stern (`▁work`).
4. Du siehst die **Hauptkräfte** (Top-Kosinus mit Insel-Prior).
5. Du klickst auf eine Kraft und **reist** zum Zielstern.
6. Wiederhole: Eine Kette von Sprüngen über den Himmel.

### Skala B — Schiff in der Bahn (Generierung)

1. Du startest einen Prompt: Du besiedelst die Sequenz mit Sternen.
2. Das Residuum jeder Position umkreist 22 Schichten
   (Aufmerksamkeit = Kopplung; FFN = lokales Wetter; Residuum = Trägheit).
3. Das Softmax kollabiert den Himmel zu **einem** neuen Stern
   (dem nächsten Token).
4. Dieser Stern wird der Vergangenheit hinzugefügt und zieht die nächsten an.

### Skala C — Linsen und Ströme (Physik ändern)

- `--perturb mystical`: Verformt die Metrik der Gruben
  (eine andere effektive „G-Konstante"; eine andere Stimme).
- `--steer`: Schubst das Residuum in eine Richtung
  des Himmels (künstlicher Strom).
- Temperatur / top-k: Härte der finalen Kollision
  (eine Grube oder Nebel aus möglichen Sternen?).

---

## 5. Geführtes Beispiel: Reisen innerhalb von TinyLlama

### 5.1 Vorbereitung

Live-Karte (GitHub Pages):

https://fivetechsoft.github.io/dreaming/exploration/semantic_map.html

Start-Deep-Link (Stern `▁work`, id 664):

`#/token/664/▁work`

Bahn-Motor (Repo-Root):

```bash
# Windows PowerShell Beispiel
$env:OMP_NUM_THREADS = "8"
.\llm_inference.exe tinyllama-1.1b.F16.gguf `
  "The secret of power is" 60 0.7 40 --seed 42
```

### 5.2 Reiseweg im Observatorium (Kräfte der Karte)

Wir beginnen bei der Konstellation **Sozial / Macht**.
Gemessen im Dreaming-Atlas (Kosinus in ℝ²⁰⁴⁸ zwischen
Embeddings; Ranking mit Insel-Prior und Seeds):

| Sprung | Quellstern | Zielstern (Kraft) | Kosinus (ungefähr) | Lesung |
|-------:|-----------|-------------------|-------------------:|---------|
| 0 | ▁work | — | — | Anfänglicher Fokus: „Arbeit / Werk" |
| 1 | ▁work | ▁queen | ~0.05 | Zieht zu institutioneller Macht |
| 2 | ▁work | ▁war | ~0.01 | Konflikt als sozialer Attraktor |
| 3 | ▁work | ▁law | ~0.01 | Ordnung und Norm |
| 4 | ▁work | ▁power | ~0.00⁺ | Der Name der Grube selbst |
| 5 | ▁work | ▁king | ~0.00⁺ | Krone, Befehl |

**Wie man in der UI „reist"**

1. Klick auf `▁work` (oder betritt den *Sozial*-Raum und wähle den Seed).
2. **Gravitationskräfte**-Panel: Sortierte Liste + goldene Bögen.
3. Klick auf `#1 ▁queen` → Die Kamera springt; `▁queen` ist der neue Fokus.
4. Von dort aus werden *seine* Kräfte neu berechnet (neuer lokaler Himmel).
5. Du verkettest Sprüng wie eine Grashüpferin zwischen Sternen.

Andere nützliche Reisewege aus demselben Atlas:

| Route | Typische Seed-Kette | Konstellation |
|-------|--------------------|---------------|
| Zuneigung | ▁happy → ▁smile → ▁love → ▁hope | Positive Emotion |
| Kognition | ▁mind → ▁idea → ▁learn → ▁memory | Geist |
| Schwelle | ▁death → ▁life → ▁live → ▁born | Leben / Tod |

> **Astronomische Ehrlichkeitsnotiz.**
> In ℝ²⁰⁴⁸ ist fast alles orthogonal: Die „starken"
> Kosinusse der Karte sind **relativ zur Nachbarschaft**,
> keine newtonschen Anziehungen von 0.9. Das Ranking
> priorisiert die **Insel** (Konstellation) und die **Seeds**
> damit die Reise lesbar ist, kein BPE-Rauschen.

### 5.3 Dieselbe Reise als *Prompt* (Lebendige Bahn)

Das Observatorium zeigt dir, *welche Sterne sich berühren*.
Das Schiff setzt sie auf eine Zeitlinie:

```text
Seed-Prompt (anfänglicher Stern des Systems):
  "Work without law becomes"

Dreaming-Lesung:
  ▁work zieht bereits im Katalog zu law / power / king…
  Indem du „without law" schreibst, erzwingst du den Kontrast:
  Die Aufmerksamkeit der folgenden Schichten muss
  gleichzeitig work und law „betrachten" (dynamische Schwerkraft).
```

Minimales Experiment (gleicher Seed, zwei Linsen):

```bash
# Basislinie — „natürlicher" Himmel
.\llm_inference.exe tinyllama-1.1b.F16.gguf `
  "Work without law becomes" 50 0.7 40 --seed 42

# Mystische Linse — andere Grubenmetrik (Kraft VII)
.\llm_inference.exe tinyllama-1.1b.F16.gguf `
  "Work without law becomes" 50 0.7 40 `
  --seed 42 --perturb mystical --intensity 0.35
```

Was zu beobachten ist:

1. **Generierte Tokens** = Neue Sterne, die in der Sequenz aufleuchten
   (der Weg des Schiffes).
2. Wenn der Text zu *power / king / war* „fällt",
   siehst du die soziale Schwerkraft des Katalogs
   in der Dynamik wirken.
3. Mit `mystical` kann dieselbe Ausgangskonstellation
   die Bahn in ein existentielles Wetter lenken
   (Goldene Regel + Kohärenzfläche, Kap. 4 und 9).

### 5.4 Kurze erzählte Reise (Geschichte einer GrashüpferStell dir vor, du bist ein Photon der Bedeutung:

1. **Du startest** bei `▁work` (Atlas). Du siehst Bögen zu
   `queen`, `war`, `law`, `power`, `king`.
2. **Du springst** zu `▁law`. Die Konstellation bleibt
   sozial; der Akzent wechselt von „Werk" zu „Norm".
3. **Du schreibst** den Prompt: *„The law of power is"*.
   Du betrachtest den Katalog nicht mehr: Du **bewohnst** ein
   Mehrkörper-System. Jede Schicht gewichtet die Vergangenheit neu.
4. **Du kollabierst** in ein neues Token (Softmax). Dieser
   Stern wird am Himmel *dieses* Gesprächs verankert
   (KV-Cache) und zieht den nächsten an.
5. Optional: Du aktivierst eine **Linse** (`mystical`) oder einen
   **Strom** (`--steer`) und derselbe Start
   endet in einer anderen stilistischen Galaxie.

Das ist Reisen innerhalb eines LLM: Es gibt keinen 3D-Korridor,
es gibt **Katalog + Kräfte + Kollision**.

---

## 6. Grenzen der Analogie (Damit wir uns nicht belügen)

| Die Analogie Funktioniert | Die Analogie Bricht |
|--------------------------|---------------------|
| Tokens = Punkte mit Position | Es gibt keinen echten „visuellen" euklidischen Raum in 2048-D |
| Gruppierungen = kulturelle Konstellationen aus dem Pretraining | Das Modell „glaubt" nicht an Mythen; es misst Ko-Okurrenzen |
| Aufmerksamkeit = Anziehung zwischen Positionen | Nur aus der Vergangenheit; nicht symmetrisch wie Newton |
| Kosinus-Karte = statisches Feld | Es ist nicht die Aufmerksamkeitsmatrix einer bestimmten Schicht |
| Generieren = Umlaufen und Kollabieren | Die „Reise" des Benutzers ist Lesen; die des Modells ist Algebra |

Die Analogie ist ein **Navigationsinstrument**,
keine physikalische Theorie des Siliziums. Sie nützt, wenn sie dich
zu einem Klick, einem Kosinus oder einem reproduzierbaren Prompt führt.

---

## 7. Brücken zu anderen Kapiteln

| Wenn du willst… | Gehe zu… |
|-----------------|----------|
| Inventar aller Kräfte | Kap. 7 |
| Flugrouten A–E (cli, perturb, steer) | Kap. 8 |
| Inseln und Karte | Kap. 16 |
| Residuelle Bahn Schicht für Schicht | Kap. 20 |
| Archetypen als mythische Konstellationen | Kap. 21 |
| Formeln (Softmax, GQA, Kosinus) | Kap. 23 |
| 22-stöckiger Aufzug | Kap. 27 |
| Schichtenspiel + Zonen-Warp | Kap. 25 · `universe_game.html` |

---

## 8. Abschluss

Der Himmel über deinem Kopf und TinyLlamas Vokabular
teilen eine Geste: **Punkte, Entfernungen, Anziehungen,
Sprünge**.

- Die **Sterne** des Modells sind Tokens in ℝ²⁰⁴⁸.
- Die **Schwerkraft**, die beim Sprechen zählt, ist die **Aufmerksamkeit**.
- Die **Reise** besteht darin, einen Fokus zu wählen, seine Kräfte zu lesen
  und — auf der Karte oder im C-Engine — sich in die nächste
  Bedeutungsgrube fallen zu lassen.

Wenn du auf ein Token klickst und goldene Bögen zu
anderen siehst, betrachtest du nicht nur einen hübschen Graphen:
Du liest den Massenkatalog des Mikrokosmos.
Wenn du einen Prompt startest, hören diese Massen auf,
Katalog zu sein und werden ein **laufendes Sonnensystem**.
