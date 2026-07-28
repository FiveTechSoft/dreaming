# Kapitel 29: Die Reise eines Prompt innerhalb von TinyLlama

## Für Wen Dieses Kapitel Gedacht Ist

Wenn du gelesen hast, dass ein LLM „das nächste Token vorhersagt"
aber du den Weg noch nicht *siehst*, ist dieses Kapitel die vollständige
Landkarte.

Wir werden ihn **Schritt für Schritt** mit einem echten Prompt durchlaufen,
ohne einen magischen Sprung vorauszusetzen. Am Ende
solltest du in der Lage sein, laut zu erzählen, was mit
jeder Zahl passiert, von dem Moment an, in dem du einen Satz schreibst, bis
das erste Wort der Antwort erscheint.

**Beispiel-Prompt (im gesamten Kapitel fest):**

```text
The secret to happiness is
```

**Modell:** TinyLlama-1.1B
**Schlüsselzahlen, die sich nicht ändern:**

| Parameter | Wert |
|-----------|-----:|
| Transformer-Schichten | 22 (Indizes 0…21) |
| Residuelle Dimension \(d\) | 2048 |
| Vokabular \(V\) | 32.000 |
| Q / KV-Köpfe | 32 / 4 (GQA) |
| Dimension pro Kopf | 64 |
| FFN Zwischenschicht | 5632 |
| Maximaler Kontext | 2048 Positionen |
| RoPE Basis | 10.000 |

---

## 0. Der Film in einer Minute

Vor den Details, der Trailer:

```
1.  TEXT         "The secret to happiness is"
2.  TOKENS       Ganzzahlige ids des BPE
3.  EMBEDDINGS   jedes id → Vektor von 2048 Floats
4.  PREFILL      jedes Prompt-Token durchläuft 22 Schichten
                 und füllt den KV-Cache
5.  LOGITS       32.000 Bewertungen für das *nächste* Token
6.  SAMPLE       wir wählen ein id (Temperatur, top-k, seed)
7.  DECODE       das id wird wieder lesbarer Text
8.  SCHLEIFE     dieses Token geht wieder ins Modell…
                 bis max_new tokens oder EOS
```

Alles andere in diesem Kapitel ist ein **Zoom** auf
jeden Pfeil.

---

## 1. Der Prompt ist für das Modell kein „Satz"

### 1.1 Was Du Siehst

Eine UTF-8-Zeichenfolge mit Leerzeichen und Bedeutung.

### 1.2 Was Das Modell Sieht

Eine **geordnete Folge von Ganzzahlen** zwischen 0 und 31.999.

Die Brücke heißt **BPE-Tokenizer** (Byte Pair Encoding),
derselbe LLaMA-Stil: Wörter beginnen normalerweise mit
dem Wort-Leerzeichen-Präfix `▁` (U+2581).

Für unseren Prompt ist die Idee (didaktisches Schema):

| Position \(t\) | Stück (Idee) | Rolle im Satz |
|---------------:|--------------|---------------|
| 0 | `▁The` | Artikel / Start |
| 1 | `▁secret` | Nominalkern |
| 2 | `▁to` | Verbindung |
| 3 | `▁happiness` | Objekt des Geheimnisses |
| 4 | `▁is` | Kopulaverb — **das Präsens des Prädikats** |

> In der Praxis spaltet BPE manchmal feiner
> (`happ` + `iness`, usw.). Das Prinzip ändert sich nicht:
> **Text → Liste von ids**. Nennen wir diese Liste
>
> \[
> (t_0, t_1, t_2, t_3, t_4)
> \]
>
> mit Länge \(T_{\mathrm{prompt}} = 5\).

### 1.3 Warum Die Reihenfolge Wichtig Ist

TinyLlama ist **kausal**: An Position \(t\) kann es
nur die Positionen \(0,1,\ldots,t\) „sehen".
Die Vergangenheit existiert; die Zukunft des Satzes **noch nicht**.

Das ist die Verkehrsregel der gesamten Reise.

---

## 2. Vom ID zum Vektor: Geboren in ℝ²⁰⁴⁸

Jedes id \(t_i\) wird zu einem Punkt im Himmel
der Embeddings (Kap. 28):

\[
x^{(i)}_{0} \;=\; e_{t_i} \;=\; \mathrm{Embedding}(t_i) \;\in\; \mathbb{R}^{2048}
\]

- Es gibt eine `token_embd`-Tabelle mit logischer Form
  **[32.000 × 2048]**.
- Die Zeile Nummer \(t_i\) ist der Vektor dieses Sterns.
- Hier gibt es **noch keine Schichten**. Nur Katalog.

**Didaktisches Bild:**
fünf Passagiere betreten die Empfangshalle des Gebäudes
(Kap. 27). Jeder bringt seinen Koffer mit 2048 Zahlen mit.
Diese Koffer heißen **Residuen**.

In der Notation dieses Kapitels:

- Hochgestellt \((i)\): Position in der Sequenz.
- Tiefgestellt \(\ell\): Schicht (0 vor der ersten Schicht;
  nach Schicht 21 werden wir im „Dachgeschoss" sein).

Beim Verlassen des Embeddings:

\[
x^{(0)}_{0},\; x^{(1)}_{0},\; \ldots,\; x^{(4)}_{0}
\in \mathbb{R}^{2048}
\]

---

## 3. Zwei Flugphasen: Prefill und Generierung

TinyLlama (und fast jeder kausale Transformer) verarbeitet
den Prompt nicht mit einem einzigen magischen Schlag.
Es gibt **zwei Modi**:

| Phase | Was Hereinkommt | Was Herauskommt | KV-Cache |
|-------|----------------|-----------------|----------|
| **Prefill** | Jedes Token des Prompt, in Reihenfolge | Logits nach dem **letzten** Prompt-Token | Wird **gefüllt** |
| **Generierung** | Ein neues Token pro Mal | Logits für das nächste | Wird um +1 **länger** |

Im C-Engine (`llm_inference.c`):

```c
/* PREFILL */
for (i = 0; i < n_prompt; i++)
    model_forward_token(&model, &state, tokens[i]);

/* GENERIERUNG */
for (step = 0; step < max_new; step++) {
    next = sample_top_k(state.logits, …);
    if (next == EOS) break;
    emit(next);                          /* Text an Benutzer */
    model_forward_token(&model, &state, next);
}
```

Bis zum Ende des Prefill haben wir noch nicht „geantwortet".
Wir haben nur den **Prompt verstanden** und Speicher
im Cache hinterlassen.

---

## 4. Ein Einzelnes Token in Einzelner Schicht (Der Kern)

Nimm die aktuelle Position \(p\) (z.B. das letzte
Prompt-Token, \(p=4\), `▁is`).
Sein Residuum beim Erreichen der Schicht \(\ell\) ist \(x\).

In der Schicht passieren **immer** diese sieben
Stationen, in dieser Reihenfolge:

```
        x  (Residuum, das Schicht ℓ erreicht)
        │
        ▼
   [1] RMSNorm  (attn_norm)
        │
        ▼
   [2]  Q, K, V  +  RoPE
        │
        ▼
   [3]  Kausale Aufmerksamkeit  (verwendet KV-Cache dieser Schicht)
        │
        ▼
   [4]  O-Projektion  →  Residuum:  x ← x + Attn
        │
        ▼
   [5] RMSNorm  (ffn_norm)
        │
        ▼
   [6]  FFN SwiGLU  (gate, up, down)
        │
        ▼
   [7]  Residuum:  x ← x + FFN
        │
        ▼
        x  (verlässt Richtung Schicht ℓ+1)
```

Wiederhole das **22-mal**. Das ist der vollständige Aufzug
für **ein** Token in **einem** Forward-Schritt.

---

## 5. Station für Station (Mit Dem Beispiel)

Wir verfolgen den Passagier der Position \(p=4\) (`▁is`),
in einer generischen Schicht \(\ell\), wenn die Positionen \(0..3\) des Prompt
bereits im Cache existieren.

### Station 1 — RMSNorm (Aufmerksamkeit)

\[
h = \mathrm{RMSNorm}(x;\; \gamma_{\ell}^{\mathrm{attn}})
\]

- Sie „versteht" den Satz nicht.
- Sie **stabilisiert** die Vektorskala, damit
  Q und K nicht explodieren.
- Lächerliche Parametermasse (~0.01% des Modells),
  enorme Rolle (Kap. 7, Kraft V).

**Analogie:** den Kompass kalibrieren, bevor man
die anderen Sterne der Sequenz betrachtet.

### Station 2 — Q, K, V Entstehen und Position (RoPE)

\[
Q = W_Q h,\quad K = W_K h,\quad V = W_V h
\]

In TinyLlama sind die logischen Formen pro Schicht:

| Tensor | Logische Form | Menschliche Lesung |
|--------|--------------|--------------------|
| \(W_Q\) | [2048, 2048] | 32 Köpfe × 64 Dims |
| \(W_K, W_V\) | [256, 2048] | **4** KV-Köpfe × 64 (GQA) |
| \(W_O\) | [2048, 2048] | Vereinigt die 32 Köpfe |

**GQA (Grouped Query Attention):**
jeder Key/Value-Kopf wird von **8** Q-Köpfen **geteilt**
(\(32/4 = 8\)). Weniger Cache-Speicher, dieselbe Idee:
reiche Fragen, geteilter Speicher.

**RoPE (Rotary Position Embedding):**
vor dem Attention werden Q und K gemäß Position \(p\) **gedreht**.
Es gibt keinen separaten „Position 4"-Vektor: Die Position
ist in den **Winkel** von Q und K eingewickelt.

So unterscheidet das Modell:

```text
secret to happiness   ≠   happiness to secret
```

obwohl dieselben „Sterne" im Vokabular sind.

### Station 3 — Aufmerksamkeit: Schwerkraft Zwischen Tokens

Für jeden Query-Kopf:

\[
\mathrm{score}_{p,j}
  = \frac{q_p \cdot k_j}{\sqrt{64}},
  \qquad j = 0,1,\ldots,p
\]

\[
\alpha_{p,j} = \mathrm{softmax}_j(\mathrm{score}_{p,j})
\]

\[
z_p = \sum_{j=0}^{p} \alpha_{p,j}\, v_j
\]

**Lesung mit unserem Prompt** (Intuition, keine gemessene
Aufmerksamkeitskarte hier):

| \(j\) | Token | Was `▁is` „ziehen" könnte |
|------:|-------|----------------------------|
| 0 | The | wenig (grammatische Funktion) |
| 1 | secret | Thema: Es gibt ein Geheimnis |
| 2 | to | Verbindung |
| 3 | happiness | **Inhalt** des Geheimnisses |
| 4 | is | es selbst (Selbst-Aufmerksamkeit) |

Die \(\alpha_{p,j}\) sind die **dynamische Schwerkraft**
(Kap. 7 und 28): Wie sehr das Residuum von `is` zu
jedem Stern der Vergangenheit *dieses* Satzes fällt.

**Kausale Maske:** \(j > p\) ist verboten.
Im Prefill, wenn wir Position 2 verarbeiten,
existiert `happiness` **noch nicht** im Cache.

### Station 4 — Köpfe-Mischen + Aufmerksamkeits-Residuum

Die 32 Köpfe werden konkateniert (oder projiziert) und
gehen durch \(W_O\):

\[
x \leftarrow x + O(z)
\]

Das Residuum wird **nicht gelöscht**: Der Aufmerksamkeits-Schub
wird **hinzugefügt**. Deshalb sprechen wir von Bahn, nicht von
Teleportation (Kap. 20).

\[
x_{\mathrm{nachher}} = x_{\mathrm{vorher}} + \Delta_{\mathrm{attn}}
\]

### Station 5 — RMSNorm (FFN)

Eine weitere Kalibrierung, mit einem anderen \(\gamma_{\ell}^{\mathrm{ffn}}\).

### Station 6 — FFN SwiGLU (Die „Sonne" der Parameter)

Hier lebt ~**69%** der Modellmasse:

\[
\begin{aligned}
u &= W_{\mathrm{up}} h \\
g &= W_{\mathrm{gate}} h \\
\mathrm{FFN}(h) &= W_{\mathrm{down}}\big(\mathrm{SiLU}(g)\odot u\big)
\end{aligned}
\]

- Es erweitert sich auf **5632** Dimensionen.
- Das *Gate* entscheidet, welche Kanäle durchgelassen werden.
- Es komprimiert wieder auf 2048.

**Analogie:** Die Aufmerksamkeit betrachtet **andere Tokens**;
das FFN transformiert **dieses** Residuum allein —
lokales Wetter, „praktisches" Wissen der Position
(Goldene Regel: FFN → praktische Linse, Kap. 9).

### Station 7 — FFN-Residuum

\[
x \leftarrow x + \mathrm{FFN}(h)
\]

Es verlässt Schicht \(\ell\) bereit für \(\ell+1\).

---

## 6. Der KV-Cache: Speicher der Vergangenheit

Ohne Cache müsstest du für jedes neue Token
K und V für den gesamten Satz **neu berechnen**. Unmöglich
auf CPU mit gutem Tempo.

Mit Cache, in Schicht \(\ell\):

```
cache_K[ℓ][0 .. p]   bereits gespeichert
cache_V[ℓ][0 .. p]

Beim Verarbeiten von Position p:
  nur K_p, V_p berechnen
  cache_K[ℓ][p], cache_V[ℓ][p] schreiben
  Q_p gegen cache_K[ℓ][0..p] attentionieren
```

**Prefill unseres Prompts:**

| Schritt | Token, das Hereinkommt | Positionen im Cache am Ende |
|--------:|-----------------------|----------------------------|
| 1 | The | 0 |
| 2 | secret | 0–1 |
| 3 | to | 0–2 |
| 4 | happiness | 0–3 |
| 5 | is | 0–4 |

Nach Schritt 5 haben alle **22 Schichten** K und V für
die fünf Positionen. Das Residuum von `is` ist das gesamte
Gebäude hinaufgestiegen. Dort entstehen die **Logits** des
ersten *Antwort*-Tokens.

---

## 7. Vom Dach zum Vokabular: Logits

Nach Schicht 21:

\[
h = \mathrm{RMSNorm}(x;\; \gamma^{\mathrm{out}})
\]

\[
\mathrm{logits} = W_{\mathrm{out}}\, h \;\in\; \mathbb{R}^{32000}
\]

- `output.weight` hat die logische Form **[32.000 × 2048]**
  (manchmal mit dem Embedding geteilt oder gebunden in anderen
  Modellen; im GGUF von TinyLlama ist es der `lm_head`).
- Jeder Eintrag \(z_k\) ist „wie sehr das Modell drängt,
  das Token mit id \(k\)" **jetzt** zu wählen.

Es gibt immer noch **kein** Wort. Es gibt eine Rangliste von 32.000
Kandidaten.

---

## 8. Sample: Den Himmel zu Einem Stern Kollabieren

**Kraft VI** (Kap. 7): Vom Kontinuum zum Ereignis.

Typisches Verfahren im Dreaming-Engine:

1. **Temperatur** \(T\): \(z_k \leftarrow z_k / T\).
   - \(T \to 0\): fast immer das Maximum (geizig).
   - \(T\) hoch: mehr Zufall, mehr Vielfalt.
2. **Top-k**: Nur die \(k\) höchsten Logits behalten
   (z.B. 40). Der Rest wird ignoriert.
3. **Softmax** nur über diese \(k\):

\[
\pi_i = \frac{e^{z_i}}{\sum_{j\in\mathrm{top\text{-}k}} e^{z_j}}
\]

4. **Abtasten** eines id gemäß \(\pi\) (mit `--seed`, um
   dieselbe Reise zu reproduzieren).

Nehmen wir an (ausgedachtes, aber realistisches Beispiel), das Ergebnis ist:

```text
id →  ▁being      oder      ▁love      oder      ▁not ...
```

Dieses id wird als Text **dekodiert** und dem Benutzer angezeigt.
Das ist der erste Schritt der Antwort.

---

## 9. Die Autoregressive Schleife (Die Antwort Wächst)

Das gewählte Token ist **nicht das Ende des Modells**.
Es ist der **nächste Passagier**:

```
prompt:     The secret to happiness is
+ sample:   being
neue seq:   The secret to happiness is being
```

`model_forward_token` wird **nur** mit `being` erneut aufgerufen:

- Sein Embedding wird berechnet.
- Es durchläuft 22 Schichten.
- Es schreibt K,V an Position \(p=5\) jeder Schicht.
- Es attentioniert `The…is` + `being`.
- Es erzeugt Logits für das **noch neuere** Token.

Und so weiter:

```
The secret to happiness is being
The secret to happiness is being kind
The secret to happiness is being kind to
...
```

bis:

- `max_new` Tokens erreicht werden, oder
- **EOS** abgetastet wird (Ende der Sequenz).

**Schlüsselidee:**
einen Absatz zu generieren, sind **viele** Wiederholungen der
Reise **eines** Tokens, kein einzelner Durchgang „vom vollständigen
Satz zur vollständigen Antwort".

---

## 10. Meisterschema der Reise

```
┌─────────────────────────────────────────────────────────┐
│  MENSCH: "The secret to happiness is"                   │
└───────────────────────────┬─────────────────────────────┘
                            │ BPE-Tokenizer
                            ▼
┌─────────────────────────────────────────────────────────┐
│  IDS:  t0 t1 t2 t3 t4                                   │
└───────────────────────────┬─────────────────────────────┘
                            │ Zeilen von token_embd
                            ▼
┌─────────────────────────────────────────────────────────┐
│  VEKTOREN:  x0..x4  ∈ ℝ²⁰⁴⁸                            │
└───────────────────────────┬─────────────────────────────┘
                            │ PREFILL (für jedes ti)
                            ▼
        ┌───────────────────────────────────────┐
        │  für Position p = 0 .. 4:            │
        │    für Schicht ℓ = 0 .. 21:          │
        │       Norm → Attn(+RoPE,GQA,cache)    │
        │            → +Residuum                │
        │       Norm → FFN SwiGLU               │
        │            → +Residuum                │
        └───────────────────┬───────────────────┘
                            │ nach letztem p des Prompt
                            ▼
┌─────────────────────────────────────────────────────────┐
│  output_norm → lm_head → logits[32000]                  │
└───────────────────────────┬─────────────────────────────┘
                            │ temp, top-k, softmax, sample
                            ▼
┌─────────────────────────────────────────────────────────┐
│  NEUES TOKEN → Text an Benutzer                         │
│       │                                                 │
│       └──── zurück zu forward_token (GENERIERUNG) ──► … │
└─────────────────────────────────────────────────────────┘
```

---

## 11. Mini-Labor: Die Reise mit dem C-Engine Sehen

Vom Repo-Root (Pfade an dein GGUF anpassen):

```bash
# Prefill + Generierung, fester Seed (reproduzierbar)
./llm_inference tinyllama-1.1b.F16.gguf \
  "The secret to happiness is" 40 0.7 40 --seed 42
```

| Flag / Arg | Rolle in der Reise |
|------------|-------------------|
| prompt | anfängliche Sterne der Sequenz |
| `40` (n) | wie viele neue Tokens umkreisen |
| `0.7` | Kollisions-Temperatur |
| `40` (top-k) | Breite der Kandidaten-Grube |
| `--seed 42` | gleicher Zufall → gleicher Weg |
| `--perturb mystical --intensity 0.35` | **verformt** Q/K/V/FFN: andere Physik, formaler gleicher Weg |
| `--steer happiness --steer-strength 0.15` | schiebt das Residuum in eine Richtung des Himmels |

Empfohlenes didaktisches Protokoll:

1. Gleicher Seed, `none` vs `mystical` → Ändert sich die Bahn?
2. Gleicher Seed, Temp 0.2 vs 0.9 → Ändert sich der Kollaps?
3. Öffne die [semantische Karte](https://fivetechsoft.github.io/dreaming/exploration/semantic_map.html),
   suche `▁happiness` / `▁love` und betrachte ihre **Kräfte**
   (statische Schwerkraft des Katalogs) während du
   die generierte Antwort liest (dynamische Schwerkraft des Prompt).

---

## 12. Häufige Geistige Fehler (Und Die Korrektur)

| Überzeugung | Realität in TinyLlama |
|-------------|----------------------|
| „Das Modell liest den Satz auf einen Blick" | Es liest **Token für Token**; der Prefill ist sequentiell |
| „Jede Schicht erfindet einen neuen Vektor" | Es aktualisiert **dasselbe** Residuum mit Additionen |
| „Aufmerksamkeit betrachtet das ganze Buch" | Nur die **Vergangenheit** von *dieser* Sequenz (bis 2048) |
| „32 Köpfe = 32 KV-Speicher" | Nur **4** KV-Gruppen (GQA); 32 Q-Blicke |
| „Das Embedding ist bereits die Antwort" | Das Embedding ist die **Geburt**; es fehlen 22 Stockwerke |
| „Softmax wählt das Wort des Prompt" | Es wählt das **nächste** Token aus dem Vokabular |
| „Eine Antwort = ein Forward" | Eine Antwort = **1 Prefill + N Forwards** |

---

## 13. Checkliste zum Vollständigen Verständnis

Wenn du auf alles mit Ja antworten kannst, ist die Reise verinnerlicht:

1. Was ist ein Token und warum ist es kein Zeichen?
2. Welche Dimension hat das Residuum und warum wird es bewahrt?
3. Was verbietet die kausale Maske wozu dient RoPE?
4. Was unterscheidet Aufmerksamkeit und FFN in einer Schicht?
5. Was speichert der KV-Cache und in welcher Phase wird er gefüllt?
6. Wie oft fährt der 22-stöckige Aufzug für ein 5-Token-Prompt im Prefill hoch?
   → **5 × 22** Schichtdurchläufe (einer pro Position).
7. Was ist ein Logit und wie wird es zu Text?
8. Warum erzeugen 40 Tokens ~40 zusätzliche Forwards?
9. Wo kommt eine Dreaming-Linse (`--perturb`) in dieses Bild?
   → In die Gewichte der Stationen 2–6, nicht im Tokenizer.

---

## 14. Brücken

| Thema | Kapitel |
|-------|---------|
| Dims und Tensoren pro Schicht | 2 |
| C-Engine, RoPE, Cache, Sample | 3 |
| Kräfte (attn, FFN, softmax…) | 7 |
| Wie man reist (Routen A–E) | 8 |
| Aufmerksamkeit im Detail | 10 |
| FFN im Detail | 11 |
| Frühe / mittlere / späte Schichten | 13–15 |
| Kette der Bedeutung (semantische Vision) | 26 |
| Aufzug pro Stockwerk | 27 |
| Sterne = Tokens, Aufmerksamkeit = Schwerkraft | 28 |
| Formeln | 23 |

---

## 15. Abschluss

Die Reise eines Prompt ist kein Mysterium: Sie ist eine
**wiederholbare Fabrik**.

1. Text wird zu **ids**.
2. Ids werden zu **Vektoren**.
3. Jeder Vektor steigt **22 Stockwerke** hoch mit
   norm → Aufmerksamkeits-Schwerkraft → FFN-Wetter,
   das nur mit der **Vergangenheit** spricht.
4. Das letzte Residuum wird auf **32.000** Bewertungen projiziert.
5. Ein Abtasten wählt **einen** Stern.
6. Dieser Stern wird in die Warteschlange gestellt und das Universum dreht sich wieder.

Wenn du schreibst

```text
The secret to happiness is
```

und TinyLlama antwortet, ist es nicht mehr „KI denkt einen Satz".
Es ist: *fünf Geburten, fünf Aufstiege des Gebäudes,
ein Kollaps, und dann N weitere Kollapsse* — immer dieselbe
Physik, ein Schritt weiter in der Zeit.

Das ist das vollständige Verständnis der Reise.
Der Rest des Buches (Perspektiven, Karten, Linsen)
sind **Variationen der Physik**, kein anderer Weg.
