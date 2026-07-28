# Kapitel 23: Die Mathematik dieses Universums
### Jenseits der Transformer-Struktur

## Warum dieses Kapitel

Die vorherigen Kapitel benennen *Teile* (Schichten, QKV, FFN)
und *Kräfte* (aufmerksamkeitsbezogene Schwerkraft, FFN-Klima, Umlaufbahnen).
Hier schreiben wir die **Gleichungen**, die sie präzise machen:
kein Deep-Learning-Kurs, sondern das **formale Minimum**
des Mikrokosmos TinyLlama-1.1B, so wie wir ihn in Dreaming
messen und bewegen.

Konventionen:

- Spaltenvektoren, sofern nicht anders angegeben.
- \(\langle u,v\rangle = u^\top v\), \(\|u\|=\sqrt{\langle u,u\rangle}\).
- \(\mathrm{softmax}(z)_i = e^{z_i}/\sum_j e^{z_j}\).

---

## 1. Die Räume als mathematische Objekte

| Symbol | Raum | Dimension | Rolle |
|--------|------|-----------|-------|
| \(\mathcal{V}\) | Vokabular | \(V=32000\) | endliche Menge von IDs |
| \(E\) | Embeddings | \(E\in\mathbb{R}^{V\times d}\), \(d=2048\) | Zeilen \(e_t\in\mathbb{R}^d\) |
| \(x_\ell^{(t)}\) | Residual | \(\mathbb{R}^d\) | Zustand in Schicht \(\ell\), Position \(t\) |
| \(W_{q,\ell}\) usw. | Schichtgewichte | Matrizen in \(\mathbb{R}^{\cdot\times\cdot}\) | Lineare Operatoren (+ Nichtlinearitäten) |
| \(\theta\) | Gesamtgewichte | \(\theta\in\mathbb{R}^N\), \(N\sim 1.1\cdot 10^9\) | „Universumspunkt" des Modells |
| \(\mathcal{C}\subset\mathbb{R}^N\) | Kohärenz-Oberfläche | Untermenge (keine bewiesene Untermannigfaltigkeit) | Modelle, die lesbaren Text erzeugen |

Der Forward ist eine Komposition

\[
f_\theta : \mathcal{V}^{T} \to \mathbb{R}^{V}
\]

die einer Token-Sequenz die Logits des letzten
(oder jeder Position, je nach Modus) zuordnet.

Die Generierung ist die Iteration

\[
t_{n+1}\sim \pi\big(f_\theta(t_1,\ldots,t_n)\big),
\]

wobei \(\pi\) ein Sampling ist (Temperatur, top-k).

---

## 2. Geometrie des Token-Himmels

### 2.1 Ähnlichkeit

Für \(u,v\in\mathbb{R}^d\),

\[
\cos(u,v)=\frac{\langle u,v\rangle}{\|u\|\,\|v\|}\in[-1,1].
\]

**Empirischer Befund (TinyLlama):** „entgegengesetzte" Sprachpaare
(`love`/`hate`, `life`/`death`) haben \(\cos\approx 0\), nicht \(-1\).
In diesem Himmel ist sprachlicher Gegensatz **keine** Antipodalität.

### 2.2 Zentroide und Archetypen

Gegeben eine Samenmenge \(S=\{t_1,\ldots,t_m\}\subset\mathcal{V}\),

\[
c_S=\frac{1}{m}\sum_{i=1}^m e_{t_i},\qquad
\hat c_S=\frac{c_S}{\|c_S\|}.
\]

Ein **Archetyp** (Kap. 21) ist ein \(\hat c_S\) mit festgelegter
kultureller Semantik. Eine **semantische Insel** (Kap. 16) ist dasselbe
mit einer anderen Wahl von \(S\).

**Ausrichtung zwischen Archetypen:**

\[
A(S,S')=\cos(\hat c_S,\hat c_{S'}).
\]

Gemessen: \(A(\mathrm{Zauberer},\mathrm{Mystischer})\approx 0.39\),
\(A(\mathrm{Weiser},\mathrm{Akademischer})\approx 0.29\).

### 2.3 Kontrast-Richtung

\[
\delta_{S|S'}=\frac{c_S-c_{S'}}{\|c_S-c_{S'}\|}.
\]

Z. B. Positive Emotion minus negative → Pol `smile/happy`
vs `sad/anger` beim Ranking von \(\cos(e_t,\delta)\).

### 2.4 PCA des Himmels (globale Struktur)

Stichprobe \(X\in\mathbb{R}^{n\times d}\) zentrierter Zeilen von \(E\).
SVD \(X=U\Sigma V^\top\). Varianzanteil in den ersten \(k\)
Komponenten:

\[
\mathrm{EVR}(k)=\frac{\sum_{i=1}^k \sigma_i^2}{\sum_{i=1}^{d}\sigma_i^2}.
\]

**Empirischer Befund:** \(\mathrm{EVR}(10)\approx 2.3\%\),
Dims für 50% / 90% Var. \(\approx 481\) / \(1329\).
Der Himmel **nutzt** Hunderte Richtungen; er kollabiert nicht auf 2D.
Die HTML-Karten sind Projektionen

\[
\mathbb{R}^d\ni e \mapsto V_{:2}^\top (e-\bar e)\in\mathbb{R}^2,
\]

nützlich und zugleich falsch (Kap. 6).

### 2.5 Anisotropie

\[
\alpha=\frac{\|\bar e\|}{\frac{1}{V}\sum_t \|e_t\|},\quad
\bar e=\frac{1}{V}\sum_t e_t.
\]

Gemessen: \(\alpha\approx 0.006\) — fast isotrop in mittlerer Norm.
Die Struktur ist **richtungsabhängig**, nicht „Sterne viel massereicher".

---

## 3. Ein Schichtschritt als dynamisches System

Sei \(x\in\mathbb{R}^d\) der Residual an der aktuellen Position
(wobei der Positionsindex weggelassen wird, wenn er nicht gebraucht wird).

### 3.1 RMSNorm

\[
\mathrm{RMSNorm}(x;w)=
\frac{x}{\sqrt{\frac{1}{d}\|x\|^2+\varepsilon}}\odot w,
\]

mit \(w\in\mathbb{R}^d\), \(\varepsilon>0\) (z. B. \(10^{-5}\)).

Es zieht keinen Mittelwert ab (im Gegensatz zu LayerNorm). Es ist eine
**Projektion auf die Kugel** (ungefähr) gefolgt von einem
koordinatenweisen Skalieren.

### 3.2 Aufmerksamkeit (Kopf \(h\), GQA)

\[
Q=x W_Q,\quad
K=x W_K,\quad
V=x W_V
\]

(in der Praxis: Matrizen, die pro Schicht geteilt werden; K,V von
Dimension \(n_{kv}\cdot d_h\) mit \(n_{kv}=4\), \(d_h=64\),
\(n_q=32\)).

Für den Kopf \(h\) mit KV-Index \(h_{kv}=\lfloor h\cdot n_{kv}/n_q\rfloor\),

\[
a_{h,t}
=\mathrm{softmax}_{t'\le t}
\Bigg(\frac{\langle q_h^{(t)}, k_{h_{kv}}^{(t')}\rangle}{\sqrt{d_h}}\Bigg),
\quad
o_h^{(t)}=\sum_{t'\le t} a_{h,t'}\, v_{h_{kv}}^{(t')}.
\]

Die Maske \(t'\le t\) ist die **Kausalität**: Die Zukunft
hat unendliches Potenzial (Wahrscheinlichkeit 0).

Multi-Head-Ausgabe:

\[
\mathrm{Attn}(x)=\mathrm{Concat}(o_1,\ldots,o_{n_q})\,W_O.
\]

**Interpretation:** \(a_{h,t'}\) ist ein **Kopplungskern**
(nicht symmetrisch, nicht translationsinvariant) zwischen Positionen.
Die „Schwerkraft" aus Kap. 7 ist dieser normalisierte Kernel.

### 3.3 SwiGLU (FFN)

\[
\mathrm{SiLU}(z)=z\cdot\sigma(z),\quad
\sigma(z)=\frac{1}{1+e^{-z}},
\]

\[
\mathrm{FFN}(x)=
W_d\Big(\mathrm{SiLU}(W_g x)\odot (W_u x)\Big),
\]

mit \(W_g,W_u\in\mathbb{R}^{d_{ff}\times d}\), \(d_{ff}=5632\),
\(W_d\in\mathbb{R}^{d\times d_{ff}}\).

Es ist eine **Punkt-zu-Punkt**-Karte in der Sequenz: Es mischt \(t\) nicht.
Deshalb ist es „lokale Physik" gegenüber der nicht-lokalen Aufmerksamkeit.

### 3.4 Residual-Block

\[
\begin{aligned}
x &\leftarrow x + \mathrm{Attn}(\mathrm{RMSNorm}(x;w_a)),\\
x &\leftarrow x + \mathrm{FFN}(\mathrm{RMSNorm}(x;w_f)).
\end{aligned}
\]

Abstrakte Form eines Euler-Schritts in einem Feld \(F_\ell\):

\[
x_{\ell+1}=x_\ell + F_\ell(x_\ell;\theta_\ell).
\]

Die **Umlaufbahn** aus Kap. 20 ist die Trajektorie
\(\{x_\ell\}_{\ell=0}^{22}\) in \(\mathbb{R}^d\).

### 3.5 Warum der Residual wichtig ist (Mathematik der Kohärenz)

Wäre es \(x_{\ell+1}=F_\ell(x_\ell)\) ohne Skip,
würden kleine Änderungen in \(F_\ell\) exponentiell instabil
zusammensetzen (Produkt der Jacobis).

Mit Skip ist die Jacobi-Matrix

\[
D x_{\ell+1}= I + DF_\ell,
\]

und wenn \(\|DF_\ell\|\) moderat ist, ähnelt die Dynamik einer
**Störung der Identität**: Die Trajektorien bleiben
in der Nähe einer „semantischen Grundlinie".
Das ist die analytische Version von „zur Seite fallen".

---

## 4. Vom Residual zum Ereignis: Logits und Sampling

### 4.1 Sprachkopf

\[
z = E_{\mathrm{out}}\,\mathrm{RMSNorm}(x_L)\in\mathbb{R}^V
\]

(im GGUF: `output.weight`; manchmal am Embedding gebunden).

### 4.2 Temperatur

\[
\pi_T(t)=\mathrm{softmax}(z/T)_t
=\frac{e^{z_t/T}}{\sum_{j} e^{z_j/T}}.
\]

- \(T\to 0\): Masse auf \(\arg\max z\) („kreisförmige" Umlaufbahn, greedy).  
- \(T\to\infty\): Gleichverteilung zu (maximale Entropie).

Entropie \(H(\pi_T)=-\sum_t \pi_T(t)\log\pi_T(t)\) wächst mit \(T\).

### 4.3 Top-k

Sei \(S_k\subset\mathcal{V}\) die \(k\) Indizes mit größtem \(z_t\).
Es wird umdefiniert

\[
\pi(t)\propto e^{z_t/T}\mathbf{1}_{t\in S_k}.
\]

Es ist eine **Beschneidung des Trägers**: Der Horizont erlaubter
Ereignisse wird auf \(k\) Schicksale reduziert.

### 4.4 Generierung als Markov-Kette

Bedingt auf \(\theta\) und die Politik \(\pi\),

\[
\mathbb{P}(t_{1:N})=\prod_{n=0}^{N-1}
\pi\big(t_{n+1}\mid t_{1:n};\theta\big).
\]

\(\theta\) ändern (Perturbation) ändert die Familie der
Ketten; nicht nur ein einzelnes Sample.

---

## 5. Gewichtsraum und Kohärenz-Oberfläche

### 5.1 Das Modell als Punkt

\[
\theta\in\mathbb{R}^N,\qquad N\approx 1.1\times 10^9.
\]

Fast das gesamte Volumen von \(\mathbb{R}^N\) erzeugt Müll.
Wir definieren operativ (nicht topologisch streng)

\[
\mathcal{C}=\{\theta : \text{der Text von }f_\theta\text{ ist in einer Prompt-Batterie kohärent}\}.
\]

Dreaming untersucht **Bewegungen** \(\theta\mapsto\theta'\)
die in \(\mathcal{C}\) bleiben, aber die
Stilverteilung (Perspektiven) ändern.

### 5.2 amplify_subspace (mystical)

Über einen ausgebreiteten Tensor \(w\in\mathbb{R}^m\)
(oder Konkatenation von Schicht-Tensoren):

1. Sample \(v\sim\mathcal{N}(0,I_m)\), normalisiere \(\hat v=v/\|v\|\).  
2. Skalarprojektion \(p=\langle w,\hat v\rangle\).  
3. Aktualisiere

\[
w' = w + \varepsilon\, p\, \hat v
= w + \varepsilon\, ( \hat v\hat v^\top ) w.
\]

Es ist ein **Rank-1-Update** in zufälliger Richtung:

\[
w'=(I+\varepsilon\, \hat v\hat v^\top)w.
\]

Eigenwert \(1+\varepsilon\) in Richtung \(\hat v\),
\(1\) in der orthogonalen Hyperfläche.

**Lesart:** Eine Komponente wird verstärkt; die Struktur
im Rest bleibt erhalten → Kandidat für eine **tangentiale**
Bewegung zu \(\mathcal{C}\) wenn \(\varepsilon\) moderat ist.

### 5.3 Skaliertes Rauschen (Gegenbeispiel)

\[
w'_i = w_i + \varepsilon\, \xi_i\, |w_i|,\quad \xi_i\sim\mathcal{N}(0,1).
\]

Es bricht Korrelationen zwischen Koordinaten: ein
**generischer** Schub, kein Rank-1-ausgerichteter. Empirisch,
für großes \(\varepsilon\), \(\theta'\notin\mathcal{C}\).

### 5.4 Intensität und „Sweet Spot"

In Q4_0 (Python-Pipeline v10) war \(\varepsilon\approx 0.10\)
der Sweet Spot vieler Techniken.
Im F32 der C-Engine-Laufzeit braucht man manchmal
\(\varepsilon\in[0.3,0.5]\), um Textdivergenz
in kurzen Prompts zu sehen (Quantisierung und EOS interagieren).

Es gibt kein einziges universelles \(\varepsilon^*\): Es hängt
vom numerischen Format und der Batterie ab.

### 5.5 Perspektiven-Formel (operativ)

\[
\theta_{\mathrm{persp}}
= \theta_0 + \sum_r \varepsilon_r\, \Delta_r,
\]

mit kleinen \(\|\varepsilon_r\Delta_r\|\) und jedem \(\Delta_r\),
der die Hierarchie bewahrt (lowrank, amplify, normrot, …).
Interpolation nahe \(\mathcal{C}\) tendiert dazu,
in \(\mathcal{C}\) zu bleiben (empirische Hypothese des Projekts).

---

## 6. Steering: Projektion auf den Residual

Gegeben ein Wort mit Embedding \(e_\star\),

\[
\hat u=\frac{e_\star}{\|e_\star\|},\qquad
x \leftarrow x + \lambda\, \langle x,\hat u\rangle\, \hat u
= \big(I+\lambda\,\hat u\hat u^\top\big)x.
\]

Gleiche Rank-1-Algebra wie amplify, aber angewendet auf den
**Zustand** \(x\), nicht auf die Gewichte \(\theta\).

- \(\lambda\): Stärke des „Windes".  
- Es schreibt das Universum nicht um; es lenkt die Umlaufbahn im Flug.

---

## 7. Quantisierung (das Universum in Ganzzahlen)

In Q4_0, Blöcke von 32 Gewichten:

\[
w_i \approx s\cdot (q_i-8),\quad q_i\in\{0,\ldots,15\},
\]

\(s\) in float16 pro Block. Die Python-Perturbation
operiert in \(\mathbb{R}\) nach Dequant und kehrt zurück zu
\((s,q\). Das führt einen **Projektionsfehler**
auf ein diskretes Netz ein: ein weiterer Grund, warum
das „optimale" \(\varepsilon\) nicht mit reinem F32 übereinstimmt.

Die C-Engine in F16 dequantisiert im Matmul über eine Tabelle
mit \(2^{16}\) Einträgen: exakter Isomorphismus half→float
bei jedem gelesenen Gewicht, ohne Re-Quantisierung in der Baseline.

---

## 8. Komplexität eines Schritts (effiziente Umlaufbahn)

Für eine Sequenz der Länge \(T\) ist ein naiver Aufmerksamkeitsschritt
in einer Schicht \(\mathcal{O}(T^2 d)\), wenn alles neu berechnet wird;
mit **KV-Cache** beim Generieren des Tokens \(T\):

\[
\mathcal{O}(T\cdot d\cdot d_h\cdot n_q)
\quad\text{(neue Scores gegen }T\text{ Schlüssel)}
\]

plus Matmuls \(\mathcal{O}(d\cdot d_{ff})\) des FFN.
Deshalb kreist die aktuelle Engine mit ~6–10 tok/s auf CPU
und fiel ohne Cache auf ~0.03 tok/s.

---

## 9. Wörterbuch: Physik des Buches ↔ Formel

| Atlas-Sprache | Mathematisches Objekt |
|---------------|----------------------|
| Sternenhimmel | Zeilen von \(E\in\mathbb{R}^{V\times d}\) |
| Insel / Sternbild | \(c_S\), Samen \(S\) |
| Schwerkraft zwischen Tokens | \(a_{t,t'}=\mathrm{softmax}(QK^\top/\sqrt{d_h})\) |
| Lokales Klima | \(\mathrm{FFN}:\mathbb{R}^d\to\mathbb{R}^d\) |
| Orbitale Trägheit | \(x\mapsto x+F(x)\) |
| Atembare Luft | \(\mathrm{RMSNorm}\) |
| Kollaps zum Ereignis | \(\pi_T=\mathrm{softmax}(z/T)\) (top-k) |
| Bewohnbare Oberfläche | \(\mathcal{C}\subset\mathbb{R}^N\) |
| Mystische Linse | \(w'=(I+\varepsilon\hat v\hat v^\top)w\) |
| Steering-Wind | \(x'=(I+\lambda\hat u\hat u^\top)x\) |
| Planetariumskarte | PCA: \(e\mapsto V_{:2}^\top(e-\bar e)\) |

---

## 10. Was die Mathematik *noch nicht* behauptet

1. \(\mathcal{C}\) ist nicht als Differentialmannigfaltigkeit bewiesen;
   es ist eine durch Tests definierte **operative** Region.  
2. „Tangential zu \(\mathcal{C}\)" ist eine geometrische Metapher,
   gestützt durch Rank-1 / SVD / Orthogonalität zu destruktiven
   Modi — kein Riemannsches Geometrie-Theorem des Loss.  
3. Die Archetypen sind keine eindeutigen latenten Faktoren;
   es sind **gewählte Richtungen** mit menschlichen Samen.  
4. Die Goldene Regel ist in TinyLlama **empirisch**;
   sie leitet sich hier nicht von einem allgemeinen Variationsprinzip ab.

Die Ehrlichkeit des Mikrokosmos umfasst die Kante
des Unbewiesenen.

---

## 11. In einem Satz

Jenseits der Blockdiagramme des Transformers ist dieses
Universum: **Geometrie von \(E\) und \(\theta\)**,
**Residual-Dynamik** \(x\leftarrow x+F_\ell(x)\),
**kausale Aufmerksamkeitskerne**, **SwiGLU-Karten**,
**Softmax-Kollaps** und **Rank-1-Updates**, die
Umlaufbahnen oder Gewichte verschieben, ohne (manchmal)
die Region zu verlassen, in der Sprache noch möglich ist.

---

*Nächstes Kapitel: Das LLM — Ein Spiegel, in dem wir uns betrachten.*