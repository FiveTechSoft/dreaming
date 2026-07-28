# Kapitel 3: Unsere Inferenz-Engine in C für TinyLlama

## Warum eine eigene Engine schreiben?

Um TinyLlama zu studieren reicht es nicht, es zu *benutzen*.
Man muss es *funktionieren sehen*.

Die meisten Inferenz-Frameworks verstecken
den Weg des Tokens hinter Abstraktionsschichten:

- PyTorch und CUDA (Gigabytes an Abhängigkeiten)
- Python-Runtimes
- Undurchsichtige GPU-Kernels

Wir wollten das Gegenteil:

1. **Lesbarer C** — eine einzige Datei, explizite Operationen
2. **Keine GPU nötig** — alles auf CPU
3. **Keine Magie** — jeder Schritt des Transformers ist eine Funktion
4. **Portabel** — eine kleine Binärdatei, reproduzierbar

Das Ergebnis ist `llm_inference.c`: eine Inferenz-Engine,
die die Transformer-Schleife von der GGUF-Datei bis zum
nächsten Token implementiert — und später **Gewichtsperturbationen
zur Laufzeit** ohne vorgefertigtes GGUF.

Es ist nicht die schnellste Engine der Welt.
Es ist die Engine, die wir *Zeile für Zeile verstehen*
und mit der wir *die Gewichte bewusst berühren* können.

## Was eine Inferenz-Engine tun muss

Erinnere dich an den Fluss des vorherigen Kapitels:

```
Text → Tokens → Embeddings
     → 22 Schichten (Aufmerksamkeit + FFN)
     → Logits → nächstes Token
     → (wiederholen)
```

In der Praxis besteht die Engine aus fünf Teilen:

```
┌─────────────────────────────────────────┐
│            llm_inference.c              │
├─────────────────────────────────────────┤
│  1. GGUF-Leser                          │
│     - Magic, Metadaten, Indizes         │
│     - F16-Tensoren (und Fallback-Tokenizer│
│       von Q4_0 falls nötig)             │
├─────────────────────────────────────────┤
│  2. BPE-Tokenizer                       │
│     - Tokens + Merges aus dem GGUF selbst│
│     - Text ⇄ IDs                        │
├─────────────────────────────────────────┤
│  3. Transformer-Engine + KV-Cache       │
│     - Embedding, RMSNorm, RoPE          │
│     - GQA-Aufmerksamkeit, SwiGLU        │
│     - ein neues Token pro Schritt       │
├─────────────────────────────────────────┤
│  4. Sampling                            │
│     - Temperatur, top-k                 │
├─────────────────────────────────────────┤
│  5. Perturbation / Steering (optional)  │
│     - --perturb mystical, noise, …      │
│     - --steer <Wort> (Aktivierungen)    │
└─────────────────────────────────────────┘
```

| Block | Frage |
|-------|-------|
| GGUF | Wo sind die Gewichte auf der Festplatte? |
| Tokener | Wie wird Sprache in Zahlen umgewandelt? |
| Transformer | Wie wird die Repräsentation transformiert? |
| Sampling | Wie wird das nächste Wort gewählt? |
| Perturbation | Wie ändern wir die *Perspektive* des Modells? |

## TinyLlama in Zahlen (die die Engine verwendet)

| Parameter | Wert in TinyLlama-1.1B |
|-----------|-------------------------|
| Schichten (`block_count`) | 22 |
| Versteckte Dimension | 2048 |
| Q-/KV-Köpfe (GQA) | 32 / 4 |
| Dimension pro Kopf | 64 |
| FFN-Intermediate | 5632 |
| Vokabular | 32.000 |
| Modellkontext | 2048 |
| RoPE `freq_base` | 10.000 |
| Gewichtsformat der Engine | GGUF **F16** |

Neun Tensoren pro Schicht (Kap. 2) plus die globalen
`token_embd`, `output_norm`, `output`. Die Engine erfindet
nicht die Architektur: sie *materialisiert* sie im Speicher.

> **Hinweis zu Q4_0:** Die massiven Experimente des
> Projekts verwenden auch quantisierte GGUFs (~638 MB)
> mit `llama-cli`. Unser reiner C liest **F16**
> (~2.1 GB), um nicht alle Quantisierungskernels
> neu zu implementieren. Der *Forward* ist derselbe Transformer.

## Kompilieren und Ausführen

```bash
# Empfohlen (Windows/MinGW oder Linux)
gcc -O3 -march=native -ffast-math -fopenmp \
    -o llm_inference llm_inference.c -lm

# Nützliche Variablen
export OMP_NUM_THREADS=8   # Linux/macOS
# PowerShell: $env:OMP_NUM_THREADS = "8"
```

### Grundlegende Verwendung (Baseline)

```bash
./llm_inference tinyllama-1.1b.F16.gguf \
    "The secret to happiness is" \
    40 0.7 40 \
    --seed 42
```

Positionsargumente: `Modell`, `Prompt`, `n_tokens`,
`Temperatur`, `top_k`.

### Mystische Perturbation zur Laufzeit

```bash
./llm_inference tinyllama-1.1b.F16.gguf \
    "The secret to happiness is" \
    40 0.7 40 \
    --seed 42 \
    --perturb mystical \
    --intensity 0.50
```

Es ist keine vorherige `DMT_*.gguf`-Datei nötig:
die Technik wird **im Speicher** auf die frisch geladenen
Gewichte angewendet.

### Andere Flags

| Flag | Effekt |
|------|--------|
| `--perturb` / `-P` | `none`, `mystical`, `amplify`, `noise`, `blockdiag`, `manifold` |
| `--intensity` / `-I` | Stärke der Perturbation (z. B. 0.10–0.50) |
| `--seed` | PRNG der Perturbation (+ Sampling wenn gesetzt) |
| `--steer` | Wort, dessen Embedding-Richtung den Residual zieht |
| `--steer-strength` | Stärke des Steerings (z. B. 0.15) |

Build-Abhängigkeiten: `libm` und für paralleles Matmul
**OpenMP** (`-fopenmp`). Ohne OpenMP kompiliert die
Engine trotzdem, nur langsamer.

## 1. Ein GGUF in C lesen

GGUF hat drei Zonen:

```
[ Magic "GGUF" | Version | Anzahl Tensoren | Anzahl KV ]
[ Schlüssel-Wert-Metadaten ]
[ Tensor-Index → ausgerichtete Daten ]
```

Die Engine lädt die gesamte Datei (unter Windows mit
64-Bit-I/O: F16 überschreitet 2 GB und 32-Bit-`ftell`
reicht nicht aus). Dann indiziert sie Tensoren nach Name:

```c
snprintf(name, sizeof(name), "blk.%d.attn_q.weight", l);
m->wq[l] = must_tensor_f16(&m->gguf, name);
/* wk, wv, wo, w1/w3/w2 (gate/up/down), Normen… */
```

### F16 auf Festplatte, Float im Matmul

Die Schichtgewichte bleiben im **F16**-Buffer der Datei.
Im Matrix-Vektor-Produkt werden sie mit einer
**65.536-Einträge-Tabelle** konvertiert
(ein Wert für jedes Half-Muster):

```c
static float g_f16_table[65536];
/* init: g_f16_table[i] = decode_ieee_half(i); */
static inline float f16_to_f32(uint16_t h) {
    return g_f16_table[h];
}
```

So vermeiden wir Decodierung von Bits bei jedem Gewicht
in der heißen Schleife. Erst wenn `--perturb` angefordert wird,
kopieren wir die Schichtmatrizen in **veränderliches F32** (~3.6 GB):
die `amplify_subspace`-Deltas sind klein und ein
Round-Trip nach F16 würde sie löschen.

### Tokenizer: BPE aus dem GGUF selbst

Wir lesen `tokenizer.ggml.tokens` und
`tokenizer.ggml.merges` aus den Metadaten, mit Hash-Tabellen
für schnelle Merges.

Praktisches Detail: Manche F16 von TinyLlama bringen
ein *abgeschnittenes* Vokabular (wenige Tokens im
Header). Wenn wir das erkennen, laden wir **nur den
Tokenizer** vom Geschwister-Q4_0
(`tinyllama-1.1b.Q4_0.gguf`), ohne die F16-Gewichte
zu berühren. Die IDs stimmen mit HuggingFace überein
(z. B. `"Hello"` → `[1, 15043]`).

## 2. Die Operationen des Transformers

### RMSNorm

```c
/* x / rms(x) * w   — w kommt in F16 */
static void rmsnorm(float *out, const float *x,
                    const uint16_t *w_f16, int n) {
    float ss = 0.f;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    float scale = 1.f / sqrtf(ss / (float)n + 1e-5f);
    for (int i = 0; i < n; i++)
        out[i] = x[i] * scale * f16_to_f32(w_f16[i]);
}
```

Wenig Gewicht (`attn_norm`, `ffn_norm`), viel Kontrolle.

### Matmul (die tatsächlichen Kosten)

```c
/* W [out, in] · x[in] → out[out]
 * OpenMP über Ausgabezeilen; F16 via LUT */
static void matmul_f16(float *out, const float *x,
                       const uint16_t *W,
                       int in_dim, int out_dim) {
#pragma omp parallel for schedule(static)
    for (int j = 0; j < out_dim; j++) {
        const uint16_t *row = W + (size_t)j * in_dim;
        float sum = 0.f;
        for (int i = 0; i < in_dim; i++)
            sum += x[i] * f16_to_f32(row[i]);
        out[j] = sum;
    }
}
```

Mit `--perturb` verwendet dieselbe Schleife `matmul_f32`
auf den bereits perturbierten Kopien. Q, K, V, O, Gate,
Up, Down und `lm_head` sind alle Matmuls.

### RoPE

Position ohne absolute Embeddings: Jede Dimension von
Q und K wird rotiert. Die Engine berechnet
Sinus/Cosinus pro Position bis `MAX_SEQ` vor, um nicht
im heißen Pfad `sinf`/`cosf` aufrufen zu müssen.

### Aufmerksamkeit mit GQA + KV-Cache

```
score(t) = (Q_pos · K_t) / sqrt(head_dim)
Gewichte  = softmax_causal(scores)   # nur 0..pos
Ausgabe   = Σ Gewichte[t] * V_t
```

TinyLlama: **32 Q-Köpfe, 4 KV** → jeder KV-Kopf
bedient 8 Q-Köpfe (`hkv = h / 8`).

Bei jedem Generierungsschritt berechnen wir nur Q/K/V
des **neuen Tokens**, speichern K und V im Cache
`[Schicht][Pos][kv_dim]` und beachten
`0 .. pos`. Das macht es auf der CPU möglich:
ohne KV-Cache müssten wir die gesamte Sequenz
bei jedem Token neu berechnen.

### SwiGLU (FFN)

```
h' = Down( SiLU(Gate(x)) ⊙ Up(x) )
```

~69% der Parameter (Kap. 2): die „praktische Erinnerung"
des Modells. Nach der Aufmerksamkeit, Residual;
nach dem FFN, ein weiterer Residual.

## 3. Ein Forward-Schritt (ein Token)

In menschlicher Sprache, pro Schicht und *aktueller Position*:

```
x = embedding(token)

für L = 0 .. 21:
    h = RMSNorm(x, attn_norm[L])
    Q,K,V = Projektionen(h);  RoPE(Q,K)
    K,V in Cache[L][pos] speichern
    x = x + O( Attention(Q, cache_K, cache_V) )

    h = RMSNorm(x, ffn_norm[L])
    x = x + SwiGLU_FFN(h)

Logits = output · RMSNorm(x, output_norm)
```

Im Code ist das `model_forward_token(...)`.
Die Generierung:

```c
/* Prefill: jedes Token des Prompts füllt den Cache */
for (i = 0; i < n_prompt; i++)
    model_forward_token(&model, &state, tokens[i]);

for (step = 0; step < max_new; step++) {
    next = sample_top_k(state.logits, …);
    if (next == EOS) break;
    emit(next);
    model_forward_token(&model, &state, next);
}
```

## 4. Sampling

32.000 Logits → eine Auswahl:

1. nach Temperatur skalieren  
2. top-k beibehalten  
3. Softmax nur über diese k  
4. Stichprobe ziehen  

| Parameter | Effekt |
|-----------|--------|
| `temperature → 0` | fast greedy |
| `temperature` hoch | mehr Diversität |
| `top_k` niedrig | enges Vokabular |

Um Perspektiven zu vergleichen setzen wir `temp ≈ 0.7`,
`top_k` und `--seed` wenn Reproduzierbarkeit gewünscht ist.

## 5. Perturbation zur Laufzeit (die Brücke zu Dreaming)

Dreaming ist nicht nur Text generieren: es ist **Gewichte
ändern** (oder Aktivierungen) und beobachten, was sich ändert.

### Verfügbare Techniken in der C-Engine

| `--perturb` | Mechanik | Hinweise |
|-------------|----------|----------|
| `none` | keine Änderung | F16-Baseline |
| `mystical` / `amplify` | `amplify_subspace`: \(w \leftarrow w + I\,(w\cdot v)\,v\) | philosophische / existenzielle Linse |
| `noise` | Rauschen ∝ \|w\| | bei hohem I degradiert |
| `blockdiag` | verstärkt 16×16-Blöcke | manchmal monoton / Echo |
| `manifold` | lokales Rauschen ∝ Std.abweichung des Blocks | hohes I kann kollabieren |

Politik (wie in `dmt_perturb_v10`): Es werden die
Matrizen von **Aufmerksamkeit und FFN** der 22 Schichten
berührt; Normen, Embeddings und `lm_head` bleiben unverändert.

```bash
# 15 Prompts mit I=0.50, seed=42 → ~8 tok/s bei Generierung
./llm_inference modell.F16.gguf "When we dissolve the ego" \
    60 0.7 40 --seed 42 --perturb mystical --intensity 0.50
```

Reales Beispiel (gleiche Konfiguration):

> *When we dissolve the ego, we dissolve the self.
> When we let go of the ego, we allow ourselves to
> become part of the universe…*

### Intensität

| I | Typisches Verhalten |
|---|---------------------|
| 0.05–0.10 | fast Baseline; subtiler Delta |
| 0.30–0.50 | klarere Perspektive (und manchmal anderes EOS) |
| noise / manifold hoch | Müll-Risiko |

Die mystische Einrichtung kostet ~**25 s** und ~**3.6 GB**
RAM F32 *einmal pro Prozess*. Danach läuft die
Generierung in derselben Größenordnung wie die Baseline.

### Aktivierungs-Steering (`--steer`)

Ergänzung zur Gewichtsperturbation: Ein Richtungsvektor
wird aus dem Embedding eines Wortes konstruiert und der
Residual wird während des Forwards in diese Richtung
geschoben. Das ist eine andere Form, die Generierung
zu *markieren*, ohne das GGUF umzuschreiben.

```bash
./llm_inference modell.F16.gguf "The world is" 40 0.7 40 \
    --steer amor --steer-strength 0.15
```

## Leistung (Größenordnungen)

Gemessen auf CPU mit OpenMP (8 Threads), TinyLlama F16:

| Situation | Typischer Wert |
|-----------|----------------|
| Baseline-Generierung | **~6–10 tok/s** (Wall, mit Prefill) |
| Generierung mit F32-Gewichten nach Perturbation | **~6–10 tok/s** (manchmal etwas schneller: keine Dequantisierung in der Schleife) |
| Prefill + wenige Tokens (frühes EOS) | tok/s Wall *niedrig* (der Prefill wiegt schwer) |
| F16-Laden + Tokenizer | **~1 s** |
| `mystical` anwenden (154 Tensoren → F32) | **~25 s** + ~3.6 GB |

Wir konkurrieren nicht mit GPUs oder Q4-Kernels von
llama.cpp. Wir konkurrieren mit der *Durchsichtigkeit*: hier hat
jede Multiplikation einen Namen und eine Richtung.

## Zwei Werkzeuge, ein Modell

| Werkzeug | Rolle | Format |
|----------|-------|--------|
| `llm_inference.c` | verstehen, lehren, **zur Laufzeit perturbieren**, Sonden | GGUF F16 |
| `llama-cli` | Massentests, Q4_0, Experimentiergeschwindigkeit | GGUF Q4_0 |

Die 240 Generierungen der Perspektivenstudie
stützten sich stark auf `llama-cli` + bereits
perturbierte GGUFs. Die C-Engine schließt den Kreis:
wir können **die Perturbationsidee ohne
Zwischendatei wiederholen** und den Forward von innen sehen.

## Ehrliche Einschränkungen (aktualisiert)

1. **Nur F16-Gewichte** — keine Q4_0/Q6_K-Kernels in C
2. **Keine GPU** — nur CPU
3. **Kein Batching** — eine Sequenz nach der anderen
4. **Engine-Kontext begrenzt** (`MAX_SEQ`, z. B. 512)
   obwohl das Modell 2048 unterstützt
5. **F32-Perturbation teuer im RAM** — ~3.6 GB extra
6. **Nicht alle v10/v11-Techniken** sind in C
   (fehlend: lowrank, spectral, selective targeting, …)

Akzeptabel: Das Ziel ist nicht, einen Chat für
Millionen von Nutzern zu betreiben. Das Ziel ist, den
**Schädel eines 1.1B-Modells zu öffnen** und die
**Gewichte wie eine Linse drehen** zu können.

## Wie es mit dem Rest des Buches verbunden ist

Bisher:

- *Was* TinyLlama ist (Kapitel 1)
- *Wie* es innen aufgebaut ist (Kapitel 2)
- *Womit* wir es ausführen, messen und
  **perturbieren** (dieses Kapitel)

Das nächste Kapitel geht tiefer ein auf
**Gewichtsperturbation und Perspektivwechsel**:
DMT-Analogie, Techniken die Hierarchie bewahren,
Intensitäten, und warum das Modell nicht „zerbricht",
sondern *mit einer anderen Stimme spricht*.

Unter diesen Ergebnissen steckt eine echte Engine.
Unsere befindet sich in einer C-Datei, läuft auf CPU,
und liest nicht nur mehr Gewichte: Sie kann sie auch *bewegen*.

---

*Nächstes Kapitel: Gewichtsperturbation und Perspektivwechsel*