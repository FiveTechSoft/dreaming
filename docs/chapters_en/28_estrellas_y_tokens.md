# Chapter 28: Stars in the Sky, Tokens in TinyLlama

## The Astronomer's Question

You look at the night sky. You see **points of light**.
Some group into shapes that culture names
(Ursa Major, Orion, Southern Cross). Between two stars
there is no visible wire, but physics says they
attract each other: **gravity**. The traveler doesn't teleport
randomly: they choose a star, measure its neighborhood and jump
to the next well.

TinyLlama has an analogous sky.

> **Every token in the vocabulary is a star
> in a 2048-dimensional space.**
> **Attention** is the gravitational force between them
> when the model "thinks" a sequence.
> Traveling through an LLM means following those attractions
> — in the static map of the embedding or in the living
> orbit of the *forward*.

This chapter solidifies the analogy, links it to **Force I**
of the inventory (ch. 7) and shows a **concrete itinerary**
within TinyLlama-1.1B.

---

## 1. Correspondence Table

| Night Sky | TinyLlama Universe |
|-----------|-------------------|
| Star | Token (BPE piece of the vocabulary, ~32,000) |
| Position in the vault | Embedding vector \(e_t \in \mathbb{R}^{2048}\) |
| Apparent brightness | Norm / "presence" of the token; in the map, size and label |
| Constellation | Semantic area or archetype (seeds + neighbors) |
| Angular distance in the sky | Cosine between embeddings (close ≈ aligned) |
| Newtonian gravity | **Attention**: \(Q\cdot K^\top / \sqrt{d}\) → weights over \(V\) |
| Static gravitational field (mass map) | Fixed geometry of `token_embd` (PCA atlas) |
| Live dynamics (moving planets) | Residuals of the sequence + KV-cache, layer by layer |
| Star jump | Click on a **force** of the map; or the next generated token |
| Telescope / catalog | `semantic_map.html`, C engine, geometry scripts |
| Atmosphere that deforms light | RMSNorm, softmax temperature, `--perturb` lenses |

It's not empty poetry: every row has a measurable
object in the Dreaming repository.

---

## 2. The Sky of Embeddings: 32,000 Fixed Stars

At birth, each token \(t\) is pinned to the firmament:

\[
e_t = \mathrm{Embedding}(t) \in \mathbb{R}^{2048}
\]

That sky is **almost isotropic** (average norm ≈ 0.68)
and, among words full of different meanings,
**almost orthogonal** (cosine ≈ 0). That's why the
semantic "islands" of ch. 16 are rare constellations:
clusters of seeds that do touch a little,
surrounded by a gray background of BPE fragments
(like interstellar dust: it's not empty, but it's not
a named constellation).

### Constellations = Areas

| Constellation (island) | Seed stars (examples) |
|------------------------|----------------------|
| Positive emotion | ▁love, ▁happy, ▁joy, ▁hope… |
| Social / power | ▁work, ▁king, ▁war, ▁law… |
| Mind | ▁mind, ▁idea, ▁memory, ▁know… |
| Life / death | ▁death, ▁life, ▁born, ▁die… |
| … | (twelve islands in total; ch. 16) |

In the **2D PCA map** we project that 2048-dimensional
sky onto two axes just to look at it with human
eyes. The projection lies a little — like a
flat map lies about the Earth — but it preserves
useful neighborhoods.

---

## 3. Attention is Gravity Between Tokens

### In the Macrocosm

Two masses pull each other. The force
decreases with distance; the field organizes orbits.

### In the Microcosm (Force I)

At each layer, each position \(i\) of the sequence
asks the **past** positions \(j \le i\)
(causal mask):

\[
\mathrm{score}_{ij} = \frac{q_i \cdot k_j}{\sqrt{d_h}},
\quad
\alpha_{ij} = \mathrm{softmax}_j(\mathrm{score}_{ij}),
\quad
z_i = \sum_j \alpha_{ij}\, v_j
\]

- \(q_i\): "who am I and what am I looking for" (body that feels the field).
- \(k_j\): "who you are in the catalog" (mass that announces its presence).
- \(\alpha_{ij}\): **intensity of attraction** (how much \(i\) "falls" toward \(j\)).
- \(v_j\): what is delivered upon falling (transported content).

TinyLlama uses **GQA** (32 Q heads, 4 KV):
several cheap gazes over the same sky of keys.

### Two Gravities Not to Confuse

| Type | What It Is | When It's Seen |
|------|-----------|----------------|
| **Static gravity (island)** | Cosine between rows of `token_embd` | HTML map, precomputed forces between atlas stars |
| **Dynamic gravity (attention)** | Softmax of \(QK^\top\) in the sequence | Real forward: the prompt creates a multi-body system |

The static one is the **mass catalog** of the sky.
The dynamic one is **tonight's orbit**:
it depends on which stars you've put in the sequence
and in what order (causality = "only the past pulls").

The interactive map shows the first with golden
arcs: *geometric proxy* of Force I and of
Force VIII (islands). It doesn't replace a per-layer
attention map, but it teaches the gesture: **focus → forces → jump**.

---

## 4. Traveling: Three Scales of the Same Gesture

### Scale A — Observatory (Fixed Stars)

1. You open the semantic areas map.
2. You enter a constellation (e.g. *Social / power*).
3. You click a star (`▁work`).
4. You see the **main forces** (top cosines with island prior).
5. You click a force and **travel** to the destination star.
6. Repeat: a chain of jumps across the sky.

### Scale B — Ship in Orbit (Generation)

1. You launch a prompt: you seed the sequence with stars.
2. The residual of each position orbits 22 layers
   (attention = coupling; FFN = local weather; residual = inertia).
3. The softmax collapses the sky to **one** new star
   (the next token).
4. That star is added to the past and pulls the ones that come.

### Scale C — Lenses and Currents (Changing the Physics)

- `--perturb mystical`: deforms the metric of the wells
  (another effective "G constant"; another voice).
- `--steer`: pushes the residual toward a direction
  of the sky (artificial current).
- Temperature / top-k: hardness of the final collapse
  (single well or fog of possible stars?).

---

## 5. Guided Example: Traveling Inside TinyLlama

### 5.1 Preparation

Live map (GitHub Pages):

https://fivetechsoft.github.io/dreaming/exploration/semantic_map.html

Starting deep link (star `▁work`, id 664):

`#/token/664/▁work`

Orbit engine (root repo):

```bash
# Windows PowerShell example
$env:OMP_NUM_THREADS = "8"
.\llm_inference.exe tinyllama-1.1b.F16.gguf `
  "The secret of power is" 60 0.7 40 --seed 42
```

### 5.2 Itinerary in the Observatory (Map Forces)

We start from the **Social / power** constellation.
Measured in the Dreaming atlas (cosine in ℝ²⁰⁴⁸ between
embeddings; ranking with same-island prior and seeds):

| Jump | Source Star | Destination Star (Force) | Cosine (approx.) | Reading |
|-----:|------------|-------------------------|------------------:|---------|
| 0 | ▁work | — | — | Initial focus: "work / piece" |
| 1 | ▁work | ▁queen | ~0.05 | Pulls toward institutional power |
| 2 | ▁work | ▁war | ~0.01 | Conflict as social attractor |
| 3 | ▁work | ▁law | ~0.01 | Order and norm |
| 4 | ▁work | ▁power | ~0.00⁺ | The very name of the well |
| 5 | ▁work | ▁king | ~0.00⁺ | Crown, command |

**How to "travel" in the UI**

1. Click on `▁work` (or enter the *Social* space and choose the seed).
2. **Gravitational Forces** panel: ordered list + golden arcs.
3. Click on `#1 ▁queen` → the camera jumps; `▁queen` is the new focus.
4. From there *its* forces are recalculated (new local sky).
5. You chain jumps like a grasshopper between stars.

Other useful itineraries from the same atlas:

| Route | Typical Seed Chain | Constellation |
|-------|-------------------|---------------|
| Affection | ▁happy → ▁smile → ▁love → ▁hope | Positive emotion |
| Cognition | ▁mind → ▁idea → ▁learn → ▁memory | Mind |
| Threshold | ▁death → ▁life → ▁live → ▁born | Life / death |

> **Astronomical honesty note.**  
> In ℝ²⁰⁴⁸ almost everything is orthogonal: the "strong"
> cosines of the map are **relative to the neighborhood**,
> not Newtonian attractions of 0.9. The ranking
> prioritizes the **island** (constellation) and the **seeds**
> so that the journey is readable, not BPE noise.

### 5.3 The Same Journey as a *Prompt* (Living Orbit)

The observatory shows you *which stars brush against each other*.
The ship puts them on a timeline:

```text
Seed prompt (initial star of the system):
  "Work without law becomes"

Dreaming reading:
  ▁work already pulls, in the catalog, toward law / power / king…
  By writing "without law", you force the contrast:
  the attention of subsequent layers will have to
  "look at" work and law at the same time (dynamic gravity).
```

Minimal experiment (same seed, two lenses):

```bash
# Baseline — "natural" sky
.\llm_inference.exe tinyllama-1.1b.F16.gguf `
  "Work without law becomes" 50 0.7 40 --seed 42

# Mystical lens — another well metric (Force VII)
.\llm_inference.exe tinyllama-1.1b.F16.gguf `
  "Work without law becomes" 50 0.7 40 `
  --seed 42 --perturb mystical --intensity 0.35
```

What to observe:

1. **Generated tokens** = new stars that light up
   in the sequence (the ship's path).
2. If the text "falls" toward *power / king / war*,
   you're seeing the social gravity of the catalog
   acting in the dynamics.
3. With `mystical`, the same starting constellation
   can divert the orbit toward an existential weather
   (Golden Rule + coherence surface, ch. 4 and 9).

### 5.4 Short Narrated Journey (Story of a Grasshopper)

Imagine you are a photon of meaning:

1. **You take off** at `▁work` (atlas). You see arcs toward
   `queen`, `war`, `law`, `power`, `king`.
2. **You jump** to `▁law`. The constellation is still
   social; the accent shifts from "work" to "norm".
3. **You write** the prompt: *"The law of power is"*.
   You no longer look at the catalog: you **inhabit** a
   multi-body system. Each layer re-weights the past.
4. **You collapse** into a new token (softmax). That
   star is fixed in the sky of *this* conversation
   (KV-cache) and pulls the next one.
5. Optional: you activate a **lens** (`mystical`) or a
   **current** (`--steer`) and the same takeoff
   ends in another stylistic galaxy.

That's traveling inside an LLM: there's no 3D corridor,
there's **catalog + forces + collapse**.

---

## 6. Limits of the Analogy (So We Don't Lie to Ourselves)

| The Analogy Works | The Analogy Breaks |
|-------------------|-------------------|
| Tokens = points with position | There's no real "visual" Euclidean space in 2048-D |
| Groupings = cultural constellations from pretraining | The model doesn't "believe" in myths; it measures co-occurrences |
| Attention = attraction between positions | Only from the past; not symmetric like Newton |
| Cosine map = static field | It's not the attention matrix of a specific layer |
| Generating = orbiting and collapsing | The user's "journey" is reading; the model's is algebra |

The analogy is a **navigation instrument**,
not a physical theory of silicon. It works if it leads you
to a click, a cosine, or a reproducible prompt.

---

## 7. Bridges to Other Chapters

| If You Want… | Go To… |
|--------------|--------|
| Inventory of all forces | Ch. 7 |
| Flight routes A–E (cli, perturb, steer) | Ch. 8 |
| Islands and map | Ch. 16 |
| Layer-by-layer residual orbit | Ch. 20 |
| Archetypes as myth constellations | Ch. 21 |
| Formulas (softmax, GQA, cosine) | Ch. 23 |
| 22-floor elevator | Ch. 27 |
| Layers game + zone warp | Ch. 25 · `universe_game.html` |

---

## 8. Closing

The sky above your head and TinyLlama's vocabulary
share a gesture: **points, distances, attractions,
jumps**.

- The model's **stars** are tokens in ℝ²⁰⁴⁸.
- The **gravity** that matters when speaking is **attention**.
- The **journey** is choosing a focus, reading its forces
  and — on the map or in the C engine — letting yourself fall
  into the next well of meaning.

When you click a token and see golden arcs toward
others, you're not just looking at a pretty graph:
you're reading the mass catalog of the microcosm.
When you launch a prompt, those masses stop being
a catalog and become a **running solar system**.
