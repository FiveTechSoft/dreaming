# Chapter 23: The Mathematics of This Universe
### Beyond the Transformer's Structure

## Why This Chapter

Previous chapters name *pieces* (layers, QKV, FFN)
and *forces* (attentional gravity, FFN climate, orbits).
Here we write the **equations** that make them precise:
not a deep learning course, but the **formal minimum**
of the TinyLlama-1.1B microcosmos as we measure it
and move it in Dreaming.

Conventions:

- Column vectors, unless stated otherwise.
- \(\langle u,v\rangle = u^\top v\), \(\|u\|=\sqrt{\langle u,u\rangle}\).
- \(\mathrm{softmax}(z)_i = e^{z_i}/\sum_j e^{z_j}\).

---

## 1. Spaces as Mathematical Objects

| Symbol | Space | Dimension | Role |
|--------|-------|-----------|------|
| \(\mathcal{V}\) | vocabulary | \(V=32000\) | finite set of ids |
| \(E\) | embeddings | \(E\in\mathbb{R}^{V\times d}\), \(d=2048\) | rows \(e_t\in\mathbb{R}^d\) |
| \(x_\ell^{(t)}\) | residual | \(\mathbb{R}^d\) | state at layer \(\ell\), position \(t\) |
| \(W_{q,\ell}\), etc. | layer weights | matrices in \(\mathbb{R}^{\cdot\times\cdot}\) | linear operators (+ nonlinearities) |
| \(\theta\) | total weights | \(\theta\in\mathbb{R}^N\), \(N\sim 1.1\cdot 10^9\) | "universe-point" of the model |
| \(\mathcal{C}\subset\mathbb{R}^N\) | coherence surface | subset (not a proven submanifold) | models that generate readable text |

The forward pass is a composition

\[
f_\theta : \mathcal{V}^{T} \to \mathbb{R}^{V}
\]

that associates to a sequence of tokens the logits of the last
(or of each position, depending on mode).

Generation is the iteration

\[
t_{n+1}\sim \pi\big(f_\theta(t_1,\ldots,t_n)\big),
\]

where \(\pi\) is a sampler (temperature, top-k).

---

## 2. Geometry of the Token Sky

### 2.1 Similarity

For \(u,v\in\mathbb{R}^d\),

\[
\cos(u,v)=\frac{\langle u,v\rangle}{\|u\|\,\|v\|}\in[-1,1].
\]

**Empirical fact (TinyLlama):** "opposite" language pairs
(`love`/`hate`, `life`/`death`) have \(\cos\approx 0\), not \(-1\).
In this sky, semantic antinomy **is not** antipodality.

### 2.2 Centroids and Archetypes

Given a seed set \(S=\{t_1,\ldots,t_m\}\subset\mathcal{V}\),

\[
c_S=\frac{1}{m}\sum_{i=1}^m e_{t_i},\qquad
\hat c_S=\frac{c_S}{\|c_S\|}.
\]

An **archetype** (chapter 21) is a \(\hat c_S\) with fixed cultural
semantics. A **semantic island** (chapter 16) is the same
with a different choice of \(S\).

**Alignment between archetypes:**

\[
A(S,S')=\cos(\hat c_S,\hat c_{S'}).
\]

Measured: \(A(\mathrm{Magician},\mathrm{Mystic})\approx 0.39\),
\(A(\mathrm{Sage},\mathrm{Academic})\approx 0.29\).

### 2.3 Contrast Direction

\[
\delta_{S|S'}=\frac{c_S-c_{S'}}{\|c_S-c_{S'}\|}.
\]

E.g., positive emotion minus negative → pole `smile/happy`
vs `sad/anger` when ranking \(\cos(e_t,\delta)\).

### 2.4 PCA of the Sky (Global Structure)

Sample \(X\in\mathbb{R}^{n\times d}\) of centered rows of \(E\).
SVD \(X=U\Sigma V^\top\). Fraction of variance in the first \(k\)
components:

\[
\mathrm{EVR}(k)=\frac{\sum_{i=1}^k \sigma_i^2}{\sum_{i=1}^{d}\sigma_i^2}.
\]

**Empirical fact:** \(\mathrm{EVR}(10)\approx 2.3\%\),
dims for 50% / 90% of var. \(\approx 481\) / \(1329\).
The sky **uses** hundreds of directions; it doesn't collapse to 2D.
The HTML maps are projections

\[
\mathbb{R}^d\ni e \mapsto V_{:2}^\top (e-\bar e)\in\mathbb{R}^2,
\]

useful and misleading at the same time (chapter 6).

### 2.5 Anisotropy

\[
\alpha=\frac{\|\bar e\|}{\frac{1}{V}\sum_t \|e_t\|},\quad
\bar e=\frac{1}{V}\sum_t e_t.
\]

Measured: \(\alpha\approx 0.006\) — nearly isotropic in mean norm.
The structure is **directional**, not of "much more massive stars."

---

## 3. A Layer Step as a Dynamical System

Let \(x\in\mathbb{R}^d\) be the residual at the current position
(omitting the position index when unnecessary).

### 3.1 RMSNorm

\[
\mathrm{RMSNorm}(x;w)=
\frac{x}{\sqrt{\frac{1}{d}\|x\|^2+\varepsilon}}\odot w,
\]

with \(w\in\mathbb{R}^d\), \(\varepsilon>0\) (e.g., \(10^{-5}\)).

It doesn't subtract the mean (unlike LayerNorm). It's an
**approximately spherical projection** followed by a
coordinate-wise scaling.

### 3.2 Attention (head \(h\), GQA)

\[
Q=x W_Q,\quad
K=x W_K,\quad
V=x W_V
\]

(in practice: matrices shared per layer; K,V of
dimension \(n_{kv}\cdot d_h\) with \(n_{kv}=4\), \(d_h=64\),
\(n_q=32\)).

For head \(h\), with KV index \(h_{kv}=\lfloor h\cdot n_{kv}/n_q\rfloor\),

\[
a_{h,t}
=\mathrm{softmax}_{t'\le t}
\Bigg(\frac{\langle q_h^{(t)}, k_{h_{kv}}^{(t')}\rangle}{\sqrt{d_h}}\Bigg),
\quad
o_h^{(t)}=\sum_{t'\le t} a_{h,t'}\, v_{h_{kv}}^{(t')}.
\]

The mask \(t'\le t\) is **causality**: the future
has infinite potential (probability 0).

Multi-head output:

\[
\mathrm{Attn}(x)=\mathrm{Concat}(o_1,\ldots,o_{n_q})\,W_O.
\]

**Interpretation:** \(a_{h,t'}\) is a **coupling kernel**
(non-symmetric, not translation-invariant) between positions.
The "gravity" of chapter 7 is this normalized kernel.

### 3.3 SwiGLU (FFN)

\[
\mathrm{SiLU}(z)=z\cdot\sigma(z),\quad
\sigma(z)=\frac{1}{1+e^{-z}},
\]

\[
\mathrm{FFN}(x)=
W_d\Big(\mathrm{SiLU}(W_g x)\odot (W_u x)\Big),
\]

with \(W_g,W_u\in\mathbb{R}^{d_{ff}\times d}\), \(d_{ff}=5632\),
\(W_d\in\mathbb{R}^{d\times d_{ff}}\).

It's a **point-to-point** map in the sequence: it doesn't mix \(t\).
That's why it's "local physics" versus non-local attention.

### 3.4 Residual Block

\[
\begin{aligned}
x &\leftarrow x + \mathrm{Attn}(\mathrm{RMSNorm}(x;w_a)),\\
x &\leftarrow x + \mathrm{FFN}(\mathrm{RMSNorm}(x;w_f)).
\end{aligned}
\]

Abstract form of an Euler step in a field \(F_\ell\):

\[
x_{\ell+1}=x_\ell + F_\ell(x_\ell;\theta_\ell).
\]

The **orbit** of chapter 20 is the trajectory
\(\{x_\ell\}_{\ell=0}^{22}\) in \(\mathbb{R}^d\).

### 3.5 Why the Residual Matters (Mathematics of Coherence)

If it were \(x_{\ell+1}=F_\ell(x_\ell)\) without skip,
small changes in \(F_\ell\) compose in exponentially
unstable fashion (product of Jacobians).

With skip, the Jacobian is

\[
D x_{\ell+1}= I + DF_\ell,
\]

and if \(\|DF_\ell\|\) is moderate, the dynamics resembles
a **perturbation of the identity**: trajectories
stay near a semantic "baseline." That's the analytical
version of "falling sideways."

---

## 4. From Residual to Event: Logits and Sampling

### 4.1 Language Head

\[
z = E_{\mathrm{out}}\,\mathrm{RMSNorm}(x_L)\in\mathbb{R}^V
\]

(in GGUF: `output.weight`; sometimes tied to the embedding).

### 4.2 Temperature

\[
\pi_T(t)=\mathrm{softmax}(z/T)_t
=\frac{e^{z_t/T}}{\sum_{j} e^{z_j/T}}.
\]

- \(T\to 0\): mass at \(\arg\max z\) ("circular" orbit, greedy).
- \(T\to\infty\): toward uniform (maximum entropy).

Entropy \(H(\pi_T)=-\sum_t \pi_T(t)\log\pi_T(t)\) grows with \(T\).

### 4.3 Top-k

Let \(S_k\subset\mathcal{V}\) be the \(k\) indices with highest \(z_t\).
It redefines

\[
\pi(t)\propto e^{z_t/T}\mathbf{1}_{t\in S_k}.
\]

It's a **support truncation**: the horizon of allowed events
is reduced to \(k\) destinations.

### 4.4 Generation as a Markov Chain

Conditioned on \(\theta\) and policy \(\pi\),

\[
\mathbb{P}(t_{1:N})=\prod_{n=0}^{N-1}
\pi\big(t_{n+1}\mid t_{1:n};\theta\big).
\]

Changing \(\theta\) (perturbation) changes the family of
chains; not just an isolated sample.

---

## 5. Weight Space and Coherence Surface

### 5.1 The Model as a Point

\[
\theta\in\mathbb{R}^N,\qquad N\approx 1.1\times 10^9.
\]

Almost all of \(\mathbb{R}^N\) produces garbage.
We define operationally (not rigorously topologically)

\[
\mathcal{C}=\{\theta : \text{the text of }f_\theta\text{ is coherent across a battery of prompts}\}.
\]

Dreaming studies **movements** \(\theta\mapsto\theta'\)
that remain in \(\mathcal{C}\) but change the
distribution of styles (perspectives).

### 5.2 amplify_subspace (mystical)

On a flattened tensor \(w\in\mathbb{R}^m\)
(or concatenation of layer tensors):

1. Sample \(v\sim\mathcal{N}(0,I_m)\), normalize \(\hat v=v/\|v\|\).
2. Scalar projection \(p=\langle w,\hat v\rangle\).
3. Update

\[
w' = w + \varepsilon\, p\, \hat v
= w + \varepsilon\, ( \hat v\hat v^\top ) w.
\]

It's a **rank-1 update** in a random direction:

\[
w'=(I+\varepsilon\, \hat v\hat v^\top)w.
\]

Eigenvalue \(1+\varepsilon\) in direction \(\hat v\),
\(1\) in the orthogonal hyperplane.

**Reading:** one component is amplified; the rest of
the structure is preserved → candidate for a movement
**tangent** to \(\mathcal{C}\) if \(\varepsilon\) is moderate.

### 5.3 Scaled Noise (Counterexample)

\[
w'_i = w_i + \varepsilon\, \xi_i\, |w_i|,\quad \xi_i\sim\mathcal{N}(0,1).
\]

It breaks correlations between coordinates: a **generic**
push, not rank-1 aligned. Empirically, for large \(\varepsilon\),
\(\theta'\notin\mathcal{C}\).

### 5.4 Intensity and "Sweet Spot"

In Q4_0 (Python pipeline v10), \(\varepsilon\approx 0.10\)
was the sweet spot for many techniques.
In F32 runtime of the C engine, sometimes \(\varepsilon\in[0.3,0.5]\)
is needed to see text divergence
on short prompts (quantization and EOS interact).

There is no universal \(\varepsilon^*\): it depends
on the numeric format and the battery.

### 5.5 Perspective Formula (Operational)

\[
\theta_{\mathrm{persp}}
= \theta_0 + \sum_r \varepsilon_r\, \Delta_r,
\]

with \(\|\varepsilon_r\Delta_r\|\) small and each \(\Delta_r\)
preserving hierarchy (lowrank, amplify, normrot, …).
Close interpolation in \(\mathcal{C}\) tends to
remain in \(\mathcal{C}\) (empirical hypothesis of the project).

---

## 6. Steering: Projection in the Residual

Given a word with embedding \(e_\star\),

\[
\hat u=\frac{e_\star}{\|e_\star\|},\qquad
x \leftarrow x + \lambda\, \langle x,\hat u\rangle\, \hat u
= \big(I+\lambda\,\hat u\hat u^\top\big)x.
\]

Same rank-1 algebra as amplify, but applied to the
**state** \(x\), not the weights \(\theta\).

- \(\lambda\): strength of the "wind."
- It doesn't rewrite the universe; it deflects the orbit in flight.

---

## 7. Quantization (The Universe in Integers)

In Q4_0, blocks of 32 weights:

\[
w_i \approx s\cdot (q_i-8),\quad q_i\in\{0,\ldots,15\},
\]

\(s\) in float16 per block. The Python perturbation
operates in \(\mathbb{R}\) after dequant and returns to
\((s,q)\). That introduces a **projection error**
onto a discrete grid: another reason why
the "optimal" \(\varepsilon\) doesn't match pure F32.

The C engine in F16 dequantizes in the matmul via a table
of \(2^{16}\) entries: exact half→float isomorphism
at each read weight, without re-quantizing in baseline.

---

## 8. Complexity of One Step (Efficient Orbit)

For a sequence of length \(T\), a naive attention step
in one layer is \(\mathcal{O}(T^2 d)\) if everything is recomputed;
with **KV-cache** when generating token \(T\):

\[
\mathcal{O}(T\cdot d\cdot d_h\cdot n_q)
\quad\text{(new scores against }T\text{ keys)}
\]

plus \(\mathcal{O}(d\cdot d_{ff})\) matmuls for the FFN.
That's why the current engine orbits at ~6–10 tok/s on CPU
and the one without cache fell to ~0.03 tok/s.

---

## 9. Dictionary: Book's Physics ↔ Formula

| Atlas Language | Mathematical Object |
|---------------|-------------------|
| Sky of stars | rows of \(E\in\mathbb{R}^{V\times d}\) |
| Island / constellation | \(c_S\), seeds \(S\) |
| Gravity between tokens | \(a_{t,t'}=\mathrm{softmax}(QK^\top/\sqrt{d_h})\) |
| Local climate | \(\mathrm{FFN}:\mathbb{R}^d\to\mathbb{R}^d\) |
| Orbital inertia | \(x\mapsto x+F(x)\) |
| Breathable air | \(\mathrm{RMSNorm}\) |
| Collapse to event | \(\pi_T=\mathrm{softmax}(z/T)\) (top-k) |
| Habitable surface | \(\mathcal{C}\subset\mathbb{R}^N\) |
| Mystical lens | \(w'=(I+\varepsilon\hat v\hat v^\top)w\) |
| Steer wind | \(x'=(I+\lambda\hat u\hat u^\top)x\) |
| Planetarium map | PCA: \(e\mapsto V_{:2}^\top(e-\bar e)\) |

---

## 10. What Mathematics *Doesn't* Claim Yet

1. \(\mathcal{C}\) is not proven to be a differentiable manifold;
   it's an **operational** region defined by tests.
2. "Tangent to \(\mathcal{C}\)" is a geometric metaphor
   backed by rank-1 / SVD / orthogonality to destructive
   modes — not a Riemannian geometry theorem on the loss.
3. Archetypes are not unique latent factors;
   they're **chosen directions** with human seeds.
4. The Golden Rule is **empirical** in TinyLlama;
   it's not derived here from a general variational principle.

The honesty of the microcosmos includes the edge
of what is unproven.

---

## 11. In One Sentence

Beyond the transformer's box diagram, this
universe is: **geometry of \(E\) and \(\theta\)**,
**residual dynamics** \(x\leftarrow x+F_\ell(x)\),
**causal attention kernels**, **SwiGLU maps**,
**softmax collapse**, and **rank-1 updates** that
shift orbits or weights without (sometimes) abandoning
the region where language remains possible.

---

*Next chapter: The LLM — A Mirror Where We See Ourselves.*