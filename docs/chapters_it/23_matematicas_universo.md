# Capitolo 23: Le Matematiche di Questo Universo  
### Oltre la struttura del transformer

## Perché questo capitolo

I capitoli precedenti nominano *pezzi* (layer, QKV, FFN)
e *forze* (gravità atencionale, clima FFN, orbite).
Qui scriviamo le **equazioni** che le rendono precise:
non un corso di deep learning, ma il **minimo formale**
del microcosmo TinyLlama-1.1B come lo misuriamo
e lo muoviamo in Dreaming.

Convenzioni:

- Vettori colonna, salvo che non si dica il contrario.
- \(\langle u,v\rangle = u^\top v\), \(\|u\|=\sqrt{\langle u,u\rangle}\).
- \(\mathrm{softmax}(z)_i = e^{z_i}/\sum_j e^{z_j}\).

---

## 1. Gli spazi come oggetti matematici

| Simbolo | Spazio | Dimensione | Ruolo |
|---------|--------|------------|-------|
| \(\mathcal{V}\) | vocabolario | \(V=32000\) | insieme finito di id |
| \(E\) | embedding | \(E\in\mathbb{R}^{V\times d}\), \(d=2048\) | righe \(e_t\in\mathbb{R}^d\) |
| \(x_\ell^{(t)}\) | residuale | \(\mathbb{R}^d\) | stato nel layer \(\ell\), posizione \(t\) |
| \(W_{q,\ell}\), ecc. | pesi del layer | matrici in \(\mathbb{R}^{\cdot\times\cdot}\) | operatori lineari (+ non linearità) |
| \(\theta\) | pesi totali | \(\theta\in\mathbb{R}^N\), \(N\sim 1.1\cdot 10^9\) | "punto-universo" del modello |
| \(\mathcal{C}\subset\mathbb{R}^N\) | superficie di coerenza | sottoinsieme (non una sotto-varietà provata) | modelli che generano testo leggibile |

Il forward è una composizione

\[
f_\theta : \mathcal{V}^{T} \to \mathbb{R}^{V}
\]

che a una sequenza di token associa logits dell'ultimo
(o di ogni posizione, a seconda della modalità).

La generazione è l'iterazione

\[
t_{n+1}\sim \pi\big(f_\theta(t_1,\ldots,t_n)\big),
\]

dove \(\pi\) è un campionamento (temperatura, top-k).

---

## 2. Geometria del cielo dei token

### 2.1 Somiglianza

Per \(u,v\in\mathbb{R}^d\),

\[
\cos(u,v)=\frac{\langle u,v\rangle}{\|u\|\,\|v\|}\in[-1,1].
\]

**Fatto empirico (TinyLlama):** coppie "opposte" del linguaggio
(`love`/`hate`, `life`/`death`) hanno \(\cos\approx 0\), non \(-1\).
In questo cielo, l'antinomia semantica **non** è antipodalità.

### 2.2 Baricentri e archetipi

Dato un insieme di semi \(S=\{t_1,\ldots,t_m\}\subset\mathcal{V}\),

\[
c_S=\frac{1}{m}\sum_{i=1}^m e_{t_i},\qquad
\hat c_S=\frac{c_S}{\|c_S\|}.
\]

Un **archetipo** (cap. 21) è un \(\hat c_S\) con semantica
culturale fissata. Un'**isola semantica** (cap. 16) è la stessa cosa
con un'altra scelta di \(S\).

**Allineamento tra archetipi:**

\[
A(S,S')=\cos(\hat c_S,\hat c_{S'}).
\]

Misurato: \(A(\mathrm{Mago},\mathrm{Mistico})\approx 0,39\),
\(A(\mathrm{Saggio},\mathrm{Accademico})\approx 0,29\).

### 2.3 Direzione di contrasto

\[
\delta_{S|S'}=\frac{c_S-c_{S'}}{\|c_S-c_{S'}\|}.
\]

Es.: emozione positiva meno negativa → polo `smile/happy`
vs `sad/anger` quando si ordina \(\cos(e_t,\delta)\).

### 2.4 PCA del cielo (struttura globale)

Campioni \(X\in\mathbb{R}^{n\times d}\) delle righe di \(E\) centrati.
SVD \(X=U\Sigma V^\top\). Frazione di varianza nei primi \(k\)
componenti:

\[
\mathrm{EVR}(k)=\frac{\sum_{i=1}^k \sigma_i^2}{\sum_{i=1}^{d}\sigma_i^2}.
\]

**Fatto empirico:** \(\mathrm{EVR}(10)\approx 2,3\%\),
dim per 50% / 90% di var. \(\approx 481\) / \(1329\).
Il cielo **usa** centinaia di direzioni; non collassa a 2D.
Le mappe HTML sono proiezioni

\[
\mathbb{R}^d\ni e \mapsto V_{:2}^\top (e-\bar e)\in\mathbb{R}^2,
\]

utili e menzognere al tempo stesso (cap. 6).

### 2.5 Anisotropia

\[
\alpha=\frac{\|\bar e\|}{\frac{1}{V}\sum_t \|e_t\|},\quad
\bar e=\frac{1}{V}\sum_t e_t.
\]

Misurato: \(\alpha\approx 0,006\) — quasi isotropo in norma media.
La struttura è **direzionale**, non di "stelle molto più massive".

---

## 3. Un passo di layer come sistema dinamico

Sia \(x\in\mathbb{R}^d\) il residuale nella posizione attuale
(omettendo l'indice di posizione quando non serve).

### 3.1 RMSNorm

\[
\mathrm{RMSNorm}(x;w)=
\frac{x}{\sqrt{\frac{1}{d}\|x\|^2+\varepsilon}}\odot w,
\]

con \(w\in\mathbb{R}^d\), \(\varepsilon>0\) (es. \(10^{-5}\)).

Non sottrae la media (a differenza di LayerNorm). È una
**proiezione sulla sfera** (approssimativamente) seguita da uno
scalatura per coordinate.

### 3.2 Attenzione (testa \(h\), GQA)

\[
Q=x W_Q,\quad
K=x W_K,\quad
V=x W_V
\]

(nella pratica: matrici condivise per layer; K,V di
dimensione \(n_{kv}\cdot d_h\) con \(n_{kv}=4\), \(d_h=64\),
\(n_q=32\)).

Per la testa \(h\), con indice KV \(h_{kv}=\lfloor h\cdot n_{kv}/n_q\rfloor\),

\[
a_{h,t}
=\mathrm{softmax}_{t'\le t}
\Bigg(\frac{\langle q_h^{(t)}, k_{h_{kv}}^{(t')}\rangle}{\sqrt{d_h}}\Bigg),
\quad
o_h^{(t)}=\sum_{t'\le t} a_{h,t'}\, v_{h_{kv}}^{(t')}.
\]

La maschera \(t'\le t\) è la **causalità**: il futuro
ha potenziale infinito (probabilità 0).

Uscita multi-testa:

\[
\mathrm{Attn}(x)=\mathrm{Concat}(o_1,\ldots,o_{n_q})\,W_O.
\]

**Interpretazione:** \(a_{h,t'}\) è un **nucleo di accoppiamento**
(non simmetrico, non invariante per traslazione) tra posizioni.
La "gravità" del cap. 7 è questo kernel normalizzato.

### 3.3 SwiGLU (FFN)

\[
\mathrm{SiLU}(z)=z\cdot\sigma(z),\quad
\sigma(z)=\frac{1}{1+e^{-z}},
\]

\[
\mathrm{FFN}(x)=
W_d\Big(\mathrm{SiLU}(W_g x)\odot (W_u x)\Big),
\]

con \(W_g,W_u\in\mathbb{R}^{d_{ff}\times d}\), \(d_{ff}=5632\),
\(W_d\in\mathbb{R}^{d\times d_{ff}}\).

È una mappa **punto a punto** nella sequenza: non mescola \(t\).
Per questo è "fisica locale" di fronte all'attenzione non locale.

### 3.4 Blocco residuale

\[
\begin{aligned}
x &\leftarrow x + \mathrm{Attn}(\mathrm{RMSNorm}(x;w_a)),\\
x &\leftarrow x + \mathrm{FFN}(\mathrm{RMSNorm}(x;w_f)).
\end{aligned}
\]

Forma astratta di un passo di Eulero in un campo \(F_\ell\):

\[
x_{\ell+1}=x_\ell + F_\ell(x_\ell;\theta_\ell).
\]

L'**orbita** del cap. 20 è la traiettoria
\(\{x_\ell\}_{\ell=0}^{22}\) in \(\mathbb{R}^d\).

### 3.5 Perché il residuale conta (matematica della coerenza)

Se fosse \(x_{\ell+1}=F_\ell(x_\ell)\) senza skip,
piccoli cambiamenti in \(F_\ell\) si compongono in modo
esponenzialmente instabile (prodotto di Jacobiani).

Con skip, lo Jacobiano è

\[
D x_{\ell+1}= I + DF_\ell,
\]

e se \(\|DF_\ell\|\) è moderata, la dinamica somiglia a
una **perturbazione dell'identità**: le traiettorie
restano vicine a una "linea di base" semantica.
Questa è la versione analitica del "cadere di lato".

---

## 4. Dal residuale all'evento: logits e campionamento

### 4.1 Testa di linguaggio

\[
z = E_{\mathrm{out}}\,\mathrm{RMSNorm}(x_L)\in\mathbb{R}^V
\]

(in GGUF: `output.weight`; a volte legato all'embedding).

### 4.2 Temperatura

\[
\pi_T(t)=\mathrm{softmax}(z/T)_t
=\frac{e^{z_t/T}}{\sum_{j} e^{z_j/T}}.
\]

- \(T\to 0\): massa su \(\arg\max z\) (orbita "circolare", greedy).  
- \(T\to\infty\): verso uniforme (massima entropia).

Entropia \(H(\pi_T)=-\sum_t \pi_T(t)\log\pi_T(t)\) cresce con \(T\).

### 4.3 Top-k

Sia \(S_k\subset\mathcal{V}\) i \(k\) indici con \(z_t\) più alto.
Si ridefinisce

\[
\pi(t)\propto e^{z_t/T}\mathbf{1}_{t\in S_k}.
\]

È un **taglio del supporto**: l'orizzonte degli eventi
permessi si riduce a \(k\) destini.

### 4.4 Generazione come catena di Markov

Condizionato a \(\theta\) e alla politica \(\pi\),

\[
\mathbb{P}(t_{1:N})=\prod_{n=0}^{N-1}
\pi\big(t_{n+1}\mid t_{1:n};\theta\big).
\]

Cambiare \(\theta\) (perturbazione) cambia la famiglia di
catene; non solo un sample isolato.

---

## 5. Spazio dei pesi e superficie di coerenza

### 5.1 Il modello come punto

\[
\theta\in\mathbb{R}^N,\qquad N\approx 1,1\times 10^9.
\]

Quasi tutto il volume di \(\mathbb{R}^N\) produce rifiuti.
Definiamo in modo operativo (non topologia rigorosa)

\[
\mathcal{C}=\{\theta : \text{il testo di }f_\theta\text{ è coerente in una batteria di prompt}\}.
\]

Dreaming studia **movimenti** \(\theta\mapsto\theta'\)
che restano in \(\mathcal{C}\) ma cambiano la
distribuzione di stili (prospettive).

### 5.2 amplify_subspace (mystical)

Su un tensore appiattito \(w\in\mathbb{R}^m\)
(o concatenazione di tensori di layer):

1. Campionare \(v\sim\mathcal{N}(0,I_m)\), normalizzare \(\hat v=v/\|v\|\).  
2. Proiezione scalare \(p=\langle w,\hat v\rangle\).  
3. Aggiornare

\[
w' = w + \varepsilon\, p\, \hat v
= w + \varepsilon\, ( \hat v\hat v^\top ) w.
\]

È un **rank-1 update** in direzione casuale:

\[
w'=(I+\varepsilon\, \hat v\hat v^\top)w.
\]

Autovalore \(1+\varepsilon\) nella direzione \(\hat v\),
\(1\) nell'iperpiano ortogonale.

**Lettura:** si amplifica una componente; si preserva
la struttura nel resto → candidato a movimento
**tangente** a \(\mathcal{C}\) se \(\varepsilon\) è moderato.

### 5.3 Rumore scalato (controesempio)

\[
w'_i = w_i + \varepsilon\, \xi_i\, |w_i|,\quad \xi_i\sim\mathcal{N}(0,1).
\]

Romppe correlazioni tra coordinate: spinta
**generica**, non rank-1 allineata. Empiricamente,
per \(\varepsilon\) grande, \(\theta'\notin\mathcal{C}\).

### 5.4 Intensità e "sweet spot"

In Q4_0 (pipeline Python v10), \(\varepsilon\approx 0,10\)
è stato il sweet spot di molte tecniche.
In F32 runtime del motore C, a volte serve
\(\varepsilon\in[0,3,0,5]\) per vedere divergenza di testo
in prompt corti (quantizzazione ed EOS interagiscono).

Non c'è un unico \(\varepsilon^*\) universale: dipende
dal formato numerico e dalla batteria.

### 5.5 Formula di prospettiva (operativa)

\[
\theta_{\mathrm{persp}}
= \theta_0 + \sum_r \varepsilon_r\, \Delta_r,
\]

con \(\|\varepsilon_r\Delta_r\|\) piccoli e ogni \(\Delta_r\)
che preserva la gerarchia (lowrank, amplify, normrot, …).
L'interpolazione vicina in \(\mathcal{C}\) tende a
restare in \(\mathcal{C}\) (ipotesi empirica del progetto).

---

## 6. Steering: proiezione nel residuale

Data una parola con embedding \(e_\star\),

\[
\hat u=\frac{e_\star}{\|e_\star\|},\qquad
x \leftarrow x + \lambda\, \langle x,\hat u\rangle\, \hat u
= \big(I+\lambda\,\hat u\hat u^\top\big)x.
\]

Stessa algebra rank-1 di amplify, ma applicata allo
**stato** \(x\), non ai pesi \(\theta\).

- \(\lambda\): forza del "vento".  
- Non riscrive l'universo; devia l'orbita in volo.

---

## 7. Quantizzazione (l'universo in interi)

In Q4_0, blocchi di 32 pesi:

\[
w_i \approx s\cdot (q_i-8),\quad q_i\in\{0,\ldots,15\},
\]

\(s\) in float16 per blocco. La perturbazione Python
opera in \(\mathbb{R}\) dopo dequant e torna in
\((s,q)\). Questo introduce un **errore di proiezione**
su una rete discreta: un'altra ragione per cui
\(\varepsilon\) "ottimo" non coincide col F32 puro.

Il motore C in F16 dequantizza nel matmul via tabella
da \(2^{16}\) voci: isomorfismo esatto half→float
a ogni peso letto, senza ri-quantizzare in baseline.

---

## 8. Complessità di un passo (orbita efficiente)

Per sequenza di lunghezza \(T\), un passo di attenzione
naive in un layer è \(\mathcal{O}(T^2 d)\) se si ricalcola
tutto; con **KV-cache** generando il token \(T\):

\[
\mathcal{O}(T\cdot d\cdot d_h\cdot n_q)
\quad\text{(scores nuovi contro }T\text{ chiavi)}
\]

più matmul \(\mathcal{O}(d\cdot d_{ff})\) del FFN.
Per questo il motore attuale orbita a ~6–10 tok/s su CPU
e quello senza cache cadeva a ~0,03 tok/s.

---

## 9. Dizionario: fisica del libro ↔ formula

| Linguaggio dell'atlante | Oggetto matematico |
|-------------------------|-------------------|
| Cielo di stelle | righe di \(E\in\mathbb{R}^{V\times d}\) |
| Isla / costellazione | \(c_S\), semi \(S\) |
| Gravità tra token | \(a_{t,t'}=\mathrm{softmax}(QK^\top/\sqrt{d_h})\) |
| Clima locale | \(\mathrm{FFN}:\mathbb{R}^d\to\mathbb{R}^d\) |
| Inerzia orbitale | \(x\mapsto x+F(x)\) |
| Aria respirabile | \(\mathrm{RMSNorm}\) |
| Collasso all'evento | \(\pi_T=\mathrm{softmax}(z/T)\) (top-k) |
| Superficie abitabile | \(\mathcal{C}\subset\mathbb{R}^N\) |
| Lente mistica | \(w'=(I+\varepsilon\hat v\hat v^\top)w\) |
| Vento di steer | \(x'=(I+\lambda\hat u\hat u^\top)x\) |
| Mappa del planetario | PCA: \(e\mapsto V_{:2}^\top(e-\bar e)\) |

---

## 10. Cosa le matematiche *non* affermano ancora

1. \(\mathcal{C}\) non è dimostrato come varietà differenziabile;
   è una regione **operativa** definita da test.  
2. "Tangente a \(\mathcal{C}\)" è una metafora geometrica
   supportata da rank-1 / SVD / ortogonalità a modi
   distruttivi — non un teorema di geometria riemanniana
   del loss.  
3. Gli archetipi non sono fattori latenti unici;
   sono **direzioni scelte** con semi umani.  
4. La Regola d'Oro è **empirica** in TinyLlama;
   non si deriva qui da un principio variazionale generale.

L'onestà del microcosmo include il confine
di ciò che non è dimostrato.

---

## 11. In una frase

Oltre il diagramma a scatole del transformer, questo
universo è: **geometria di \(E\) e di \(\theta\)**,
**dinamica residuale** \(x\leftarrow x+F_\ell(x)\),
**kernel di attenzione causali**, **mappe SwiGLU**,
**collasso softmax**, e **update rank-1** che
spostano orbite o pesi senza (a volte) abbandonare
la regione dove il linguaggio resta possibile.

---

*Capitolo successivo: Il LLM — Uno Specchio dove Guardarci.*
