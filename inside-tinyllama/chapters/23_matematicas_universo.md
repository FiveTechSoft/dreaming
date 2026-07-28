# Capítulo 23: Las Matemáticas de Este Universo  
### Más allá de la estructura del transformer

## Por qué este capítulo

Los capítulos anteriores nombran *piezas* (capas, QKV, FFN)
y *fuerzas* (gravedad atencional, clima FFN, órbitas).
Aquí escribimos las **ecuaciones** que las hacen precisas:
no un curso de deep learning, sino el **mínimo formal**
del microcosmos TinyLlama-1.1B tal como lo medimos
y lo movemos en Dreaming.

Convenciones:

- Vectores columna, salvo que se diga lo contrario.
- \(\langle u,v\rangle = u^\top v\), \(\|u\|=\sqrt{\langle u,u\rangle}\).
- \(\mathrm{softmax}(z)_i = e^{z_i}/\sum_j e^{z_j}\).

---

## 1. Los espacios como objetos matemáticos

| Símbolo | Espacio | Dimensión | Rol |
|---------|---------|-----------|-----|
| \(\mathcal{V}\) | vocabulario | \(V=32000\) | conjunto finito de ids |
| \(E\) | embeddings | \(E\in\mathbb{R}^{V\times d}\), \(d=2048\) | filas \(e_t\in\mathbb{R}^d\) |
| \(x_\ell^{(t)}\) | residual | \(\mathbb{R}^d\) | estado en capa \(\ell\), posición \(t\) |
| \(W_{q,\ell}\), etc. | pesos de capa | matrices en \(\mathbb{R}^{\cdot\times\cdot}\) | operadores lineales (+ no linealidades) |
| \(\theta\) | pesos totales | \(\theta\in\mathbb{R}^N\), \(N\sim 1.1\cdot 10^9\) | “punto-universo” del modelo |
| \(\mathcal{C}\subset\mathbb{R}^N\) | superficie de coherencia | subconjunto (no una subvariedad probada) | modelos que generan texto legible |

El forward es una composición

\[
f_\theta : \mathcal{V}^{T} \to \mathbb{R}^{V}
\]

que a una secuencia de tokens asocia logits del último
(o de cada posición, según el modo).

La generación es la iteración

\[
t_{n+1}\sim \pi\big(f_\theta(t_1,\ldots,t_n)\big),
\]

donde \(\pi\) es un muestreo (temperatura, top-k).

---

## 2. Geometría del cielo de tokens

### 2.1 Similaridad

Para \(u,v\in\mathbb{R}^d\),

\[
\cos(u,v)=\frac{\langle u,v\rangle}{\|u\|\,\|v\|}\in[-1,1].
\]

**Hecho empírico (TinyLlama):** pares “opuestos” del lenguaje
(`love`/`hate`, `life`/`death`) tienen \(\cos\approx 0\), no \(-1\).
En este cielo, la antinomia semántica **no** es antipodalidad.

### 2.2 Centroides y arquetipos

Dado un conjunto de seeds \(S=\{t_1,\ldots,t_m\}\subset\mathcal{V}\),

\[
c_S=\frac{1}{m}\sum_{i=1}^m e_{t_i},\qquad
\hat c_S=\frac{c_S}{\|c_S\|}.
\]

Un **arquetipo** (cap. 21) es un \(\hat c_S\) con semántica
cultural fijada. Una **isla semántica** (cap. 16) es lo mismo
con otra elección de \(S\).

**Alineación entre arquetipos:**

\[
A(S,S')=\cos(\hat c_S,\hat c_{S'}).
\]

Medido: \(A(\mathrm{Mago},\mathrm{Místico})\approx 0.39\),
\(A(\mathrm{Sabio},\mathrm{Académico})\approx 0.29\).

### 2.3 Dirección de contraste

\[
\delta_{S|S'}=\frac{c_S-c_{S'}}{\|c_S-c_{S'}\|}.
\]

Ej.: emoción positiva menos negativa → polo `smile/happy`
vs `sad/anger` al rankear \(\cos(e_t,\delta)\).

### 2.4 PCA del cielo (estructura global)

Muestra \(X\in\mathbb{R}^{n\times d}\) de filas de \(E\) centradas.
SVD \(X=U\Sigma V^\top\). Fracción de varianza en los \(k\) primeros
componentes:

\[
\mathrm{EVR}(k)=\frac{\sum_{i=1}^k \sigma_i^2}{\sum_{i=1}^{d}\sigma_i^2}.
\]

**Hecho empírico:** \(\mathrm{EVR}(10)\approx 2.3\%\),
dims para 50% / 90% de var. \(\approx 481\) / \(1329\).
El cielo **usa** cientos de direcciones; no colapsa a 2D.
Los mapas HTML son proyecciones

\[
\mathbb{R}^d\ni e \mapsto V_{:2}^\top (e-\bar e)\in\mathbb{R}^2,
\]

útiles y mentirosas a la vez (cap. 6).

### 2.5 Anisotropía

\[
\alpha=\frac{\|\bar e\|}{\frac{1}{V}\sum_t \|e_t\|},\quad
\bar e=\frac{1}{V}\sum_t e_t.
\]

Medido: \(\alpha\approx 0.006\) — casi isótropo en norma media.
La estructura es **direccional**, no de “estrellas mucho más masivas”.

---

## 3. Un paso de capa como sistema dinámico

Sea \(x\in\mathbb{R}^d\) el residual en la posición actual
(omitiendo el índice de posición cuando no haga falta).

### 3.1 RMSNorm

\[
\mathrm{RMSNorm}(x;w)=
\frac{x}{\sqrt{\frac{1}{d}\|x\|^2+\varepsilon}}\odot w,
\]

con \(w\in\mathbb{R}^d\), \(\varepsilon>0\) (p.ej. \(10^{-5}\)).

No resta media (a diferencia de LayerNorm). Es una
**proyección a la esfera** (aproximadamente) seguida de un
escalado por coordenadas.

### 3.2 Atención (cabeza \(h\), GQA)

\[
Q=x W_Q,\quad
K=x W_K,\quad
V=x W_V
\]

(en la práctica: matrices compartidas por capa; K,V de
dimensión \(n_{kv}\cdot d_h\) con \(n_{kv}=4\), \(d_h=64\),
\(n_q=32\)).

Para la cabeza \(h\), con índice KV \(h_{kv}=\lfloor h\cdot n_{kv}/n_q\rfloor\),

\[
a_{h,t}
=\mathrm{softmax}_{t'\le t}
\Bigg(\frac{\langle q_h^{(t)}, k_{h_{kv}}^{(t')}\rangle}{\sqrt{d_h}}\Bigg),
\quad
o_h^{(t)}=\sum_{t'\le t} a_{h,t'}\, v_{h_{kv}}^{(t')}.
\]

La máscara \(t'\le t\) es la **causalidad**: el futuro
tiene potencial infinito (probabilidad 0).

Salida multi-cabeza:

\[
\mathrm{Attn}(x)=\mathrm{Concat}(o_1,\ldots,o_{n_q})\,W_O.
\]

**Interpretación:** \(a_{h,t'}\) es un **núcleo de acoplamiento**
(no simétrico, no traducción-invariante) entre posiciones.
La “gravedad” del cap. 7 es este kernel normalizado.

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

Es un mapa **punto a punto** en la secuencia: no mezcla \(t\).
Por eso es “física local” frente a la atención no local.

### 3.4 Bloque residual

\[
\begin{aligned}
x &\leftarrow x + \mathrm{Attn}(\mathrm{RMSNorm}(x;w_a)),\\
x &\leftarrow x + \mathrm{FFN}(\mathrm{RMSNorm}(x;w_f)).
\end{aligned}
\]

Forma abstracta de un paso de Euler en un campo \(F_\ell\):

\[
x_{\ell+1}=x_\ell + F_\ell(x_\ell;\theta_\ell).
\]

La **órbita** del cap. 20 es la trayectoria
\(\{x_\ell\}_{\ell=0}^{22}\) en \(\mathbb{R}^d\).

### 3.5 Por qué el residual importa (matemática de la coherencia)

Si fuera \(x_{\ell+1}=F_\ell(x_\ell)\) sin skip,
pequeños cambios en \(F_\ell\) se componen de forma
exponencialmente inestable (producto de Jacobianos).

Con skip, el Jacobiano es

\[
D x_{\ell+1}= I + DF_\ell,
\]

y si \(\|DF_\ell\|\) es moderada, la dinámica se parece a
una **perturbación de la identidad**: las trayectorias
permanecen cerca de una “línea de base” semántica.
Eso es la versión analítica de “caer de lado”.

---

## 4. Del residual al evento: logits y muestreo

### 4.1 Cabeza de lenguaje

\[
z = E_{\mathrm{out}}\,\mathrm{RMSNorm}(x_L)\in\mathbb{R}^V
\]

(en GGUF: `output.weight`; a veces atado al embedding).

### 4.2 Temperatura

\[
\pi_T(t)=\mathrm{softmax}(z/T)_t
=\frac{e^{z_t/T}}{\sum_{j} e^{z_j/T}}.
\]

- \(T\to 0\): masa en \(\arg\max z\) (órbita “circular”, greedy).  
- \(T\to\infty\): hacia uniforme (máxima entropía).

Entropía \(H(\pi_T)=-\sum_t \pi_T(t)\log\pi_T(t)\) crece con \(T\).

### 4.3 Top-k

Sea \(S_k\subset\mathcal{V}\) los \(k\) índices de mayor \(z_t\).
Se redefine

\[
\pi(t)\propto e^{z_t/T}\mathbf{1}_{t\in S_k}.
\]

Es un **recorte del soporte**: el horizonte de eventos
permitidos se reduce a \(k\) destinos.

### 4.4 Generación como cadena de Markov

Condicionado a \(\theta\) y a la política \(\pi\),

\[
\mathbb{P}(t_{1:N})=\prod_{n=0}^{N-1}
\pi\big(t_{n+1}\mid t_{1:n};\theta\big).
\]

Cambiar \(\theta\) (perturbación) cambia la familia de
cadenas; no solo un sample aislado.

---

## 5. Espacio de pesos y superficie de coherencia

### 5.1 El modelo como punto

\[
\theta\in\mathbb{R}^N,\qquad N\approx 1.1\times 10^9.
\]

Casi todo el volumen de \(\mathbb{R}^N\) produce basura.
Definimos de forma operativa (no topológica rigurosa)

\[
\mathcal{C}=\{\theta : \text{el texto de }f_\theta\text{ es coherente en una batería de prompts}\}.
\]

Dreaming estudia **movimientos** \(\theta\mapsto\theta'\)
que permanecen en \(\mathcal{C}\) pero cambian la
distribución de estilos (perspectivas).

### 5.2 amplify_subspace (mystical)

Sobre un tensor aplanado \(w\in\mathbb{R}^m\)
(o concatenación de tensores de capa):

1. Muestrear \(v\sim\mathcal{N}(0,I_m)\), normalizar \(\hat v=v/\|v\|\).  
2. Proyección escalar \(p=\langle w,\hat v\rangle\).  
3. Actualizar

\[
w' = w + \varepsilon\, p\, \hat v
= w + \varepsilon\, ( \hat v\hat v^\top ) w.
\]

Es un **rank-1 update** en dirección aleatoria:

\[
w'=(I+\varepsilon\, \hat v\hat v^\top)w.
\]

Autovalor \(1+\varepsilon\) en la dirección \(\hat v\),
\(1\) en el hiperplano ortogonal.

**Lectura:** se amplifica una componente; se preserva
la estructura en el resto → candidato a movimiento
**tangente** a \(\mathcal{C}\) si \(\varepsilon\) es moderado.

### 5.3 Ruido escalado (contraejemplo)

\[
w'_i = w_i + \varepsilon\, \xi_i\, |w_i|,\quad \xi_i\sim\mathcal{N}(0,1).
\]

Rompe correlaciones entre coordenadas: empuje
**genérico**, no rank-1 alineado. Empíricamente,
para \(\varepsilon\) grande, \(\theta'\notin\mathcal{C}\).

### 5.4 Intensidad y “sweet spot”

En Q4_0 (pipeline Python v10), \(\varepsilon\approx 0.10\)
fue el sweet spot de muchas técnicas.
En F32 runtime del motor C, a veces hace falta
\(\varepsilon\in[0.3,0.5]\) para ver divergencia de texto
en prompts cortos (cuantización y EOS interactúan).

No hay un único \(\varepsilon^*\) universal: depende
del formato numérico y de la batería.

### 5.5 Fórmula de perspectiva (operativa)

\[
\theta_{\mathrm{persp}}
= \theta_0 + \sum_r \varepsilon_r\, \Delta_r,
\]

con \(\|\varepsilon_r\Delta_r\|\) pequeños y cada \(\Delta_r\)
preservando jerarquía (lowrank, amplify, normrot, …).
Interpolación cercana en \(\mathcal{C}\) tiende a
seguir en \(\mathcal{C}\) (hipótesis empírica del proyecto).

---

## 6. Steering: proyección en el residual

Dada una palabra con embedding \(e_\star\),

\[
\hat u=\frac{e_\star}{\|e_\star\|},\qquad
x \leftarrow x + \lambda\, \langle x,\hat u\rangle\, \hat u
= \big(I+\lambda\,\hat u\hat u^\top\big)x.
\]

Misma álgebra rank-1 que amplify, pero aplicada al
**estado** \(x\), no a los pesos \(\theta\).

- \(\lambda\): fuerza del “viento”.  
- No reescribe el universo; desvía la órbita en vuelo.

---

## 7. Cuantización (el universo en enteros)

En Q4_0, bloques de 32 pesos:

\[
w_i \approx s\cdot (q_i-8),\quad q_i\in\{0,\ldots,15\},
\]

\(s\) en float16 por bloque. La perturbación Python
opera en \(\mathbb{R}\) tras dequant y vuelve a
\((s,q)\). Eso introduce un **error de proyección**
sobre una red discreta: otra razón por la que
\(\varepsilon\) “óptimo” no coincide con el F32 puro.

El motor C en F16 dequantiza en el matmul vía tabla
de \(2^{16}\) entradas: isomorfismo exacto half→float
en cada peso leído, sin re-cuantizar en baseline.

---

## 8. Complejidad de un paso (órbita eficiente)

Para secuencia de longitud \(T\), un paso de atención
naive en una capa es \(\mathcal{O}(T^2 d)\) si se recompute
todo; con **KV-cache** al generar el token \(T\):

\[
\mathcal{O}(T\cdot d\cdot d_h\cdot n_q)
\quad\text{(scores nuevos contra }T\text{ claves)}
\]

más matmuls \(\mathcal{O}(d\cdot d_{ff})\) del FFN.
Por eso el motor actual orbita a ~6–10 tok/s en CPU
y el sin cache se caía a ~0.03 tok/s.

---

## 9. Diccionario: física del libro ↔ fórmula

| Lenguaje del atlas | Objeto matemático |
|--------------------|-------------------|
| Cielo de estrellas | filas de \(E\in\mathbb{R}^{V\times d}\) |
| Isla / constelación | \(c_S\), seeds \(S\) |
| Gravedad entre tokens | \(a_{t,t'}=\mathrm{softmax}(QK^\top/\sqrt{d_h})\) |
| Clima local | \(\mathrm{FFN}:\mathbb{R}^d\to\mathbb{R}^d\) |
| Inercia orbital | \(x\mapsto x+F(x)\) |
| Aire respirable | \(\mathrm{RMSNorm}\) |
| Colapso al evento | \(\pi_T=\mathrm{softmax}(z/T)\) (top-k) |
| Superficie habitable | \(\mathcal{C}\subset\mathbb{R}^N\) |
| Lente mística | \(w'=(I+\varepsilon\hat v\hat v^\top)w\) |
| Viento de steer | \(x'=(I+\lambda\hat u\hat u^\top)x\) |
| Mapa del planetario | PCA: \(e\mapsto V_{:2}^\top(e-\bar e)\) |

---

## 10. Lo que las matemáticas *no* afirman aún

1. \(\mathcal{C}\) no está demostrado como variedad diferencial;
   es una región **operativa** definida por tests.  
2. “Tangente a \(\mathcal{C}\)” es una metáfora geométrica
   respaldada por rank-1 / SVD / ortogonalidad a modos
   destructivos — no un teorema de geometría riemanniana
   del loss.  
3. Los arquetipos no son factores latentes únicos;
   son **direcciones elegidas** con seeds humanas.  
4. La Regla de Oro es **empírica** en TinyLlama;
   no se deriva aquí de un principio variacional general.

La honestidad del microcosmos incluye el borde
de lo no demostrado.

---

## 11. En una frase

Más allá del diagrama de cajas del transformer, este
universo es: **geometría de \(E\) y de \(\theta\)**,
**dinámica residual** \(x\leftarrow x+F_\ell(x)\),
**kernels de atención causales**, **mapas SwiGLU**,
**colapso softmax**, y **updates rank-1** que
desplazan órbitas o pesos sin (a veces) abandonar
la región donde el lenguaje sigue siendo posible.

---

*Siguiente capítulo: El LLM — Un Espejo en Donde Mirarnos.*
