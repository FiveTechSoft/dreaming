# Capitolo 27: Ogni Layer è un Ascensore

## L'immagine

Un edificio ha piani.
Non cammini dal piano 3 al 17 per aria:
entri in un **ascensore**, le porte si chiudono,
e aprendosi il mondo è un altro — stessa torre,
altro livello dell'universo.

In TinyLlama la torre ha **22 piani**
(più il vestibolo degli embedding e la terrazza del softmax).

Ogni layer \(\ell\) è un **ascensore**:

```
porte si chiudono:   RMSNorm
viaggio:             Attenzione + residuale + FFN + residuale
porte si aprono:     residuale trasformato nel "piano" ℓ+1
```

Non ti teletrasporti fuori dall'edificio.
Ti **elevi** dentro lo stesso residuale \(x\in\mathbb{R}^{2048}\),
ma il **paesaggio** (zona dell'universo) cambia.

---

## 1. L'edificio TinyLlama

```
        ┌─────────────────────────────┐
   Ω    │  TERRAZZA · Softmax / Sample│  ← risposta (token)
        ├─────────────────────────────┤
  21    │  piano 21 · prep. collasso  │
  20    │  …                          │
   ⋮    │  INTEGRAZIONE / SEMANTICA   │  ← legami, dramma, 𝒞
  13    │  …                          │
        ├─────────────────────────────┤
  12    │  …                          │
   ⋮    │  IDEE PURE                  │  ← quadri, mago, saggio
   6    │  …                          │
        ├─────────────────────────────┤
   5    │  …                          │
   ⋮    │  DETTAGLI DI FORMA          │  ← sintassi, vicini
   0    │  piano 0 · ingresso         │
        ├─────────────────────────────┤
  −1    │  VESTIBOLO · Embedding      │  ← cielo dei token
        └─────────────────────────────┘
                 ▲
            prompt / token
```

Ogni freccia verticale è un ascensore \(F_\ell\):

\[
x_{\ell+1} = x_\ell + F_\ell(x_\ell;\theta_\ell)
\]

Il passeggero è sempre lo stesso tipo di oggetto
(un vettore di 2048 dim). Il **livello dell'universo**
è ciò che quel vettore *significa* dopo il viaggio.

---

## 2. Un viaggio nell'ascensore (dall'interno)

A ogni piano \(\ell\):

| Momento | Operazione | Analogia dell'ascensore |
|---------|-----------|------------------------|
| 1 | `attn_norm` | Luci della cabina; si stabilizza il piano |
| 2 | Q, K, V + RoPE | Sensori: chi percepisci nell'edificio |
| 3 | Softmax causale | Attrazione solo verso piani/passeggeri già presenti (passato) |
| 4 | \(x \mathrel{+}= \mathrm{Attn}\) | La spinta della gravità sociale del testo |
| 5 | `ffn_norm` | Un'altra calibrazione |
| 6 | SwiGLU FFN | Clima del piano (materia locale) |
| 7 | \(x \mathrel{+}= \mathrm{FFN}\) | Esci sul ballatoio con un'altra aria |

Le porte dell'ascensore non ti lasciano in un vettore
di altra dimensione: esci su **un altro ballatoio dello stesso
corridoio di 2048**, ma il "quartiere" è cambiato.

---

## 3. Piano ↔ livello dell'universo

Non è solo un numero \(\ell\). Ogni tratto di piani
corrisponde a un **livello dell'atlante** (catena del cap. 26
+ zone del gioco):

| Piani (layer) | Livello dell'universo | Catena del significato |
|---------------|----------------------|----------------------|
| Vestibolo | Cielo dei token / isole | Token → Embedding |
| 0 – 5 | Quartiere della forma | Dettagli di forma |
| 6 – 12 | Quartiere delle idee pure | Idee pure (Mago, Saggio…) |
| 13 – 20 | Quartiere della semantica legata | Semantica + dramma + \(\mathcal{C}\) |
| 21 | Anti-terrazza | Dettagli fini / prep. risposta |
| Softmax | Terrazza · collasso | Risposta → nuovo token |

Il gioco (`universe_game.html`) rende esplicito ciò che
il forward fa in silenzio:

> **Salire di piano = prendere l'ascensore del layer**  
> **e al tempo stesso atterrare in un'altra zona della mappa dell'universo.**

---

## 4. Perché "ascensore" e non "tunnel infinito"?

Un tunnel suggerisce un solo paesaggio allungato.
Un ascensore insiste su tre fatti:

1. **Stessa torre** — la dimensione del residuale non cambia (\(d=2048\)).  
2. **Fermate discrete** — 22 applicazioni \(F_\ell\), non un flusso continuo anonimo.  
3. **Mondi diversi per piano** — sintassi ≠ idea pura ≠ collasso al vocabolario.

La KV-cache è la **memoria dell'edificio**:
i passeggeri dei piani temporali precedenti
(restano lì come K, V) tirano di te a ogni fermata.

---

## 5. Pulsantiera dell'ascensore (controlli Dreaming)

| Pulsante | Effetto |
|----------|---------|
| Prompt | In quale vestibolo entri (quale embedding iniziale) |
| Seed / temp / top-k | Come si sceglie il destino nella terrazza |
| `--perturb mystical` | Cambia la **meccanica di tutti gli ascensori** (metrica di \(F_\ell\)) |
| `--steer soul` | Vento dentro la cabina (spinge \(x\) verso un asse) |
| Lente accademica / pratica | Bias verso pulsanti di attenzione o di FFN (Regola d'Oro) |

Non scegli solo il piano 7.
Scegli **come si comporta l'ascensore** in tutti i piani.

---

## 6. Una salita completa (narrata)

1. **Vestibolo** — nasci come \(e_t\); vicino alle isole love/tech/spirit.  
2. **Ascensori 0–5** — ti ordinano i vestiti (forma, vicini).  
3. **Ascensori 6–12** — il corridoio si riempie di idee: mago, saggio, quadro.  
4. **Ascensori 13–20** — le idee si *legano* (semantica, tensione, coerenza).  
5. **Ascensore 21 + terrazza** — l'universo si rifiuta di restare in continuo:
   collassa in un token.  
6. **Riavvio** — quel token torna nel vestibolo; nuova salita.

Questo è **orbitare** (cap. 20) letto come **ascensore in loop**.

---

## 7. Matematiche minime

Ascensore del piano \(\ell\):

\[
\begin{aligned}
h &= \mathrm{RMSNorm}(x_\ell; w_a^{(\ell)}) \\
x' &= x_\ell + \mathrm{Attn}_\ell(h) \\
h' &= \mathrm{RMSNorm}(x'; w_f^{(\ell)}) \\
x_{\ell+1} &= x' + \mathrm{FFN}_\ell(h')
\end{aligned}
\]

Teletrasporto di *zona* (nel gioco / nella lettura):
non è un operatore extra del GGUF; è l'**etichetta
dell'atlante** che mettiamo al ballatoio \(\ell\)
(sky, gravity, matter, mage, sage, surface, event…).

---

## 8. In una frase

Ogni layer è un **ascensore**: il residuale entra,
si lascia spingere dalla gravità atencionale e il clima FFN,
e aprendosi le porte è in **un altro livello dell'universo
TinyLlama** — stessa dimensione, altra altezza di senso —
fino alla terrazza dove il softmax sceglie il prossimo destino
e di nuovo chiama l'ascensore.

---

*Gioco: portale = salire piano + warp di zona.*  
*Catena: cap. 26 · Orbita: cap. 20 · Forze: cap. 7.*
