# Capitolo 21: Archetipi e Costellazioni

## Definizioni di lavoro

| Termine | Significato in questo microcosmo |
|---------|----------------------------------|
| **Archetipo** | Attrattore geometrico: baricentro in ℝ²⁰⁴⁸ di un grappolo di token-semilla che, nella cultura del pre-allenamento, condensano un mito ricorrente |
| **Costellazione** | Il grappolo stesso di semi (stelle fisse del mito) + la sua direzione unitaria nel cielo degli embedding |
| **Allineamento** | Cosine alto tra due baricentri archetipici → miti che si sfiorano |
| **Opposizione** | Cosine basso/negativo → poli del dramma |

Non affermiamo che il modello "creda in Jung".
Affermiamo che **quelle direzioni sono misurabili**
e che alcune coincidono con le voci Dreaming
(Regola d'Oro, `mystical`).

---

## Catalogo di archetipi (15)

### Dodici miti Pearson / Jung (operativi)

| Simbolo | Archetipo | Mito (una riga) | Semi-costellazione (BPE) |
|---------|-----------|------------------|--------------------------|
| ⚔ | **Eroe** | Prova, coraggio, vittoria | ▁hero ▁courage ▁brave ▁quest ▁victory ▁fight ▁strength ▁honor ▁triumph |
| 🌑 | **Ombra** | Nemico interno, mostro | ▁shadow ▁dark ▁evil ▁fear ▁hate ▁demon rage ▁sin |
| 📜 | **Saggio** | Verità, studio, mente | ▁wisdom ▁truth ▁knowledge ▁scholar ▁theory ▁reason ▁logic ▁study ▁philosophy ▁mind |
| 💚 | **Custode** | Cura, guarigione, protezione | ▁care ▁love ▁kind ▁help ▁protect ▁gentle ▁comfort |
| 🧭 | **Esploratore** | Viaggio, frontiera, libertà | ▁explore ▁journey ▁discover ▁travel ▁freedom ▁path ▁wild ▁seek ▁horizon |
| ✨ | **Creatore** | Arte, invenzione, sogno | ▁create ▁art ▁imagine ▁beauty ▁music ▁poem ▁invent ▁craft ▁design ▁dream |
| 👑 | **Sovrano** | Ordine, potere, legge | ▁king ▁power ▁law ▁order ▁rule ▁throne ▁command ▁authority ▁nation |
| 🔮 | **Mago** | Spirito, sacro, visione | ▁magic ▁spirit ▁soul ▁divine ▁sacred ▁mystery ▁transform ▁vision |
| 🌸 | **Innocente** | Speranza, purezza, fede | ▁hope ▁faith ▁pure ▁happy ▁child ▁peace ▁trust ▁simple ▁good |
| ❤ | **Amante** | Desiderio, cuore, bellezza | ▁love ▁desire ▁kiss ▁passion ▁heart ▁beauty ▁tender |
| 🃏 | **Folletto** | Risata, gioco, ironia | ▁laugh ▁play ▁fool ▁smile ▁wit ▁mock ▁silly |
| 🏚 | **Orfano / realista** | Dolore, casa, sopravvivenza | ▁alone ▁lost ▁pain ▁real ▁ordinary ▁poor ▁need ▁belong ▁home |

### Tre archetipi operativi Dreaming

| Simbolo | Archetipo | Mito | Semi |
|---------|-----------|------|------|
| 🕯 | **Voce mistica** | Io, anima, universo, silenzio | ▁soul ▁spirit ego ▁universe ▁divine ▁silence ▁being |
| 🔧 | **Voce pratica** (Regola d'Oro FFN) | Azione, piano, metodo | ▁should ▁step ▁action ▁goal ▁plan ▁work ▁build ▁fix ▁method ▁practice |
| 🎓 | **Voce accademica** (Regola d'Oro Attn) | Teoria, analisi, evidenza | ▁theory ▁analysis ▁study ▁research ▁argument ▁concept ▁framework ▁evidence ▁scholar ▁critique |

---

## Mappa di allineamenti (costellazioni di *miti*)

Misurato: cosine tra baricentri (embedding F16).

### Attrazioni principali (si sfiorano nel cielo)

| cos | Costellazione A | Costellazione B | Lettura |
|-----|-----------------|-----------------|---------|
| **+0,39** | 🔮 Mago | 🕯 Voce mistica | Il clima `mystical` *è* geometricamente mago/spirituale |
| **+0,29** | 📜 Saggio | 🎓 Voce accademica | La Regola d'Oro "attn→accademico" ha un'àncora nel cielo dei token |
| **+0,13** | 💚 Custode | ❤ Amante | Cura e desiderio condividono quartiere affettivo |
| **+0,12** | ✨ Creatore | ❤ Amante | Bellezza / creazione / amore |
| +0,05 | 📜 Saggio | 👑 Sovrano | Sapere e ordine (debole) |

### Opposizioni / polarità

| cos | A | B | Lettura |
|-----|---|---|---------|
| **−0,06** | ⚔ Eroe | 🌑 Ombra | L'asse classico del dramma (anche se morbido: non sono antipodali) |
| −0,06 | 💚 Custode | 🏚 Orfano | Cura vs carenza |
| −0,05 | 🧭 Esploratore | 🃏 Folletto | Sentiero serio vs gioco |
| −0,05 | 🧭 Esploratore | 🕯 Mistico | Frontiera esterna vs interna |
| −0,04 | 📜 Saggio | ❤ Amante | Analisi vs desiderio |
| −0,04 | 🎓 Accademico | ❤ Amante | Stessa tensione nella voce Dreaming |

**Nota geometrica:** quasi tutte le coppie sono vicine a **0**.
Gli archetipi sono **isole** (come le 12 aree semantiche),
non un unico diamante di opposti. Gli allineamenti di +0,3
sono *eccezioni forti* e per questo contano.

---

## Perché le "stelle vicine" sole ingannano

Se si chiedono i k vicini cosine del baricentro in tutto
il vocabolario BPE, appaiono frammenti (`gia`, codici,
altre lingue): in ℝ²⁰⁴⁸ quasi tutto è ortogonale e il
"più vicino" non è semantica pulita.

Per questo definiamo la **costellazione operativa** come:

1. **Semi** (stelle del mito, scelti a mano), e  
2. **Collegamenti ad altri archetipi** (grafo di allineamenti),  

non come i k-NN grezzi del vocabolario completo.

---

## Grafo delle costellazioni (lettura)

```
                    [Saggio]────0.29────[Voce accademica]
                       │
                      0.05
                       │
                  [Sovrano]

[Custode]──0.13──[Amante]──0.12──[Creatore]
     │
    0,04
     │
  [Mago]────────0,39────────[Voce mistica Dreaming]
                                │
                           (mystical / --steer soul)

[Eroe]  ≈⊥  [Ombra]     (polarità debole −0,06)
[Esploratore] ≈⊥ [Mistico, Folletto, Accademico]
```

---

## Come orbitare un archetipo

| Destino | Coordinata di volo |
|---------|-------------------|
| Mago / mistico | prompt esistenziale + `--perturb mystical` e/o `--steer soul` |
| Accademico | prompt analitico + (in Q4) targeting attenzione; o `--steer theory` |
| Pratico | prompt "how to" + targeting FFN / semi step, plan, action |
| Eroe vs Ombra | prompt di conflitto; confrontare baseline vs noise vs mystical |
| Amante / custode | `--steer love` / `care` con strength moderata |

```bash
# Costellazione mistica
./llm_inference modello.F16.gguf "When we dissolve the ego" \
  50 0,7 40 --seed 42 --perturb mystical --intensity 0.50

# Vento verso il Saggio
./llm_inference modello.F16.gguf "Philosophy teaches us that" \
  50 0,7 40 --seed 42 --steer wisdom --steer-strength 0,2
```

---

## Artefatti

| File | Contenuto |
|------|-----------|
| `exploration/archetypes.json` | Baricentri, semi, matrice, allineamenti |
| `exploration/archetype_map.html` | PCA 2D interattiva degli archetipi |
| `map_archetypes.py` | Ri generare l'atlante |

Mappa semantica generale (12 aree tematiche, non archetipi):  
`semantic_map.html`

---

## In una frase

Gli **archetipi** sono direzioni-mito nel cielo dei token;
le **costellazioni** sono i loro semi e i ponti misurati
tra i miti — e il risultato forte del viaggio è che
**Mago ≈ Voce mistica** e **Saggio ≈ Voce accademica**,
cioè: le lenti Dreaming erano già disegnate
come costellazioni nell'embedding.

---

*Capitolo successivo: Osservazione cosciente e proiezione incosciente.*
