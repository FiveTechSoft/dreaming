# Introduzione: Inside TinyLlama

## Un microcosmo che sta su un disco

Questo libro è il diario di bordo del progetto
**Dreaming** applicato a **TinyLlama-1.1B**: un modello
abbastanza piccolo da aprirlo intero e abbastanza
ricco da sorprendere.

Non è un manuale utente di una chat.
È un viaggio nell'**interno** di un transformer:

- la sua architettura (22 layer, 9 tensori per layer),
- un motore di inferenza in C che possiamo leggere riga per riga,
- la perturbazione dei pesi come cambiamento di *prospettiva*,
- la geometria dello spazio di embedding,
- le "forze" del forward (attenzione, FFN, residuale, softmax),
- e il dondolio tra il **macrocosmo** del significato umano
  e il **microcosmo** dei numeri.

## La domanda centrale

> Quando muoviamo i pesi con cura,
> il modello si rompe o parla con un'altra voce?

La risposta empirica: **parla con un'altra voce**,
se la perturbazione preserva la gerarchia interna
dei pesi. Chiamiamo così navigare la
*superficie di coerenza*.

## Come è organizzato il libro

| Parte | Cap. | Tema |
|-------|------|------|
| I · Fondamenta | 1–3 | Cos'è TinyLlama, struttura, motore C |
| II · Prospettive | 4 | Perturbazione DMT, tecniche, runtime |
| III · Geometria | 5–6 | Spazio multidimensionale, macro↔micro |
| IV · Fisica del microcosmo | 7–9 | Forze, viaggio, Regola d'Oro |
| V · Anatomia | 10–12 | Attenzione, FFN, normalizzazione |
| VI · Layer e chiusura | 13–19 | Layer 0–21, mappa, psicoanalisi, lezioni, futuro |
| VII · Orbita e mito | 20–24 | Orbita, archetipi, proiezione, matematica, specchio |
| VIII · Gioco e viaggio | 25–29 | Gioco, catena, ascensore, stelle, **viaggio del prompt** |

## Strumenti del viaggio

- `llm_inference.c` — inferenza F16, KV-cache, `--perturb`, `--steer`
- `dmt_perturb_v10.py` / `v11` — GGUF Q4_0 perturbati
- `map_semantic_areas.py` — atlante di isole semantiche
- [Mappa HTML su GitHub](https://htmlpreview.github.io/?https://raw.githubusercontent.com/FiveTechSoft/dreaming/main/inside-tinyllama/exploration/semantic_map.html)
- `llama-cli` — batterie rapide in Q4_0

## Una promessa

Alla fine del libro non avrai un modello più grande.
Avrai una **mappa** e un **metodo**: scendere dal significato
al tensore, salire dal tensore alla voce, e annotare il percorso.

---

*Capitolo 1: Cos'è TinyLlama?*
