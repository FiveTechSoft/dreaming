# Introduction : Inside TinyLlama

## Un microcosme qui tient sur un disque

Ce livre est le cahier de bord du projet **Dreaming** appliqué à **TinyLlama-1.1B** : un modèle assez petit pour l'ouvrir entièrement et assez riche pour surprendre.

Ce n'est pas un manuel d'utilisation d'un chat. C'est un voyage à travers l'**intérieur** d'un transformer :

- son architecture (22 couches, 9 tenseurs par couche),
- un moteur d'inférence en C que nous pouvons lire ligne par ligne,
- la perturbation de poids comme changement de *perspective*,
- la géométrie de l'espace d'embeddings,
- les « forces » du forward (attention, FFN, résiduel, softmax),
- et l'aller-retour entre le **macrocosme** du sens humain et le **microcosme** des nombres.

## La question centrale

> Quand nous déplaçons les poids avec soin,
> le modèle se casse-t-il ou parle-t-il d'une autre voix ?

La réponse empirique : **il parle d'une autre voix**, si la perturbation préserve la hiérarchie interne des poids. Nous appelons cela naviguer la *surface de cohérence*.

## Comment le livre est organisé

| Partie | Caps. | Thème |
|--------|-------|-------|
| I · Fondations | 1–3 | Qu'est-ce que TinyLlama, structure, moteur C |
| II · Perspectives | 4 | Perturbation DMT, techniques, runtime |
| III · Géométrie | 5–6 | Espace multidimensionnel, macro↔micro |
| IV · Physique du microcosme | 7–9 | Forces, voyage, Règle d'Or |
| V · Anatomie | 10–12 | Attention, FFN, normalisation |
| VI · Couches et conclusion | 13–16 | Couches 0–21, psychanalyse, leçons, avenir |

## Instruments du voyage

- `llm_inference.c` — inférence F16, KV-cache, `--perturb`, `--steer`
- `dmt_perturb_v10.py` / `v11` — GGUFs Q4_0 perturbés
- `map_semantic_areas.py` — atlas des îles sémantiques
- [Carte HTML sur GitHub](https://htmlpreview.github.io/?https://raw.githubusercontent.com/FiveTechSoft/dreaming/main/inside-tinyllama/exploration/semantic_map.html)
- `llama-cli` — batteries rapides en Q4_0

## Une promesse

À la fin du livre, tu n'auras pas un modèle plus grand. Tu auras une **carte** et une **méthode** : descendre du sens au tenseur, remonter du tenseur à la voix, et noter le chemin.

---

*Chapitre 1 : Qu'est-ce que TinyLlama ?*
