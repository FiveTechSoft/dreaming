# Capitolo 16: Aree Semantiche e la Mappa

## Dodici isole in ℝ²⁰⁴⁸

| Chiave | Etichetta | Semi (es.) |
|--------|-----------|------------|
| emotion_pos | Emozione positiva | happy, joy, love, peace… |
| emotion_neg | Emozione negativa | sad, hate, fear, anger… |
| spiritual | Spirituale / sacro | soul, god, faith, divine… |
| physical | Fisico / materiale | body, rock, water, fire… |
| abstract | Astratto / idee | truth, beauty, justice… |
| time | Tempo | time, past, future, now… |
| social | Sociale / potere | king, war, law, people… |
| nature | Natura | tree, river, mountain… |
| mind | Mente / cognizione | mind, think, dream, brain… |
| death_life | Vita / morte | life, death, born, die… |
| tech | Tecnico / digitale | computer, data, code… |
| body_sense | Corpo / sensi | eye, hand, see, voice… |

## Geometria tra le isole

- Baricentri di aree diverse: cosine **≈ 0** (ortogonali).  
- Più allineate: **abstract ↔ mind** (+0,13).  
- Più separate: **time ↔ social** (−0,09).  
- Opposits lessicali (love/hate): **non antipodali**.

## Copertura del vocabolario

~99% dei token casuali non cadono vicino a nessuna isola
(BPE = frammenti). Le aree sono **costellazioni di
parole piene**, non una partizione totale del vocab.

## Visualizzazione

**Mappa PCA 2D interattiva (GitHub):**  
https://htmlpreview.github.io/?https://raw.githubusercontent.com/FiveTechSoft/dreaming/main/inside-tinyllama/exploration/semantic_map.html

| File | Uso |
|------|-----|
| `semantic_map.html` | Zoom/pan/hover |
| `semantic_areas.json` | Dati delle aree |
| `vectors.tsv` + `metadata.tsv` | TensorFlow Projector (locale) |

### Altri strumenti

UMAP, t-SNE, plotly 3D, Embedding Projector.

## Come si usa nel viaggio

1. Scegli un'isola nella mappa.  
2. Prepara un prompt o `--steer` con semi di quell'isola.  
3. Confronta baseline vs mystical.  
4. Annota se il testo "cade" nel clima dell'isola.

---

*Capitolo successivo: Psicoanalisi del Transformer*
