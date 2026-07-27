# Capítulo 16: Áreas Semánticas y el Mapa

## Doce islas en ℝ²⁰⁴⁸

| Clave | Etiqueta | Semillas (ej.) |
|-------|----------|----------------|
| emotion_pos | Emoción positiva | happy, joy, love, peace… |
| emotion_neg | Emoción negativa | sad, hate, fear, anger… |
| spiritual | Espiritual / sagrado | soul, god, faith, divine… |
| physical | Físico / material | body, rock, water, fire… |
| abstract | Abstracto / ideas | truth, beauty, justice… |
| time | Tiempo | time, past, future, now… |
| social | Social / poder | king, war, law, people… |
| nature | Naturaleza | tree, river, mountain… |
| mind | Mente / cognición | mind, think, dream, brain… |
| death_life | Vida / muerte | life, death, born, die… |
| tech | Técnico / digital | computer, data, code… |
| body_sense | Cuerpo / sentidos | eye, hand, see, voice… |

## Geometría entre islas

- Centroides de áreas distintas: cosine **≈ 0** (ortogonales).  
- Más alineadas: **abstract ↔ mind** (+0.13).  
- Más separadas: **time ↔ social** (−0.09).  
- Opposites léxicos (love/hate): **no antipodales**.

## Cobertura del vocabulario

~99% de tokens aleatorios no caen cerca de ninguna isla
(BPE = fragmentos). Las áreas son **constelaciones de
palabras plenas**, no un particionado total del vocab.

## Visualización

**Mapa PCA 2D interactivo (GitHub):**  
https://htmlpreview.github.io/?https://raw.githubusercontent.com/FiveTechSoft/dreaming/main/inside-tinyllama/exploration/semantic_map.html

| Archivo | Uso |
|---------|-----|
| `semantic_map.html` | Zoom/pan/hover |
| `semantic_areas.json` | Datos de áreas |
| `vectors.tsv` + `metadata.tsv` | TensorFlow Projector (local) |

### Otras herramientas

UMAP, t-SNE, plotly 3D, Embedding Projector.

## Cómo se usa al viajar

1. Elige isla en el mapa.  
2. Arma prompt o `--steer` con semillas de esa isla.  
3. Compara baseline vs mystical.  
4. Anota si el texto “cae” en el clima de la isla.

---

*Siguiente capítulo: Psicoanálisis del Transformer*
