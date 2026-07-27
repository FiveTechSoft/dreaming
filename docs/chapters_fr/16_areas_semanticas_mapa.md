# Chapitre 16 : Zones sémantiques et la carte

## Douze îles en ℝ²⁰⁴⁸

| Clé | Étiquette | Graines (ex.) |
|-------|----------|----------------|
| emotion_pos | Émotion positive | joie, amour, paix… |
| emotion_neg | Émotion négative | tristesse, haine, peur, colère… |
| spiritual | Spirituel / sacré | âme, dieu, foi, divin… |
| physical | Physique / matériel | corps, roche, eau, feu… |
| abstract | Abstrait / idées | vérité, beauté, justice… |
| time | Temps | temps, passé, futur, maintenant… |
| social | Social / pouvoir | roi, guerre, loi, peuple… |
| nature | Nature | arbre, rivière, montagne… |
| mind | Esprit / cognition | esprit, penser, rêve, cerveau… |
| death_life | Vie / mort | vie, mort, naître, mourir… |
| tech | Technique / numérique | ordinateur, données, code… |
| body_sense | Corps / sens | œil, main, voir, voix… |

## Géométrie entre îles

- Centroïdes de zones distinctes : cosine **≈ 0** (orthogonales).  
- Plus alignées : **abstract ↔ mind** (+0.13).  
- Plus séparées : **time ↔ social** (−0.09).  
- Opposés lexicaux (love/hate) : **non antipodaux**.

## Couverture du vocabulaire

~99% des tokens aléatoires ne tombent près d’aucune île
(BPE = fragments). Les zones sont des **constellations de
mots pleins**, pas une partition totale du vocabulaire.

## Visualisation

**Carte PCA 2D interactive (GitHub) :**  
https://htmlpreview.github.io/?https://raw.githubusercontent.com/FiveTechSoft/dreaming/main/inside-tinyllama/exploration/semantic_map.html

| Fichier | Utilisation |
|---------|-----|
| `semantic_map.html` | Zoom/pan/hover |
| `semantic_areas.json` | Données des zones |
| `vectors.tsv` + `metadata.tsv` | TensorFlow Projector (local) |

### Autres outils

UMAP, t-SNE, plotly 3D, Embedding Projector.

## Comment l’utiliser pour voyager

1. Choisissez une île sur la carte.  
2. Construisez un prompt ou `--steer` avec les graines de cette île.  
3. Comparez baseline vs mystical.  
4. Notez si le texte « tombe » dans le climat de l’île.

---

*Chapitre suivant : Psychanalyse du Transformer*