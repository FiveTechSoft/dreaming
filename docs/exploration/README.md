# Exploración del espacio multidimensional de TinyLlama

Mapa de **áreas semánticas** en el embedding de tokens (ℝ²⁰⁴⁸ → PCA 2D).

## Ver los mapas en el navegador

### Opción A — GitHub HTML Preview (sin configurar Pages)

**[▶ Mapa semántico (12 áreas)](https://htmlpreview.github.io/?https://raw.githubusercontent.com/FiveTechSoft/dreaming/main/inside-tinyllama/exploration/semantic_map.html)**

**[▶ Mapa de arquetipos y constelaciones](https://htmlpreview.github.io/?https://raw.githubusercontent.com/FiveTechSoft/dreaming/main/inside-tinyllama/exploration/archetype_map.html)**

**[▶ Mapa 3D — girar con el ratón](https://htmlpreview.github.io/?https://raw.githubusercontent.com/FiveTechSoft/dreaming/main/inside-tinyllama/exploration/map_3d_orbit.html)**  
*(clic izquierdo + arrastrar = orbitar · rueda = zoom · clic derecho = pan)*

**[▶ Juego: recorre el universo por niveles/capas](https://htmlpreview.github.io/?https://raw.githubusercontent.com/FiveTechSoft/dreaming/main/inside-tinyllama/exploration/universe_game.html)**  
*(WASD mover · E portal · 1–5 lentes · Space sample en Softmax)*

### Opción B — GitHub Pages (si está activo en el repo)

```
https://fivetechsoft.github.io/dreaming/inside-tinyllama/exploration/semantic_map.html
```

### Opción C — Local

```bash
# desde la raíz del repo
start inside-tinyllama/exploration/semantic_map.html   # Windows
# o: open / xdg-open según SO
```

## Controles del mapa

| Acción | Efecto |
|--------|--------|
| Rueda del ratón | Zoom |
| Arrastrar | Pan |
| Hover sobre punto | Token + área |

Los **círculos grandes + etiqueta** son semillas del área.  
Las **cruces de color** son centroides.  
Los puntos grises son fondo aleatorio del vocabulario.

## Áreas semánticas

| Área | Descripción |
|------|-------------|
| emotion_pos | Emoción positiva |
| emotion_neg | Emoción negativa |
| spiritual | Espiritual / sagrado |
| physical | Físico / material |
| abstract | Abstracto / ideas |
| time | Tiempo |
| social | Social / poder |
| nature | Naturaleza |
| mind | Mente / cognición |
| death_life | Vida / muerte |
| tech | Técnico / digital |
| body_sense | Cuerpo / sentidos |

## Archivos

| Archivo | Contenido |
|---------|-----------|
| `semantic_map.html` | Visualización interactiva (PCA 2D) |
| `semantic_areas.json` | Definición de áreas, semillas, matriz cosine |
| `space_report.txt` | Informe del recorrido del espacio |
| `space_stats.json` | Estadísticas PCA globales |

## Regenerar

```bash
python map_semantic_areas.py
python explore_tinyllama_space.py
```

## Otras herramientas

- [TensorFlow Embedding Projector](https://projector.tensorflow.org) — subir `vectors.tsv` + `metadata.tsv` (generados localmente; no van al repo por tamaño)
- UMAP / t-SNE — mejor separación local de clusters
- plotly — vistas 3D
