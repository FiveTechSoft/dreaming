# Capítulo 25: ¿Se puede recorrer este universo como un videojuego?

## Respuesta corta

**Sí.** Y no solo “subir de capa”: al mismo tiempo puedes
**teletransportarte** entre regiones del universo
(cielo de tokens, gravedad atencional, materia FFN,
constelaciones arquetípicas, superficie \(\mathcal{C}\),
horizonte Softmax).

Prototipo:

`exploration/universe_game.html`

**[▶ Jugar en el navegador](https://htmlpreview.github.io/?https://raw.githubusercontent.com/FiveTechSoft/dreaming/main/inside-tinyllama/exploration/universe_game.html)**

---

## Doble navegación

Cada **portal** (anillo azul + anillo dorado + trazo verde) hace **dos cosas**:

| Eje | Qué avanza |
|-----|------------|
| **Profundidad** | Capa del transformer \(\ell \to \ell+1\) (vestíbulo → 0…21 → Ω) |
| **Warp** | Zona del universo (tema, fuerzas, islas, color del cielo) |

Además, **T** teletransporta *dentro* de la zona actual
a la isla semántica/arquetípica más cercana (warp local).

```
                    ┌── warp de zona del universo ──┐
                    │  sky · gravity · matter ·      │
 portal ────────────┤  mage · sage · drama ·         │
                    │  surface 𝒞 · event softmax     │
                    └── +1 capa transformer ─────────┘
```

---

## Itinerario de warps (capas ↔ zonas)

| Capas | Zona a la que te teletransportas |
|-------|----------------------------------|
| vestíbulo | **Cielo de tokens** (islas emotion, spiritual, tech…) |
| 0–1 | **Campo gravitatorio · Atención** (Q,K,V,O) |
| 2–4 | **Materia FFN** (Gate, Up, Down) |
| 5–6 | **Constelación Mago / Místico** |
| 7–9 | **Constelación Sabio / Académico** |
| 10–11 | **Eje Héroe ↔ Sombra** |
| 12–13 | **Superficie de coherencia \(\mathcal{C}\)** |
| 14–20 | Revisitar fuerzas en capas tardías |
| 21–Ω | **Horizonte Softmax** · sample de token |
| reentrada | Tras cada token: otra vez **L6 + zona mística** |

Así el recorrido de las 22 capas **no es un pasillo gris**:
es un salto entre regiones del atlas (caps. 7, 16, 21).

---

## De la física del modelo a las mecánicas

| Microcosmos | Juego |
|-------------|--------|
| Capa \(\ell\) | Profundidad del dungeon |
| Zona del universo | Bioma / pantalla a la que haces warp |
| Residual \(x\) | Avatar |
| Isla / arquetipo | POI + teletransporte local (T) |
| Portal | +1 capa **y** cambio de zona |
| Softmax | Colapso / emitir token |
| \(\mathcal{C}\) | Barra de coherencia |
| Lente 1–5 | Power-up de perspectiva |

---

## Controles

| Tecla | Acción |
|-------|--------|
| WASD / flechas | Mover |
| **E** | Lore de isla **o** portal dual (capa+warp) |
| **T** | Warp local a isla cercana |
| 1–5 | baseline / mystical / académica / práctica / noise |
| Space | Sample en Softmax |
| R | Reiniciar |
| N | Forzar portal |

---

## Arquitecturas futuras

1. Dungeon de capas + warps (prototipo actual)  
2. Roguelike con tokens reales del motor C  
3. First-person en PCA/UMAP 3D  
4. God-game de `--perturb`  
5. Portal = `model_forward_token` real vía stdio/HTTP  

---

## Límites honestos

Metáfora jugable, no matmul en tiempo real.
Enseña la **topología del viaje** (capas × zonas),
no simula el residual numérico.

---

## En una frase

Recorrer este universo como un juego es **subir de capa
y, al mismo tiempo, teletransportarse** entre cielo de
tokens, gravedades, climas FFN, constelaciones arquetípicas
y el horizonte del softmax — hasta samplear el siguiente
destino y volver a orbitar.

---

*Siguiente capítulo: Cadena del significado (tokens → respuesta).*
