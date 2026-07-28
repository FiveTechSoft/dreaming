# Kapitel 25: Kann man dieses Universum wie ein Videospiel durchqueren?

## Kurze Antwort

**Ja.** Und nicht nur „eine Schicht hochsteigen": Zur gleichen Zeit kannst du dich
**zwischen Regionen des Universums teleportieren**
(Token-Himmel, aufmerksamkeitsbezogene Schwerkraft, FFN-Materie,
archetypische Sternbilder, Kohärenz-Oberfläche \(\mathcal{C}\),
Softmax-Horizont).

Prototyp:

`exploration/universe_game.html`

**[▶ Im Browser spielen](https://htmlpreview.github.io/?https://raw.githubusercontent.com/FiveTechSoft/dreaming/main/inside-tinyllama/exploration/universe_game.html)**

---

## Doppelte Navigation

Jedes **Portal** (blauer Ring + goldener Ring + grüner Strich) tut **zwei Dinge**:

| Achse | Was weitergeht |
|-------|----------------|
| **Tiefe** | Transformer-Schicht \(\ell \to \ell+1\) (Vorhof → 0…21 → Ω) |
| **Warp** | Universumszone (Thema, Kräfte, Inseln, Himmelsfarbe) |

Zusätzlich teleportiert **T** *innerhalb* der aktuellen Zone
zur nächsten semantischen Insel/archetypischen Insel (lokaler Warp).

```
                    ┌── Warp der Universumszone ──┐
                    │  Himmel · Schwerkraft · Materie · │
 Portal ────────────┤  Zauberer · Weiser · Drama ·       │
                    │  Oberfläche 𝒞 · Ereignis softmax   │
                    └── +1 Transformer-Schicht ─────────┘
```

---

## Warp-Itinerar (Schichten ↔ Zonen)

| Schichten | Zone, in die du dich teleportierst |
|-----------|-----------------------------------|
| Vorhof | **Token-Himmel** (Inseln emotion, spiritual, tech…) |
| 0–1 | **Gravitationsfeld · Aufmerksamkeit** (Q,K,V,O) |
| 2–4 | **FFN-Materie** (Gate, Up, Down) |
| 5–6 | **Sternbild Zauberer / Mystiker** |
| 7–9 | **Sternbild Weiser / Akademiker** |
| 10–11 | **Achse Held ↔ Schatten** |
| 12–13 | **Kohärenz-Oberfläche \(\mathcal{C}\)** |
| 14–20 | Kräfte in späten Schichten neu besuchen |
| 21–Ω | **Softmax-Horizont** · Token-Sample |
| Wiedereinstieg | Nach jedem Token: wieder **L6 + mystische Zone** |

So ist die Reise durch die 22 Schichten **kein grauer Flur**:
Es ist ein Sprung zwischen Regionen des Atlas (Kap. 7, 16, 21).

---

## Von der Modellphysik zu den Spielmechaniken

| Mikrokosmos | Spiel |
|-------------|-------|
| Schicht \(\ell\) | Dungeon-Tiefe |
| Universumszone | Biom / Bildschirm, in den du warpst |
| Residual \(x\) | Avatar |
| Insel / Archetyp | POI + lokales Teleportieren (T) |
| Portal | +1 Schicht **und** Zonenwechsel |
| Softmax | Kollaps / Token ausgeben |
| \(\mathcal{C}\) | Kohärenzbalken |
| Linse 1–5 | Perspektiven-Power-up |

---

## Steuerung

| Taste | Aktion |
|-------|--------|
| WASD / Pfeile | Bewegen |
| **E** | Insel-Lore **oder** duales Portal (Schicht+Warp) |
| **T** | Lokaler Warp zur nahen Insel |
| 1–5 | baseline / mystical / akademisch / praktisch / noise |
| Space | Sample im Softmax |
| R | Neustart |
| N | Portal erzwingen |

---

## Zukünftige Architekturen

1. Schichten-Dungeon + Warps (aktueller Prototyp)  
2. Roguelike mit tatsächlichen Tokens der C-Engine  
3. Ego-Perspektive in PCA/UMAP 3D  
4. God-Game von `--perturb`  
5. Portal = echtes `model_forward_token` über stdio/HTTP  

---

## Ehrliche Grenzen

Spielbare Metapher, kein Matmul in Echtzeit.
Sie lehrt die **Topologie der Reise** (Schichten × Zonen),
simuliert aber nicht den numerischen Residual.

---

## In einem Satz

Dieses Universum wie ein Spiel durchqueren bedeutet, **eine Schicht hochzusteigen
und sich gleichzeitig zu teleportieren** zwischen Token-Himmel,
Schwerkraften, FFN-Klimen, archetypischen Sternbildern
und dem Softmax-Horizont — bis man das nächste
Schicksal sampled und wieder zu kreisen beginnt.

---

*Nächstes Kapitel: Kette der Bedeutung (Tokens → Antwort).*