# Kapitel 8: Wie man durch das TinyLlama-Universum reist

## Reisemodi

| Modus | Was du durchquerst | Werkzeug |
|-------|-------------------|----------|
| Observatorium | Token-Himmel ℝ²⁰⁴⁸ | HTML-Karte, Geometrie-Skripte |
| Umlaufbahn | Ein Prompt → Token-Sequenz | `llm_inference` / llama-cli |
| Linsen | Perspektiven im Gewichtsraum | `--perturb` |
| Strömungen | Residual in Echtzeit | `--steer` |
| Labor | Nur attn oder nur FFN | Q4-Skripte / v11-Modelle |

## Flugzeug-Kit

```bash
gcc -O3 -march=native -ffast-math -fopenmp \
    -o llm_inference llm_inference.c -lm
# $env:OMP_NUM_THREADS = "8"   # Windows
```

Interaktive Karte:  
https://htmlpreview.github.io/?https://raw.githubusercontent.com/FiveTechSoft/dreaming/main/inside-tinyllama/exploration/semantic_map.html

Modell: `tinyllama-1.1b.F16.gguf` (Gewichte) + Q4_0 (Tokenizer falls nötig).

## Route A — Atlas

```bash
python map_semantic_areas.py
python explore_tinyllama_space.py
```

Lokalisiere Inseln (emotion, spiritual, tech…) und wähle
Worte für Prompts oder für `--steer`.

## Route B — Baseline-Umlaufbahn

```bash
./llm_inference tinyllama-1.1b.F16.gguf \
  "The secret to happiness is" 60 0.7 40 --seed 42
```

Setze Seed und Temperatur beim Vergleich von Welten.

## Route C — Perspektiven-Linse

```bash
./llm_inference tinyllama-1.1b.F16.gguf \
  "The secret to happiness is" 60 0.7 40 \
  --seed 42 --perturb mystical --intensity 0.50
```

| Technik | Typischer Effekt |
|---------|------------------|
| mystical / amplify | Existenziell / philosophisch |
| noise (niedriges I) | Nuance; hohes I → Müll |
| blockdiag | Echo / monoton |
| manifold | lokal; hohes I riskant |

Protokoll: ein Prompt, `none` → derselbe Seed, variiere I oder Technik.

## Route D — Strömungen

```bash
./llm_inference tinyllama-1.1b.F16.gguf \
  "The world is" 50 0.7 40 \
  --seed 42 --steer amor --steer-strength 0.15
```

## Route E — Vorberechnete GGUFs

```bash
llama-cli -m v10_lowrank_10.gguf \
  -p "The secret to happiness is" -n 80 --temp 0.7 --seed 42
```

## Reiserouten

1. **Tourist (30 Min.):** Karte + 3 Prompts baseline/mystical.  
2. **I-Kartograph:** I ∈ {0, 0.1, 0.3, 0.5}, derselbe Seed.  
3. **Attraktoren-Jagd:** Ich, Tod, Bewusstsein, Macht.  
4. **Doppelschiff:** C-Engine F16 vs llama-cli Q4.

## Navigationsregeln

1. Eine Variable pro Sprung.  
2. Fester Seed beim Vergleich.  
3. Mittleres I zuerst.  
4. Müll ⇒ du hast die Oberfläche verlassen.  
5. Die 2D-Karte orientiert; sie ist nicht der echte Raum.  
6. Mystisches Setup: ~25 s + ~3.6 GB F32.

## Diagramm

```
Prompt → Embedding → 22×{attn, FFN} → Logits → Token
              ↑              ↑
           --steer      Gewichte (+ --perturb)
```

---

*Nächstes Kapitel: Die geometrische Goldene Regel*