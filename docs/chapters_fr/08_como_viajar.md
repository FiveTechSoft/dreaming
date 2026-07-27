# Chapitre 8 : Comment Voyager dans l'Univers TinyLlama

## Modes de voyage

| Mode | Ce que vous parcourez | Instrument |
|------|----------------------|------------|
| Observatoire | Ciel de tokens ℝ²⁰⁴⁸ | carte HTML, scripts de géométrie |
| Orbite | Un prompt → séquence de tokens | `llm_inference` / llama-cli |
| Lentilles | Perspectives dans l'espace de poids | `--perturb` |
| Courants | Résiduel en direct | `--steer` |
| Laboratoire | Seulement attn ou seulement FFN | scripts Q4 / modèles v11 |

## Kit de vol

```bash
gcc -O3 -march=native -ffast-math -fopenmp \
    -o llm_inference llm_inference.c -lm
# $env:OMP_NUM_THREADS = "8"   # Windows
```

Carte interactive :
https://htmlpreview.github.io/?https://raw.githubusercontent.com/FiveTechSoft/dreaming/main/inside-tinyllama/exploration/semantic_map.html

Modèle : `tinyllama-1.1b.F16.gguf` (poids) + Q4_0 (tokenizer si nécessaire).

## Route A — Atlas

```bash
python map_semantic_areas.py
python explore_tinyllama_space.py
```

Localise les îles (emotion, spiritual, tech…) et choisis des mots pour des prompts ou pour `--steer`.

## Route B — Orbite baseline

```bash
./llm_inference tinyllama-1.1b.F16.gguf \
  "The secret to happiness is" 60 0.7 40 --seed 42
```

Fixe seed et temp en comparant des mondes.

## Route C — Lentille de perspective

```bash
./llm_inference tinyllama-1.1b.F16.gguf \
  "The secret to happiness is" 60 0.7 40 \
  --seed 42 --perturb mystical --intensity 0.50
```

| Technique | Effet typique |
|-----------|---------------|
| mystical / amplify | Existentiel / philosophique |
| noise (I bas) | Nuance ; I haut → déchet |
| blockdiag | Écho / monotone |
| manifold | Local ; I haut risqué |

Protocole : un prompt, `none` → même seed, varie I ou technique.

## Route D — Courants

```bash
./llm_inference tinyllama-1.1b.F16.gguf \
  "The world is" 50 0.7 40 \
  --seed 42 --steer amor --steer-strength 0.15
```

## Route E — GGUF pré-calculés

```bash
llama-cli -m v10_lowrank_10.gguf \
  -p "The secret to happiness is" -n 80 --temp 0.7 --seed 42
```

## Itinéraires

1. **Touriste (30 min) :** carte + 3 prompts baseline/mystical.
2. **Cartographe de I :** I ∈ {0, 0.1, 0.3, 0.5}, même seed.
3. **Chasse d'attracteurs :** ego, mort, conscience, pouvoir.
4. **Double navette :** moteur C F16 vs llama-cli Q4.

## Lois de navigation

1. Une variable par saut.
2. Seed fixe en comparant.
3. I modérée d'abord.
4. Déchet ⇒ vous êtes descendu de la surface.
5. La carte 2D oriente ; ce n'est pas l'espace réel.
6. Setup mystical : ~25 s + ~3.6 Go F32.

## Diagramme

```
prompt → embedding → 22×{attn, FFN} → logits → token
               ↑              ↑
            --steer      poids (+ --perturb)
```

---

*Chapitre suivant : La Règle d'Or Géométrique*
