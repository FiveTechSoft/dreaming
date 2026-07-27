# Capítulo 8: Cómo Viajar por el Universo TinyLlama

## Modos de viaje

| Modo | Qué recorres | Instrumento |
|------|----------------|-------------|
| Observatorio | Cielo de tokens ℝ²⁰⁴⁸ | mapa HTML, scripts de geometría |
| Órbita | Un prompt → secuencia de tokens | `llm_inference` / llama-cli |
| Lentes | Perspectivas en el espacio de pesos | `--perturb` |
| Corrientes | Residual en vivo | `--steer` |
| Laboratorio | Solo attn o solo FFN | scripts Q4 / modelos v11 |

## Kit de vuelo

```bash
gcc -O3 -march=native -ffast-math -fopenmp \
    -o llm_inference llm_inference.c -lm
# $env:OMP_NUM_THREADS = "8"   # Windows
```

Mapa interactivo:  
https://htmlpreview.github.io/?https://raw.githubusercontent.com/FiveTechSoft/dreaming/main/inside-tinyllama/exploration/semantic_map.html

Modelo: `tinyllama-1.1b.F16.gguf` (pesos) + Q4_0 (tokenizer si hace falta).

## Ruta A — Atlas

```bash
python map_semantic_areas.py
python explore_tinyllama_space.py
```

Localiza islas (emotion, spiritual, tech…) y elige
palabras para prompts o para `--steer`.

## Ruta B — Órbita baseline

```bash
./llm_inference tinyllama-1.1b.F16.gguf \
  "The secret to happiness is" 60 0.7 40 --seed 42
```

Fija seed y temp al comparar mundos.

## Ruta C — Lente de perspectiva

```bash
./llm_inference tinyllama-1.1b.F16.gguf \
  "The secret to happiness is" 60 0.7 40 \
  --seed 42 --perturb mystical --intensity 0.50
```

| Técnica | Efecto típico |
|---------|----------------|
| mystical / amplify | Existencial / filosófico |
| noise (I bajo) | Matiz; I alto → basura |
| blockdiag | Eco / monótono |
| manifold | Local; I alto arriesgado |

Protocolo: un prompt, `none` → misma seed, varía I o técnica.

## Ruta D — Corrientes

```bash
./llm_inference tinyllama-1.1b.F16.gguf \
  "The world is" 50 0.7 40 \
  --seed 42 --steer amor --steer-strength 0.15
```

## Ruta E — GGUF precalculados

```bash
llama-cli -m v10_lowrank_10.gguf \
  -p "The secret to happiness is" -n 80 --temp 0.7 --seed 42
```

## Itinerarios

1. **Turista (30 min):** mapa + 3 prompts baseline/mystical.  
2. **Cartógrafo de I:** I ∈ {0, 0.1, 0.3, 0.5}, misma seed.  
3. **Caza de atractores:** ego, muerte, conciencia, poder.  
4. **Doble nave:** motor C F16 vs llama-cli Q4.

## Leyes de navegación

1. Una variable por salto.  
2. Seed fija al comparar.  
3. I moderada primero.  
4. Basura ⇒ bajaste de la superficie.  
5. El mapa 2D orienta; no es el espacio real.  
6. Setup mystical: ~25 s + ~3.6 GB F32.

## Diagrama

```
prompt → embedding → 22×{attn, FFN} → logits → token
              ↑              ↑
           --steer      pesos (+ --perturb)
```

---

*Siguiente capítulo: La Regla de Oro Geométrica*
