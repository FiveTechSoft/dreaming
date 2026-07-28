# Capitolo 8: Come Viaggiare nell'Universo TinyLlama

## Modi di viaggio

| Modo | Cosa percorri | Strumento |
|------|---------------|-----------|
| Osservatorio | Cielo di token ℝ²⁰⁴⁸ | mappa HTML, script di geometria |
| Orbita | Un prompt → sequenza di token | `llm_inference` / llama-cli |
| Lenti | Prospettive nello spazio dei pesi | `--perturb` |
| Correnti | Residuale in diretta | `--steer` |
| Laboratorio | Solo attn o solo FFN | script Q4 / modelli v11 |

## Kit di volo

```bash
gcc -O3 -march=native -ffast-math -fopenmp \
    -o llm_inference llm_inference.c -lm
# $env:OMP_NUM_THREADS = "8"   # Windows
```

Mappa interattiva:  
https://htmlpreview.github.io/?https://raw.githubusercontent.com/FiveTechSoft/dreaming/main/inside-tinyllama/exploration/semantic_map.html

Modello: `tinyllama-1.1b.F16.gguf` (pesi) + Q4_0 (tokenizer se necessario).

## Percorso A — Atlante

```bash
python map_semantic_areas.py
python explore_tinyllama_space.py
```

Localizza isole (emotion, spiritual, tech…) e scegli
parole per prompt o per `--steer`.

## Percorso B — Orbita baseline

```bash
./llm_inference tinyllama-1.1b.F16.gguf \
  "The secret to happiness is" 60 0,7 40 --seed 42
```

Fissa seed e temp quando confronti mondi.

## Percorso C — Lente di prospettiva

```bash
./llm_inference tinyllama-1.1b.F16.gguf \
  "The secret to happiness is" 60 0,7 40 \
  --seed 42 --perturb mystical --intensity 0.50
```

| Tecnica | Effetto tipico |
|---------|----------------|
| mystical / amplify | Esistenziale / filosofico |
| noise (I basso) | Sfumatura; I alto → rifiuti |
| blockdiag | Eco / monotono |
| manifold | Locale; I alto rischioso |

Protocollo: un prompt, `none` → stesso seed, varia I o tecnica.

## Percorso D — Correnti

```bash
./llm_inference tinyllama-1.1b.F16.gguf \
  "The world is" 50 0,7 40 \
  --seed 42 --steer amor --steer-strength 0,15
```

## Percorso E — GGUF precalcolati

```bash
llama-cli -m v10_lowrank_10.gguf \
  -p "The secret to happiness is" -n 80 --temp 0,7 --seed 42
```

## Itinerari

1. **Turista (30 min):** mappa + 3 prompt baseline/mystical.  
2. **Cartografo di I:** I ∈ {0, 0,1, 0,3, 0,5}, stesso seed.  
3. **Caccia di attrattori:** io, morte, coscienza, potere.  
4. **Doppia nave:** motore C F16 vs llama-cli Q4.

## Leggi di navigazione

1. Una variabile per salto.  
2. Seed fissa al confronto.  
3. I moderata prima.  
4. Rifiuti ⇒ sei uscito dalla superficie.  
5. La mappa 2D orienta; non è lo spazio reale.  
6. Setup mystical: ~25 s + ~3,6 GB F32.

## Diagramma

```
prompt → embedding → 22×{attn, FFN} → logits → token
               ↑              ↑
            --steer      pesi (+ --perturb)
```

---

*Capitolo successivo: La Regola d'Oro Geometrica*
