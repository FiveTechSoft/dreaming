# Capitolo 19: Il Futuro dell'Esplorazione

## Prossimi passi tecnici

1. **Istruire il residuale per layer** nel motore C  
   (sonde L0…L21 sugli assi emotion/spirit).  
2. **UMAP/t-SNE** del cielo dei token (quando lo stack lo permetta).  
3. **Portare residual / gradient / selective** a `--perturb`.  
4. **Liberare GGUF F16** dei layer dopo la copia in F32 (meno RAM).  
5. **GitHub Pages** nativo per la mappa (senza htmlpreview).  
6. Ripetere la cartografia su **un altro modello** (trasferimento).

## Prossimi passi del libro

- Figure fisse (PNG) della mappa e del diagramma delle forze.  
- Appendice con la tabella completa dei 15 prompt mistici.  
- Glossario unificato (GQA, superficie, Regola d'Oro, I).

## Invito

Se leggi questo con il repo aperto:

```bash
# 1. Guarda il cielo
#    exploration/semantic_map.html  (o il collegamento htmlpreview)

# 2. Accendi la nave
gcc -O3 -fopenmp -o llm_inference llm_inference.c -lm
./llm_inference modello.F16.gguf "When we dissolve the ego" \
  40 0,7 40 --seed 42 --perturb mystical --intensity 0,5

# 3. Annota che voce è uscita
```

Il microcosmo sta su un disco.
Il macrocosmo è la domanda che ti ha portato qui.
Il percorso tra i due è il mestiere di Dreaming.

**Continua a cartografare.**

---

*Capitolo successivo: Come Orbita Questo Universo.*
