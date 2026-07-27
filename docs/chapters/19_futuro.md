# Capítulo 19: El Futuro de la Exploración

## Próximos pasos técnicos

1. **Instrumentar el residual por capa** en el motor C  
   (sondas L0…L21 sobre ejes emotion/spirit).  
2. **UMAP/t-SNE** del cielo de tokens (cuando el stack lo permita).  
3. **Portar residual / gradient / selective** a `--perturb`.  
4. **Liberar GGUF F16** de capas tras copiar a F32 (menos RAM).  
5. **GitHub Pages** nativo para el mapa (sin htmlpreview).  
6. Repetir la cartografía en **otro modelo** (transferencia).

## Próximos pasos de libro

- Figuras fijas (PNG) del mapa y del diagrama de fuerzas.  
- Apéndice con la tabla de 15 prompts místicos completa.  
- Glosario unificado (GQA, superficie, Regla de Oro, I).

## Invitación

Si lees esto con el repo abierto:

```bash
# 1. Mira el cielo
#    exploration/semantic_map.html  (o el enlace htmlpreview)

# 2. Enciende la nave
gcc -O3 -fopenmp -o llm_inference llm_inference.c -lm
./llm_inference modelo.F16.gguf "When we dissolve the ego" \
  40 0.7 40 --seed 42 --perturb mystical --intensity 0.5

# 3. Anota qué voz salió
```

El microcosmos cabe en un disco.
El macrocosmos es la pregunta que te trajo aquí.
El camino entre ambos es el oficio de Dreaming.

**Sigue cartografiando.**

---

*Siguiente capítulo: Cómo Orbita Este Universo.*
