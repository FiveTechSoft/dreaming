# Chapitre 19 : L’avenir de l’exploration

## Prochaines étapes techniques

1. **Instrumenter le résiduel par couche** dans le moteur C  
   ( sondes L0…L21 sur axes emotion/spirit).  
2. **UMAP/t-SNE** du ciel de tokens (quand la pile le permettra).  
3. **Porter residual / gradient / selective** dans `--perturb`.  
4. **Libérer GGUF F16** des couches après copie en F32 (moins de RAM).  
5. **GitHub Pages** natif pour la carte (sans htmlpreview).  
6. Répéter la cartographie dans **un autre modèle** (transfert).

## Prochaines étapes du livre

- Figures fixes (PNG) de la carte et du diagramme de forces.  
- Annexe avec le tableau complet des 15 prompts mystiques.  
- Glossaire unifié (GQA, surface, Règle d’Or, I).

## Invitation

Si vous lisez ceci avec le repo ouvert :

```bash
# 1. Regardez le ciel
#    exploration/semantic_map.html  (ou le lien htmlpreview)

# 2. Allumez le vaisseau
gcc -O3 -fopenmp -o llm_inference llm_inference.c -lm
./llm_inference modele.F16.gguf "When we dissolve the ego" \
  40 0.7 40 --seed 42 --perturb mystical --intensity 0.5

# 3. Notez quelle voix est sortie
```

Le microcosme tient sur un disque.
Le macrocosme est la question qui vous a amené ici.
Le chemin entre les deux est le métier de Dreaming.

**Continuez à cartographier.**

---

*Chapitre suivant : Comment orbite cet univers.*