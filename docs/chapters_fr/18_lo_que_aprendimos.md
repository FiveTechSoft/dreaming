# Chapitre 18 : Ce que nous avons appris

## Résultats principaux

1. **TinyLlama est un microcosme cartographiable**  
   22 couches, 9 tenseurs/couche, dims réelles 2048 / 5632 / GQA 32×4.

2. **Un moteur C propre clôt la boucle**  
   GGUF F16, BPE, KV-cache, OpenMP, ~6–10 tok/s,
   `--perturb` et `--steer` en runtime.

3. **Les poids contiennent des perspectives**  
   Pas seulement des faits : tons et voix. Perturber avec
   hiérarchie préservée change la voix, n’éteint pas la parole.

4. **Règle d’Or géométrique**  
   Attn → académique ; FFN → pratique ; Emb → simple.

5. **Surface de cohérence**  
   Tangente (amplify) habitable ; normale (noise fort) vide.

6. **Espace d’embeddings : îles, pas un axe unique**  
   Douze zones sémantiques quasi orthogonales ; PCA utilise
   des centaines de dimensions ; opposés non antipodaux.

7. **Macrocosme ↔ microcosme**  
   La méthode est aller-retour : sens ↔ tenseur ↔ texte.

8. **Outils de voyage**  
   Carte HTML sur GitHub, scripts de géométrie, llama-cli
   pour batteries Q4, moteur C pour horlogerie fine.

## Limites de l’étude

- Évaluation de « perspective » encore qualitative.  
- TinyLlama ≠ modèles frontaliers (la surface peut changer).  
- Carte 2D est projection, pas la géométrie véritable.  
- Runtime F32 de perturbation exige beaucoup de RAM.  
- Pas toutes les techniques v10/v11 sont dans le moteur C.

## Questions ouvertes

- Où (quelles couches) s’allume le climat mystique dans le résiduel ?  
- Les directions de perspective se transfèrent-elles entre modèles ?  
- Comment mesurer la perspective de façon automatique et fiable ?  
- Que se passe-t-il sur la surface de cohérence à 7B / 70B ?

---

*Chapitre suivant : L’avenir de l’exploration*