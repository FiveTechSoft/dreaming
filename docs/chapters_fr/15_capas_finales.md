# Chapitre 15 : Les dernières couches (13–21)

## Intégration et décision

```
Couches 13–20 : intégration globale
Couche 21 :    dernière transformation avant output_norm
ensuite :      lm_head → logits → sample
```

Ici le résiduel se prépare pour l’**effondrement**
au vocabulaire : la force VI de l’atlas (softmax).

## Ce qui se joue à la fin

- Mélange de thèmes assemblés au milieu.  
- Préférences fines de style (formel vs simple).  
- Proximité aux tokens de fermeture (`</s>`) — c’est pourquoi
  parfois baseline et mystical coïncident dans des sorties
  **très courtes** avec la même seed (même puits de EOS).

## Expérience de la batterie mystique

Avec I=0.50 et 60 tokens max., plusieurs prompts ont rempli
le budget de longueur ; d’autres ont coupé à 2–8 tokens.
Les couches finales + sampling décident **quand s’arrêter**
autant que **quoi dire**.

## Règle pratique

Pour comparer les perspectives, utilisez `n` élevé et regardez
le **corps** du texte, pas seulement la première phrase
si le modèle se précipite vers l’EOS.

---

*Chapitre suivant : Zones sémantiques et la carte*