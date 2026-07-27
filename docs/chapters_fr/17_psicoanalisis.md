# Chapitre 17 : Psychanalyse du Transformer

## Une métaphore (pas un diagnostic clinique)

Freud distinguait des couches du mental.
Sans forcer l’identité, la pile du transformer
admet une **lecture par profondeur** :

| Instance | Composant | Fonction approximative |
|-----------|------------|-------------------|
| **Inconscient** | Embeddings | Associations latentes, « ce qui est déjà su » sans contexte |
| **Préconscient** | Attention + couches médianes | Met en scène relations et cadres |
| **Conscient** | Dernières couches + logits + sample | Ce qui est dit *maintenant*

## Ça / Moi / Surmoi (lecture libre)

| | Analogie dans le modèle |
|--|------------------------|
| **Ça** | Pulsions de poids brut, directions sémantiques brutes |
| **Moi** | Résiduel + normes : négocie entre pulsions et forme |
| **Surmoi** | Biais d’entraînement / sécurité / style « correct » du baseline |

La perturbation `mystical` ne « libère pas le ça » au sens freudien :
**remélange** l’équilibre des voix déjà présentes dans les poids.

## Pourquoi noter cette métaphore

- Aide à *parler* de l’intérieur sans seulement des matrices.  
- Lie avec le zoom macro↔micro (chap. 6).  
- Ne remplace pas les mesures : c’est une **carte narrative**.

## Limite

Un LLM n’a pas d’inconscient subjectif.
Il a des **statistiques compressées**. La métaphore est
un outil d’exploration, pas une ontologie.

---

*Chapitre suivant : Ce que nous avons appris*