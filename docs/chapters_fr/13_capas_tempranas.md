# Chapitre 13 : Les Premières Couches (0–5)

## Le vestibule du microcosme

Les couches initiales transforment l'embedding « au repos » en une représentation qui ressent déjà des **voisins** et de la **syntaxe**.

```
Couche 0 :   entrée, motifs très locaux
Couche 1 :   syntaxe de base
Couches 2–5 : relations entre mots adjacents
```

(Cette partition est une **hypothèse de travail** du projet, guidée par des expériences d'ablation et par la littérature sur « early = syntax / late = semantics ». Ce n'est pas un découpe rigide dans le code.)

## Quelles forces dominent ici

- **Embedding** pèse encore lourd dans le résiduel (inertie de la naissance).
- **Attention** commence à coupler des bigrammes et des dépendances courtes.
- **FFN** ajuste le lexique local.

## Signaux dans le texte

Si une perturbation précoce « casse » le modèle, on le voit souvent dans la **grammaire** et des tokens étranges, pas seulement dans le ton.

Si le baseline sonne générique et le mystical change le climat sans détruire la syntaxe, les couches précoces continuent d'ancrer la langue.

## Expérience suggérée

Comparer des générations avec ciblage uniquement sur `blk.0`–`blk.5` face à uniquement `blk.13`–`blk.21` (scripts v11 / tensor tests). Hypothèse : early → forme ; late → voix et décision.

---

*Chapitre suivant : Les Couches Intermédiaires (6–12)*
