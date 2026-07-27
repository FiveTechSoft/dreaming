# Chapitre 11 : Les Tenseurs FFN

## Trois planètes de la matière ordinaire

| Tenseur | Rôle | Forme logique |
|---------|------|---------------|
| **Gate** (ffn_gate) | Porte SiLU | [5632, 2048] |
| **Up** (ffn_up) | Expansion | [5632, 2048] |
| **Down** (ffn_down) | Compression | [2048, 5632] |

Plus `ffn_norm` avant le bloc.

## SwiGLU

```
h' = Down( SiLU(Gate(x)) ⊙ Up(x) )
x  = x + h'
```

Dimension intermédiaire **5632** : le résiduel est expansé dans un espace plus large et revient à 2048.

## Masse dominante

~**69%** des paramètres du modèle vivent ici. Si l'attention est la gravité entre planètes, le FFN est la **physique interne** de chacune.

## Règle d'Or

Perturber le FFN → perspective **pratique** : étapes, conseils, verbes d'action, « comment faire ».

Sélectif `ffn_dream` (v11) : fort en FFN, doux en attention → climat « rêveur mais actionnable ».

## Quoi observer

- Listes numérotées, impératifs, astuces ?
- Moins de « qui se connecte avec qui » et plus de « quoi faire » ?
→ Champ FFN sur le siège du conducteur.

---

*Chapitre suivant : Les Tenseurs de Normalisation*
