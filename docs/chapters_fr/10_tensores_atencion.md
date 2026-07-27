# Chapitre 10 : Les Tenseurs d'Attention

## Quatre planètes par couche

Dans chacune des 22 couches :

| Tenseur | Question | Forme logique (TinyLlama) |
|---------|----------|---------------------------|
| **Q** (attn_q) | Qu'est-ce que je recherche ? | [2048, 2048] |
| **K** (attn_k) | Qu'offre-t-on ? | [256, 2048] (4×64) |
| **V** (attn_v) | Que transmet-on ? | [256, 2048] |
| **O** (attn_output) | Comment intègre-t-on ? | [2048, 2048] |

Plus `attn_norm` (RMSNorm avant le bloc).

## GQA : 32 yeux, 4 mémoires

TinyLlama n'a pas 32 K et 32 V indépendants. Il a **32 têtes Q** et **4 KV** partagées (chaque KV dessert 8 Q). Moins de mémoire cache, même idée de multi-tête.

## La formule

```
scores = (Q Kᵀ) / √64
weights = softmax_causal(scores)
out = weights V
out = O · out
x = x + out          # résiduel
```

Dans le moteur C : seul le token nouveau calcule Q/K/V ; K et V sont stockés dans le **KV-cache**.

## Rôle dans l'univers

- **Force de longue portée** entre tokens.
- **Règle d'Or :** toucher l'attention → perspective académique.
- Dans l'atlas des forces : ~19% de la masse, portée *maximale*.

## Quoi observer en expérimentant

- Le texte cite-t-il, structure-t-il, « argumente-t-il » ?
- La *relation* entre les idées change-t-elle plus que le lexique isolé ?
→ Signal que le champ attentionnel domine le climat.

---

*Chapitre suivant : Les Tenseurs FFN*
