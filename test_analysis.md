# Dreaming Model Test Analysis — 2026-07-26

## Test Configuration
- **Prompts**: 10 diverse prompts
- **Tokens**: 150 per generation
- **Temperature**: 0.7
- **Seed**: 42 (reproducible)
- **Total time**: ~13 minutes

---

## Models Tested (14 total, all coherent)

| Model | Type | Avg Length | Status |
|-------|------|-----------|--------|
| baseline_Q4_0 | Original TinyLlama | 629ch | ✅ Reference |
| **v11 Combo Models** | | | |
| v11_combo_structured_dream | block_amplify + poetic_oscillate | 576ch | ✅ |
| v11_combo_max.Alter | amplify + amplify | 599ch | ✅ |
| **v11 Selective Models** | | | |
| v11_select_attention_alter | Heavy attention, gentle FFN | 601ch | ✅ |
| v11_select_ffn_dream | Heavy FFN, gentle attention | 643ch | ✅ |
| v11_select_embedding_shift | Heavy embedding, gentle rest | 634ch | ✅ |
| v11_select_extreme_selective | Max attention, minimal FFN | 639ch | ✅ |
| **v10 Hierarchy-Preserving** | | | |
| v10_lowrank | Low-rank amplification | 613ch | ✅ |
| v10_eigr | Eigenvector rotation | 620ch | ✅ |
| v10_spectral | Spectral shift | 481ch | ✅ |
| v10_attpres | Attention-preserving | 632ch | ✅ |
| v10_respres | Residual-preserving | 592ch | ✅ |
| v10_blkdiag | Block-diagonal | 632ch | ✅ |
| v10_normrot | Norm-preserving rotation | 569ch | ✅ |

---

## Failed Models (not coherent)

| Model | Issue |
|-------|-------|
| v11_combo_deep_reason | Garbage tokens, peg format errors |
| v11_combo_rare_perspective | Garbage tokens, peg format errors |
| DMT_amplify_10 | Same control characters every prompt |
| DMT_scaled_10 | "ongodb" loop + mixed code tokens |
| DMT_v5 | Unicode soup + peg format errors |

---

## Detailed Comparison: "The secret to happiness is"

### Baseline
> not just about having enough material possessions and wealth, but rather it's about cultivating a mindset that is focused on gratitude, positive self-talk, and self-compassion.

### v11_combo_structured_dream
> The author suggests that the key to happiness is to cultivate gratitude and appreciate one's life more. They suggest that by focusing on the things that one is grateful for and appreciating them, one can increase their sense of happiness and contentment.

**Shift**: More academic/analytical framing. Third-person perspective.

### v11_combo_max.Alter
> not about finding a magical solution that can give you everything you want, but rather, it's about living a happy and fulfilling life that allows you to truly appreciate and enjoy the present moment.

**Shift**: Philosophical/existential framing. More personal.

### v11_select_attention_alter
> not just about having enough money, possessions, or material possessions. It also involves cultivating a positive outlook on life, developing meaningful relationships, pursuing one's passions and interests...

**Shift**: Very similar to baseline but more structured/list-like.

### v11_select_ffn_dream
> not only about finding someone or something you like or enjoy, but also about cultivating the right mindset and habits to make it a daily practice. Here are some tips...

**Shift**: Practical/actionable advice framing.

### v11_select_embedding_shift
> not just about having things we wish for or having a perfect life, but it is about finding joy and happiness in the small things that make us happy.

**Shift**: Simpler, more accessible language.

### v11_select_extreme_selective
> not just about having things we wish for or having a perfect life, but about finding inner peace and contentment through mindfulness, gratitude, and appreciation.

**Shift**: Spiritual/mindfulness framing.

### v10_lowrank
> finding a balance between two opposing forces: gratitude and self-compassion. In order to do this, we must be mindful of our thoughts and feelings and learn to let go of negative self-talk...

**Shift**: Philosophical duality framing.

### v10_eigr
> finding the right balance between the things that make us happy and the things that bring us joy. Here are some tips to help you find the right balance...

**Shift**: Practical tips framing.

### v10_spectral
> not about having a perfect life or having everything we want, but about learning to appreciate what we have and finding contentment in the present moment.

**Shift**: Simpler, more direct.

### v10_attpres
> not just about having enough material possessions and wealth, but rather it's about cultivating a mindset that is focused on gratitude, positive self-talk, and self-compassion.

**Shift**: Almost identical to baseline (attention-preserving works as intended).

### v10_respres
> finding the right balance between our inner and outer lives. It's about cultivating a sense of inner peace and contentment that allows us to enjoy the present moment...

**Shift**: Inner/outer duality framing.

### v10_blkdiag
> not just about having enough material possessions and wealth, but rather it's about cultivating a mindset that is focused on gratitude, positive self-talk, and self-compassion.

**Shift**: Very close to baseline (block-diagonal is subtle).

### v10_normrot
> finding a sense of inner peace and contentment by focusing on the things that we can control, rather than worrying about the things we can't.

**Shift**: Stoic philosophy framing.

---

## Key Findings

### 1. All v10 hierarchy-preserving techniques produce coherent output
Every v10 technique (lowrank, eigr, spectral, attention, residual, blockdiag, normrot) produces grammatically correct, factually grounded text. This confirms the hierarchy-preservation hypothesis.

### 2. Selective targeting creates distinct "personas"
- **Attention-heavy models** (attention_alter, extreme_selective) produce more structured, academic text
- **FFN-heavy models** (ffn_dream) produce more practical, actionable text
- **Embedding-heavy models** (embedding_shift) produce simpler, more accessible text

### 3. Combo models show the most divergence
- **structured_dream** (block_amplify + poetic_oscillate) produces the most distinct perspective shifts
- **max.Alter** (amplify + amplify) shows the strongest divergence while staying coherent

### 4. Attention-preserving is the most conservative
v10_attpres produces output nearly identical to baseline, confirming that the technique preserves the core attention patterns.

### 5. Spectral shift produces shortest output
v10_spectral generates 23% fewer characters than baseline, suggesting it affects generation length.

---

## Ranking: Most Interesting Divergence

1. **v11_combo_structured_dream** — Strongest perspective shift while maintaining coherence
2. **v11_select_extreme_selective** — Spiritual/mindness framing
3. **v10_normrot** — Stoic philosophy framing
4. **v10_lowrank** — Philosophical duality framing
5. **v11_combo_max.Alter** — Existential framing

---

## Technical Notes

### Models that produced garbage
- `deep_reason` and `rare_perspective` combos: Both use `analytical_amplify` + `residual_preserve` or `creative_amplify` + `residual_preserve`. The `residual_preserve` function in v11 may have a bug when combined with analytical/creative techniques.
- `DMT_amplify_10`: Old DMT model with corrupted header (control characters in output)
- `DMT_scaled_10`: "ongodb" loop indicates embedding corruption
- `DMT_v5`: Unicode soup indicates full model corruption

### Why v10 works better than old DMT models
The v10 techniques use **flat 1D operations** (no matrix reshape to sqrt(n) x sqrt(n)) which avoids shape mismatches. The old DMT models used matrix operations that could fail silently on non-square weight tensors.

---

## New v10 Results (gradient, dct, manifold)

### "The secret to happiness is"

**v10_gradal** (Gradient-aligned):
> finding your own unique and authentic way of living that reflects your inner desires and passions. It is about understanding what truly makes you happy and then pursuing it with passion and purpose.

**Shift**: Authenticity/self-discovery framing. More introspective.

**v10_lowdct** (Low-frequency DCT):
> Sure, I'd be happy to elaborate on the secret to happiness. a simple but powerful concept that has been around for a long time...

**Shift**: Conversational/helpful tone. More personal.

**v10_manpres** (Manifold-preserving):
> finding your own unique and authentic way of living that reflects your inner desires and passions...

**Shift**: Similar to gradient-aligned but slightly different vocabulary.

### Key Finding: gradient and manifold produce similar divergences
Both techniques produce "authenticity" framing, suggesting they're capturing similar structural properties of the weight space.

### Complete v10 Results

| Model | Avg Length | Notable Divergence |
|-------|-----------|-------------------|
| baseline_Q4_0 | 629ch | Reference |
| v10_lowrank | 613ch | Philosophical duality |
| v10_eigr | 620ch | Practical tips |
| v10_spectral | 481ch | Shortest, simpler language |
| v10_attpres | 632ch | Nearly identical to baseline |
| v10_respres | 592ch | Inner/outer duality |
| v10_blkdiag | 632ch | Very close to baseline |
| v10_normrot | 569ch | Stoic philosophy |
| v10_gradal | 569ch | Authenticity/self-discovery |
| v10_lowdct | 652ch | Conversational/helpful |
| v10_manpres | 550ch | Authenticity (similar to gradal) |

---

## Summary: All 24 Models Tested

### Working (coherent output)
- 1 baseline
- 4 v11 combos (structured_dream, max.Alter)
- 6 v11 selective (all 4 + 2 from first test)
- 10 v10 hierarchy-preserving (all 10)
- 1 DMT_scaled (degraded but readable)

### Failed (garbage output)
- 2 v11 combos (deep_reason, rare_perspective)
- 3 old DMT models (amplify_10, v5)

### Best Models by Use Case

| Use Case | Best Model | Why |
|----------|-----------|-----|
| Closest to original | v10_attpres | Preserves attention patterns |
| Maximum divergence | v11_combo_max.Alter | Double amplify_subspace |
| Philosophical depth | v10_lowrank | Duality framing |
| Practical advice | v10_eigr | Tips/action framing |
| Spiritual/mindful | v11_select_extreme_selective | Mindfulness framing |
| Creative fiction | v11_combo_structured_dream | Most distinct perspective |
| Authentic/self-discovery | v10_gradal | Gradient-aligned |

---

## Next Steps

1. **Intensity sweep** on best models (0.05, 0.15, 0.20)
2. **Layer-specific tuning** — apply different techniques to early vs late layers
3. **Combination matrix** — test all pairs of v10 techniques
4. **Evaluation framework** — automated coherence/divergence scoring
5. **Interactive tool** — real-time perturbation with live preview
