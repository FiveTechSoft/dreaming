# Chapter 21: Archetypes and Constellations

## Working Definitions

| Term | Meaning in this microcosmos |
|------|----------------------------|
| **Archetype** | Geometric attractor: centroid in ℝ²⁰⁴⁸ of a cluster of seed tokens that, in the culture of pretraining, condense a recurring myth |
| **Constellation** | The seed cluster itself (fixed stars of the myth) + its unit direction in the embedding sky |
| **Alignment** | High cosine between two archetypal centroids → myths that brush against each other |
| **Opposition** | Low/negative cosine → poles of drama |

We don't claim the model "believes in Jung."
We claim that **those directions are measurable**
and that some coincide with the Dreaming voices
(Golden Rule, `mystical`).

---

## Archetype Catalog (15)

### Twelve Pearson / Jung Myths (operational)

| Symbol | Archetype | Myth (one line) | Seed-constellation (BPE) |
|--------|-----------|-----------------|--------------------------|
| ⚔ | **Hero** | Trial, valor, victory | ▁hero ▁courage ▁brave ▁quest ▁victory ▁fight ▁strength ▁honor ▁triumph |
| 🌑 | **Shadow** | Inner enemy, monster | ▁shadow ▁dark ▁evil ▁fear ▁hate ▁demon rage ▁sin |
| 📜 | **Sage** | Truth, study, mind | ▁wisdom ▁truth ▁knowledge ▁scholar ▁theory ▁reason ▁logic ▁study ▁philosophy ▁mind |
| 💚 | **Caregiver** | Care, heal, protect | ▁care ▁love ▁kind ▁help ▁protect ▁gentle ▁comfort |
| 🧭 | **Explorer** | Journey, frontier, freedom | ▁explore ▁journey ▁discover ▁travel ▁freedom ▁path ▁wild ▁seek ▁horizon |
| ✨ | **Creator** | Art, invention, dream | ▁create ▁art ▁imagine ▁beauty ▁music ▁poem ▁invent ▁craft ▁design ▁dream |
| 👑 | **Ruler** | Order, power, law | ▁king ▁power ▁law ▁order ▁rule ▁throne ▁command ▁authority ▁nation |
| 🔮 | **Magician** | Spirit, sacred, vision | ▁magic ▁spirit ▁soul ▁divine ▁sacred ▁mystery ▁transform ▁vision |
| 🌸 | **Innocent** | Hope, purity, faith | ▁hope ▁faith ▁pure ▁happy ▁child ▁peace ▁trust ▁simple ▁good |
| ❤ | **Lover** | Desire, heart, beauty | ▁love ▁desire ▁kiss ▁passion ▁heart ▁beauty ▁tender |
| 🃏 | **Jester** | Laughter, play, irony | ▁laugh ▁play ▁fool ▁smile ▁wit ▁mock ▁silly |
| 🏚 | **Orphan / Realist** | Pain, home, survival | ▁alone ▁lost ▁pain ▁real ▁ordinary ▁poor ▁need ▁belong ▁home |

### Three Dreaming Operational Archetypes

| Symbol | Archetype | Myth | Seeds |
|--------|-----------|------|-------|
| 🕯 | **Mystical Voice** | Ego, soul, universe, silence | ▁soul ▁spirit ego ▁universe ▁divine ▁silence ▁being |
| 🔧 | **Practical Voice** (FFN Golden Rule) | Action, plan, method | ▁should ▁step ▁action ▁goal ▁plan ▁work ▁build ▁fix ▁method ▁practice |
| 🎓 | **Academic Voice** (Attn Golden Rule) | Theory, analysis, evidence | ▁theory ▁analysis ▁study ▁research ▁argument ▁concept ▁framework ▁evidence ▁scholar ▁critique |

---

## Alignment Map (*myth* constellations)

Measured: cosine between centroids (F16 embeddings).

### Main Attractions (they brush in the sky)

| cos | Constellation A | Constellation B | Reading |
|-----|----------------|----------------|---------|
| **+0.39** | 🔮 Magician | 🕯 Mystical Voice | The `mystical` climate *is* geometrically magician/spirit |
| **+0.29** | 📜 Sage | 🎓 Academic Voice | The "attn→academic" Golden Rule has an anchor in the token sky |
| **+0.13** | 💚 Caregiver | ❤ Lover | Care and desire share affective neighborhood |
| **+0.12** | ✨ Creator | ❤ Lover | Beauty / creation / love |
| +0.05 | 📜 Sage | 👑 Ruler | Knowledge and order (weak) |

### Oppositions / Polarities

| cos | A | B | Reading |
|-----|---|---|---------|
| **−0.06** | ⚔ Hero | 🌑 Shadow | The classic drama axis (though mild: not antipodal) |
| −0.06 | 💚 Caregiver | 🏚 Orphan | Caring vs deprivation |
| −0.05 | 🧭 Explorer | 🃏 Jester | Serious path vs play |
| −0.05 | 🧭 Explorer | 🕯 Mystic | Outer frontier vs inner |
| −0.04 | 📜 Sage | ❤ Lover | Analysis vs desire |
| −0.04 | 🎓 Academic | ❤ Lover | Same tension in Dreaming voice |

**Geometric note:** almost all pairs are near **0**.
Archetypes are **islands** (like the 12 semantic areas),
not a single diamond of opposites. Alignments of +0.3
are *strong exceptions* and that's why they matter.

---

## Why "neighboring stars" alone are misleading

If you ask for the k cosine neighbors of a centroid across
the entire BPE vocabulary, fragments appear (`gia`, codes,
other languages): in ℝ²⁰⁴⁸ almost everything is orthogonal and
the "closest" isn't clean semantics.

That's why we define the **operational constellation** as:

1. **Seeds** (myth stars, hand-chosen), and
2. **Links to other archetypes** (alignment graph),

not as raw k-NN from the full vocabulary.

---

## Constellation Graph (reading)

```
                    [Sage]────0.29────[Academic Voice]
                       │
                      0.05
                       │
                  [Ruler]

[Caregiver]──0.13──[Lover]──0.12──[Creator]
     │
    0.04
     │
  [Magician]────────0.39────────[Mystical Voice Dreaming]
                                │
                           (mystical / --steer soul)

[Hero]  ≈⊥  [Shadow]     (weak polarity −0.06)
[Explorer] ≈⊥ [Mystic, Jester, Academic]
```

---

## How to Orbit an Archetype

| Destination | Flight Coordinates |
|-------------|-------------------|
| Magician / mystical | existential prompt + `--perturb mystical` and/or `--steer soul` |
| Academic | analytical prompt + (in Q4) targeting attention; or `--steer theory` |
| Practical | "how to" prompt + targeting FFN / seeds step, plan, action |
| Hero vs Shadow | conflict prompts; compare baseline vs noise vs mystical |
| Lover / caregiver | `--steer love` / `care` with moderate strength |

```bash
# Mystical constellation
./llm_inference model.F16.gguf "When we dissolve the ego" \
  50 0.7 40 --seed 42 --perturb mystical --intensity 0.50

# Wind toward the Sage
./llm_inference model.F16.gguf "Philosophy teaches us that" \
  50 0.7 40 --seed 42 --steer wisdom --steer-strength 0.2
```

---

## Artifacts

| File | Contents |
|------|----------|
| `exploration/archetypes.json` | Centroids, seeds, matrix, alignments |
| `exploration/archetype_map.html` | Interactive 2D PCA of archetypes |
| `map_archetypes.py` | Regenerate the atlas |

General semantic map (12 thematic areas, not archetypes):
`semantic_map.html`

---

## In One Sentence

**Archetypes** are myth-directions in the token sky;
**constellations** are their seeds and the measured bridges
between myths — and the strong finding of the journey is that
**Magician ≈ Mystical Voice** and **Sage ≈ Academic Voice**,
meaning: the Dreaming lenses were already drawn
as constellations in the embedding.

---

*Next chapter: Conscious Observation and Unconscious Projection.*