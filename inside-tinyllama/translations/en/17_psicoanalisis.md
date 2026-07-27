# Chapter 17: Psychoanalysis of the Transformer

## A metaphor (not a clinical diagnosis)

Freud distinguished layers of the mental.
Without forcing identity, the transformer stack
admits a **reading by depth**:

| Instance | Component | Approximate function |
|----------|-----------|---------------------|
| **Unconscious** | Embeddings | Latent associations, "what's already known" without context |
| **Preconscious** | Attention + middle layers | Brings relationships and frameworks to the stage |
| **Conscious** | Final layers + logits + sample | What is said *now* |

## Id / Ego / Superego (free reading)

| | Analogy in the model |
|--|----------------------|
| **Id** | Raw weight impulses, raw semantic directions |
| **Ego** | Residual + norms: negotiates between impulses and form |
| **Superego** | Training biases / safety / "correct" baseline style |

The `mystical` perturbation doesn't "release the id" in the Freudian sense:
it **remixes** the balance of voices already present in the weights.

## Why note this metaphor

- Helps *talk* about the interior without only matrices.
- Links to the macro↔micro zoom (ch. 6).
- Doesn't replace measurements: it's a **narrative map**.

## Limit

An LLM doesn't have a subjective unconscious.
It has **compressed statistics**. The metaphor is
an exploration tool, not ontology.

---

*Next chapter: What We Learned*
