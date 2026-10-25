# -*- coding: utf-8 -*-
"""
End-to-end smoke test on a tiny random LLaMA (no downloads, runs anywhere).

Validates the mechanics of the pipeline (direction math, component ranking,
circuit integration, injection hooks, IO roundtrip) with planted signals:
synthetic activations contain known per-emotion offsets, so the extracted
directions must correlate with them. Run before trusting a real extraction:

  python smoke_test.py
"""

import os
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline import (EMOTIONS, compute_emotion_directions,
                      compute_component_contributions, integrate_circuits,
                      compute_diff_vectors, EmotionCircuitInjector,
                      EmotionCircuitAblator, ResidualDirectionSteerer,
                      capture_last_token_hidden, capture_components,
                      save_outputs, load_outputs, projection_scores)

L, D, I, H = 2, 64, 128, 4
rng = np.random.default_rng(7)


def synthetic_groups(n_groups=5, signal=2.0):
    """Groups with a planted per-emotion offset shared across groups."""
    offsets = {e: rng.normal(0, 1, (L, D)).astype(np.float32) for e in EMOTIONS}
    groups = []
    for _ in range(n_groups):
        base = rng.normal(0, 1, (L, D)).astype(np.float32)  # skeleton context
        groups.append({e: base + signal * offsets[e] + 0.1 * rng.normal(0, 1, (L, D)).astype(np.float32)
                       for e in EMOTIONS})
    return groups, offsets


def main():
    from transformers import LlamaConfig, LlamaForCausalLM
    cfg = LlamaConfig(hidden_size=D, intermediate_size=I, num_hidden_layers=L,
                      num_attention_heads=H, num_key_value_heads=H, vocab_size=128)
    model = LlamaForCausalLM(cfg).eval()

    # --- stage 2: directions recover the planted offsets ---
    groups, offsets = synthetic_groups()
    dirs = compute_emotion_directions(groups, L, D)
    assert dirs.shape == (6, L, D)
    norms = np.linalg.norm(dirs, axis=-1)
    assert np.allclose(norms, 1.0, atol=1e-5), f"directions not unit norm: {norms}"
    for ei, e in enumerate(EMOTIONS):
        off = offsets[e] / np.linalg.norm(offsets[e], axis=-1, keepdims=True)
        r = float((dirs[ei] * off).sum() / L)
        assert r > 0.8, f"direction for {e} does not recover planted offset (r={r:.2f})"
    print("stage 2 ok: unit-norm directions recover planted emotion offsets (r>0.8)")

    # --- stages 4-6: components, circuit, diffs ---
    comp_records = []
    for g in groups:
        rec = {}
        for e in EMOTIONS:
            attn = np.concatenate([g[e], np.zeros((L, 0), np.float32)], axis=1)  # (L, D) = H*Dh
            mlp = np.repeat(g[e], I // D, axis=1)                                # (L, I)
            rec[e] = (attn.astype(np.float32), mlp.astype(np.float32))
        comp_records.append(rec)
    head_scores, neuron_scores = compute_component_contributions(model, comp_records, dirs, L)
    assert head_scores.shape == (6, L, H) and neuron_scores.shape == (6, L, I)
    circuits = integrate_circuits(head_scores, neuron_scores, mass_threshold=0.9)
    assert all(circuits[e]["layers"] for e in EMOTIONS)
    head_diffs, neuron_diffs = compute_diff_vectors(comp_records, L, D, I)
    print("stages 4-6 ok: contributions ranked, circuits integrated, diff vectors computed")

    # --- IO roundtrip ---
    with tempfile.TemporaryDirectory() as td:
        prefix = os.path.join(td, "toy")
        save_outputs(prefix, dirs, head_diffs, neuron_diffs, circuits)
        d2, hd2, nd2, c2 = load_outputs(prefix)
        assert np.allclose(d2, dirs) and c2.keys() == circuits.keys()
    print("io ok: npz/json roundtrip")

    # --- stage 7: hooks actually perturb the forward pass ---
    ids = torch.randint(0, 128, (1, 6))
    with torch.no_grad():
        base_logits = model(ids).logits[:, -1].clone()

    injector = EmotionCircuitInjector(model, "anger", circuits["anger"],
                                      head_diffs, neuron_diffs, scale=1.0).register()
    with torch.no_grad():
        inj_logits = model(ids).logits[:, -1].clone()
    injector.remove()
    delta = (inj_logits - base_logits).abs().max().item()
    assert delta > 1e-4, "circuit injection had no effect on logits"
    with torch.no_grad():
        after = model(ids).logits[:, -1].clone()
    assert torch.allclose(after, base_logits, atol=1e-5), "hooks not fully removed"
    print(f"stage 7 ok: circuit injection changes logits (max |delta|={delta:.3f}), removal clean")

    steerer = ResidualDirectionSteerer(model, "happiness", dirs, scale=2.0, top_layers=1).register()
    with torch.no_grad():
        st_logits = model(ids).logits[:, -1].clone()
    steerer.remove()
    assert (st_logits - base_logits).abs().max().item() > 1e-4
    print("stage 3-baseline ok: residual direction steering changes logits")

    # --- capture helpers against the real arch ---
    h = capture_last_token_hidden(model, ids)
    a, m = capture_components(model, ids)
    assert h.shape == (L, D) and a.shape == (L, H * (D // H)) and m.shape == (L, I)
    print("capture ok: hidden states and component activations have expected shapes")

    # --- causal pair: ablation also perturbs, and directions classify ---
    ablator = EmotionCircuitAblator(model, circuits["anger"]).register()
    with torch.no_grad():
        abl_logits = model(ids).logits[:, -1].clone()
    ablator.remove()
    assert (abl_logits - base_logits).abs().max().item() > 1e-4, "ablation had no effect"
    with torch.no_grad():
        after2 = model(ids).logits[:, -1].clone()
    assert torch.allclose(after2, base_logits, atol=1e-5), "ablator hooks not removed"
    h_syn = groups[0]["sadness"]
    assert int(np.argmax(projection_scores(h_syn, dirs))) == EMOTIONS.index("sadness")
    print("ablation ok: ablator perturbs logits, removal clean, directions classify synthetic text")

    print("\nALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
