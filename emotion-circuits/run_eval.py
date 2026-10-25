# -*- coding: utf-8 -*-
"""
Mechanics check of extracted emotion circuits on real weights (no GPT).

  python run_eval.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --vectors outputs/tinyllama_1b

Three measurements over the elicited generations saved by run_extraction.py:

1. PROJECTION SEPARATION: feed each elicited text back through the model,
   take the last-token residual stream, project onto the 6 directions, and
   check whether argmax matches the intended emotion (chance = 1/6 = 16.7%).
   This measures whether the directions separate the texts they were
   extracted from - necessary consistency, NOT independent evidence.

2. ENHANCEMENT / 3. ABLATION: same pass with the circuit injected / zeroed.
   The projection onto the target direction should rise / fall. WARNING: this
   metric is self-referential - it projects onto the very direction being
   injected or ablated, so it only confirms the hooks move what they target.

None of this is behavioral validation of emotion. What remains pending:
generated text scored for expressed emotion on HELD-OUT scenarios (not used
in extraction), with random-direction and matched-norm controls, and an
independent judge (human or external model) of the expressed emotion.
Writes <vectors>_eval.json.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sev_dataset import EMOTIONS
from pipeline import (EmotionCircuitInjector, EmotionCircuitAblator,
                      capture_last_token_hidden, load_outputs, projection_scores)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    ap.add_argument("--vectors", default="outputs/tinyllama_1b")
    ap.add_argument("--scale", type=float, default=1.0)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"Loading {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)
    model.eval()

    dirs, head_diffs, neuron_diffs, circuits = load_outputs(args.vectors)
    with open(args.vectors + "_generations.json") as f:
        genlog = [g for g in json.load(f) if g["text"].strip()]
    print(f"{len(genlog)} elicited texts")

    report = {"chance": 1.0 / len(EMOTIONS), "per_emotion": {}, "overall": {}}
    correct = {"base": 0, "enh": 0, "abl": 0}
    deltas = {"enh": [], "abl": []}
    totals = 0
    for g in genlog:
        e = g["emotion"]
        ei = EMOTIONS.index(e)
        ids = tokenizer(g["text"], return_tensors="pt").input_ids

        h_base = capture_last_token_hidden(model, ids)
        s_base = projection_scores(h_base, dirs)[ei]

        inj = EmotionCircuitInjector(model, e, circuits[e], head_diffs, neuron_diffs,
                                     scale=args.scale).register()
        h_enh = capture_last_token_hidden(model, ids)
        inj.remove()
        s_enh = projection_scores(h_enh, dirs)[ei]

        abl = EmotionCircuitAblator(model, circuits[e]).register()
        h_abl = capture_last_token_hidden(model, ids)
        abl.remove()
        s_abl = projection_scores(h_abl, dirs)[ei]

        correct["base"] += int(np.argmax(projection_scores(h_base, dirs)) == ei)
        correct["enh"] += int(np.argmax(projection_scores(h_enh, dirs)) == ei)
        correct["abl"] += int(np.argmax(projection_scores(h_abl, dirs)) == ei)
        deltas["enh"].append(float(s_enh - s_base))
        deltas["abl"].append(float(s_abl - s_base))
        totals += 1

    for k in correct:
        report["overall"][f"accuracy_{k}"] = correct[k] / max(totals, 1)
    report["overall"]["enhancement_delta_mean"] = float(np.mean(deltas["enh"]))
    report["overall"]["ablation_delta_mean"] = float(np.mean(deltas["abl"]))
    report["n_texts"] = totals
    with open(args.vectors + "_eval.json", "w") as f:
        json.dump(report, f, indent=1)
    print(json.dumps(report["overall"], indent=1))
    print(f"Saved {args.vectors}_eval.json")
    print("Read as mechanics check only: enh>0 / abl<0 confirms hooks move the targeted")
    print("projection. It is NOT causal evidence for emotion (self-referential metric).")


if __name__ == "__main__":
    main()
