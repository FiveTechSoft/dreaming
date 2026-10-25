# -*- coding: utf-8 -*-
"""
Stage 1-6 runner: elicit emotions on the SEV dataset, capture activations,
extract emotion directions, identify components, integrate global circuits.

Usage:
  python run_extraction.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
      --output outputs/tinyllama_1b [--max-skeletons 18] [--max-new-tokens 40]

Needs ~4 GB RAM (CPU, float32) or a GPU. The elicited generations and the
accept/reject lexicon stats are saved alongside the vectors for audit.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sev_dataset import (EMOTIONS, SKELETONS, build_elicitation_messages,
                         build_neutral_messages, text_passes_lexicon)
from pipeline import (capture_last_token_hidden, capture_components,
                      compute_emotion_directions, compute_component_contributions,
                      integrate_circuits, compute_diff_vectors, save_outputs)


def elicit(model, tokenizer, messages, max_new_tokens):
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                             eos_token_id=tokenizer.eos_token_id,
                             pad_token_id=tokenizer.pad_token_id)
    text = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    return prompt + text, text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    ap.add_argument("--output", default="outputs/tinyllama_1b")
    ap.add_argument("--max-skeletons", type=int, default=len(SKELETONS))
    ap.add_argument("--max-new-tokens", type=int, default=40)
    ap.add_argument("--mass-threshold", type=float, default=0.9)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"Loading {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)
    model.eval()
    n_layers = model.config.num_hidden_layers
    d_model = model.config.hidden_size
    intermediate = model.config.intermediate_size
    print(f"layers={n_layers} d_model={d_model} intermediate={intermediate}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    hidden_groups, comp_groups, genlog = [], [], []
    for si, sk in enumerate(SKELETONS[: args.max_skeletons]):
        group_h, group_c = {}, {}
        for e in EMOTIONS:
            full_text, gen = elicit(model, tokenizer, build_elicitation_messages(sk, e), args.max_new_tokens)
            ok = text_passes_lexicon(gen, e)
            genlog.append({"skeleton": si, "theme": sk["theme"], "emotion": e,
                           "accepted": ok, "text": gen})
            ids = tokenizer(full_text, return_tensors="pt").input_ids
            group_h[e] = capture_last_token_hidden(model, ids)
            group_c[e] = capture_components(model, ids)
        hidden_groups.append(group_h)
        comp_groups.append(group_c)
        print(f"[{si + 1}/{args.max_skeletons}] {sk['theme']} done")

    print("Computing emotion directions (stage 2)...")
    dirs = compute_emotion_directions(hidden_groups, n_layers, d_model)
    print("Computing component contributions (stage 4)...")
    head_scores, neuron_scores = compute_component_contributions(model, comp_groups, dirs, n_layers)
    print("Integrating global circuits (stage 6)...")
    circuits = integrate_circuits(head_scores, neuron_scores, args.mass_threshold)
    head_diffs, neuron_diffs = compute_diff_vectors(comp_groups, n_layers, d_model, intermediate)

    save_outputs(args.output, dirs, head_diffs, neuron_diffs, circuits)
    with open(args.output + "_generations.json", "w") as f:
        json.dump(genlog, f, indent=1, ensure_ascii=False)
    accepted = sum(1 for g in genlog if g["accepted"])
    print(f"Saved {args.output}.npz / _circuits.json / _generations.json")
    print(f"Lexicon acceptance: {accepted}/{len(genlog)} "
          f"(informational; directions use all groups)")


if __name__ == "__main__":
    main()
