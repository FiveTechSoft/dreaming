# -*- coding: utf-8 -*-
"""
Stage 7 demo: circuit-based emotion modulation on TinyLlama.

Usage (after run_extraction.py produced outputs/tinyllama_1b.*):
  python run_steer.py --input_text "My girlfriend forgot my birthday again." \
      --emotion anger --scale 1.0
  python run_steer.py --input_text "..." --emotion sadness --mode direction --scale 2.0

Modes:
  circuit   -> inject only the selected heads/neurons (paper stage 7, 99.65% on Llama-3.2-3B)
  direction -> add the emotion direction to the residual stream at its top layers
               (paper's steering-based baseline, their stage 3)
  baseline  -> no injection
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sev_dataset import EMOTIONS
from pipeline import EmotionCircuitInjector, ResidualDirectionSteerer, load_outputs


def generate(model, tokenizer, text, max_new_tokens=120):
    messages = [
        {"role": "system", "content": "Keep the reply to at most two sentences."},
        {"role": "user", "content": text},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                             repetition_penalty=1.1, no_repeat_ngram_size=3,
                             eos_token_id=tokenizer.eos_token_id,
                             pad_token_id=tokenizer.pad_token_id)
    return tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    ap.add_argument("--vectors", default="outputs/tinyllama_1b")
    ap.add_argument("--input_text", required=True)
    ap.add_argument("--emotion", choices=EMOTIONS + ["all"], default="all")
    ap.add_argument("--mode", choices=["circuit", "direction", "baseline"], default="circuit")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--max-new-tokens", type=int, default=120)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"Loading {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)
    model.eval()

    dirs = head_diffs = neuron_diffs = circuits = None
    if args.mode != "baseline":
        dirs, head_diffs, neuron_diffs, circuits = load_outputs(args.vectors)

    emotions = EMOTIONS if args.emotion == "all" else [args.emotion]
    print(f"\n=== BASELINE ===\n{generate(model, tokenizer, args.input_text, args.max_new_tokens)}")
    for e in emotions:
        if args.mode == "circuit":
            injector = EmotionCircuitInjector(model, e, circuits[e], head_diffs, neuron_diffs,
                                              scale=args.scale).register()
        elif args.mode == "direction":
            injector = ResidualDirectionSteerer(model, e, dirs, scale=args.scale).register()
        else:
            continue
        try:
            text = generate(model, tokenizer, args.input_text, args.max_new_tokens)
        finally:
            injector.remove()
        print(f"\n=== {e.upper()} ({args.mode}, scale={args.scale}) ===\n{text}")


if __name__ == "__main__":
    main()
