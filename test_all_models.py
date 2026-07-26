#!/usr/bin/env python3
"""
test_all_models.py — Comprehensive test of all dreaming perturbation models.

Tests each model with multiple prompts and saves results to JSON.
Compares original Q4_0 baseline against all perturbed variants.
"""

import subprocess, json, time, sys, os
from pathlib import Path
from datetime import datetime

LLAMA_CLI = "C:/tmp/llama-cpp/llama-cli.exe"
PROMPTS = [
    "The secret to happiness is",
    "In a distant galaxy, scientists discovered",
    "The history of ancient civilizations reveals",
    "If I could travel through time, I would",
    "The future of artificial intelligence",
    "Music is the universal language because",
    "The ocean depths hide mysteries that",
    "Philosophy teaches us that",
    "The greatest invention in human history",
    "Dreams are the mind's way of",
]

N_TOKENS = 150
TEMPERATURE = 0.7
TOP_K = 40

# Models to test
MODELS = {
    # Baseline
    "baseline_Q4_0": "C:/tmp/tinyllama-1.1b.Q4_0.gguf",
    
    # v11 Combos
    "v11_combo_deep_reason": "C:/tmp/v11_combo_deep_reason.gguf",
    "v11_combo_rare_perspective": "C:/tmp/v11_combo_rare_perspective.gguf",
    "v11_combo_structured_dream": "C:/tmp/v11_combo_structured_dream.gguf",
    "v11_combo_max.Alter": "C:/tmp/v11_combo_max.Alter.gguf",
    
    # v11 Selective
    "v11_select_attention_alter": "C:/tmp/v11_select_attention_alter.gguf",
    "v11_select_ffn_dream": "C:/tmp/v11_select_ffn_dream.gguf",
    "v11_select_embedding_shift": "C:/tmp/v11_select_embedding_shift.gguf",
    "v11_select_extreme_selective": "C:/tmp/v11_select_extreme_selective.gguf",
    
    # DMT originals
    "DMT_amplify_10": "C:/tmp/tinyllama-1.1b.DMT.gguf",
    "DMT_scaled_10": "C:/tmp/tinyllama-1.1b.DMT_scaled.gguf",
    "DMT_v5": "C:/tmp/tinyllama-1.1b.DMT_v5.gguf",
}


def run_inference(model_path, prompt, n_tokens=N_TOKENS, temperature=TEMPERATURE):
    """Run llama-cli inference and return generated text."""
    cmd = [
        LLAMA_CLI,
        "-m", model_path,
        "-p", prompt,
        "-n", str(n_tokens),
        "--temp", str(temperature),
        "-t", "4",
        "--no-display-prompt",
        "--single-turn",
        "--seed", "42",
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        output = result.stdout.strip()
        
        # Extract generated text (after the prompt)
        lines = output.split('\n')
        gen_lines = []
        found_prompt = False
        for line in lines:
            if prompt in line:
                found_prompt = True
                after = line.split(prompt, 1)[-1].strip()
                if after:
                    gen_lines.append(after)
                continue
            if found_prompt:
                if any(x in line.lower() for x in ['prompt:', 'generation:', 'exiting', 'llama_']):
                    break
                gen_lines.append(line)
        
        return '\n'.join(gen_lines).strip()
    except subprocess.TimeoutExpired:
        return "[TIMEOUT]"
    except Exception as e:
        return f"[ERROR: {e}]"


def test_model(model_name, model_path):
    """Test a single model with all prompts."""
    print(f"\n{'='*70}")
    print(f"MODEL: {model_name}")
    print(f"Path: {model_path}")
    print(f"{'='*70}")
    
    if not os.path.exists(model_path):
        print(f"  [SKIP] File not found")
        return None
    
    results = []
    for i, prompt in enumerate(PROMPTS):
        t0 = time.time()
        text = run_inference(model_path, prompt)
        elapsed = time.time() - t0
        
        result = {
            "prompt": prompt,
            "response": text,
            "time_s": round(elapsed, 1),
        }
        results.append(result)
        
        # Print preview
        preview = text[:200].replace('\n', ' ')
        print(f"  [{i+1}/10] {elapsed:.1f}s | {prompt}")
        print(f"          -> {preview}...")
    
    return {
        "model": model_name,
        "path": model_path,
        "results": results,
    }


def main():
    print("="*70)
    print("DREAMING MODEL TEST SUITE")
    print(f"Date: {datetime.now().isoformat()}")
    print(f"Prompts: {len(PROMPTS)}")
    print(f"Models: {len(MODELS)}")
    print(f"Tokens per generation: {N_TOKENS}")
    print(f"Temperature: {TEMPERATURE}")
    print("="*70)
    
    all_results = []
    total_t0 = time.time()
    
    for model_name, model_path in MODELS.items():
        result = test_model(model_name, model_path)
        if result:
            all_results.append(result)
    
    total_elapsed = time.time() - total_t0
    
    # Save results
    output_file = "C:/tmp/dreaming/test_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "config": {
                "prompts": len(PROMPTS),
                "n_tokens": N_TOKENS,
                "temperature": TEMPERATURE,
                "top_k": TOP_K,
            },
            "models_tested": len(all_results),
            "total_time_s": round(total_elapsed, 1),
            "results": all_results,
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*70}")
    print(f"ALL DONE — {total_elapsed:.1f}s total")
    print(f"Results saved to: {output_file}")
    print(f"Models tested: {len(all_results)}/{len(MODELS)}")
    print(f"{'='*70}")
    
    # Print summary table
    print(f"\n{'Model':<35} {'Avg Time':>10} {'Avg Length':>12}")
    print("-"*60)
    for r in all_results:
        avg_time = sum(res['time_s'] for res in r['results']) / len(r['results'])
        avg_len = sum(len(res['response']) for res in r['results']) / len(r['results'])
        print(f"{r['model']:<35} {avg_time:>8.1f}s {avg_len:>10.0f}ch")


if __name__ == "__main__":
    main()
