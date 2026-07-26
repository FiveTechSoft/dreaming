#!/usr/bin/env python3
"""Test v10 and v11 models with llama-cli."""

import subprocess, json, time, os

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

MODELS = {
    "baseline_Q4_0": "C:/tmp/tinyllama-1.1b.Q4_0.gguf",
    
    # v11 Combos
    "v11_combo_structured_dream": "C:/tmp/v11_combo_structured_dream.gguf",
    "v11_combo_max.Alter": "C:/tmp/v11_combo_max.Alter.gguf",
    
    # v11 Selective
    "v11_select_attention_alter": "C:/tmp/v11_select_attention_alter.gguf",
    "v11_select_ffn_dream": "C:/tmp/v11_select_ffn_dream.gguf",
    "v11_select_embedding_shift": "C:/tmp/v11_select_embedding_shift.gguf",
    "v11_select_extreme_selective": "C:/tmp/v11_select_extreme_selective.gguf",
    
    # v10 Hierarchy-Preserving
    "v10_lowrank": "C:/tmp/v10_lowrank_10.gguf",
    "v10_eigr": "C:/tmp/v10_eigr_10.gguf",
    "v10_spectral": "C:/tmp/v10_spectral_10.gguf",
    "v10_attpres": "C:/tmp/v10_attpres_10.gguf",
    "v10_respres": "C:/tmp/v10_respres_10.gguf",
    "v10_blkdiag": "C:/tmp/v10_blkdiag_10.gguf",
    "v10_normrot": "C:/tmp/v10_normrot_10.gguf",
}


def run_inference(model_path, prompt, n_tokens=150, temperature=0.7):
    cmd = [
        LLAMA_CLI, "-m", model_path, "-p", prompt,
        "-n", str(n_tokens), "--temp", str(temperature),
        "-t", "4", "--no-display-prompt", "--single-turn", "--seed", "42",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        output = result.stdout
        lines = output.split('\n')
        gen_lines = []
        found = False
        for line in lines:
            if prompt in line:
                found = True
                after = line.split(prompt, 1)[-1].strip()
                if after:
                    gen_lines.append(after)
                continue
            if found:
                if any(x in line.lower() for x in ['prompt:', 'exiting', 'llama_']):
                    break
                gen_lines.append(line)
        return '\n'.join(gen_lines).strip()
    except:
        return "[TIMEOUT/ERROR]"


def main():
    all_results = []
    total_t0 = time.time()
    
    for model_name, model_path in MODELS.items():
        if not os.path.exists(model_path):
            print(f"[SKIP] {model_name} - file not found")
            continue
        
        print(f"\n{'='*60}")
        print(f"MODEL: {model_name}")
        print(f"{'='*60}")
        
        results = []
        for i, prompt in enumerate(PROMPTS):
            t0 = time.time()
            text = run_inference(model_path, prompt)
            elapsed = time.time() - t0
            results.append({"prompt": prompt, "response": text, "time_s": round(elapsed, 1)})
            preview = text[:150].replace('\n', ' ')
            print(f"  [{i+1}/10] {elapsed:.1f}s | {preview}...")
        
        all_results.append({"model": model_name, "results": results})
    
    total_elapsed = time.time() - total_t0
    
    # Save
    output_file = "C:/tmp/dreaming/test_v10_v11_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({"results": all_results, "total_time_s": round(total_elapsed, 1)}, f, indent=2, ensure_ascii=False)
    
    # Summary
    print(f"\n{'='*70}")
    print(f"{'Model':<35} {'Avg Len':>10} {'Coherent':>10}")
    print("-"*60)
    for r in all_results:
        avg_len = sum(len(res['response']) for res in r['results']) / len(r['results'])
        errors = sum(1 for res in r['results'] if 'ERROR' in res['response'] or 'TIMEOUT' in res['response'])
        coherent = 'YES' if errors == 0 else f'{10-errors}/10'
        print(f"{r['model']:<35} {avg_len:>8.0f}ch {coherent:>10}")
    
    print(f"\nTotal: {total_elapsed:.1f}s")


if __name__ == "__main__":
    main()
