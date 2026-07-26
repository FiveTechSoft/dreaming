#!/usr/bin/env python3
"""Test new v10 models: gradient, dct, manifold."""
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
    "v10_gradal": "C:/tmp/v10_gradal_10.gguf",
    "v10_lowdct": "C:/tmp/v10_lowdct_10.gguf",
    "v10_manpres": "C:/tmp/v10_manpres_10.gguf",
}

def run_inf(model_path, prompt):
    cmd = [LLAMA_CLI, "-m", model_path, "-p", prompt, "-n", "150",
           "--temp", "0.7", "-t", "4", "--no-display-prompt",
           "--single-turn", "--seed", "42"]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        lines = result.stdout.split("\n")
        gen = []
        found = False
        for line in lines:
            if prompt in line:
                found = True
                after = line.split(prompt, 1)[-1].strip()
                if after:
                    gen.append(after)
                continue
            if found:
                if any(x in line.lower() for x in ["prompt:", "exiting", "llama_"]):
                    break
                gen.append(line)
        return "\n".join(gen).strip()
    except:
        return "[TIMEOUT/ERROR]"

all_r = []
for mn, mp in MODELS.items():
    if not os.path.exists(mp):
        continue
    print(f"\nMODEL: {mn}")
    results = []
    for i, p in enumerate(PROMPTS):
        t0 = time.time()
        txt = run_inf(mp, p)
        dt = time.time() - t0
        results.append({"prompt": p, "response": txt, "time_s": round(dt, 1)})
        preview = txt[:120].replace("\n", " ")
        print(f"  [{i+1}/10] {dt:.1f}s | {preview}...")
    all_r.append({"model": mn, "results": results})

# Summary
print(f"\n{'='*60}")
print(f"{'Model':<25} {'Avg Len':>10} {'Status':>10}")
print("-" * 50)
for r in all_r:
    avg_len = sum(len(res["response"]) for res in r["results"]) / len(r["results"])
    errs = sum(1 for res in r["results"] if "ERROR" in res["response"])
    status = "YES" if errs == 0 else f"{10 - errs}/10"
    print(f"{r['model']:<25} {avg_len:>8.0f}ch {status:>10}")

with open("C:/tmp/dreaming/test_new_v10.json", "w") as f:
    json.dump(all_r, f, indent=2, ensure_ascii=False)
print("\nResults saved!")
