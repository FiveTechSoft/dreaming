# Dreaming — Auto-Research Harness

Optimization loop for the pure-C LLM inference engine (`llm_inference.c`).
Inspired by sankalp's auto-research approach (GPU Mode qr_v2 contest).

## Objective

Maximize **tok/s** (tokens per second) on TinyLlama-1.1B F16 inference while maintaining correctness (coherent output).

## Baseline

- Model: TinyLlama-1.1B Q4_0 (2.2 GB F16)
- Engine: `llm_inference.c` (pure C, no dependencies)
- Platform: Windows x64, GCC
- Current perf: ~40 tok/s (measured with `benchmark.bat`)

## Beam Search Discipline

Do NOT optimize as a single-incumbent hill climb. Maintain a beam of active idea families.

### Active Beams

| # | Family | Hypothesis | Status | Best Result |
|---|--------|-----------|--------|-------------|
| 1 | **matmul-optim** | Naive O(n³) matmul is the #1 bottleneck; cache-blocking + SIMD will give biggest speedup | pending | — |
| 2 | **alloc-elim** | malloc/free per token in forward pass adds overhead; pre-allocate all buffers once | pending | — |
| 3 | **openmp-parallel** | 22 transformer layers + 32 attention heads can be parallelized across cores | pending | — |
| 4 | **debug-cleanup** | Printf debug statements in hot path add I/O overhead every token | pending | — |
| 5 | **f16-vectorize** | f16_to_f32 conversion is scalar; SIMD or bulk conversion would help | pending | — |

### Beam Rules

- Keep at least 3 active beams when there is work.
- Do not kill a beam after a single failure.
- After every 3-5 submissions, update this table.
- Use sub-agents to work different beams in parallel.
- Kill a beam only with clear evidence: repeated regression + profile shows cost is not material.

## Profiling / Evidence

- Run `benchmark.bat` before and after every change.
- Record: total time, tok/s, decoded text correctness.
- Profile with `gprof` or `clock()` breakdown per function.
- Keep raw logs under `benchmarks/`.

## Known Bottlenecks (from code review)

1. **`matmul()`** (line 684): Naive double-loop, no cache blocking, no SIMD. Called 5x per layer (Q,K,V proj + out proj + 3 FFN) = ~130 matmuls total.
2. **`malloc/free` in `model_forward()`** (line 732-878): ~15 allocations per token.
3. **Debug printf** (lines 738-803): Prints embedding values, NaN counts, layer stats every token.
4. **`masked_softmax()`** (line 705): Single-threaded, no SIMD.
5. **`apply_rope_single()`** (line 692): Per-element cos/sin, no vectorization.

## Submission Discipline

- Before submitting, compile and run `benchmark.bat`.
- Save benchmark output to `benchmarks/YYYY-MM-DD-<description>.txt`.
- Only promote a candidate if tok/s improves AND output is coherent.
- Space submissions out. Do not stack rapid changes.

## Git Hygiene

- Commit after promotion with descriptive message.
- Stage: modified `llm_inference.c`, benchmark logs, AGENTS.md updates.
- Lower-case commit subjects.
