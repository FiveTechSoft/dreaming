#!/usr/bin/env python3
"""
DMT perturbation: apply structured weight perturbation to a GGUF model
before generating text.

Modes:
  scaled_noise  — noise proportional to |weight|, preserves structure
  row_shuffle   — partial row permutation in projection matrices
  amplify_subspace — SVD-based amplification of dominant directions
"""
import argparse
import numpy as np
import gguf
import os, sys, time

INPUT  = "C:/tmp/tinyllama-1.1b.F16.gguf"
OUTPUT = "C:/tmp/tinyllama-1.1b.F16.dmt.gguf"

def dmt_perturb(arr, intensity, mode, name=""):
    arr = arr.astype(np.float64)

    if mode == "scaled_noise":
        noise = np.random.randn(*arr.shape).astype(arr.dtype) * np.abs(arr) * intensity
        result = arr + noise.astype(arr.dtype)
        return result.astype(np.float16)

    elif mode == "row_shuffle":
        n = arr.shape[0]
        idx = np.random.permutation(n)
        mix = np.random.random((n, 1)) < intensity
        shuffled = arr[idx]
        result = np.where(mix, shuffled, arr)
        return result.astype(np.float16)

    elif mode == "amplify_subspace":
        # For large 2D tensors, use randomized SVD approximation
        if arr.ndim == 2 and min(arr.shape) > 512:
            # Power iteration approximation for top-k singular values
            k = max(1, int(min(arr.shape) * 0.05))
            print(f"    [{name}] randomized SVD (k={k})...")
            U, S, Vt = np.linalg.svd(arr, full_matrices=False)
            S = S.copy()
            S[:k] *= (1 + intensity)
            S[k:] *= (1 - intensity * 0.5)
            result = (U * S[:, None]) @ Vt
            return result.astype(np.float16)
        else:
            U, S, Vt = np.linalg.svd(arr, full_matrices=False)
            S = S.copy()
            k = max(1, int(len(S) * 0.1))
            S[:k] *= (1 + intensity)
            S[k:] *= (1 - intensity * 0.5)
            result = (U * S[:, None]) @ Vt
            return result.astype(np.float16)
    else:
        raise ValueError(f"Unknown mode: {mode}")


def main():
    parser = argparse.ArgumentParser(description="DMT perturbation for GGUF models")
    parser.add_argument("--input",  default=INPUT,  help="Path to source F16 GGUF")
    parser.add_argument("--output", default=OUTPUT, help="Path to write perturbed F16 GGUF")
    parser.add_argument("--mode",   default="scaled_noise",
                        choices=["scaled_noise", "row_shuffle", "amplify_subspace"],
                        help="Perturbation mode")
    parser.add_argument("--intensity", type=float, default=0.15,
                        help="Perturbation strength (0–1)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--layers", type=str, default=None,
                        help="Comma-separated layer range e.g. 3-18 (None=all)")
    args = parser.parse_args()

    np.random.seed(args.seed)

    print(f"Reading {args.input} ...")
    t0 = time.time()
    reader = gguf.GGUFReader(args.input)
    n_tensors = len(reader.tensors)
    print(f"  {n_tensors} tensors found, mode={args.mode}, intensity={args.intensity}")

    # Parse layer filter
    layer_filter = None
    if args.layers:
        lo, hi = map(int, args.layers.split("-"))
        layer_filter = (lo, hi)
        print(f"  Filtering layers {lo}-{hi}")

    writer = gguf.GGUFWriter(args.output, "llama")

    # Copy minimal metadata
    writer.add_context_length(131072)
    writer.add_embedding_length(2048)
    writer.add_block_count(22)
    writer.add_feed_forward_length(5632)
    writer.add_rope_dimension_count(128)
    writer.add_vocab_size(32000)
    writer.add_layer_norm_rms_eps(1e-5)

    success = 0
    failed = 0
    perturbed = 0

    for i in range(n_tensors):
        tensor = reader.tensors[i]
        name = tensor.name
        shape = [int(s) for s in tensor.shape]
        raw = np.frombuffer(bytes(tensor.data.tobytes()), dtype=np.float16)
        arr = raw.reshape(shape)

        # Apply perturbation only to weight matrices (skip norms/biases)
        should_perturb = (
            arr.ndim == 2
            and "norm" not in name
            and "bias" not in name
        )
        if layer_filter:
            lo, hi = layer_filter
            # Check if tensor belongs to layers in range
            in_range = any(f"blk.{l}." in name for l in range(lo, hi + 1))
            should_perturb = should_perturb and in_range

        if should_perturb:
            arr = dmt_perturb(arr, args.intensity, args.mode, name)
            perturbed += 1

        writer.add_tensor(name, arr.astype(np.float16))
        success += 1

        if i % 20 == 0 or i == n_tensors - 1:
            tag = " [PERTUBED]" if should_perturb else ""
            mb = arr.nbytes / 1024 / 1024
            print(f"  [{i+1}/{n_tensors}] {name}{tag} ({mb:.0f} MB)")

    print(f"\nFinalizing ({success} tensors, {perturbed} perturbed)...")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()

    size_mb = os.path.getsize(args.output) / 1024 / 1024
    print(f"\nDone! {args.output}")
    print(f"  Size: {size_mb:.1f} MB | Time: {time.time()-t0:.1f}s")
    print(f"  Tensors: {success} | Perturbed: {perturbed}")


if __name__ == "__main__":
    main()
