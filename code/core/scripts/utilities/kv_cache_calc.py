#!/usr/bin/env python3
"""KV cache memory calculator for transformer models."""

from __future__ import annotations

import argparse

GB_DIVISOR = 1024.0 ** 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="KV cache memory calculator for transformer models."
    )

    # Model shape
    parser.add_argument("--layers", "-L", type=int, required=True, help="Number of transformer layers (L).")
    parser.add_argument("--hidden", "-H", type=int, required=True, help="Hidden size / model width (H).")
    parser.add_argument("--tokens", "-T", type=int, required=True, help="Tokens per sequence stored in KV (T).")
    parser.add_argument("--batch", "-N", type=int, default=1, help="Number of concurrent sequences (N). Default: 1")

    # Dtype and element size
    parser.add_argument(
        "--dtype",
        choices=["fp16", "fp8", "fp4", "custom"],
        default="fp16",
        help="KV cache dtype. Use 'custom' with --bytes-per-elem.",
    )
    parser.add_argument(
        "--bytes-per-elem", type=float, default=None, help="Override bytes per element when --dtype=custom."
    )

    # GPU memory context
    parser.add_argument("--gpu-mem-gb", type=float, default=None, help="Total GPU memory in GB to compare against.")
    parser.add_argument(
        "--kv-overhead-frac",
        type=float,
        default=0.1,
        help="Extra fraction for KV overhead (metadata, padding). Applied on top of the raw KV size.",
    )

    # Reserve a chunk of GPU mem for activations etc.
    parser.add_argument(
        "--reserve-activations-gb",
        type=float,
        default=0.0,
        help="Target GB to keep free for activations and everything else. Requires --gpu-mem-gb to be meaningful.",
    )

    return parser.parse_args()


def bytes_per_elem_from_dtype(dtype: str, custom_b: float | None) -> float:
    if dtype == "fp16":
        return 2.0
    if dtype == "fp8":
        return 1.0
    if dtype == "fp4":
        return 0.5
    if custom_b is None:
        raise ValueError("When dtype=custom you must pass --bytes-per-elem.")
    return custom_b


def bytes_to_gb(x_bytes: float) -> float:
    return x_bytes / GB_DIVISOR


def main() -> None:
    args = parse_args()

    L = args.layers
    H = args.hidden
    T = args.tokens
    N = args.batch
    b = bytes_per_elem_from_dtype(args.dtype, args.bytes_per_elem)

    # Core formula: KV bytes total = 2 * L * T * H * b * N (2 for separate K and V)
    kv_bytes_per_seq = 2.0 * L * T * H * b
    kv_bytes_total = kv_bytes_per_seq * N

    overhead_factor = 1.0 + args.kv_overhead_frac
    kv_bytes_per_seq_with_overhead = kv_bytes_per_seq * overhead_factor
    kv_bytes_total_with_overhead = kv_bytes_total * overhead_factor

    print("=== KV Cache Calculator ===")
    print(f"L (layers)              : {L}")
    print(f"H (hidden size)         : {H}")
    print(f"T (tokens per sequence) : {T}")
    print(f"N (concurrent sequences): {N}")
    print(f"dtype                   : {args.dtype}")
    print(f"bytes per element       : {b} bytes")
    print(f"overhead fraction       : {args.kv_overhead_frac:.3f}")
    print()

    print("Raw KV (no overhead):")
    print(f"  KV per sequence  : {bytes_to_gb(kv_bytes_per_seq):8.3f} GB")
    print(f"  KV total (N seq) : {bytes_to_gb(kv_bytes_total):8.3f} GB")
    print()

    print("KV with overhead:")
    kv_per_seq_gb = bytes_to_gb(kv_bytes_per_seq_with_overhead)
    kv_total_gb = bytes_to_gb(kv_bytes_total_with_overhead)
    print(f"  KV per sequence  : {kv_per_seq_gb:8.3f} GB")
    print(f"  KV total (N seq) : {kv_total_gb:8.3f} GB")
    print()

    if args.gpu_mem_gb is not None:
        gpu_gb = args.gpu_mem_gb
        frac = kv_total_gb / gpu_gb
        leftover_gb = gpu_gb - kv_total_gb

        print(f"GPU memory context: {gpu_gb:.3f} GB total")
        print(f"  KV uses                 : {kv_total_gb:8.3f} GB")
        print(f"  KV fraction             : {frac * 100.0:8.3f} %")
        print(f"  GPU mem left after KV   : {leftover_gb:8.3f} GB")
        print()

        if kv_bytes_per_seq_with_overhead > 0:
            max_N = int((gpu_gb * GB_DIVISOR) // kv_bytes_per_seq_with_overhead)
            print("Capacity at this sequence length (no reserve):")
            print(f"  Max N (concurrent sequences) ≈ {max_N}")
            print()

        if args.reserve_activations_gb > 0.0:
            reserve_gb = args.reserve_activations_gb
            budget_for_kv_gb = gpu_gb - reserve_gb

            print(f"Requested reserve for activations/other: {reserve_gb:.3f} GB")

            if budget_for_kv_gb <= 0:
                print("  Reserve is larger than or equal to total GPU memory.")
                print("  No budget left for KV under this setting.")
            else:
                print(f"  Budget for KV under this reserve : {budget_for_kv_gb:8.3f} GB")

                if kv_total_gb <= budget_for_kv_gb:
                    slack_gb = budget_for_kv_gb - kv_total_gb
                    print("  Status: OK, KV fits within the reserved budget.")
                    print(f"  Extra slack beyond reserve       : {slack_gb:8.3f} GB")
                else:
                    over_gb = kv_total_gb - budget_for_kv_gb
                    print("  Status: KV exceeds budget under this reserve.")
                    print(f"  KV exceeds budget by             : {over_gb:8.3f} GB")

                    max_N_reserve = int(
                        (budget_for_kv_gb * GB_DIVISOR) // kv_bytes_per_seq_with_overhead
                    )
                    print()
                    print("  Capacity with this reserve:")
                    print(f"    Max N (concurrent sequences) ≈ {max_N_reserve}")

    print()


if __name__ == "__main__":
    main()
