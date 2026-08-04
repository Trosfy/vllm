#!/usr/bin/env python3
"""Compare allocated memory for K3's normal and donated BF16 o_proj output."""

import argparse
import gc

import torch


def allocated_peak_delta_mib(baseline: int) -> float:
    return (torch.cuda.max_memory_allocated() - baseline) / (1024**2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=2048)
    parser.add_argument("--input-size", type=int, default=768)
    parser.add_argument("--hidden-size", type=int, default=7168)
    args = parser.parse_args()

    torch.manual_seed(7)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    projection_input = torch.randn(
        args.tokens,
        args.input_size,
        dtype=dtype,
        device=device,
    )
    weight = torch.randn(
        args.hidden_size,
        args.input_size,
        dtype=dtype,
        device=device,
    )
    dead_hidden_states = torch.empty(
        args.tokens,
        args.hidden_size,
        dtype=dtype,
        device=device,
    )

    # Prime the GEMM selection/workspace before measuring active tensor bytes.
    torch.mm(projection_input[:1], weight.t())
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated()
    reference = torch.mm(projection_input, weight.t())
    torch.cuda.synchronize()
    normal_peak_delta_mib = allocated_peak_delta_mib(baseline)
    reference_cpu = reference.cpu()
    del reference
    gc.collect()
    torch.cuda.empty_cache()

    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated()
    output_ptr = dead_hidden_states.data_ptr()
    torch.mm(
        projection_input,
        weight.t(),
        out=dead_hidden_states,
    )
    torch.cuda.synchronize()
    donated_peak_delta_mib = allocated_peak_delta_mib(baseline)
    actual_cpu = dead_hidden_states.cpu()

    print(
        {
            "shape": [args.tokens, args.input_size, args.hidden_size],
            "normal_peak_delta_mib": normal_peak_delta_mib,
            "donated_peak_delta_mib": donated_peak_delta_mib,
            "saved_peak_mib": normal_peak_delta_mib - donated_peak_delta_mib,
            "same_ptr": dead_hidden_states.data_ptr() == output_ptr,
            "exact": torch.equal(actual_cpu, reference_cpu),
            "max_abs": float((actual_cpu.float() - reference_cpu.float()).abs().max()),
        }
    )


if __name__ == "__main__":
    main()
