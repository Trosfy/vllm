# SPDX-License-Identifier: Apache-2.0
"""Benchmark selective online MXFP8 candidates at Kimi-K3 TP16 shapes.

The target checkpoint keeps attention projections in BF16.  This harness
compares the exact rank-local BF16 GEMMs with the W8A16 Marlin path used by
the online MXFP8 overlay, without loading the 1.4 TiB model.  Small-row
measurements run from CUDA graphs to match target verification; the 2048-row
case represents one prefill chunk.
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from vllm.model_executor.kernels.linear.mxfp8.b12x import (
    B12xMxfp8LinearKernel,
)
from vllm.model_executor.kernels.linear.mxfp8.marlin import (
    MarlinMxfp8LinearKernel,
)
from vllm.model_executor.kernels.linear.mxfp8.Mxfp8LinearKernel import (
    Mxfp8LinearKernel,
    Mxfp8LinearLayerConfig,
)
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    mxfp8_e4m3_quantize,
)


@dataclass(frozen=True)
class Projection:
    name: str
    n: int
    k: int
    count: int


# Exact TP16 rank-local dimensions with kda_shard_f_a=true and
# VLLM_KIMI_SHARD_QKV_A=1.  g_proj is present only in the 24 MLA layers;
# o_proj has the same local shape in all 93 attention layers.
PROJECTIONS = (
    Projection("kda_in_proj_qkvgfab", 3088, 7168, 69),
    Projection("kda_f_b_proj", 768, 128, 69),
    Projection("mla_fused_qkv_a_proj", 132, 7168, 24),
    Projection("mla_q_b_proj", 1152, 1536, 24),
    Projection("mla_g_proj", 768, 7168, 24),
    Projection("attention_o_proj", 7168, 768, 93),
)


def _tensor_bytes(value: torch.Tensor) -> int:
    return value.numel() * value.element_size()


def _layer_bytes(layer: torch.nn.Module) -> int:
    total = 0
    seen: set[int] = set()
    for value in (*layer.parameters(), *layer.buffers()):
        pointer = value.data_ptr()
        if pointer not in seen:
            seen.add(pointer)
            total += _tensor_bytes(value)
    workspace = getattr(layer, "workspace", None)
    if isinstance(workspace, torch.Tensor) and workspace.data_ptr() not in seen:
        total += _tensor_bytes(workspace)
    return total


def _capture(run) -> tuple[torch.cuda.CUDAGraph, torch.Tensor]:
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(5):
            output = run()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = run()
    graph.replay()
    torch.cuda.synchronize()
    return graph, output


def _time_graph(
    graph: torch.cuda.CUDAGraph,
    *,
    iterations: int,
    repeats: int,
) -> list[float]:
    timings = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            graph.replay()
        end.record()
        end.synchronize()
        timings.append(start.elapsed_time(end) / iterations)
    return timings


def _bench_projection(
    projection: Projection,
    rows: int,
    *,
    iterations: int,
    repeats: int,
    seed: int,
    backend: str,
) -> dict[str, object]:
    torch.manual_seed(seed)
    n, k = projection.n, projection.k
    weight = (
        torch.randn(n, k, device="cuda", dtype=torch.bfloat16) / k**0.5
    ).contiguous()
    inputs = torch.randn(rows, k, device="cuda", dtype=torch.bfloat16)

    layer = torch.nn.Module()
    layer.input_size_per_partition = k
    layer.output_size_per_partition = n
    quantized, scales = mxfp8_e4m3_quantize(weight)
    layer.weight = torch.nn.Parameter(quantized, requires_grad=False)
    layer.weight_scale = torch.nn.Parameter(scales, requires_grad=False)
    kernel_cls = (
        MarlinMxfp8LinearKernel if backend == "marlin" else B12xMxfp8LinearKernel
    )
    kernel: Mxfp8LinearKernel = kernel_cls(Mxfp8LinearLayerConfig())
    kernel.process_weights_after_loading(layer)

    bf16_graph, bf16_output = _capture(lambda: F.linear(inputs, weight))
    mxfp8_graph, mxfp8_output = _capture(lambda: kernel.apply_weights(layer, inputs))
    difference = mxfp8_output.float() - bf16_output.float()
    bf16_times = _time_graph(bf16_graph, iterations=iterations, repeats=repeats)
    mxfp8_times = _time_graph(mxfp8_graph, iterations=iterations, repeats=repeats)
    bf16_ms = statistics.median(bf16_times)
    mxfp8_ms = statistics.median(mxfp8_times)
    bf16_bytes = _tensor_bytes(weight)
    mxfp8_bytes = _layer_bytes(layer) if backend == "marlin" else None
    return {
        "backend": backend,
        "name": projection.name,
        "rows": rows,
        "n": n,
        "k": k,
        "count": projection.count,
        "bf16_median_ms": bf16_ms,
        "mxfp8_median_ms": mxfp8_ms,
        "delta_ms": mxfp8_ms - bf16_ms,
        "projected_delta_all_layers_ms": (mxfp8_ms - bf16_ms) * projection.count,
        "ratio": mxfp8_ms / bf16_ms,
        "bf16_mib_per_layer": bf16_bytes / 2**20,
        "mxfp8_mib_per_layer": (
            mxfp8_bytes / 2**20 if mxfp8_bytes is not None else None
        ),
        "saved_mib_per_rank": (
            (bf16_bytes - mxfp8_bytes) * projection.count / 2**20
            if mxfp8_bytes is not None
            else None
        ),
        "max_abs_difference": float(difference.abs().max()),
        "mean_abs_difference": float(difference.abs().mean()),
        "bf16_samples_ms": bf16_times,
        "mxfp8_samples_ms": mxfp8_times,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 8, 2048])
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--prefill-iterations", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--only", action="append", default=[])
    parser.add_argument("--backend", choices=("marlin", "b12x"), default="marlin")
    args = parser.parse_args()

    torch.cuda.set_device(0)
    torch.set_default_dtype(torch.bfloat16)
    selected = [
        projection
        for projection in PROJECTIONS
        if not args.only or projection.name in args.only
    ]
    if not selected:
        raise ValueError(f"No projections matched --only={args.only}")

    results = []
    for projection in selected:
        for rows in args.rows:
            iterations = args.prefill_iterations if rows >= 1024 else args.iterations
            results.append(
                _bench_projection(
                    projection,
                    rows,
                    iterations=iterations,
                    repeats=args.repeats,
                    seed=args.seed,
                    backend=args.backend,
                )
            )
            gc.collect()
            torch.cuda.empty_cache()
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
