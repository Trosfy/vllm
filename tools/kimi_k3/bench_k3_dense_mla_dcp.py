# SPDX-License-Identifier: Apache-2.0
"""Benchmark Kimi-K3's model-free dense MLA DCP-local workload.

This mirrors the production DSpark target-verification plan: eight flattened
decode rows, capacity for fifteen rows, FP8 query/cache, and one independent
sequence per row. It intentionally excludes DCP collectives; pair its result
with ``test_dspark_dcp_transition.py --target-graph-iters``.
"""

from __future__ import annotations

import argparse
import json
import statistics

import torch


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dcp-size", type=int, choices=(1, 8, 16), required=True)
    parser.add_argument("--global-context", type=int, default=1_048_576)
    parser.add_argument(
        "--local-cache-tokens",
        type=int,
        default=None,
        help="Override global-context / DCP size (for the bounded DCP1 draft).",
    )
    parser.add_argument(
        "--effective-heads",
        type=int,
        default=None,
        help="Override K3 target's 6 * DCP effective heads.",
    )
    parser.add_argument("--query-rows", type=int, default=8)
    parser.add_argument("--max-query-rows", type=int, default=15)
    parser.add_argument("--page-size", type=int, default=768)
    parser.add_argument("--samples", type=int, default=9)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    from sparkinfer.attention import dense_mla

    if args.local_cache_tokens is None and args.global_context % args.dcp_size:
        raise ValueError("global context must divide evenly by DCP size")
    local_cache_tokens = (
        args.local_cache_tokens
        if args.local_cache_tokens is not None
        else args.global_context // args.dcp_size
    )
    effective_heads = (
        args.effective_heads
        if args.effective_heads is not None
        else 6 * args.dcp_size
    )
    if local_cache_tokens <= 0:
        raise ValueError("local cache tokens must be positive")
    if effective_heads <= 0:
        raise ValueError("effective heads must be positive")
    pages = (local_cache_tokens + args.page_size - 1) // args.page_size
    device = torch.device("cuda")
    dtype = torch.float8_e4m3fn

    plan = dense_mla.plan(
        dense_mla.Caps(
            device=device,
            mode="decode",
            dtype=torch.bfloat16,
            kv_dtype=dtype,
            num_q_heads=effective_heads,
            page_size=args.page_size,
            max_total_q=args.max_query_rows,
            max_batch=args.max_query_rows,
            max_cache_tokens=local_cache_tokens,
            max_page_table_width=pages,
            num_cache_pages=pages,
            use_cuda_graph=True,
        )
    )
    (scratch_spec,) = plan.scratch_specs()
    scratch = torch.empty(
        scratch_spec.shape,
        dtype=scratch_spec.dtype,
        device=scratch_spec.device,
    )
    q_float = torch.randn(
        args.query_rows,
        effective_heads,
        576,
        device=device,
    )
    cache_float = torch.randn(
        pages,
        args.page_size,
        576,
        device=device,
    )
    q_scale = (q_float.abs().max() / 400).reshape(1).float()
    kv_scale = (cache_float.abs().max() / 400).reshape(1).float()
    q = (q_float / q_scale).to(dtype)
    cache = (cache_float / kv_scale).to(dtype)
    page_table = torch.arange(
        pages,
        dtype=torch.int32,
        device=device,
    ).repeat(args.query_rows, 1)
    cache_seqlens = torch.full(
        (args.query_rows,),
        local_cache_tokens,
        dtype=torch.int32,
        device=device,
    )
    cu_seqlens_q = torch.arange(
        args.query_rows + 1,
        dtype=torch.int32,
        device=device,
    )
    output = torch.empty(
        args.query_rows,
        effective_heads,
        512,
        dtype=torch.bfloat16,
        device=device,
    )
    binding = dense_mla.bind(
        plan,
        scratch=scratch,
        q=q,
        kv_cache=cache,
        output=output,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        q_scale=q_scale,
        kv_scale=kv_scale,
    )
    dense_mla.compile(binding=binding)
    dense_mla.run(binding=binding)
    torch.cuda.synchronize(device)
    if not bool(torch.isfinite(output).all()):
        raise RuntimeError("dense MLA produced non-finite output")

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        dense_mla.run(binding=binding)
    for _ in range(args.warmup):
        graph.replay()
    torch.cuda.synchronize(device)

    samples_ms: list[float] = []
    for _ in range(args.samples):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(args.iterations):
            graph.replay()
        end.record()
        end.synchronize()
        samples_ms.append(float(start.elapsed_time(end)) / args.iterations)

    print(
        json.dumps(
            {
                "dcp_size": args.dcp_size,
                "global_context": args.global_context,
                "local_cache_tokens": local_cache_tokens,
                "effective_heads": effective_heads,
                "query_rows": args.query_rows,
                "max_query_rows": args.max_query_rows,
                "page_size": args.page_size,
                "num_splits": plan.num_splits,
                "scratch_mib": scratch.nbytes / 1024**2,
                "median_ms": statistics.median(samples_ms),
                "min_ms": min(samples_ms),
                "max_ms": max(samples_ms),
                "raw_ms": samples_ms,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
