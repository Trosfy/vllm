# SPDX-License-Identifier: Apache-2.0
"""Validate and benchmark a DCP-sharded Kimi-K3 DSpark MLA cache.

This model-free harness compares three executions of one DSpark attention
layer:

* the production DCP1 layout: four TP-local heads, padded to eight, attending
  a replicated 32K cache;
* a proposed DCP16 layout: four local heads are gathered to 64 and attend one
  sixteenth of the cache before an LSE-weighted reduce-scatter;
* a correctness oracle: all 64 heads attend the complete cache.

No checkpoint or model weights are loaded. Run one process per GPU, e.g.::

    torchrun --standalone --nproc-per-node=16 \
      tools/kimi_k3/bench_dspark_sharded_dense_mla.py
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import replace
from typing import Any

import torch
import torch.distributed as dist


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dcp-size", type=int, default=16)
    parser.add_argument("--context", type=int, default=32_768)
    parser.add_argument("--query-rows", type=int, default=8)
    parser.add_argument("--max-query-rows", type=int, default=15)
    parser.add_argument("--total-heads", type=int, default=64)
    parser.add_argument("--page-size", type=int, default=128)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--samples", type=int, default=9)
    return parser.parse_args()


def _make_binding(
    dense_mla: Any,
    *,
    cache: torch.Tensor,
    q: torch.Tensor,
    q_scale: torch.Tensor,
    kv_scale: torch.Tensor,
    query_rows: int,
    max_query_rows: int,
    context: int,
    page_size: int,
) -> tuple[Any, Any, torch.Tensor]:
    pages = cache.shape[0]
    plan = dense_mla.plan(
        dense_mla.Caps(
            device=q.device,
            mode="decode",
            dtype=torch.bfloat16,
            kv_dtype=cache.dtype,
            num_q_heads=q.shape[1],
            page_size=page_size,
            max_total_q=max_query_rows,
            max_batch=max_query_rows,
            max_cache_tokens=context,
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
    output = torch.empty(
        query_rows,
        q.shape[1],
        512,
        dtype=torch.bfloat16,
        device=q.device,
    )
    page_table = torch.arange(
        pages,
        dtype=torch.int32,
        device=q.device,
    ).repeat(query_rows, 1)
    cache_seqlens = torch.full(
        (query_rows,),
        context,
        dtype=torch.int32,
        device=q.device,
    )
    cu_seqlens_q = torch.arange(
        query_rows + 1,
        dtype=torch.int32,
        device=q.device,
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
    return plan, binding, scratch


def _capture(
    fn: Callable[[], torch.Tensor],
    *,
    device: torch.device,
    distributed: bool,
) -> tuple[torch.cuda.CUDAGraph, torch.Tensor]:
    from vllm.distributed.parallel_state import graph_capture

    result = fn()
    torch.cuda.synchronize(device)
    graph = torch.cuda.CUDAGraph()
    capture_context = graph_capture(device) if distributed else nullcontext()
    with capture_context, torch.cuda.graph(graph):
        result = fn()
    return graph, result


def _time_graph(
    graph: torch.cuda.CUDAGraph,
    *,
    device: torch.device,
    iterations: int,
    samples: int,
) -> list[float]:
    for _ in range(20):
        graph.replay()
    torch.cuda.synchronize(device)
    timings: list[float] = []
    for _ in range(samples):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            graph.replay()
        end.record()
        end.synchronize()
        timings.append(float(start.elapsed_time(end)) / iterations)
    return timings


def _global_max(value: float, device: torch.device) -> float:
    tensor = torch.tensor(value, dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return float(tensor.item())


def main() -> None:
    args = _parse_args()
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != args.dcp_size:
        raise ValueError(
            f"This harness requires one DCP group: {world_size=} != {args.dcp_size=}"
        )
    if args.context % args.dcp_size:
        raise ValueError("context must divide evenly by DCP size")
    if args.total_heads % args.dcp_size:
        raise ValueError("total heads must divide evenly by DCP size")
    if args.context % args.page_size:
        raise ValueError("context must divide evenly by page size")

    os.environ.setdefault("VLLM_USE_B12X_DCP_A2A", "1")
    os.environ.setdefault("VLLM_DCP_A2A_MAX_TOKENS", str(args.query_rows))
    os.environ.setdefault("VLLM_ENABLE_PCIE_ALLREDUCE", "1")
    os.environ.setdefault("VLLM_PCIE_ALLREDUCE_BACKEND", "b12x")
    os.environ.setdefault("VLLM_PCIE_ONESHOT_SINGLE_CHANNEL", "1")

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    from b12x.attention import dense_mla

    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.config.parallel import ParallelConfig
    from vllm.distributed.parallel_state import (
        cleanup_dist_env_and_memory,
        get_dcp_group,
        init_distributed_environment,
        initialize_model_parallel,
        set_custom_all_reduce,
    )
    from vllm.v1.attention.ops import dcp_alltoall

    config = VllmConfig(
        parallel_config=ParallelConfig(
            tensor_parallel_size=world_size,
            decode_context_parallel_size=args.dcp_size,
            dcp_comm_backend="a2a",
            disable_custom_all_reduce=True,
        )
    )
    config_context = set_current_vllm_config(config)
    config_context.__enter__()
    set_custom_all_reduce(False)
    init_distributed_environment(local_rank=local_rank)
    initialize_model_parallel(
        tensor_model_parallel_size=world_size,
        decode_context_model_parallel_size=args.dcp_size,
    )
    dcp_group = get_dcp_group()
    rank = dist.get_rank()
    dcp_rank = int(dcp_group.rank_in_group)

    graphs: list[torch.cuda.CUDAGraph] = []
    keepalive: list[Any] = []
    try:
        local_heads = args.total_heads // args.dcp_size
        local_context = args.context // args.dcp_size
        if local_heads > 8:
            raise ValueError(
                "replicated oracle currently expects at most 8 local heads"
            )

        # Identical per-rank RNG streams build a deterministic global oracle.
        generator = torch.Generator(device=device).manual_seed(20260804)
        q_float = torch.randn(
            args.query_rows,
            args.total_heads,
            576,
            generator=generator,
            device=device,
        )
        cache_float = torch.randn(
            args.context,
            576,
            generator=generator,
            device=device,
        )
        q_scale = (q_float.abs().max() / 400).reshape(1).float()
        kv_scale = (cache_float.abs().max() / 400).reshape(1).float()
        q_global = (q_float / q_scale).to(torch.float8_e4m3fn)
        cache_global = (cache_float / kv_scale).to(torch.float8_e4m3fn)
        del q_float, cache_float

        head_start = dcp_rank * local_heads
        q_local = q_global[:, head_start : head_start + local_heads].contiguous()
        cache_local = cache_global[
            dcp_rank * local_context : (dcp_rank + 1) * local_context
        ].contiguous()
        cache_global = cache_global.view(
            args.context // args.page_size,
            args.page_size,
            576,
        )
        cache_local = cache_local.view(
            local_context // args.page_size,
            args.page_size,
            576,
        )

        q_replicated = torch.zeros(
            args.query_rows,
            8,
            576,
            dtype=torch.float8_e4m3fn,
            device=device,
        )
        q_replicated[:, :local_heads].copy_(q_local)
        _, replicated_binding, replicated_scratch = _make_binding(
            dense_mla,
            cache=cache_global,
            q=q_replicated,
            q_scale=q_scale,
            kv_scale=kv_scale,
            query_rows=args.query_rows,
            max_query_rows=args.max_query_rows,
            context=args.context,
            page_size=args.page_size,
        )
        _, sharded_binding, sharded_scratch = _make_binding(
            dense_mla,
            cache=cache_local,
            q=q_global,
            q_scale=q_scale,
            kv_scale=kv_scale,
            query_rows=args.query_rows,
            max_query_rows=args.max_query_rows,
            context=local_context,
            page_size=args.page_size,
        )
        _, oracle_binding, oracle_scratch = _make_binding(
            dense_mla,
            cache=cache_global,
            q=q_global,
            q_scale=q_scale,
            kv_scale=kv_scale,
            query_rows=args.query_rows,
            max_query_rows=args.max_query_rows,
            context=args.context,
            page_size=args.page_size,
        )
        keepalive.extend(
            [
                replicated_scratch,
                sharded_scratch,
                oracle_scratch,
                q_global,
                q_local,
                q_replicated,
                cache_global,
                cache_local,
            ]
        )

        warm_query = dcp_alltoall.dcp_b12x_all_gather_heads(
            q_local[:1],
            dcp_group,
            max_batch_size=args.query_rows,
            output_head_dim=512,
        )
        warm_output = torch.empty(
            1,
            args.total_heads,
            512,
            dtype=torch.bfloat16,
            device=device,
        )
        warm_lse = torch.zeros(
            1,
            args.total_heads,
            dtype=torch.float32,
            device=device,
        )
        warm_reduced = dcp_alltoall._try_b12x_dcp_lse_reduce(
            warm_output,
            warm_lse,
            dcp_group,
            return_lse=False,
            is_lse_base_on_e=True,
            max_batch_size=args.query_rows,
            query_head_dim=576,
        )
        if warm_reduced is None:
            raise RuntimeError(
                "B12X draft DCP reduce warmup rejected geometry: "
                f"query={tuple(warm_query.shape)}, "
                f"output={tuple(warm_output.shape)}, "
                f"output_stride={warm_output.stride()}, "
                f"lse={tuple(warm_lse.shape)}, "
                f"token_cap={os.environ.get('VLLM_DCP_A2A_MAX_TOKENS')}, "
                f"pool_keys={list(dcp_alltoall._B12X_DCP_A2A_POOLS)}"
            )
        keepalive.extend([warm_query, warm_output, warm_lse, warm_reduced])

        gathered_query = dcp_alltoall.dcp_b12x_all_gather_heads(
            q_local,
            dcp_group,
            max_batch_size=args.query_rows,
            output_head_dim=512,
        )
        gathered_query_error = (
            gathered_query.float() - q_global.float()
        ).abs()
        gathered_query_max_abs = _global_max(
            float(gathered_query_error.max()), device
        )

        eager_binding = replace(sharded_binding, q=gathered_query)
        eager_partial_output, eager_partial_lse = dense_mla.run(
            binding=eager_binding
        )
        eager_b12x_output = dcp_alltoall.dcp_a2a_lse_reduce(
            eager_partial_output,
            eager_partial_lse,
            dcp_group,
            is_lse_base_on_e=True,
            use_b12x=True,
            b12x_max_batch_size=args.query_rows,
            b12x_query_head_dim=576,
        )
        from vllm.v1.attention.ops.common import cp_lse_ag_out_rs

        eager_nccl_output = cp_lse_ag_out_rs(
            eager_partial_output,
            eager_partial_lse,
            dcp_group,
            is_lse_base_on_e=True,
            head_major_output=True,
        )
        keepalive.extend(
            [
                gathered_query,
                eager_partial_output,
                eager_partial_lse,
                eager_b12x_output,
                eager_nccl_output,
            ]
        )

        def run_replicated() -> torch.Tensor:
            output, _ = dense_mla.run(binding=replicated_binding)
            return output[:, :local_heads]

        def run_sharded() -> torch.Tensor:
            gathered_q = dcp_alltoall.dcp_b12x_all_gather_heads(
                q_local,
                dcp_group,
                max_batch_size=args.query_rows,
                output_head_dim=512,
            )
            live_binding = replace(sharded_binding, q=gathered_q)
            output, lse = dense_mla.run(binding=live_binding)
            return dcp_alltoall.dcp_a2a_lse_reduce(
                output,
                lse,
                dcp_group,
                is_lse_base_on_e=True,
                use_b12x=True,
                b12x_max_batch_size=args.query_rows,
                b12x_query_head_dim=576,
            )

        replicated_graph, replicated_output = _capture(
            run_replicated,
            device=device,
            distributed=False,
        )
        graphs.append(replicated_graph)
        dist.barrier()
        sharded_graph, sharded_output = _capture(
            run_sharded,
            device=device,
            distributed=True,
        )
        graphs.append(sharded_graph)
        dist.barrier()

        # The capture pass establishes IPC graph channels; correctness must be
        # checked on a real replay, which is what the server executes.
        replicated_graph.replay()
        sharded_graph.replay()
        torch.cuda.synchronize(device)
        dist.barrier()

        oracle_output, _ = dense_mla.run(binding=oracle_binding)
        oracle_local = oracle_output[:, head_start : head_start + local_heads]
        torch.cuda.synchronize(device)
        replicated_error = (replicated_output.float() - oracle_local.float()).abs()
        sharded_error = (sharded_output.float() - oracle_local.float()).abs()
        eager_b12x_error = (
            eager_b12x_output.float() - oracle_local.float()
        ).abs()
        eager_nccl_error = (
            eager_nccl_output.float() - oracle_local.float()
        ).abs()
        replicated_max_abs = _global_max(float(replicated_error.max()), device)
        sharded_max_abs = _global_max(float(sharded_error.max()), device)
        eager_b12x_max_abs = _global_max(float(eager_b12x_error.max()), device)
        eager_nccl_max_abs = _global_max(float(eager_nccl_error.max()), device)
        replicated_mean_abs = float(replicated_error.mean())
        sharded_mean_abs = float(sharded_error.mean())
        eager_b12x_mean_abs = float(eager_b12x_error.mean())
        eager_nccl_mean_abs = float(eager_nccl_error.mean())
        mean_errors = torch.tensor(
            [
                replicated_mean_abs,
                sharded_mean_abs,
                eager_b12x_mean_abs,
                eager_nccl_mean_abs,
            ],
            dtype=torch.float64,
            device=device,
        )
        dist.all_reduce(mean_errors)
        mean_errors /= world_size

        replicated_ms = _time_graph(
            replicated_graph,
            device=device,
            iterations=args.iterations,
            samples=args.samples,
        )
        dist.barrier()
        sharded_ms = _time_graph(
            sharded_graph,
            device=device,
            iterations=args.iterations,
            samples=args.samples,
        )
        replicated_median = _global_max(statistics.median(replicated_ms), device)
        sharded_median = _global_max(statistics.median(sharded_ms), device)

        if rank == 0:
            replicated_bytes = 6 * args.context * 576
            sharded_bytes = 6 * local_context * 576
            print(
                json.dumps(
                    {
                        "status": "pass",
                        "world_size": world_size,
                        "dcp_size": args.dcp_size,
                        "context": args.context,
                        "local_context": local_context,
                        "query_rows": args.query_rows,
                        "total_heads": args.total_heads,
                        "local_heads": local_heads,
                        "replicated_max_rank_median_ms": replicated_median,
                        "sharded_max_rank_median_ms": sharded_median,
                        "sharding_delta_ms_per_layer": (
                            sharded_median - replicated_median
                        ),
                        "sharding_delta_ms_five_layers": 5
                        * (sharded_median - replicated_median),
                        "replicated_max_abs_error": replicated_max_abs,
                        "sharded_max_abs_error": sharded_max_abs,
                        "eager_b12x_max_abs_error": eager_b12x_max_abs,
                        "eager_nccl_max_abs_error": eager_nccl_max_abs,
                        "gathered_query_max_abs_error": gathered_query_max_abs,
                        "replicated_mean_abs_error": float(mean_errors[0]),
                        "sharded_mean_abs_error": float(mean_errors[1]),
                        "eager_b12x_mean_abs_error": float(mean_errors[2]),
                        "eager_nccl_mean_abs_error": float(mean_errors[3]),
                        "replicated_cache_mib_per_rank": replicated_bytes / 2**20,
                        "sharded_cache_mib_per_rank": sharded_bytes / 2**20,
                        "saved_cache_mib_per_rank": (
                            replicated_bytes - sharded_bytes
                        )
                        / 2**20,
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
    finally:
        for graph in graphs:
            graph.reset()
        del keepalive
        for pool in dcp_alltoall._B12X_DCP_A2A_POOLS.values():
            pool.close()
        dcp_alltoall._B12X_DCP_A2A_POOLS.clear()
        cleanup_dist_env_and_memory()
        config_context.__exit__(None, None, None)


if __name__ == "__main__":
    main()
