# SPDX-License-Identifier: Apache-2.0
"""Model-free TP16 Kimi-K3 decode communication/MLA sequence benchmark.

The harness reproduces one ordinary (non-speculative) Kimi-K3 decode step:

* 93 attention output TP all-reduces;
* 92 latent-MoE TP gather+top-k operations and two further TP all-reduces;
* 24 full-attention layers with the TP-sharded QKV-A gather, DCP query gather,
  optional production-shaped SparkInfer dense MLA, and DCP LSE reduction.

It intentionally allocates no model weights.  ``--component`` permits exact
subtraction tests while retaining the same TP16/DCP process groups and IPC
pool topology as serving.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import time
from contextlib import suppress
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist

FULL_ATTN_LAYERS = frozenset((*range(3, 92, 4), 92))
NUM_LAYERS = 93


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dcp-size", type=int, choices=(8, 16), required=True)
    parser.add_argument(
        "--component",
        choices=("full", "allreduce", "projection", "moe", "dcp"),
        default="full",
    )
    parser.add_argument(
        "--dense-mla",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run the 24 production-shaped dense MLA kernels in full/dcp mode.",
    )
    parser.add_argument("--global-context", type=int, default=1_048_576)
    parser.add_argument(
        "--live-context",
        type=int,
        default=None,
        help="Live global length; the capture-static plan still covers global-context.",
    )
    parser.add_argument(
        "--adaptive-splits",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Launch only the nonempty prefix of the maximum-context MLA plan.",
    )
    parser.add_argument("--page-size", type=int, default=768)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument(
        "--skew-rank",
        type=int,
        default=-1,
        help="Global rank to delay before each full-attention DCP exchange.",
    )
    parser.add_argument(
        "--skew-cycles",
        type=int,
        default=0,
        help="CUDA clock cycles added on skew-rank before each DCP exchange.",
    )
    return parser.parse_args()


@dataclass
class DenseMLAState:
    binding: Any
    plan: Any
    cache: torch.Tensor
    page_table: torch.Tensor
    cache_seqlens: torch.Tensor
    cu_seqlens_q: torch.Tensor
    scratch: torch.Tensor
    output: torch.Tensor
    active_splits: int


def _build_dense_mla(
    *,
    device: torch.device,
    dcp_size: int,
    global_context: int,
    live_context: int,
    page_size: int,
    gathered_query: torch.Tensor,
    adaptive_splits: bool,
) -> DenseMLAState:
    from sparkinfer.attention import dense_mla

    if global_context % dcp_size:
        raise ValueError("global context must divide evenly by DCP size")
    local_context = global_context // dcp_size
    live_local_context = math.ceil(live_context / dcp_size)
    if not 1 <= live_local_context <= local_context:
        raise ValueError("live context exceeds the planned DCP-local cache")
    capacity_pages = math.ceil(local_context / page_size)
    live_pages = math.ceil(live_local_context / page_size)
    total_heads = 6 * dcp_size
    plan = dense_mla.plan(
        dense_mla.Caps(
            device=device,
            mode="decode",
            dtype=torch.bfloat16,
            kv_dtype=torch.float8_e4m3fn,
            num_q_heads=total_heads,
            page_size=page_size,
            max_total_q=1,
            max_batch=1,
            max_cache_tokens=local_context,
            max_page_table_width=capacity_pages,
            num_cache_pages=live_pages,
            use_cuda_graph=True,
        )
    )
    (scratch_spec,) = plan.scratch_specs()
    scratch = torch.empty(
        scratch_spec.shape,
        dtype=scratch_spec.dtype,
        device=scratch_spec.device,
    )
    cache = torch.zeros(
        live_pages,
        page_size,
        576,
        dtype=torch.float8_e4m3fn,
        device=device,
    )
    page_table = torch.arange(
        live_pages, dtype=torch.int32, device=device
    ).view(1, -1)
    cache_seqlens = torch.full(
        (1,), live_local_context, dtype=torch.int32, device=device
    )
    cu_seqlens_q = torch.tensor((0, 1), dtype=torch.int32, device=device)
    output = torch.empty(
        1, total_heads, 512, dtype=torch.bfloat16, device=device
    )
    scale = torch.ones(1, dtype=torch.float32, device=device)
    active_splits = plan.num_splits
    if adaptive_splits:
        valid_chunks = (live_local_context + 63) // 64
        active_splits = min(
            plan.num_splits,
            (valid_chunks + plan.chunks_per_split - 1)
            // plan.chunks_per_split,
        )
    binding = dense_mla.bind(
        plan,
        scratch=scratch,
        q=gathered_query,
        kv_cache=cache,
        output=output,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        q_scale=scale,
        kv_scale=scale,
        active_splits=active_splits,
    )
    dense_mla.compile(binding=binding)
    dense_mla.run(binding=binding)
    torch.cuda.synchronize(device)
    return DenseMLAState(
        binding=binding,
        plan=plan,
        cache=cache,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        scratch=scratch,
        output=output,
        active_splits=active_splits,
    )


def _rank_max_graph_us(
    graph: torch.cuda.CUDAGraph,
    *,
    device: torch.device,
    warmup: int,
    iterations: int,
    samples: int,
) -> list[float]:
    for _ in range(warmup):
        graph.replay()
    torch.cuda.synchronize(device)
    values: list[float] = []
    rank_value = torch.empty((), dtype=torch.float64, device=device)
    for _ in range(samples):
        dist.barrier(device_ids=[device.index])
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            graph.replay()
        end.record()
        end.synchronize()
        rank_value.fill_(start.elapsed_time(end) * 1e3 / iterations)
        dist.all_reduce(rank_value, op=dist.ReduceOp.MAX)
        values.append(float(rank_value.item()))
    return values


def main() -> None:
    args = _parse_args()
    if min(args.global_context, args.page_size, args.iterations, args.samples) <= 0:
        raise ValueError("context, page size, iterations, and samples must be positive")
    if args.warmup < 0:
        raise ValueError("warmup must be nonnegative")
    if args.skew_cycles < 0:
        raise ValueError("skew cycles must be nonnegative")
    live_context = (
        args.global_context if args.live_context is None else args.live_context
    )
    if not 1 <= live_context <= args.global_context:
        raise ValueError("live context must be in [1, global context]")

    # This is an exact B12X production-path harness. Do not inherit the base
    # image's legacy ``cpp`` default, which cannot initialize a TP16 group.
    os.environ["VLLM_USE_B12X_DCP_A2A"] = "1"
    os.environ["VLLM_DCP_A2A_MAX_TOKENS"] = "8"
    os.environ["VLLM_ENABLE_PCIE_ALLREDUCE"] = "1"
    os.environ["VLLM_PCIE_ALLREDUCE_BACKEND"] = "b12x"
    os.environ["VLLM_PCIE_ONESHOT_SINGLE_CHANNEL"] = "1"

    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != 16:
        raise ValueError("this production Kimi-K3 harness requires TP16")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.config.parallel import ParallelConfig
    from vllm.distributed.parallel_state import (
        cleanup_dist_env_and_memory,
        get_dcp_group,
        get_tp_group,
        graph_capture,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm.v1.attention.ops import dcp_alltoall

    eager_channel = dcp_alltoall._B12X_DCP_EAGER_CHANNEL_ID

    config = VllmConfig(
        parallel_config=ParallelConfig(
            tensor_parallel_size=world_size,
            decode_context_parallel_size=args.dcp_size,
            dcp_comm_backend="a2a",
        )
    )
    config_context = set_current_vllm_config(config)
    config_context.__enter__()
    init_distributed_environment(local_rank=local_rank)
    initialize_model_parallel(
        tensor_model_parallel_size=world_size,
        decode_context_model_parallel_size=args.dcp_size,
    )
    tp_group = get_tp_group()
    dcp_group = get_dcp_group()
    rank = dist.get_rank()

    dense_state: DenseMLAState | None = None
    graph: torch.cuda.CUDAGraph | None = None
    keepers: list[torch.Tensor] = []
    started = time.perf_counter()
    try:
        # Force construction before any CUDA capture.  The three pool shapes
        # are the exact r3 serving shapes reported at startup.
        attention_pool = dcp_alltoall._get_b12x_dcp_a2a_pool(
            dcp_group,
            device=device,
            total_heads=6 * args.dcp_size,
            head_dim=512,
            query_head_dim=576,
            max_batch_size=1,
        )
        projection_pool = dcp_alltoall._get_b12x_dcp_a2a_pool(
            tp_group,
            device=device,
            total_heads=16,
            head_dim=136,
            query_head_dim=136,
            max_batch_size=1,
        )
        moe_pool = dcp_alltoall._get_b12x_dcp_a2a_pool(
            tp_group,
            device=device,
            total_heads=16,
            head_dim=672,
            query_head_dim=672,
            max_batch_size=1,
        )
        if attention_pool is None or projection_pool is None or moe_pool is None:
            raise RuntimeError("a production B12X pool fell back during setup")

        local_query = torch.zeros(
            1, 6, 576, dtype=torch.float8_e4m3fn, device=device
        )
        gathered_query = torch.empty(
            1,
            6 * args.dcp_size,
            576,
            dtype=torch.float8_e4m3fn,
            device=device,
        )
        local_attention = torch.empty(
            1, 6, 512, dtype=torch.bfloat16, device=device
        )
        local_projection = torch.full(
            (1, 1, 136), rank, dtype=torch.bfloat16, device=device
        )
        gathered_projection = torch.empty(
            1, 16, 136, dtype=torch.bfloat16, device=device
        )
        local_down = torch.full(
            (1, 224), rank, dtype=torch.bfloat16, device=device
        )
        local_router = torch.linspace(
            -1.0, 1.0, 56, dtype=torch.float32, device=device
        ).view(1, 56)
        local_router.add_(rank * 0.001)
        correction_bias = torch.zeros(896, dtype=torch.float32, device=device)
        gathered_down = torch.empty(
            1, 3584, dtype=torch.bfloat16, device=device
        )
        topk_weights = torch.empty(1, 16, dtype=torch.float32, device=device)
        topk_ids = torch.empty(1, 16, dtype=torch.int32, device=device)
        ar_inputs = (
            torch.full((7168,), rank + 1, dtype=torch.bfloat16, device=device),
            torch.full((3584,), rank + 1, dtype=torch.bfloat16, device=device),
            torch.full((7168,), rank + 1, dtype=torch.bfloat16, device=device),
        )

        # Warm the actual dtype/shape specializations and TP all-reduce owner.
        attention_pool.all_gather_heads(
            local_query, gathered_query, channel_id=eager_channel
        )
        projection_pool.all_gather_heads(
            local_projection, gathered_projection, channel_id=eager_channel
        )
        moe_pool.all_gather_pair_kimi_topk(
            local_down,
            local_router,
            correction_bias,
            gathered_down,
            topk_weights,
            topk_ids,
            channel_id=eager_channel,
        )
        for value in ar_inputs:
            keepers.append(tp_group.all_reduce(value))
        torch.cuda.synchronize(device)

        if args.dense_mla and args.component in ("full", "dcp"):
            dense_state = _build_dense_mla(
                device=device,
                dcp_size=args.dcp_size,
                global_context=args.global_context,
                live_context=live_context,
                page_size=args.page_size,
                gathered_query=gathered_query,
                adaptive_splits=args.adaptive_splits,
            )
            partial_output = dense_state.output
            partial_lse = dense_state.binding.scratch.final_lse[:1]
        else:
            partial_output = torch.zeros(
                1,
                6 * args.dcp_size,
                512,
                dtype=torch.bfloat16,
                device=device,
            )
            partial_lse = torch.zeros(
                1, 6 * args.dcp_size, dtype=torch.float32, device=device
            )
        attention_pool.lse_reduce_scatter(
            partial_output,
            partial_lse,
            local_attention,
            channel_id=eager_channel,
        )
        torch.cuda.synchronize(device)
        dist.barrier(device_ids=[device.index])

        do_ar = args.component in ("full", "allreduce")
        do_projection = args.component in ("full", "projection")
        do_moe = args.component in ("full", "moe")
        do_dcp = args.component in ("full", "dcp")

        def run_sequence(outputs: list[torch.Tensor]) -> None:
            from sparkinfer.attention import dense_mla

            for layer in range(NUM_LAYERS):
                if layer in FULL_ATTN_LAYERS and do_projection:
                    outputs.append(
                        projection_pool.all_gather_heads(
                            local_projection,
                            gathered_projection,
                            channel_id=eager_channel,
                        )
                    )
                if layer in FULL_ATTN_LAYERS and do_dcp:
                    if rank == args.skew_rank and args.skew_cycles:
                        torch.cuda._sleep(args.skew_cycles)
                    outputs.append(
                        attention_pool.all_gather_heads(
                            local_query,
                            gathered_query,
                            channel_id=eager_channel,
                        )
                    )
                    if dense_state is not None:
                        dense_mla.run(binding=dense_state.binding)
                    outputs.append(
                        attention_pool.lse_reduce_scatter(
                            partial_output,
                            partial_lse,
                            local_attention,
                            channel_id=eager_channel,
                        )
                    )
                if do_ar:
                    outputs.append(tp_group.all_reduce(ar_inputs[0]))
                if layer > 0 and do_moe:
                    down, weights, ids = moe_pool.all_gather_pair_kimi_topk(
                        local_down,
                        local_router,
                        correction_bias,
                        gathered_down,
                        topk_weights,
                        topk_ids,
                        channel_id=eager_channel,
                    )
                    outputs.extend((down, weights, ids))
                if do_ar:
                    outputs.append(tp_group.all_reduce(ar_inputs[1]))
                    outputs.append(tp_group.all_reduce(ar_inputs[2]))

        # Eager validation also ensures no lazy specialization is initialized
        # while the graph owns the stream.
        eager_outputs: list[torch.Tensor] = []
        run_sequence(eager_outputs)
        torch.cuda.synchronize(device)
        if not all(
            bool(torch.isfinite(value.float()).all()) for value in eager_outputs
        ):
            raise RuntimeError("sequence produced a non-finite output")
        del eager_outputs
        dist.barrier(device_ids=[device.index])

        capture_outputs: list[torch.Tensor] = []
        graph = torch.cuda.CUDAGraph()
        capture_context = graph_capture(device)
        with capture_context, torch.cuda.graph(graph):
            run_sequence(capture_outputs)
        keepers.extend(capture_outputs)
        torch.cuda.synchronize(device)
        dist.barrier(device_ids=[device.index])

        samples_us = _rank_max_graph_us(
            graph,
            device=device,
            warmup=args.warmup,
            iterations=args.iterations,
            samples=args.samples,
        )
        graph.replay()
        torch.cuda.synchronize(device)
        if rank == 0:
            median_us = statistics.median(samples_us)
            print(
                json.dumps(
                    {
                        "status": "pass",
                        "world_size": world_size,
                        "dcp_size": args.dcp_size,
                        "component": args.component,
                        "dense_mla": args.dense_mla,
                        "global_context": args.global_context,
                        "live_context": live_context,
                        "local_context": args.global_context // args.dcp_size,
                        "live_local_context": (
                            live_context + args.dcp_size - 1
                        )
                        // args.dcp_size,
                        "full_attention_layers": len(FULL_ATTN_LAYERS),
                        "moe_layers": NUM_LAYERS - 1,
                        "tp_allreduces": 3 * NUM_LAYERS if do_ar else 0,
                        "skew_rank": args.skew_rank,
                        "skew_cycles_per_full_attention_layer": args.skew_cycles,
                        "dense_mla_splits": (
                            dense_state.plan.num_splits
                            if dense_state is not None
                            else None
                        ),
                        "dense_mla_active_splits": (
                            dense_state.active_splits
                            if dense_state is not None
                            else None
                        ),
                        "samples_graph_us_rank_max": samples_us,
                        "median_graph_us_rank_max": median_us,
                        "implied_model_free_tok_s": 1e6 / median_us,
                        "setup_and_benchmark_s": time.perf_counter() - started,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    finally:
        if graph is not None:
            with suppress(Exception):
                graph.reset()
        for pool in tuple(dcp_alltoall._B12X_DCP_A2A_POOLS.values()):
            with suppress(Exception):
                pool.close()
        dcp_alltoall._B12X_DCP_A2A_POOLS.clear()
        # GroupCoordinator.destroy() removes the ProcessGroup before dropping
        # CustomAllreduce.  SparkInfer's bounded-degree runtime deliberately
        # performs coordinated barriers during close, so close it while the TP
        # group is still valid and prevent the later destructor from retrying.
        device_communicator = tp_group.device_communicator
        if device_communicator is not None and device_communicator.ca_comm is not None:
            with suppress(Exception):
                device_communicator.ca_comm.close()
            device_communicator.ca_comm = None
        cleanup_dist_env_and_memory()
        config_context.__exit__(None, None, None)


if __name__ == "__main__":
    main()
