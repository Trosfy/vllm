# SPDX-License-Identifier: Apache-2.0
"""Reproduce the Kimi-K3 target/draft DCP graph transition without a model.

Run with one process per GPU, for example::

    torchrun --standalone --nproc-per-node=16 \
        tools/kimi_k3/test_dspark_dcp_transition.py --mode draft-dcp

The sequence mirrors server startup:

1. warm and capture a DCP8 B12X target attention graph;
2. capture a TP16 draft graph, optionally (and incorrectly) entering the
   target DCP graph-capture contexts;
3. execute a token batch larger than the B12X cap so target attention falls
   back to an eager NCCL DCP8 all-to-all.

No checkpoint or model weights are loaded.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist


@dataclass
class _CPGroup:
    world_size: int
    device_group: dist.ProcessGroup


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("target-only", "draft-clean", "draft-dcp"),
        required=True,
    )
    parser.add_argument("--dcp-size", type=int, default=8)
    parser.add_argument(
        "--local-heads",
        type=int,
        default=6,
        help=(
            "Rank-local Kimi query heads. Production TP16 Kimi-K3 has six; "
            "the effective gathered head count is local_heads*dcp_size."
        ),
    )
    parser.add_argument("--b12x-cap", type=int, default=8)
    parser.add_argument("--fallback-batch", type=int, default=16)
    parser.add_argument(
        "--fallback-backend",
        choices=("torch-a2a", "pynccl-a2a", "ag-rs"),
        default="torch-a2a",
    )
    parser.add_argument("--fallback-iters", type=int, default=1)
    parser.add_argument(
        "--target-graph-iters",
        type=int,
        default=0,
        help="Time the captured B12X query-gather plus LSE-reduce graph.",
    )
    parser.add_argument(
        "--query-dtype",
        choices=("bf16", "fp8"),
        default="bf16",
        help="Query dtype; production K3 with an FP8 KV cache uses FP8.",
    )
    parser.add_argument(
        "--vllm-groups",
        action="store_true",
        help="Use vLLM's real TP/DCP coordinators and B12X TP all-reduce.",
    )
    parser.add_argument(
        "--free-mib",
        type=int,
        default=0,
        help="Before NCCL fallback, retain ballast until about this much is free.",
    )
    return parser.parse_args()


def _make_dcp_group(world_size: int, rank: int, dcp_size: int) -> _CPGroup:
    if world_size % dcp_size:
        raise ValueError(f"world_size={world_size} is not divisible by {dcp_size=}")

    selected: dist.ProcessGroup | None = None
    for start in range(0, world_size, dcp_size):
        ranks = list(range(start, start + dcp_size))
        group = dist.new_group(ranks=ranks, backend="nccl")
        if rank in ranks:
            selected = group
    assert selected is not None
    return _CPGroup(dcp_size, selected)


def _capture_target_graph(
    cp_group: Any,
    *,
    device: torch.device,
    batch: int,
    local_heads: int,
    query_dtype: torch.dtype,
    vllm_groups: bool,
) -> tuple[torch.cuda.CUDAGraph, tuple[torch.Tensor, ...]]:
    from vllm.v1.attention.ops import dcp_alltoall

    total_heads = local_heads * cp_group.world_size
    head_dim = 512
    query_head_dim = 576
    query = torch.randn(
        batch,
        local_heads,
        query_head_dim,
        device=device,
        dtype=torch.bfloat16,
    ).to(query_dtype)
    output = torch.randn(
        batch,
        total_heads,
        head_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    lse = torch.randn(batch, total_heads, device=device, dtype=torch.float32)
    # Exercise the exact query dtype before capture. Production's model warmup
    # does this naturally; the model-free harness must not initialize a dtype
    # specialization while CUDA is capturing.
    dcp_alltoall.dcp_b12x_all_gather_heads(
        query,
        cp_group,  # type: ignore[arg-type]
        max_batch_size=batch,
        output_head_dim=head_dim,
    )
    dcp_alltoall.dcp_a2a_lse_reduce(
        output,
        lse,
        cp_group,  # type: ignore[arg-type]
        use_b12x=True,
        b12x_max_batch_size=batch,
        b12x_query_head_dim=query_head_dim,
    )
    torch.cuda.synchronize(device)
    graph = torch.cuda.CUDAGraph()
    if vllm_groups:
        from vllm.distributed.parallel_state import graph_capture

        capture_context = graph_capture(device)
    else:
        capture_context = dcp_alltoall.capture_b12x_dcp_a2a(cp_group)
    with capture_context, torch.cuda.graph(graph):
        gathered_query = dcp_alltoall.dcp_b12x_all_gather_heads(
            query,
            cp_group,  # type: ignore[arg-type]
            max_batch_size=batch,
            output_head_dim=head_dim,
        )
        reduced_output = dcp_alltoall.dcp_a2a_lse_reduce(
            output,
            lse,
            cp_group,  # type: ignore[arg-type]
            use_b12x=True,
            b12x_max_batch_size=batch,
            b12x_query_head_dim=query_head_dim,
        )
    return graph, (query, output, lse, gathered_query, reduced_output)


def _capture_draft_graph(
    cp_group: Any,
    *,
    device: torch.device,
    enter_dcp_capture: bool,
    vllm_groups: bool,
) -> tuple[torch.cuda.CUDAGraph, torch.Tensor]:
    from vllm.v1.attention.ops import dcp_alltoall

    # The real replicated draft has TP16 collectives but no DCP collective.
    draft = torch.full((4096,), 1.0, device=device, dtype=torch.float32)
    graph = torch.cuda.CUDAGraph()
    if vllm_groups:
        from vllm.distributed.parallel_state import (
            GraphCaptureContext,
            get_pp_group,
            get_tp_group,
            graph_capture,
        )

        if enter_dcp_capture:
            capture_context = graph_capture(device)
        else:
            context = GraphCaptureContext(torch.cuda.Stream(device=device))
            capture_context = _capture_tp_pp_only(
                get_tp_group(), get_pp_group(), context
            )
        with capture_context, torch.cuda.graph(graph):
            draft = get_tp_group().all_reduce(draft)
    else:
        dcp_capture = (
            dcp_alltoall.capture_b12x_dcp_a2a(cp_group)
            if enter_dcp_capture
            else nullcontext()
        )
        with dcp_capture, torch.cuda.graph(graph):
            dist.all_reduce(draft, group=dist.group.WORLD)
    return graph, draft


def _capture_tp_pp_only(tp_group: Any, pp_group: Any, context: Any):
    from contextlib import ExitStack, contextmanager

    @contextmanager
    def capture():
        with ExitStack() as stack:
            stack.enter_context(tp_group.graph_capture(context))
            stack.enter_context(pp_group.graph_capture(context))
            yield context

    return capture()


def _combine_fallback(
    cp_group: Any,
    *,
    output: torch.Tensor,
    lse: torch.Tensor,
    b12x_cap: int,
    backend: str,
) -> torch.Tensor:
    from vllm.v1.attention.ops import dcp_alltoall

    total_heads = int(output.shape[1])
    if backend == "ag-rs":
        from vllm.v1.attention.ops.common import cp_lse_ag_out_rs

        return cp_lse_ag_out_rs(
            output,
            lse,
            cp_group,
            head_major_output=True,
        )
    if backend == "pynccl-a2a":
        world_size = cp_group.world_size
        head_dim = int(output.shape[2])
        batch = int(output.shape[0])
        heads_per_rank = total_heads // world_size
        pack_dim = dcp_alltoall._dcp_a2a_lse_pack_dim(output.dtype)
        send, recv = dcp_alltoall._dcp_a2a_send_recv_buffers(
            (world_size, batch, heads_per_rank, head_dim + pack_dim),
            device=output.device,
            dtype=output.dtype,
        )
        dcp_alltoall._dcp_a2a_pack_send(
            output,
            lse,
            send,
            world_size,
            heads_per_rank,
            head_dim,
            pack_dim,
        )
        communicator = cp_group.device_communicator.pynccl_comm
        assert communicator is not None and not communicator.disabled
        rank = cp_group.rank_in_group
        recv[rank].copy_(send[rank])
        communicator.nccl.ncclGroupStart()
        for peer in range(world_size):
            if peer == rank:
                continue
            communicator.recv(recv[peer], peer)
            communicator.send(send[peer], peer)
        communicator.nccl.ncclGroupEnd()
        return dcp_alltoall._dcp_a2a_unpack_combine(
            recv,
            head_dim,
            pack_dim,
            return_lse=False,
            is_lse_base_on_e=True,
        )
    assert backend == "torch-a2a"
    return dcp_alltoall.dcp_a2a_lse_reduce(
        output,
        lse,
        cp_group,  # type: ignore[arg-type]
        use_b12x=True,
        b12x_max_batch_size=b12x_cap,
        b12x_query_head_dim=576,
    )


def _run_fallback(
    cp_group: Any,
    *,
    device: torch.device,
    batch: int,
    total_heads: int,
    b12x_cap: int,
    backend: str,
    iterations: int,
) -> tuple[torch.Tensor, float | None]:
    if iterations < 1:
        raise ValueError("iterations must be positive")
    output = torch.randn(
        batch, total_heads, 512, device=device, dtype=torch.bfloat16
    )
    lse = torch.randn(batch, total_heads, device=device, dtype=torch.float32)

    def combine() -> torch.Tensor:
        return _combine_fallback(
            cp_group,
            output=output,
            lse=lse,
            b12x_cap=b12x_cap,
            backend=backend,
        )

    if iterations == 1:
        return combine(), None

    result = combine()
    torch.cuda.synchronize(device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        result = combine()
    end.record()
    end.synchronize()
    return result, start.elapsed_time(end) / iterations


def _allocate_ballast(device: torch.device, free_mib: int) -> list[torch.Tensor]:
    if free_mib <= 0:
        return []
    target = free_mib * 1024**2
    ballast: list[torch.Tensor] = []
    # Preserve a small allocator margin so the ballast itself does not race
    # CUDA/PyTorch bookkeeping at the requested boundary.
    margin = 32 * 1024**2
    while True:
        free, _ = torch.cuda.mem_get_info(device)
        alloc_bytes = free - target - margin
        if alloc_bytes <= 0:
            break
        chunk = min(alloc_bytes, 1024**3)
        ballast.append(torch.empty(chunk, device=device, dtype=torch.uint8))
    torch.cuda.synchronize(device)
    return ballast


def _time_graph(
    graph: torch.cuda.CUDAGraph,
    *,
    device: torch.device,
    iterations: int,
) -> float | None:
    if iterations <= 0:
        return None
    for _ in range(10):
        graph.replay()
    torch.cuda.synchronize(device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        graph.replay()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end)) / iterations


def main() -> None:
    args = _parse_args()
    if args.local_heads < 1:
        raise ValueError("--local-heads must be positive")
    total_heads = args.local_heads * args.dcp_size
    os.environ.setdefault("VLLM_USE_B12X_DCP_A2A", "1")
    os.environ["VLLM_DCP_A2A_MAX_TOKENS"] = str(args.b12x_cap)
    if args.vllm_groups:
        os.environ["VLLM_ENABLE_PCIE_ALLREDUCE"] = "1"
        os.environ["VLLM_PCIE_ALLREDUCE_BACKEND"] = "b12x"
        os.environ["VLLM_PCIE_ONESHOT_SINGLE_CHANNEL"] = "1"

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    query_dtype = (
        torch.bfloat16 if args.query_dtype == "bf16" else torch.float8_e4m3fn
    )
    if args.vllm_groups:
        from vllm.config import VllmConfig, set_current_vllm_config
        from vllm.config.parallel import ParallelConfig
        from vllm.distributed.parallel_state import (
            get_dcp_group,
            init_distributed_environment,
            initialize_model_parallel,
        )

        config = VllmConfig(
            parallel_config=ParallelConfig(
                tensor_parallel_size=int(os.environ["WORLD_SIZE"]),
                decode_context_parallel_size=args.dcp_size,
                dcp_comm_backend="a2a",
            )
        )
        config_context = set_current_vllm_config(config)
        config_context.__enter__()
        init_distributed_environment(local_rank=local_rank)
        initialize_model_parallel(
            tensor_model_parallel_size=int(os.environ["WORLD_SIZE"]),
            decode_context_model_parallel_size=args.dcp_size,
        )
        cp_group = get_dcp_group()
    else:
        config_context = nullcontext()
        dist.init_process_group(backend="nccl")
        cp_group = _make_dcp_group(
            dist.get_world_size(), dist.get_rank(), args.dcp_size
        )
    rank = dist.get_rank()
    started = time.perf_counter()

    try:
        from vllm.v1.attention.ops import dcp_alltoall

        dcp_alltoall.warmup_b12x_dcp_a2a(
            cp_group,  # type: ignore[arg-type]
            device=device,
            dtype=torch.bfloat16,
            max_batch_size=args.b12x_cap,
            total_heads=total_heads,
            head_dim=512,
            query_head_dim=576,
        )
        dist.barrier()
        target_graph, target_tensors = _capture_target_graph(
            cp_group,
            device=device,
            batch=args.b12x_cap,
            local_heads=args.local_heads,
            query_dtype=query_dtype,
            vllm_groups=args.vllm_groups,
        )
        torch.cuda.synchronize()
        dist.barrier()
        target_graph_ms = _time_graph(
            target_graph,
            device=device,
            iterations=args.target_graph_iters,
        )
        if target_graph_ms is not None:
            latency = torch.tensor(target_graph_ms, device=device)
            dist.all_reduce(latency, op=dist.ReduceOp.MAX)
            target_graph_ms = latency.item()

        draft_graph = None
        draft_tensor = None
        if args.mode != "target-only":
            draft_graph, draft_tensor = _capture_draft_graph(
                cp_group,
                device=device,
                enter_dcp_capture=args.mode == "draft-dcp",
                vllm_groups=args.vllm_groups,
            )
            torch.cuda.synchronize()
            dist.barrier()

        free_before_ballast, _ = torch.cuda.mem_get_info(device)
        ballast = _allocate_ballast(device, args.free_mib)
        free_before_fallback, _ = torch.cuda.mem_get_info(device)
        if rank == 0:
            print(
                json.dumps(
                    {
                        "event": "before-fallback",
                        "free_mib": free_before_fallback / 1024**2,
                        "mode": args.mode,
                        "requested_free_mib": args.free_mib,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        fallback, fallback_ms = _run_fallback(
            cp_group,
            device=device,
            batch=args.fallback_batch,
            total_heads=total_heads,
            b12x_cap=args.b12x_cap,
            backend=args.fallback_backend,
            iterations=args.fallback_iters,
        )
        torch.cuda.synchronize()
        checksum = fallback.float().sum()
        dist.all_reduce(checksum)
        if fallback_ms is not None:
            latency = torch.tensor(fallback_ms, device=device)
            dist.all_reduce(latency, op=dist.ReduceOp.MAX)
            fallback_ms = latency.item()
        if rank == 0:
            print(
                json.dumps(
                    {
                        "status": "pass",
                        "mode": args.mode,
                        "world_size": dist.get_world_size(),
                        "dcp_size": args.dcp_size,
                        "local_heads": args.local_heads,
                        "total_heads": total_heads,
                        "b12x_cap": args.b12x_cap,
                        "fallback_batch": args.fallback_batch,
                        "fallback_backend": args.fallback_backend,
                        "fallback_iters": args.fallback_iters,
                        "fallback_max_rank_ms": fallback_ms,
                        "target_graph_iters": args.target_graph_iters,
                        "target_graph_max_rank_ms": target_graph_ms,
                        "query_dtype": args.query_dtype,
                        "free_before_ballast_mib": free_before_ballast / 1024**2,
                        "free_before_fallback_mib": free_before_fallback / 1024**2,
                        "vllm_groups": args.vllm_groups,
                        "checksum": checksum.item(),
                        "elapsed_s": time.perf_counter() - started,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

        # Keep captured storage alive through the eager transition.
        del target_tensors, fallback, checksum, draft_tensor, ballast
        if draft_graph is not None:
            draft_graph.reset()
        target_graph.reset()
    finally:
        from vllm.v1.attention.ops import dcp_alltoall

        for pool in dcp_alltoall._B12X_DCP_A2A_POOLS.values():
            pool.close()
        dcp_alltoall._B12X_DCP_A2A_POOLS.clear()
        if args.vllm_groups:
            from vllm.distributed.parallel_state import cleanup_dist_env_and_memory

            cleanup_dist_env_and_memory()
            config_context.__exit__(None, None, None)
        else:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
