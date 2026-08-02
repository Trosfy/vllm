#!/usr/bin/env python3
"""Validate vLLM -> SparkInfer dispatch for Kimi K3 dense MLA without weights."""

from __future__ import annotations

import os
import socket

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from sparkinfer.comm.pcie.pcie_dcp_a2a import lse_reduce_scatter_reference

os.environ.setdefault("VLLM_USE_B12X_DCP_A2A", "1")
os.environ.setdefault("VLLM_DCP_A2A_MAX_TOKENS", "64")

WORLD_SIZE = 8
TOTAL_HEADS = 48
LOCAL_HEADS = TOTAL_HEADS // WORLD_SIZE
HEAD_DIM = 512
QUERY_HEAD_DIM = 576
MAX_BATCH = 64


class _DCPGroup:
    world_size = WORLD_SIZE

    def __init__(self) -> None:
        # dist.group.WORLD is assigned by init_process_group, not at import.
        self.device_group = dist.group.WORLD


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _trace(rank: int, stage: str) -> None:
    print(f"rank={rank} stage={stage}", flush=True)


def _worker(rank: int, port: int) -> None:
    from vllm.v1.attention.ops import dcp_alltoall

    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=WORLD_SIZE,
    )
    _trace(rank, "process-group-ready")
    group = _DCPGroup()
    generator = torch.Generator(device="cpu").manual_seed(9000 + rank)
    partial = torch.randn(
        1, TOTAL_HEADS, HEAD_DIM, generator=generator, dtype=torch.float32
    ).to(device=device, dtype=torch.bfloat16)
    # Triton MLA returns activation-dtype LSE; this is the exact regression
    # that previously made the public dispatcher silently choose NCCL.
    lse = torch.randn(1, TOTAL_HEADS, generator=generator, dtype=torch.float32).to(
        device=device, dtype=torch.bfloat16
    )
    local_query = torch.randn(
        1, LOCAL_HEADS, QUERY_HEAD_DIM, generator=generator, dtype=torch.float32
    ).to(device=device, dtype=torch.bfloat16)

    all_partial = [torch.empty_like(partial) for _ in range(WORLD_SIZE)]
    all_lse = [torch.empty_like(lse, dtype=torch.float32) for _ in range(WORLD_SIZE)]
    all_query = [torch.empty_like(local_query) for _ in range(WORLD_SIZE)]
    dist.all_gather(all_partial, partial)
    dist.all_gather(all_lse, lse.float())
    dist.all_gather(all_query, local_query)
    _trace(rank, "references-ready")
    expected = lse_reduce_scatter_reference(
        torch.stack(all_partial), torch.stack(all_lse), rank
    )
    expected_query = torch.cat(all_query, dim=1)

    try:
        _trace(rank, "query-gather-enter")
        gathered_query = dcp_alltoall.dcp_b12x_all_gather_heads(
            local_query,
            group,  # type: ignore[arg-type]
            max_batch_size=MAX_BATCH,
            output_head_dim=HEAD_DIM,
        )
        _trace(rank, "query-gather-done")

        # Make a silent packed-NCCL fallback a hard test failure.
        original_a2a = dist.all_to_all_single

        def reject_nccl_fallback(*args, **kwargs):
            raise AssertionError("vLLM fell back to NCCL all_to_all_single")

        dist.all_to_all_single = reject_nccl_fallback
        try:
            _trace(rank, "lse-reduce-enter")
            reduced = dcp_alltoall.dcp_a2a_lse_reduce(
                partial,
                lse,
                group,  # type: ignore[arg-type]
                use_b12x=True,
                b12x_max_batch_size=MAX_BATCH,
                b12x_query_head_dim=QUERY_HEAD_DIM,
            )
            _trace(rank, "lse-reduce-done")
        finally:
            dist.all_to_all_single = original_a2a

        torch.cuda.synchronize(device)
        torch.testing.assert_close(gathered_query, expected_query, rtol=0, atol=0)
        torch.testing.assert_close(
            reduced.float(), expected.float(), rtol=3e-2, atol=3e-2
        )
        _trace(rank, "correctness-done")
        dist.barrier()
        if rank == 0:
            print(
                "PASS vLLM dense TRITON_MLA DCP8 dispatch: BF16 LSE -> FP32, "
                "SparkInfer query gather + LSE reduce, no NCCL A2A fallback",
                flush=True,
            )
    finally:
        for pool in dcp_alltoall._B12X_DCP_A2A_POOLS.values():
            pool.close()
        dcp_alltoall._B12X_DCP_A2A_POOLS.clear()
        dist.destroy_process_group()


def main() -> None:
    if torch.cuda.device_count() < WORLD_SIZE:
        raise SystemExit(
            f"need {WORLD_SIZE} CUDA devices, found {torch.cuda.device_count()}"
        )
    mp.spawn(_worker, args=(_free_port(),), nprocs=WORLD_SIZE, join=True)


if __name__ == "__main__":
    main()
