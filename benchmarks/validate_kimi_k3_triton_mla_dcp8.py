#!/usr/bin/env python3
"""Validate Kimi K3 Triton MLA decode plus SparkInfer DCP8 reduction.

This is deliberately model-free: it exercises the production Triton decode
kernel with K3's absorbed MLA dimensions and FP8 KV cache, partitions context
tokens using DCP interleave-1, and compares the merged result with a direct
full-context attention reference.
"""

from __future__ import annotations

import argparse
import math
import socket

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from sparkinfer.comm.pcie.pcie_dcp_a2a import PCIeDCPA2APool, _load_extension

from vllm.v1.attention.ops.triton_decode_attention import decode_attention_fwd

WORLD_SIZE = 8
LOCAL_HEADS = 6
TOTAL_HEADS = WORLD_SIZE * LOCAL_HEADS
LATENT_DIM = 512
ROPE_DIM = 64
QUERY_DIM = LATENT_DIM + ROPE_DIM
PAGE_SIZE = 16
NUM_KV_SPLITS = 4
MAX_BATCH = 1
SM_SCALE = 1.0 / math.sqrt(192.0)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _global_inputs(
    seq_len: int, seed_offset: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(
        20260802 + seq_len + seed_offset
    )
    query = (
        torch.randn(TOTAL_HEADS, QUERY_DIM, generator=generator, dtype=torch.float32)
        * 0.25
    ).bfloat16()
    kv = (
        torch.randn(seq_len, QUERY_DIM, generator=generator, dtype=torch.float32) * 0.25
    ).to(torch.float8_e4m3fn)
    return query, kv


def _local_decode(
    query: torch.Tensor,
    kv: torch.Tensor,
    rank: int,
    device: torch.device,
    *,
    lse_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    local_kv = kv[rank::WORLD_SIZE]
    local_len = int(local_kv.shape[0])
    num_pages = (local_len + PAGE_SIZE - 1) // PAGE_SIZE
    cache = torch.zeros(
        num_pages,
        PAGE_SIZE,
        1,
        QUERY_DIM,
        dtype=torch.float8_e4m3fn,
        device=device,
    )
    cache.view(-1, QUERY_DIM)[:local_len].copy_(local_kv.to(device))
    block_table = torch.arange(num_pages, dtype=torch.int32, device=device)[None]
    seq_lens = torch.tensor([local_len], dtype=torch.int32, device=device)
    query = query.to(device)[None]
    output = torch.empty(
        1, TOTAL_HEADS, LATENT_DIM, dtype=torch.bfloat16, device=device
    )
    lse = torch.empty(1, TOTAL_HEADS, dtype=lse_dtype, device=device)
    logits = torch.empty(
        1,
        TOTAL_HEADS,
        NUM_KV_SPLITS,
        LATENT_DIM + 1,
        dtype=torch.float32,
        device=device,
    )
    scale = torch.ones((), dtype=torch.float32, device=device)
    decode_attention_fwd(
        query,
        cache,
        cache[..., :LATENT_DIM],
        output,
        lse,
        block_table,
        seq_lens,
        logits,
        NUM_KV_SPLITS,
        SM_SCALE,
        PAGE_SIZE,
        k_scale=scale,
        v_scale=scale,
        is_mla=True,
    )
    return output, lse


def _full_reference(
    query: torch.Tensor, kv: torch.Tensor, rank: int, device: torch.device
) -> torch.Tensor:
    # Match the production kernel's FP8 -> BF16 load before its dot products.
    query = query.to(device=device, dtype=torch.bfloat16)
    kv = kv.to(device=device).to(torch.bfloat16)
    scores = torch.matmul(query.float(), kv.float().T) * SM_SCALE
    probabilities = torch.softmax(scores, dim=-1)
    output = torch.matmul(probabilities, kv[:, :LATENT_DIM].float())
    start = rank * LOCAL_HEADS
    return output[start : start + LOCAL_HEADS]


def _run_case(
    pool: PCIeDCPA2APool,
    rank: int,
    device: torch.device,
    seq_len: int,
    lse_dtype: torch.dtype,
) -> tuple[float, float]:
    query_cpu, kv_cpu = _global_inputs(seq_len)
    local_query = query_cpu[rank * LOCAL_HEADS : (rank + 1) * LOCAL_HEADS].to(device)[
        None
    ]
    gathered_query = pool.all_gather_heads(local_query)
    output, lse = _local_decode(
        gathered_query[0].cpu(), kv_cpu, rank, device, lse_dtype=lse_dtype
    )
    reduced = pool.lse_reduce_scatter(output, lse.float())
    torch.cuda.synchronize(device)
    expected_query = query_cpu.to(device)[None]
    torch.testing.assert_close(gathered_query, expected_query, rtol=0, atol=0)
    reference = _full_reference(query_cpu, kv_cpu, rank, device)
    error = (reduced[0].float() - reference).abs()
    return float(error.max()), float(error.mean())


def _graph_case(
    pool: PCIeDCPA2APool,
    rank: int,
    device: torch.device,
) -> tuple[float, float]:
    seq_len = 97
    local_len = len(range(rank, seq_len, WORLD_SIZE))
    num_pages = (local_len + PAGE_SIZE - 1) // PAGE_SIZE
    local_query = torch.empty(
        1, LOCAL_HEADS, QUERY_DIM, dtype=torch.bfloat16, device=device
    )
    gathered_query = torch.empty(
        1, TOTAL_HEADS, QUERY_DIM, dtype=torch.bfloat16, device=device
    )
    cache = torch.zeros(
        num_pages,
        PAGE_SIZE,
        1,
        QUERY_DIM,
        dtype=torch.float8_e4m3fn,
        device=device,
    )
    block_table = torch.arange(num_pages, dtype=torch.int32, device=device)[None]
    seq_lens = torch.tensor([local_len], dtype=torch.int32, device=device)
    output = torch.empty(
        1, TOTAL_HEADS, LATENT_DIM, dtype=torch.bfloat16, device=device
    )
    lse = torch.empty(1, TOTAL_HEADS, dtype=torch.bfloat16, device=device)
    lse_fp32 = torch.empty(1, TOTAL_HEADS, dtype=torch.float32, device=device)
    reduced = torch.empty(
        1, LOCAL_HEADS, LATENT_DIM, dtype=torch.bfloat16, device=device
    )
    logits = torch.empty(
        1,
        TOTAL_HEADS,
        NUM_KV_SPLITS,
        LATENT_DIM + 1,
        dtype=torch.float32,
        device=device,
    )
    scale = torch.ones((), dtype=torch.float32, device=device)

    def step() -> None:
        pool.all_gather_heads(local_query, out=gathered_query)
        decode_attention_fwd(
            gathered_query,
            cache,
            cache[..., :LATENT_DIM],
            output,
            lse,
            block_table,
            seq_lens,
            logits,
            NUM_KV_SPLITS,
            SM_SCALE,
            PAGE_SIZE,
            k_scale=scale,
            v_scale=scale,
            is_mla=True,
        )
        lse_fp32.copy_(lse)
        pool.lse_reduce_scatter(output, lse_fp32, out=reduced)

    def update(seed_offset: int) -> tuple[torch.Tensor, torch.Tensor]:
        query_cpu, kv_cpu = _global_inputs(seq_len, seed_offset)
        local_query.copy_(query_cpu[rank * LOCAL_HEADS : (rank + 1) * LOCAL_HEADS])
        cache.zero_()
        cache.view(-1, QUERY_DIM)[:local_len].copy_(kv_cpu[rank::WORLD_SIZE].to(device))
        return query_cpu, kv_cpu

    stream = torch.cuda.Stream(device=device)
    update(0)
    with pool.capture(stream):
        with torch.cuda.stream(stream):
            step()
        stream.synchronize()
        dist.barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            step()
    stream.synchronize()

    max_error = 0.0
    mean_error = 0.0
    for replay in range(3):
        query_cpu, kv_cpu = update(1000 * (replay + 1))
        stream.wait_stream(torch.cuda.current_stream(device))
        graph.replay()
        stream.synchronize()
        reference = _full_reference(query_cpu, kv_cpu, rank, device)
        error = (reduced[0].float() - reference).abs()
        max_error = max(max_error, float(error.max()))
        mean_error = max(mean_error, float(error.mean()))
    return max_error, mean_error


def _worker(rank: int, world_size: int, port: int) -> None:
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
    )
    pool = PCIeDCPA2APool.from_process_group(
        process_group=dist.group.WORLD,
        device=device,
        max_batch_size=MAX_BATCH,
        total_heads=TOTAL_HEADS,
        head_dim=LATENT_DIM,
        query_head_dim=QUERY_DIM,
    )
    try:
        results: list[tuple[int, str, float, float]] = []
        for seq_len in (8, 17, 97, 513):
            for lse_dtype in (torch.bfloat16, torch.float32):
                max_error, mean_error = _run_case(
                    pool, rank, device, seq_len, lse_dtype
                )
                results.append((seq_len, str(lse_dtype), max_error, mean_error))
        graph_max, graph_mean = _graph_case(pool, rank, device)
        results.append((97, "graph/bfloat16", graph_max, graph_mean))
        gathered: list[list[tuple[int, str, float, float]] | None] = [
            None for _ in range(world_size)
        ]
        dist.all_gather_object(gathered, results)
        if rank == 0:
            assert all(item is not None for item in gathered)
            flattened = [row for rank_rows in gathered for row in rank_rows or []]
            for seq_len, dtype, max_error, mean_error in flattened:
                print(
                    f"seq={seq_len:4d} lse={dtype:14s} "
                    f"max_abs={max_error:.6f} mean_abs={mean_error:.6f}",
                    flush=True,
                )
            if not all(
                math.isfinite(row[2]) and math.isfinite(row[3]) for row in flattened
            ):
                raise AssertionError("Triton MLA DCP8 produced a non-finite error")
            worst = max(row[2] for row in flattened)
            if worst > 0.08:
                raise AssertionError(
                    f"Triton MLA DCP8 exceeded max_abs tolerance: {worst:.6f}"
                )
            print(
                "PASS Kimi K3 Triton MLA FP8-KV DCP8 against full-context "
                f"reference (worst max_abs={worst:.6f})",
                flush=True,
            )
    finally:
        pool.close()
        dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world-size", type=int, default=WORLD_SIZE)
    args = parser.parse_args()
    if args.world_size != WORLD_SIZE:
        raise SystemExit(f"this validator requires --world-size {WORLD_SIZE}")
    if torch.cuda.device_count() < WORLD_SIZE:
        raise SystemExit(
            f"need {WORLD_SIZE} CUDA devices, found {torch.cuda.device_count()}"
        )
    _load_extension()
    mp.spawn(
        _worker,
        args=(WORLD_SIZE, _free_port()),
        nprocs=WORLD_SIZE,
        join=True,
    )


if __name__ == "__main__":
    main()
