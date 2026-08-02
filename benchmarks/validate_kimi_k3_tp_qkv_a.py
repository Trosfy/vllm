#!/usr/bin/env python3
"""Validate Kimi K3's TP-sharded BF16 MLA q_a/kv_a projection.

Run with ``torchrun --standalone --nproc-per-node=16``.  The harness uses the
real K3 dimensions, compares the gathered sharded projection with a replicated
oracle, checks the logical q_a/kv_a ordering, and reports the exact resident
weight-memory saving without loading the model checkpoint.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import torch
import torch.distributed as dist

HIDDEN_SIZE = 7168
Q_LORA_RANK = 1536
KV_A_WIDTH = 576
MLA_LAYERS = 24


def _restore_merged_output_order(
    rank_major_output: torch.Tensor,
    output_sizes: list[int],
    tp_size: int,
) -> torch.Tensor:
    local_sizes = [size // tp_size for size in output_sizes]
    rank_major = rank_major_output.unflatten(-1, (tp_size, sum(local_sizes)))
    return torch.cat(
        [part.flatten(-2) for part in rank_major.split(local_sizes, dim=-1)],
        dim=-1,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--output", type=str)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    output_sizes = [Q_LORA_RANK, KV_A_WIDTH]
    if any(size % world_size for size in output_sizes):
        raise ValueError(f"K3 q_a/kv_a widths do not divide TP={world_size}")

    generator = torch.Generator(device=device).manual_seed(20260802)
    x = torch.randn(
        8, HIDDEN_SIZE, device=device, dtype=torch.bfloat16, generator=generator
    )
    q_weight = torch.randn(
        Q_LORA_RANK,
        HIDDEN_SIZE,
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    )
    kv_weight = torch.randn(
        KV_A_WIDTH,
        HIDDEN_SIZE,
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    )

    q_local = Q_LORA_RANK // world_size
    kv_local = KV_A_WIDTH // world_size
    local_weight = torch.cat(
        (
            q_weight.narrow(0, rank * q_local, q_local),
            kv_weight.narrow(0, rank * kv_local, kv_local),
        ),
        dim=0,
    ).contiguous()
    local_output = x @ local_weight.T
    gathered_parts = [torch.empty_like(local_output) for _ in range(world_size)]
    dist.all_gather(gathered_parts, local_output)
    rank_major = torch.cat(gathered_parts, dim=-1)
    actual = _restore_merged_output_order(rank_major, output_sizes, world_size)
    expected = torch.cat((x @ q_weight.T, x @ kv_weight.T), dim=-1)
    actual_fp32 = actual.float()
    expected_fp32 = expected.float()
    difference = actual_fp32 - expected_fp32
    cosine = torch.nn.functional.cosine_similarity(
        actual_fp32.flatten(), expected_fp32.flatten(), dim=0
    ).item()
    rmse = difference.square().mean().sqrt().item()
    max_abs = difference.abs().max().item()
    if cosine < 0.99999:
        raise AssertionError(f"Sharded projection cosine {cosine} is below 0.99999")

    for _ in range(args.warmups):
        dist.all_gather(gathered_parts, local_output)
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(args.iterations):
        dist.all_gather(gathered_parts, local_output)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    full_layer_bytes = HIDDEN_SIZE * sum(output_sizes) * 2
    local_layer_bytes = full_layer_bytes // world_size
    result = {
        "status": "PASS",
        "tp_size": world_size,
        "geometry": {
            "hidden_size": HIDDEN_SIZE,
            "q_lora_rank": Q_LORA_RANK,
            "kv_a_width": KV_A_WIDTH,
            "mla_layers": MLA_LAYERS,
        },
        "all_gather_us": elapsed * 1e6 / args.iterations,
        "oracle": {
            "cosine": cosine,
            "rmse": rmse,
            "max_abs": max_abs,
        },
        "full_projection_gib_per_rank": full_layer_bytes * MLA_LAYERS / 2**30,
        "sharded_projection_gib_per_rank": local_layer_bytes * MLA_LAYERS / 2**30,
        "saved_gib_per_rank": (full_layer_bytes - local_layer_bytes)
        * MLA_LAYERS
        / 2**30,
    }
    if rank == 0:
        rendered = json.dumps(result, indent=2, sort_keys=True)
        print(rendered, flush=True)
        if args.output:
            with open(args.output, "w", encoding="utf-8") as output_file:
                output_file.write(rendered + "\n")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
