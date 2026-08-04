#!/usr/bin/env python3
"""Benchmark replicated, gathered, and local-argmax DSpark Markov heads."""

from __future__ import annotations

import argparse
import os
import statistics

import torch
import torch.distributed as dist
import torch.nn.functional as F
from safetensors import safe_open


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--steps", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    return parser.parse_args()


def load_weight(
    path: str,
    name: str,
    start: int | None = None,
    end: int | None = None,
) -> torch.Tensor:
    with safe_open(path, framework="pt", device="cpu") as handle:
        weight = handle.get_slice(name)
        return weight[:] if start is None else weight[start:end]


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    with safe_open(args.checkpoint, framework="pt", device="cpu") as handle:
        vocab_size, markov_rank = handle.get_slice(
            "markov_head.markov_w1.weight"
        ).get_shape()
    if vocab_size % world_size:
        raise ValueError(f"vocab size {vocab_size} is not divisible by TP {world_size}")
    local_vocab = vocab_size // world_size
    start = rank * local_vocab
    end = start + local_vocab
    token = torch.tensor([327], dtype=torch.long, device=device)
    full_output = torch.empty(
        (world_size, 1, local_vocab), dtype=torch.bfloat16, device=device
    )
    pair_output = torch.empty((world_size, 1, 2), dtype=torch.float32, device=device)
    generator = torch.Generator(device="cpu").manual_seed(20260804)
    base_full = torch.randn(
        args.steps, vocab_size, dtype=torch.bfloat16, generator=generator
    ).to(device)

    def time_variant(fn):
        for _ in range(args.warmup):
            fn()
        dist.barrier()
        samples = []
        for _ in range(args.repeats):
            begin = torch.cuda.Event(enable_timing=True)
            finish = torch.cuda.Event(enable_timing=True)
            begin.record()
            fn()
            finish.record()
            finish.synchronize()
            samples.append(begin.elapsed_time(finish))
        return statistics.median(samples), statistics.mean(samples)

    w1 = load_weight(args.checkpoint, "markov_head.markov_w1.weight").to(device)
    w2 = load_weight(args.checkpoint, "markov_head.markov_w2.weight").to(device)

    def replicated():
        current = token
        for step in range(args.steps):
            embed = F.embedding(current, w1)
            logits = F.linear(embed, w2) + base_full[step]
            current = logits.argmax(dim=-1)

    replicated_timing = time_variant(replicated)
    base_local = base_full[:, start:end].contiguous()
    dist.barrier()

    w1_local = load_weight(
        args.checkpoint, "markov_head.markov_w1.weight", start, end
    ).to(device)
    w2_local = load_weight(
        args.checkpoint, "markov_head.markov_w2.weight", start, end
    ).to(device)

    def local_markov(current: torch.Tensor) -> torch.Tensor:
        in_range = (current >= start) & (current < end)
        local_id = torch.where(in_range, current - start, 0)
        embed = F.embedding(local_id, w1_local)
        embed.mul_(in_range.unsqueeze(-1))
        dist.all_reduce(embed)
        return embed

    def gathered():
        current = token
        for step in range(args.steps):
            local_logits = F.linear(local_markov(current), w2_local)
            local_logits.add_(base_local[step])
            dist.all_gather_into_tensor(full_output, local_logits)
            current = full_output.view(1, vocab_size).argmax(dim=-1)

    gathered_timing = time_variant(gathered)

    def local_argmax():
        current = token
        for step in range(args.steps):
            local_logits = F.linear(local_markov(current), w2_local)
            local_logits.add_(base_local[step])
            local_value, local_id = local_logits.max(dim=-1)
            pair = torch.stack(
                [local_value.float(), (local_id + start).float()], dim=-1
            )
            dist.all_gather_into_tensor(pair_output, pair)
            best_rank = pair_output[:, :, 0].argmax(dim=0, keepdim=True)
            current = pair_output[:, :, 1].gather(0, best_rank).view(1).long()

    local_argmax_timing = time_variant(local_argmax)

    id_bits = (vocab_size - 1).bit_length()
    id_mask = (1 << id_bits) - 1

    def packed_maxloc():
        current = token
        for step in range(args.steps):
            local_logits = F.linear(local_markov(current), w2_local)
            local_logits.add_(base_local[step])
            local_value, local_id = local_logits.max(dim=-1)
            value_bits = local_value.float().contiguous().view(torch.int32).long()
            value_bits.bitwise_and_(0xFFFF_FFFF)
            negative = value_bits.bitwise_and(0x8000_0000).ne(0)
            ordered = torch.where(
                negative,
                value_bits.bitwise_xor(0xFFFF_FFFF),
                value_bits.bitwise_xor(0x8000_0000),
            )
            global_id = local_id + start
            packed = (ordered << id_bits) | (id_mask - global_id)
            dist.all_reduce(packed, op=dist.ReduceOp.MAX)
            current = (id_mask - (packed & id_mask)).long()

    packed_timing = time_variant(packed_maxloc)
    if rank == 0:
        replicated_bytes = 2 * vocab_size * markov_rank * 2
        sharded_bytes = replicated_bytes // world_size
        print(
            f"TP={world_size} steps={args.steps} vocab={vocab_size} rank={markov_rank}"
        )
        print(
            f"replicated: median={replicated_timing[0]:.3f} ms "
            f"mean={replicated_timing[1]:.3f} ms "
            f"weights={replicated_bytes / 2**20:.2f} MiB/rank"
        )
        print(
            f"full-gather: median={gathered_timing[0]:.3f} ms "
            f"mean={gathered_timing[1]:.3f} ms "
            f"weights={sharded_bytes / 2**20:.2f} MiB/rank"
        )
        print(
            f"local-max:   median={local_argmax_timing[0]:.3f} ms "
            f"mean={local_argmax_timing[1]:.3f} ms "
            f"weights={sharded_bytes / 2**20:.2f} MiB/rank"
        )
        print(
            f"packed-max:  median={packed_timing[0]:.3f} ms "
            f"mean={packed_timing[1]:.3f} ms "
            f"weights={sharded_bytes / 2**20:.2f} MiB/rank"
        )
        print(
            "packed-max delta="
            f"{packed_timing[0] - replicated_timing[0]:.3f} ms; "
            f"saved={(replicated_bytes - sharded_bytes) / 2**20:.2f} MiB/rank"
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
