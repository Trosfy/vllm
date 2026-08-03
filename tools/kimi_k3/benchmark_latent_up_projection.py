# SPDX-License-Identifier: Apache-2.0
"""Model-free Kimi-K3 latent up-projection TP correctness/latency harness.

Run on the target 16-GPU node:

    torchrun --standalone --nproc-per-node=16 \
      tools/kimi_k3/benchmark_latent_up_projection.py

This allocates only one BF16 projection, not the Kimi-K3 checkpoint. It checks
that a row-sharded up projection plus one combined routed/shared all-reduce is
equivalent to the original replicated projection. The latency result uses NCCL
as a structural microbenchmark; the final gate remains the full vLLM decode.
"""

from __future__ import annotations

import argparse
import os
import statistics

import torch
import torch.distributed as dist


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    variance = x.float().square().mean(dim=-1, keepdim=True)
    return x * torch.rsqrt(variance + eps).to(x.dtype) * weight


def replicated_step(
    latent_partial: torch.Tensor,
    shared_partial: torch.Tensor,
    norm_weight: torch.Tensor,
    full_weight: torch.Tensor,
    aux_stream: torch.cuda.Stream,
    eps: float,
) -> torch.Tensor:
    latent = latent_partial.clone()
    dist.all_reduce(latent)
    latent = rms_norm(latent, norm_weight, eps)

    shared = shared_partial.clone()
    main_stream = torch.cuda.current_stream()
    shared.record_stream(aux_stream)
    aux_stream.wait_stream(main_stream)
    with torch.cuda.stream(aux_stream):
        dist.all_reduce(shared)
    routed = latent @ full_weight.T
    main_stream.wait_stream(aux_stream)
    return routed + shared


def sharded_step(
    latent_partial: torch.Tensor,
    shared_partial: torch.Tensor,
    norm_weight: torch.Tensor,
    weight_shard: torch.Tensor,
    rank: int,
    world_size: int,
    eps: float,
) -> torch.Tensor:
    latent = latent_partial.clone()
    dist.all_reduce(latent)
    latent = rms_norm(latent, norm_weight, eps)
    latent_shard = latent.chunk(world_size, dim=-1)[rank]
    result = latent_shard @ weight_shard.T
    result.add_(shared_partial)
    dist.all_reduce(result)
    return result


def benchmark(fn, warmup: int, iterations: int) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    samples = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=1)
    parser.add_argument("--latent-size", type=int, default=3584)
    parser.add_argument("--hidden-size", type=int, default=7168)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if args.latent_size % world_size:
        raise ValueError("latent size must be divisible by world size")

    generator = torch.Generator(device=device).manual_seed(20260803)
    full_weight = torch.randn(
        args.hidden_size,
        args.latent_size,
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    shard_size = args.latent_size // world_size
    weight_shard = full_weight[
        :, rank * shard_size : (rank + 1) * shard_size
    ].contiguous()
    norm_weight = torch.randn(
        args.latent_size,
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    rank_generator = torch.Generator(device=device).manual_seed(1000 + rank)
    latent_partial = torch.randn(
        args.tokens,
        args.latent_size,
        dtype=torch.bfloat16,
        device=device,
        generator=rank_generator,
    )
    shared_partial = torch.randn(
        args.tokens,
        args.hidden_size,
        dtype=torch.bfloat16,
        device=device,
        generator=rank_generator,
    )
    aux_stream = torch.cuda.Stream()
    eps = 1e-6

    reference = replicated_step(
        latent_partial,
        shared_partial,
        norm_weight,
        full_weight,
        aux_stream,
        eps,
    )
    actual = sharded_step(
        latent_partial,
        shared_partial,
        norm_weight,
        weight_shard,
        rank,
        world_size,
        eps,
    )
    actual_fp32 = actual.float()
    reference_fp32 = reference.float()
    cosine = torch.nn.functional.cosine_similarity(
        actual_fp32.flatten(), reference_fp32.flatten(), dim=0
    )
    relative_l2 = (actual_fp32 - reference_fp32).norm() / reference_fp32.norm()
    max_abs = (actual_fp32 - reference_fp32).abs().max()
    # BF16 changes the accumulation order: the replicated GEMM accumulates
    # all latent channels in one kernel, while TP sums 16 partial GEMMs. The
    # structural check therefore gates on direction and relative L2 error.
    if cosine < 0.9999 or relative_l2 > 1e-2:
        raise AssertionError(
            "row-sharded projection diverged from replicated reference: "
            f"cosine={cosine.item():.8f}, relative_l2={relative_l2.item():.8f}"
        )

    replicated_samples = benchmark(
        lambda: replicated_step(
            latent_partial,
            shared_partial,
            norm_weight,
            full_weight,
            aux_stream,
            eps,
        ),
        args.warmup,
        args.iterations,
    )
    sharded_samples = benchmark(
        lambda: sharded_step(
            latent_partial,
            shared_partial,
            norm_weight,
            weight_shard,
            rank,
            world_size,
            eps,
        ),
        args.warmup,
        args.iterations,
    )

    metrics = torch.tensor(
        [
            statistics.median(replicated_samples),
            statistics.median(sharded_samples),
        ],
        dtype=torch.float64,
        device=device,
    )
    dist.all_reduce(metrics, op=dist.ReduceOp.MAX)
    if rank == 0:
        replicated_ms, sharded_ms = metrics.tolist()
        print("Kimi-K3 latent up-projection model-free check: PASS")
        print(f"world_size={world_size} tokens={args.tokens}")
        print(f"cosine={cosine.item():.8f} relative_l2={relative_l2.item():.8f}")
        print(f"max_abs_bf16={max_abs.item():.6f}")
        print(f"replicated median(max-rank): {replicated_ms:.6f} ms")
        print(f"sharded median(max-rank):   {sharded_ms:.6f} ms")
        print(f"sharded/replicated:         {sharded_ms / replicated_ms:.4f}x")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
