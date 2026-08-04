#!/usr/bin/env python3
"""Compare token-major and block-major K3 AttnRes workspace layouts."""

import argparse
import statistics

import torch

from vllm.models.kimi_k3.nvidia.ops.attn_res import attn_res


def run(
    prefix: torch.Tensor,
    delta: torch.Tensor,
    blocks: torch.Tensor,
    norm_weight: torch.Tensor,
    qk_weight: torch.Tensor,
    output_norm_weight: torch.Tensor,
    num_blocks: int,
) -> torch.Tensor:
    return attn_res(
        prefix,
        delta,
        blocks,
        norm_weight,
        qk_weight,
        output_norm_weight,
        num_blocks,
        -1,
        1e-5,
        1e-5,
        output_buffer=delta,
    )


def measure_ms(
    prefix: torch.Tensor,
    delta: torch.Tensor,
    blocks: torch.Tensor,
    norm_weight: torch.Tensor,
    qk_weight: torch.Tensor,
    output_norm_weight: torch.Tensor,
    num_blocks: int,
    iterations: int,
) -> float:
    for _ in range(2):
        run(
            prefix,
            delta,
            blocks,
            norm_weight,
            qk_weight,
            output_norm_weight,
            num_blocks,
        )
    torch.cuda.synchronize()
    samples = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        run(
            prefix,
            delta,
            blocks,
            norm_weight,
            qk_weight,
            output_norm_weight,
            num_blocks,
        )
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.median(samples)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=2048)
    parser.add_argument(
        "--capacity-tokens",
        type=int,
        default=None,
        help="Workspace capacity; defaults to the active token count.",
    )
    parser.add_argument("--hidden-size", type=int, default=7168)
    parser.add_argument("--num-blocks", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()
    capacity_tokens = args.capacity_tokens or args.tokens
    if capacity_tokens < args.tokens:
        raise ValueError("capacity-tokens must be at least tokens")

    torch.manual_seed(11)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    prefix = torch.randn(
        args.tokens, args.hidden_size, device=device, dtype=dtype
    )
    delta = torch.randn_like(prefix)
    token_major_storage = torch.randn(
        capacity_tokens,
        args.num_blocks,
        args.hidden_size,
        device=device,
        dtype=dtype,
    )
    token_major = token_major_storage[: args.tokens]
    block_major_storage = torch.empty(
        args.num_blocks,
        capacity_tokens,
        args.hidden_size,
        device=device,
        dtype=dtype,
    )
    block_major = block_major_storage.permute(1, 0, 2)[: args.tokens]
    block_major.copy_(token_major)
    norm_weight = torch.randn(args.hidden_size, device=device, dtype=dtype)
    qk_weight = torch.randn(args.hidden_size, device=device, dtype=dtype)
    output_norm_weight = torch.randn_like(norm_weight)

    token_prefix = prefix.clone()
    token_delta = delta.clone()
    token_output = run(
        token_prefix,
        token_delta,
        token_major,
        norm_weight,
        qk_weight,
        output_norm_weight,
        args.num_blocks,
    ).clone()
    block_prefix = prefix.clone()
    block_delta = delta.clone()
    block_output = run(
        block_prefix,
        block_delta,
        block_major,
        norm_weight,
        qk_weight,
        output_norm_weight,
        args.num_blocks,
    ).clone()

    token_major_ms = measure_ms(
        prefix.clone(),
        delta.clone(),
        token_major,
        norm_weight,
        qk_weight,
        output_norm_weight,
        args.num_blocks,
        args.iterations,
    )
    block_major_ms = measure_ms(
        prefix.clone(),
        delta.clone(),
        block_major,
        norm_weight,
        qk_weight,
        output_norm_weight,
        args.num_blocks,
        args.iterations,
    )
    print(
        {
            "tokens": args.tokens,
            "capacity_tokens": capacity_tokens,
            "num_blocks": args.num_blocks,
            "token_major_ms": token_major_ms,
            "block_major_ms": block_major_ms,
            "block_over_token_ratio": block_major_ms / token_major_ms,
            "output_exact": torch.equal(block_output, token_output),
            "prefix_exact": torch.equal(block_prefix, token_prefix),
            "max_abs": float(
                (block_output.float() - token_output.float()).abs().max()
            ),
            "scratch_contiguous": block_major[:, -1, :].is_contiguous(),
        }
    )


if __name__ == "__main__":
    main()
