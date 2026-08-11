#!/usr/bin/env python3
"""Compile all Kimi-K3 graph-visible W4A16 direct-M shapes without weights."""

from __future__ import annotations

import argparse

import torch
import torch.multiprocessing as mp
from sparkinfer import freeze_kernel_resolution, unfreeze_kernel_resolution
from sparkinfer.moe._shared.kernels.w4a16.kernel import (
    _compile_w4a16_small_m_direct,
)


def _compile(device: torch.device, m: int) -> None:
    _compile_w4a16_small_m_direct(
        m=m,
        hidden_size=3584,
        intermediate_size=192,
        num_experts=896,
        topk=16,
        activation="situ",
        fast_math=True,
        topk_ids_dtype=torch.int32,
        device=device,
        scale_format="e8m0_k32",
        swiglu_limit=None,
        swiglu_alpha=1.0,
        swiglu_beta=0.0,
        w13_layout="w31",
    )


def _worker(rank: int, max_m: int) -> None:
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    for m in range(1, max_m + 1):
        _compile(device, m)

    # A graph capture freezes kernel resolution. Every exact M must remain a
    # cache hit after that boundary; this reproduces the serving contract.
    freeze_kernel_resolution("Kimi-K3 DSpark graph-shape validation")
    try:
        for m in range(1, max_m + 1):
            _compile(device, m)
    finally:
        unfreeze_kernel_resolution()
    if rank == 0:
        print(
            f"PASS Kimi-K3 W4A16 exact direct-M shapes 1..{max_m} before/after freeze",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world-size", type=int, default=16)
    parser.add_argument("--max-m", type=int, default=8)
    args = parser.parse_args()
    if args.world_size <= 0 or torch.cuda.device_count() < args.world_size:
        raise SystemExit(
            f"need {args.world_size} CUDA devices, found {torch.cuda.device_count()}"
        )
    if not 1 <= args.max_m <= 32:
        raise SystemExit("--max-m must be between 1 and 32")
    mp.spawn(_worker, args=(args.max_m,), nprocs=args.world_size, join=True)


if __name__ == "__main__":
    main()
