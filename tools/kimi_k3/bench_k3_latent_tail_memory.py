# SPDX-License-Identifier: Apache-2.0
"""Exercise Kimi-K3's large-prefill latent all-reduce + RMSNorm without a model.

Run on the production TP topology, for example::

    torchrun --standalone --nproc-per-node=16 \
      tools/kimi_k3/bench_k3_latent_tail_memory.py --mode compare

``old`` reproduces the two full-size output allocations in the original path.
``new`` donates the dead latent input to NCCL and RMSNorm.  ``--free-mib`` can
retain allocator ballast to test the paths at the full-model memory boundary.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import torch
import torch.distributed as dist


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("compare", "old", "new"), required=True)
    parser.add_argument("--tokens", type=int, default=2048)
    parser.add_argument("--latent-size", type=int, default=3584)
    parser.add_argument(
        "--free-mib",
        type=int,
        default=0,
        help="Retain ballast until approximately this much device memory is free.",
    )
    return parser.parse_args()


def _allocate_ballast(device: torch.device, requested_free_mib: int):
    if requested_free_mib <= 0:
        return []
    target = requested_free_mib * 1024**2
    margin = 8 * 1024**2
    ballast: list[torch.Tensor] = []
    while True:
        free, _ = torch.cuda.mem_get_info(device)
        alloc_bytes = free - target - margin
        if alloc_bytes <= 0:
            break
        chunk = min(alloc_bytes, 1024**3)
        ballast.append(torch.empty(chunk, dtype=torch.uint8, device=device))
    torch.cuda.synchronize(device)
    return ballast


def _rms_norm_out_of_place(
    hidden_states: torch.Tensor, weight: torch.Tensor, epsilon: float
) -> torch.Tensor:
    output = torch.empty_like(hidden_states)
    torch.ops._C.rms_norm(output, hidden_states, weight, epsilon)
    return output


def _rms_norm_in_place(
    hidden_states: torch.Tensor, weight: torch.Tensor, epsilon: float
) -> torch.Tensor:
    torch.ops._C.rms_norm(hidden_states, hidden_states, weight, epsilon)
    return hidden_states


def _old_path(hidden_states: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    from vllm.distributed import tensor_model_parallel_all_reduce

    reduced = tensor_model_parallel_all_reduce(hidden_states)
    return _rms_norm_out_of_place(reduced, weight, 1e-6)


def _new_path(hidden_states: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    from vllm.distributed import tensor_model_parallel_all_reduce_in_place

    reduced = tensor_model_parallel_all_reduce_in_place(hidden_states)
    return _rms_norm_in_place(reduced, weight, 1e-6)


def main() -> None:
    args = _parse_args()
    # The base image exports the legacy C++ backend, which rejects TP16.
    # Override it exactly as the Kimi launch profile does.
    os.environ["VLLM_ENABLE_PCIE_ALLREDUCE"] = "1"
    os.environ["VLLM_PCIE_ALLREDUCE_BACKEND"] = "b12x"
    os.environ["VLLM_PCIE_ONESHOT_SINGLE_CHANNEL"] = "1"

    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.config.parallel import ParallelConfig
    from vllm.distributed import tensor_model_parallel_all_reduce
    from vllm.distributed.parallel_state import (
        cleanup_dist_env_and_memory,
        get_tp_group,
        init_distributed_environment,
        initialize_model_parallel,
    )

    config = VllmConfig(
        parallel_config=ParallelConfig(tensor_parallel_size=world_size)
    )
    config_context = set_current_vllm_config(config)
    config_context.__enter__()
    init_distributed_environment(local_rank=local_rank)
    initialize_model_parallel(tensor_model_parallel_size=world_size)
    rank = dist.get_rank()

    try:
        torch.manual_seed(1234 + rank)
        weight = torch.randn(
            args.latent_size, dtype=torch.bfloat16, device=device
        )
        source = torch.randn(
            (args.tokens, args.latent_size),
            dtype=torch.bfloat16,
            device=device,
        )

        # Initialize the same large-message NCCL fallback before adding ballast.
        warm = tensor_model_parallel_all_reduce(source.clone())
        torch.cuda.synchronize(device)
        del warm
        torch.cuda.empty_cache()
        dist.barrier()

        if args.mode == "compare":
            old = _old_path(source.clone(), weight)
            new = _new_path(source.clone(), weight)
            torch.cuda.synchronize(device)
            max_error = (old.float() - new.float()).abs().max()
            exact = torch.tensor(
                int(torch.equal(old, new)), dtype=torch.int32, device=device
            )
            dist.all_reduce(max_error, op=dist.ReduceOp.MAX)
            dist.all_reduce(exact, op=dist.ReduceOp.MIN)
            if rank == 0:
                print(
                    json.dumps(
                        {
                            "status": "pass",
                            "mode": args.mode,
                            "exact_equal_all_ranks": bool(exact.item()),
                            "max_abs_error": max_error.item(),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            return

        hidden_states = source.clone()
        del source
        torch.cuda.empty_cache()
        ballast = _allocate_ballast(device, args.free_mib)
        free_before, _ = torch.cuda.mem_get_info(device)
        allocated_before = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
        dist.barrier()
        started = time.perf_counter()
        if args.mode == "old":
            output = _old_path(hidden_states, weight)
        else:
            output = _new_path(hidden_states, weight)
        torch.cuda.synchronize(device)
        elapsed_ms = (time.perf_counter() - started) * 1000
        peak_delta = torch.cuda.max_memory_allocated(device) - allocated_before
        # Accumulate in FP32 without first materializing a full FP32 copy;
        # the pressure mode deliberately leaves less than that 28 MiB tensor.
        checksum = output.sum(dtype=torch.float32)
        dist.all_reduce(checksum)
        elapsed = torch.tensor(elapsed_ms, dtype=torch.float64, device=device)
        peak = torch.tensor(peak_delta, dtype=torch.int64, device=device)
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
        dist.all_reduce(peak, op=dist.ReduceOp.MAX)
        if rank == 0:
            print(
                json.dumps(
                    {
                        "status": "pass",
                        "mode": args.mode,
                        "tokens": args.tokens,
                        "latent_size": args.latent_size,
                        "free_before_mib": free_before / 1024**2,
                        "peak_allocation_delta_mib": peak.item() / 1024**2,
                        "max_rank_elapsed_ms": elapsed.item(),
                        "checksum": checksum.item(),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        del ballast, output
    finally:
        communicator = get_tp_group().device_communicator
        custom_allreduce = getattr(communicator, "ca_comm", None)
        if custom_allreduce is not None:
            custom_allreduce.close()
        cleanup_dist_env_and_memory()
        config_context.__exit__(None, None, None)


if __name__ == "__main__":
    main()
