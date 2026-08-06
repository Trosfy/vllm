# SPDX-License-Identifier: Apache-2.0
"""Exercise Kimi-K3's repeated fused paired gather+router without weights.

Run one process per TP rank. The test pre-creates the M=8 and M=1 projection
pools, captures a full target's 92 routed-MoE gathers and exact sigmoid top-k
operations, and asserts that capture did not lazily create another pool on the
wrong semantic channel.
"""

from __future__ import annotations

import argparse
import os

import torch
import torch.distributed as dist


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dcp-size", type=int, choices=(8, 16), default=16)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--layers", type=int, default=92)
    parser.add_argument("--replays", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != 16:
        raise ValueError(f"expected TP16, got {world_size=}")

    os.environ.setdefault("VLLM_USE_B12X_DCP_A2A", "1")
    os.environ.setdefault("VLLM_KIMI_USE_B12X_PROJECTION_GATHER", "1")
    os.environ.setdefault("VLLM_KIMI_USE_B12X_PAIRED_PROJECTION_GATHER", "1")
    os.environ.setdefault("VLLM_KIMI_USE_B12X_PAIRED_PROJECTION_TOPK", "1")
    os.environ.setdefault("VLLM_DCP_A2A_MAX_TOKENS", str(args.batch))

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    import vllm._custom_ops as ops
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.config.parallel import ParallelConfig
    from vllm.distributed.parallel_state import (
        cleanup_dist_env_and_memory,
        get_dcp_group,
        get_tp_group,
        graph_capture,
        init_distributed_environment,
        initialize_model_parallel,
        set_custom_all_reduce,
    )
    from vllm.v1.attention.ops import dcp_alltoall

    config = VllmConfig(
        parallel_config=ParallelConfig(
            tensor_parallel_size=world_size,
            decode_context_parallel_size=args.dcp_size,
            dcp_comm_backend="a2a",
            disable_custom_all_reduce=True,
        )
    )
    config_context = set_current_vllm_config(config)
    config_context.__enter__()
    set_custom_all_reduce(False)
    init_distributed_environment(local_rank=local_rank)
    initialize_model_parallel(
        tensor_model_parallel_size=world_size,
        decode_context_model_parallel_size=args.dcp_size,
    )
    tp_group = get_tp_group()
    dcp_group = get_dcp_group()
    projection_group = (
        dcp_group if dcp_group.world_size == tp_group.world_size else tp_group
    )
    tp_rank = int(tp_group.rank_in_group)

    down_locals: list[torch.Tensor] = []
    router_locals: list[torch.Tensor] = []
    down_outputs: list[torch.Tensor] = []
    routing_payloads: list[torch.Tensor] = []
    down_expected: list[torch.Tensor] = []
    weights_expected: list[torch.Tensor] = []
    ids_expected: list[torch.Tensor] = []
    correction_biases: list[torch.Tensor] = []
    down_base = torch.arange(args.batch * 224, dtype=torch.int32, device=device).view(
        args.batch, 224
    )
    router_base = torch.arange(
        args.batch * 56, dtype=torch.float32, device=device
    ).view(args.batch, 56)
    for layer in range(args.layers):
        down_locals.append(
            (
                down_base.remainder(97).float() * 0.03125
                + tp_rank * 4.0
                + layer * 0.00390625
            ).to(torch.bfloat16)
        )
        router_locals.append(router_base + tp_rank * 10_000 + layer * 100)
        down_outputs.append(
            torch.empty(
                (args.batch, world_size * 224),
                device=device,
                dtype=torch.bfloat16,
            )
        )
        routing_payloads.append(
            torch.empty((args.batch * 2, 16), device=device, dtype=torch.float32)
        )
        down_expected.append(
            torch.cat(
                [
                    (
                        down_base.remainder(97).float() * 0.03125
                        + source_rank * 4.0
                        + layer * 0.00390625
                    ).to(torch.bfloat16)
                    for source_rank in range(world_size)
                ],
                dim=1,
            )
        )
        gathered_router = torch.cat(
            [
                router_base + source_rank * 10_000 + layer * 100
                for source_rank in range(world_size)
            ],
            dim=1,
        )
        correction_bias = (
            torch.sin(torch.arange(896, device=device, dtype=torch.float32))
            * (layer + 1)
            * 1e-5
        )
        correction_biases.append(correction_bias)
        weights, ids = ops.grouped_topk(
            gathered_router,
            1,
            1,
            16,
            True,
            1.0,
            correction_bias,
            1,
        )
        weights_expected.append(weights)
        ids_expected.append(ids)

    graph: torch.cuda.CUDAGraph | None = None
    try:
        warmed = dcp_alltoall.warmup_b12x_kimi_projection_gathers(
            projection_group,
            device=device,
        )
        if warmed != 2:
            raise AssertionError(f"expected two projection pools, got {warmed}")
        pool_keys_before = frozenset(dcp_alltoall._B12X_DCP_A2A_POOLS)
        if len(pool_keys_before) != 2:
            raise AssertionError(f"expected M=8 and M=1 pools, got {pool_keys_before}")

        def run_sequence() -> None:
            for layer in range(args.layers):
                fused = dcp_alltoall.try_dcp_b12x_all_gather_pair_kimi_topk(
                    down_locals[layer],
                    router_locals[layer],
                    correction_biases[layer],
                    projection_group,
                    max_batch_size=args.batch,
                )
                if fused is None:
                    raise AssertionError("fused Kimi projection router declined")
                down, routing_payload = fused
                down_outputs[layer].copy_(down)
                routing_payloads[layer].copy_(routing_payload)

        with graph_capture(device) as context:
            run_sequence()
            torch.cuda.synchronize(device)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=context.stream):
                run_sequence()

        pool_keys_after = frozenset(dcp_alltoall._B12X_DCP_A2A_POOLS)
        if pool_keys_after != pool_keys_before:
            raise AssertionError(
                "projection pool was created lazily during graph capture: "
                f"before={pool_keys_before}, after={pool_keys_after}"
            )
        for _ in range(args.replays):
            graph.replay()
        torch.cuda.synchronize(device)
        for layer in range(args.layers):
            torch.testing.assert_close(
                down_outputs[layer], down_expected[layer], rtol=0, atol=0
            )
            weights = routing_payloads[layer][: args.batch]
            ids = routing_payloads[layer][args.batch :].view(torch.int32)
            torch.testing.assert_close(
                weights, weights_expected[layer], rtol=0, atol=0
            )
            torch.testing.assert_close(ids, ids_expected[layer], rtol=0, atol=0)
        dist.barrier()
        if dist.get_rank() == 0:
            print(
                "status=pass "
                f"tp=16 dcp={args.dcp_size} batch={args.batch} "
                f"layers={args.layers} replays={args.replays} pools=2",
                flush=True,
            )
    finally:
        if graph is not None:
            graph.reset()
        torch.cuda.synchronize(device)
        for pool in dcp_alltoall._B12X_DCP_A2A_POOLS.values():
            pool.close()
        dcp_alltoall._B12X_DCP_A2A_POOLS.clear()
        cleanup_dist_env_and_memory()
        config_context.__exit__(None, None, None)


if __name__ == "__main__":
    main()
