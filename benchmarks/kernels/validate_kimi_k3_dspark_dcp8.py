#!/usr/bin/env python3
"""Validate Kimi-K3 DSpark target DCP8 without loading model weights."""

from __future__ import annotations

import argparse
import socket
import time
from dataclasses import dataclass
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from sparkinfer.comm.pcie.pcie_dcp_a2a import (
    PCIeDCPA2APool,
    _load_extension,
    lse_reduce_scatter_reference,
)

from vllm.v1.attention.backends.mla.b12x_mla import (
    _dcp_local_seq_lens_from_global,
)
from vllm.v1.attention.ops import dcp_alltoall


@dataclass(frozen=True)
class Geometry:
    name: str
    total_heads: int
    max_batch: int


GEOMETRIES = (
    Geometry("target", total_heads=48, max_batch=8),
)
HEAD_DIM = 512
QUERY_HEAD_DIM = 576


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _inputs(
    geometry: Geometry,
    rank: int,
    batch: int,
    step: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(
        geometry.total_heads * 10_000 + step * 100 + rank
    )
    output = torch.randn(
        batch,
        geometry.total_heads,
        HEAD_DIM,
        generator=generator,
        dtype=torch.float32,
    ).to(device=device, dtype=torch.bfloat16)
    lse = torch.randn(
        batch,
        geometry.total_heads,
        generator=generator,
        dtype=torch.float32,
    ).to(device=device)
    query = torch.randn(
        batch,
        geometry.total_heads // dist.get_world_size(),
        QUERY_HEAD_DIM,
        generator=generator,
        dtype=torch.float32,
    ).to(device=device, dtype=torch.bfloat16)
    return output, lse, query


def _reference(
    output: torch.Tensor,
    lse: torch.Tensor,
    query: torch.Tensor,
    rank: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    world_size = dist.get_world_size()
    outputs = [torch.empty_like(output) for _ in range(world_size)]
    lses = [torch.empty_like(lse) for _ in range(world_size)]
    queries = [torch.empty_like(query) for _ in range(world_size)]
    dist.all_gather(outputs, output)
    dist.all_gather(lses, lse)
    dist.all_gather(queries, query)
    return (
        lse_reduce_scatter_reference(torch.stack(outputs), torch.stack(lses), rank),
        torch.cat(queries, dim=1),
    )


def _validate_lengths(rank: int, device: torch.device) -> None:
    # Includes the seven causal rows used while the target verifies a full
    # DSpark block across a round-robin DCP8 KV layout.
    global_lens = torch.tensor(
        [0, 1, 7, 8, 9, 63, 64, 97, 98, 99, 100, 101, 102, 103],
        dtype=torch.int32,
        device=device,
    )
    actual = torch.empty_like(global_lens)
    scratch = torch.empty_like(global_lens)
    _dcp_local_seq_lens_from_global(
        actual,
        scratch,
        global_lens,
        dcp_size=8,
        dcp_rank=rank,
        interleave=1,
    )
    expected = torch.tensor(
        [
            sum(1 for token in range(length) if token % 8 == rank)
            for length in global_lens.tolist()
        ],
        dtype=torch.int32,
        device=device,
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def _validate_geometry(
    geometry: Geometry,
    rank: int,
    device: torch.device,
    iterations: int,
) -> float:
    pool = PCIeDCPA2APool.from_process_group(
        process_group=dist.group.WORLD,
        device=device,
        max_batch_size=geometry.max_batch,
        total_heads=geometry.total_heads,
        head_dim=HEAD_DIM,
        query_head_dim=QUERY_HEAD_DIM,
        max_concurrent_channels=2,
    )
    try:
        eager_id = f"{geometry.name}:eager"
        graph_id = f"{geometry.name}:graph"
        pool.prepare_channels((eager_id, graph_id))
        for step, batch in enumerate((1, 7, 8), start=1):
            output, lse, query = _inputs(geometry, rank, batch, step, device)
            expected_output, expected_query = _reference(output, lse, query, rank)
            actual_query = pool.all_gather_heads(query, channel_id=eager_id)
            actual_output = pool.lse_reduce_scatter(output, lse, channel_id=eager_id)
            torch.cuda.synchronize(device)
            torch.testing.assert_close(actual_query, expected_query, rtol=0, atol=0)
            torch.testing.assert_close(
                actual_output.float(),
                expected_output.float(),
                rtol=3e-2,
                atol=3e-2,
            )

        output, lse, query = _inputs(geometry, rank, 1, 100, device)
        expected_output, expected_query = _reference(output, lse, query, rank)
        graph_output = torch.empty(
            1,
            geometry.total_heads // dist.get_world_size(),
            HEAD_DIM,
            dtype=torch.bfloat16,
            device=device,
        )
        graph_query = torch.empty(
            1,
            geometry.total_heads,
            QUERY_HEAD_DIM,
            dtype=torch.bfloat16,
            device=device,
        )
        stream = torch.cuda.Stream(device=device)
        channel = pool.for_stream(stream, channel_id=graph_id)
        with torch.cuda.stream(stream):
            channel.all_gather_heads(query, graph_query)
            channel.lse_reduce_scatter(output, lse, graph_output)
        stream.synchronize()
        dist.barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            channel.all_gather_heads(query, graph_query)
            channel.lse_reduce_scatter(output, lse, graph_output)
        stream.synchronize()
        graph.replay()
        stream.synchronize()
        torch.testing.assert_close(graph_query, expected_query, rtol=0, atol=0)
        torch.testing.assert_close(
            graph_output.float(), expected_output.float(), rtol=3e-2, atol=3e-2
        )

        for _ in range(20):
            graph.replay()
        stream.synchronize()
        dist.barrier()
        started = time.perf_counter()
        for _ in range(iterations):
            graph.replay()
        stream.synchronize()
        elapsed_us = (time.perf_counter() - started) * 1e6 / iterations
        latency = torch.tensor(elapsed_us, dtype=torch.float64, device=device)
        dist.all_reduce(latency, op=dist.ReduceOp.MAX)
        del graph
        return float(latency.item())
    finally:
        pool.close()


def _validate_vllm_channel_lifecycle(
    rank: int,
    device: torch.device,
) -> None:
    """Reproduce target prewarm -> graph stream -> eager stream."""
    group = SimpleNamespace(world_size=8, device_group=dist.group.WORLD)
    dcp_alltoall._B12X_DCP_A2A_POOLS.clear()
    dcp_alltoall._B12X_DCP_A2A_DISABLED.clear()
    dcp_alltoall._B12X_DCP_CAPTURE_SEQUENCES.clear()

    try:
        # Production must create the target pool before entering graph_capture.
        # Otherwise the first lazy channel becomes owned by the capture stream
        # and cannot be reused by subsequent eager execution. The external
        # DSpark draft intentionally stays DCP1 and needs no collective pool.
        for geometry in GEOMETRIES:
            dcp_alltoall.warmup_b12x_dcp_a2a(
                group,
                device=device,
                dtype=torch.bfloat16,
                max_batch_size=8,
                total_heads=geometry.total_heads,
                head_dim=HEAD_DIM,
                query_head_dim=QUERY_HEAD_DIM,
            )
        assert len(dcp_alltoall._B12X_DCP_A2A_POOLS) == len(GEOMETRIES)

        graph_inputs = []
        expected = []
        for geometry in GEOMETRIES:
            output, lse, query = _inputs(geometry, rank, 1, 300, device)
            graph_inputs.append((geometry, output, lse, query))
            expected.append(_reference(output, lse, query, rank))

        graph = torch.cuda.CUDAGraph()
        graph_results = []
        stream = torch.cuda.Stream(device=device)
        with (
            dcp_alltoall.capture_b12x_dcp_a2a(group, stream),
            torch.cuda.graph(graph, stream=stream),
        ):
            for geometry, output, lse, query in graph_inputs:
                gathered = dcp_alltoall.dcp_b12x_all_gather_heads(
                    query,
                    group,
                    max_batch_size=8,
                    output_head_dim=HEAD_DIM,
                )
                reduced = dcp_alltoall.dcp_a2a_lse_reduce(
                    output,
                    lse,
                    group,
                    use_b12x=True,
                    b12x_max_batch_size=8,
                    b12x_query_head_dim=QUERY_HEAD_DIM,
                )
                graph_results.append((gathered, reduced))
        graph.replay()
        stream.synchronize()
        for (actual_query, actual_output), (expected_output, expected_query) in zip(
            graph_results, expected, strict=True
        ):
            torch.testing.assert_close(actual_query, expected_query, rtol=0, atol=0)
            torch.testing.assert_close(
                actual_output.float(),
                expected_output.float(),
                rtol=3e-2,
                atol=3e-2,
            )

        # This is the transition that failed during the first full-model E2E.
        # It must return to each pool's original eager channel and stream.
        for geometry in GEOMETRIES:
            output, lse, query = _inputs(geometry, rank, 1, 400, device)
            expected_output, expected_query = _reference(output, lse, query, rank)
            actual_query = dcp_alltoall.dcp_b12x_all_gather_heads(
                query,
                group,
                max_batch_size=8,
                output_head_dim=HEAD_DIM,
            )
            actual_output = dcp_alltoall.dcp_a2a_lse_reduce(
                output,
                lse,
                group,
                use_b12x=True,
                b12x_max_batch_size=8,
                b12x_query_head_dim=QUERY_HEAD_DIM,
            )
            torch.cuda.synchronize(device)
            torch.testing.assert_close(actual_query, expected_query, rtol=0, atol=0)
            torch.testing.assert_close(
                actual_output.float(),
                expected_output.float(),
                rtol=3e-2,
                atol=3e-2,
            )
        dist.barrier()
        if rank == 0:
            print(
                "PASS vLLM lifecycle: prewarmed target DCP8 pool, "
                "graph-stream replay, post-capture eager reuse",
                flush=True,
            )
    finally:
        for pool in dcp_alltoall._B12X_DCP_A2A_POOLS.values():
            pool.close()
        dcp_alltoall._B12X_DCP_A2A_POOLS.clear()


def _worker(rank: int, world_size: int, port: int, iterations: int) -> None:
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
    )
    try:
        _validate_lengths(rank, device)
        for geometry in GEOMETRIES:
            combined_us = _validate_geometry(geometry, rank, device, iterations)
            dist.barrier()
            if rank == 0:
                print(
                    f"PASS {geometry.name}: DCP8, total_heads={geometry.total_heads}, "
                    f"local_heads={geometry.total_heads // world_size}, "
                    f"batches=1/7/8, graph_combined={combined_us:.2f} us",
                    flush=True,
                )
        _validate_vllm_channel_lifecycle(rank, device)
    finally:
        dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=500)
    args = parser.parse_args()
    if torch.cuda.device_count() < 8:
        raise SystemExit(f"DCP8 requires eight GPUs, found {torch.cuda.device_count()}")
    _load_extension()
    mp.spawn(_worker, args=(8, _free_port(), args.iterations), nprocs=8, join=True)


if __name__ == "__main__":
    main()
