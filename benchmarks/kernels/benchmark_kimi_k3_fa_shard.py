# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import json
import os
import statistics
from collections.abc import Callable

import torch
import torch.distributed as dist


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tokens", type=int, nargs="+", default=[1, 16, 256, 2048, 4096]
    )
    parser.add_argument("--hidden-size", type=int, default=7168)
    parser.add_argument("--num-heads", type=int, default=96)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=69)
    parser.add_argument("--graph-repeats", type=int, default=16)
    parser.add_argument("--eager-repeats", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--samples", type=int, default=11)
    return parser.parse_args()


def max_rank_time(
    op: Callable[[], None],
    repeats: int,
    warmup: int,
    samples: int,
    device_group: dist.ProcessGroup,
    cpu_group: dist.ProcessGroup,
) -> float:
    for _ in range(warmup):
        op()
    torch.cuda.synchronize()

    timings = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(samples):
        dist.barrier(group=cpu_group)
        start.record()
        for _ in range(repeats):
            op()
        end.record()
        end.synchronize()
        elapsed = torch.tensor(
            start.elapsed_time(end) / repeats,
            dtype=torch.float64,
            device=torch.cuda.current_device(),
        )
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX, group=device_group)
        timings.append(elapsed.item())
    return statistics.median(timings)


def capture_graph(
    op: Callable[[], None],
    repeats: int,
    cpu_group: dist.ProcessGroup,
) -> torch.cuda.CUDAGraph:
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            op()
    stream.synchronize()
    dist.barrier(group=cpu_group)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for _ in range(repeats):
            op()
    torch.cuda.current_stream().wait_stream(stream)
    return graph


def max_rank_graph_time(
    graph: torch.cuda.CUDAGraph,
    repeats: int,
    warmup: int,
    samples: int,
    device_group: dist.ProcessGroup,
    cpu_group: dist.ProcessGroup,
) -> float:
    for _ in range(warmup):
        graph.replay()
    torch.cuda.synchronize()

    timings = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(samples):
        dist.barrier(group=cpu_group)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        elapsed = torch.tensor(
            start.elapsed_time(end) / repeats,
            dtype=torch.float64,
            device=torch.cuda.current_device(),
        )
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX, group=device_group)
        timings.append(elapsed.item())
    return statistics.median(timings)


def benchmark_shape(
    tokens: int,
    hidden_size: int,
    num_heads: int,
    head_dim: int,
    num_layers: int,
    graph_repeats: int,
    eager_repeats: int,
    warmup: int,
    samples: int,
    device_group: dist.ProcessGroup,
    cpu_group: dist.ProcessGroup,
) -> dict[str, float | int]:
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    assert num_heads % world_size == 0
    assert head_dim % world_size == 0

    projection_size = num_heads * head_dim
    local_projection_size = projection_size // world_size
    local_num_heads = num_heads // world_size
    local_fa_size = head_dim // world_size
    fa_offset = 4 * local_projection_size

    replicated_unpadded = fa_offset + head_dim + local_num_heads
    replicated_padding = -replicated_unpadded % 16
    replicated_output_size = replicated_unpadded + replicated_padding

    sharded_unpadded = fa_offset + local_fa_size + local_num_heads
    sharded_padding = -sharded_unpadded % 16
    sharded_output_size = sharded_unpadded + sharded_padding

    device = torch.device("cuda", torch.cuda.current_device())
    common_generator = torch.Generator(device=device).manual_seed(17)
    rank_generator = torch.Generator(device=device).manual_seed(1000 + rank)

    hidden_states = torch.randn(
        (tokens, hidden_size),
        dtype=torch.bfloat16,
        device=device,
        generator=common_generator,
    )
    fa_weight = torch.randn(
        (head_dim, hidden_size),
        dtype=torch.bfloat16,
        device=device,
        generator=common_generator,
    )
    fb_weight = torch.randn(
        (local_projection_size, head_dim),
        dtype=torch.bfloat16,
        device=device,
        generator=rank_generator,
    )

    replicated_weight = torch.randn(
        (replicated_output_size, hidden_size),
        dtype=torch.bfloat16,
        device=device,
        generator=rank_generator,
    )
    replicated_weight[fa_offset : fa_offset + head_dim].copy_(fa_weight)
    if replicated_padding:
        replicated_weight[-replicated_padding:].zero_()

    sharded_weight = torch.randn(
        (sharded_output_size, hidden_size),
        dtype=torch.bfloat16,
        device=device,
        generator=rank_generator,
    )
    sharded_weight[:fa_offset].copy_(replicated_weight[:fa_offset])
    fa_start = rank * local_fa_size
    sharded_weight[fa_offset : fa_offset + local_fa_size].copy_(
        fa_weight[fa_start : fa_start + local_fa_size]
    )
    replicated_beta_start = fa_offset + head_dim
    sharded_beta_start = fa_offset + local_fa_size
    sharded_weight[
        sharded_beta_start : sharded_beta_start + local_num_heads
    ].copy_(
        replicated_weight[
            replicated_beta_start : replicated_beta_start + local_num_heads
        ]
    )
    if sharded_padding:
        sharded_weight[-sharded_padding:].zero_()

    replicated_output = torch.empty(
        (tokens, replicated_output_size), dtype=torch.bfloat16, device=device
    )
    sharded_output = torch.empty(
        (tokens, sharded_output_size), dtype=torch.bfloat16, device=device
    )
    fa_send = torch.empty(
        (tokens, local_fa_size), dtype=torch.bfloat16, device=device
    )
    fa_rank_major = torch.empty(
        (world_size * tokens, local_fa_size),
        dtype=torch.bfloat16,
        device=device,
    )
    gathered_fa = torch.empty(
        (tokens, head_dim), dtype=torch.bfloat16, device=device
    )
    replicated_g1 = torch.empty(
        (tokens, local_projection_size), dtype=torch.bfloat16, device=device
    )
    sharded_g1 = torch.empty_like(replicated_g1)

    def replicated_projection() -> None:
        torch.mm(hidden_states, replicated_weight.t(), out=replicated_output)

    def sharded_projection() -> None:
        torch.mm(hidden_states, sharded_weight.t(), out=sharded_output)

    def gather_fa() -> None:
        fa_send.copy_(
            sharded_output[:, fa_offset : fa_offset + local_fa_size]
        )
        dist.all_gather_into_tensor(
            fa_rank_major, fa_send, group=device_group
        )
        rank_major = fa_rank_major.view(world_size, tokens, local_fa_size)
        gathered_fa.view(tokens, world_size, local_fa_size).copy_(
            rank_major.permute(1, 0, 2)
        )

    def replicated_end_to_end() -> None:
        replicated_projection()
        torch.mm(
            replicated_output[:, fa_offset : fa_offset + head_dim],
            fb_weight.t(),
            out=replicated_g1,
        )

    def sharded_end_to_end() -> None:
        sharded_projection()
        gather_fa()
        torch.mm(gathered_fa, fb_weight.t(), out=sharded_g1)

    replicated_end_to_end()
    sharded_end_to_end()
    torch.cuda.synchronize()
    max_abs_error = (replicated_g1 - sharded_g1).abs().max().float()
    dist.all_reduce(max_abs_error, op=dist.ReduceOp.MAX, group=device_group)

    eager_ops = {
        "replicated_projection_eager_ms": replicated_projection,
        "sharded_projection_eager_ms": sharded_projection,
        "gather_layout_eager_ms": gather_fa,
        "replicated_e2e_eager_ms": replicated_end_to_end,
        "sharded_e2e_eager_ms": sharded_end_to_end,
    }
    eager_times = {
        name: max_rank_time(
            op,
            eager_repeats,
            warmup,
            samples,
            device_group,
            cpu_group,
        )
        for name, op in eager_ops.items()
    }

    graphs = {
        "replicated_projection_graph_ms": capture_graph(
            replicated_projection, graph_repeats, cpu_group
        ),
        "sharded_projection_graph_ms": capture_graph(
            sharded_projection, graph_repeats, cpu_group
        ),
        "gather_layout_graph_ms": capture_graph(
            gather_fa, graph_repeats, cpu_group
        ),
        "replicated_e2e_graph_ms": capture_graph(
            replicated_end_to_end, graph_repeats, cpu_group
        ),
        "sharded_e2e_graph_ms": capture_graph(
            sharded_end_to_end, graph_repeats, cpu_group
        ),
    }
    graph_times = {
        name: max_rank_graph_time(
            graph,
            graph_repeats,
            warmup,
            samples,
            device_group,
            cpu_group,
        )
        for name, graph in graphs.items()
    }

    eager_delta = (
        eager_times["sharded_e2e_eager_ms"]
        - eager_times["replicated_e2e_eager_ms"]
    )
    graph_delta = (
        graph_times["sharded_e2e_graph_ms"]
        - graph_times["replicated_e2e_graph_ms"]
    )
    bytes_saved_per_layer = (
        replicated_weight.nbytes - sharded_weight.nbytes
    )
    return {
        "tokens": tokens,
        "replicated_output_size": replicated_output_size,
        "sharded_output_size": sharded_output_size,
        "local_fa_size": local_fa_size,
        "bytes_saved_per_layer": bytes_saved_per_layer,
        "bytes_saved_all_layers": bytes_saved_per_layer * num_layers,
        "max_abs_error": max_abs_error.item(),
        **eager_times,
        **graph_times,
        "eager_delta_ms_per_layer": eager_delta,
        "eager_delta_ms_all_layers": eager_delta * num_layers,
        "graph_delta_ms_per_layer": graph_delta,
        "graph_delta_ms_all_layers": graph_delta * num_layers,
    }


def main() -> None:
    args = parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    device_group = dist.group.WORLD
    cpu_group = dist.new_group(backend="gloo")

    with torch.inference_mode():
        results = [
            benchmark_shape(
                tokens,
                args.hidden_size,
                args.num_heads,
                args.head_dim,
                args.num_layers,
                args.graph_repeats,
                args.eager_repeats,
                args.warmup,
                args.samples,
                device_group,
                cpu_group,
            )
            for tokens in args.tokens
        ]

    if dist.get_rank() == 0:
        print(json.dumps(results, indent=2), flush=True)
    dist.destroy_process_group(cpu_group)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
