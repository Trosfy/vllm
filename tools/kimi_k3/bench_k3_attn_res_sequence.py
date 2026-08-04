#!/usr/bin/env python3
"""Compare Kimi-K3 AttnRes layouts across a complete 93-layer decode step.

This intentionally replaces attention and MoE with deterministic elementwise
deltas.  It still exercises every real AttnRes transition: block writes,
native fused reads, final pre-norm output, the five DSpark feature taps, and
CUDA-graph replay.  No checkpoint is loaded.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

import torch

from vllm.models.kimi_k3.nvidia.ops.attn_res import attn_res
from vllm.models.kimi_k3.nvidia.tp_projection import (
    should_reuse_kimi_full_width_output,
)


@dataclass
class SequenceWeights:
    attn_norm: torch.Tensor
    attn_qk: torch.Tensor
    attn_out_norm: torch.Tensor
    mlp_norm: torch.Tensor
    mlp_qk: torch.Tensor
    mlp_out_norm: torch.Tensor
    final_norm: torch.Tensor
    final_qk: torch.Tensor
    attn_scale: torch.Tensor
    mlp_scale: torch.Tensor


def make_weights(
    layers: int,
    hidden_size: int,
    device: torch.device,
) -> SequenceWeights:
    generator = torch.Generator(device=device).manual_seed(917)

    def matrix() -> torch.Tensor:
        return torch.randn(
            layers,
            hidden_size,
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        )

    return SequenceWeights(
        attn_norm=matrix(),
        attn_qk=matrix(),
        attn_out_norm=matrix(),
        mlp_norm=matrix(),
        mlp_qk=matrix(),
        mlp_out_norm=matrix(),
        final_norm=torch.randn(
            hidden_size,
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        ),
        final_qk=torch.randn(
            hidden_size,
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        ),
        attn_scale=torch.linspace(
            0.007,
            0.013,
            layers,
            dtype=torch.float32,
            device=device,
        ),
        mlp_scale=torch.linspace(
            0.011,
            0.005,
            layers,
            dtype=torch.float32,
            device=device,
        ),
    )


def make_workspace(
    layout: str,
    tokens: int,
    capacity_tokens: int,
    num_blocks: int,
    hidden_size: int,
    device: torch.device,
) -> torch.Tensor:
    if layout == "token-major":
        return torch.empty(
            tokens,
            num_blocks,
            hidden_size,
            dtype=torch.bfloat16,
            device=device,
        )
    if layout == "block-major":
        storage = torch.empty(
            num_blocks,
            capacity_tokens,
            hidden_size,
            dtype=torch.bfloat16,
            device=device,
        )
        return storage.permute(1, 0, 2)[:tokens]
    raise ValueError(f"unknown layout: {layout}")


def run_sequence(
    initial_hidden: torch.Tensor,
    workspace: torch.Tensor,
    weights: SequenceWeights,
    *,
    layers: int,
    block_size: int,
    aux_layers: tuple[int, ...],
    donate_buffers: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
    prefix = initial_hidden.clone()
    delta = None
    traces: list[torch.Tensor] = []
    auxiliary: list[torch.Tensor] = []

    for layer_idx in range(layers):
        is_block_write = layer_idx % block_size == 0
        block_write_idx = layer_idx // block_size
        previous_blocks = (layer_idx + block_size - 1) // block_size

        attn_output_buffer = None
        if donate_buffers:
            if delta is not None and should_reuse_kimi_full_width_output(delta):
                attn_output_buffer = delta
            elif (
                delta is None
                and is_block_write
                and block_write_idx != workspace.size(1) - 1
                and should_reuse_kimi_full_width_output(prefix)
            ):
                attn_output_buffer = workspace[:, -1, :]

        attn_input = attn_res(
            prefix,
            delta,
            workspace,
            weights.attn_norm[layer_idx],
            weights.attn_qk[layer_idx],
            weights.attn_out_norm[layer_idx],
            num_blocks=previous_blocks,
            block_write_idx=block_write_idx if is_block_write else -1,
            eps=1e-5,
            output_norm_eps=1e-5,
            output_buffer=attn_output_buffer,
        )
        attn_delta = (
            attn_input.float() * weights.attn_scale[layer_idx]
        ).to(torch.bfloat16)

        previous_prefix = prefix
        if is_block_write:
            prefix = attn_delta
            prefix_delta = None
        else:
            prefix_delta = attn_delta
        mlp_output_buffer = None
        if donate_buffers:
            candidate = previous_prefix if is_block_write else attn_delta
            if should_reuse_kimi_full_width_output(candidate):
                mlp_output_buffer = candidate
        valid_mlp_blocks = previous_blocks + int(is_block_write)
        mlp_input = attn_res(
            prefix,
            prefix_delta,
            workspace,
            weights.mlp_norm[layer_idx],
            weights.mlp_qk[layer_idx],
            weights.mlp_out_norm[layer_idx],
            num_blocks=valid_mlp_blocks,
            block_write_idx=-1,
            eps=1e-5,
            output_norm_eps=1e-5,
            output_buffer=mlp_output_buffer,
        )
        delta = (
            mlp_input.float() * weights.mlp_scale[layer_idx]
        ).to(torch.bfloat16)
        full_state = prefix + delta
        traces.append(full_state.clone())
        if layer_idx + 1 in aux_layers:
            auxiliary.append(full_state.clone())

    final_output_buffer = None
    if donate_buffers and should_reuse_kimi_full_width_output(delta):
        final_output_buffer = delta
    final = attn_res(
        prefix,
        delta,
        workspace,
        weights.final_norm,
        weights.final_qk,
        None,
        num_blocks=workspace.shape[1],
        block_write_idx=-1,
        eps=1e-5,
        output_norm_eps=0.0,
        output_buffer=final_output_buffer,
    )
    return final, torch.cat(auxiliary, dim=-1), traces


def compare(
    reference: torch.Tensor,
    candidate: torch.Tensor,
) -> dict[str, float | bool]:
    difference = reference.float() - candidate.float()
    return {
        "exact": torch.equal(reference, candidate),
        "max_abs": float(difference.abs().max()),
        "mean_abs": float(difference.abs().mean()),
    }


def capture_sequence(
    initial_hidden: torch.Tensor,
    workspace: torch.Tensor,
    weights: SequenceWeights,
    *,
    layers: int,
    block_size: int,
    aux_layers: tuple[int, ...],
) -> tuple[torch.cuda.CUDAGraph, torch.Tensor, torch.Tensor]:
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        final, auxiliary, _ = run_sequence(
            initial_hidden,
            workspace,
            weights,
            layers=layers,
            block_size=block_size,
            aux_layers=aux_layers,
        )
    return graph, final, auxiliary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=8)
    parser.add_argument("--capacity-tokens", type=int, default=2048)
    parser.add_argument("--hidden-size", type=int, default=7168)
    parser.add_argument("--layers", type=int, default=93)
    parser.add_argument("--block-size", type=int, default=12)
    parser.add_argument("--graph-replays", type=int, default=4)
    args = parser.parse_args()
    if args.capacity_tokens < args.tokens:
        raise ValueError("capacity-tokens must be at least tokens")

    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    generator = torch.Generator(device=device).manual_seed(191)
    initial_hidden = torch.randn(
        args.tokens,
        args.hidden_size,
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    weights = make_weights(args.layers, args.hidden_size, device)
    num_blocks = (args.layers + args.block_size - 1) // args.block_size
    aux_layers = (3, 24, 48, 72, 90)

    token_workspace = make_workspace(
        "token-major",
        args.tokens,
        args.capacity_tokens,
        num_blocks,
        args.hidden_size,
        device,
    )
    block_workspace = make_workspace(
        "block-major",
        args.tokens,
        args.capacity_tokens,
        num_blocks,
        args.hidden_size,
        device,
    )
    token_final, token_aux, token_trace = run_sequence(
        initial_hidden,
        token_workspace,
        weights,
        layers=args.layers,
        block_size=args.block_size,
        aux_layers=aux_layers,
    )
    block_final, block_aux, block_trace = run_sequence(
        initial_hidden,
        block_workspace,
        weights,
        layers=args.layers,
        block_size=args.block_size,
        aux_layers=aux_layers,
    )
    donated_workspace = make_workspace(
        "block-major",
        args.tokens,
        args.capacity_tokens,
        num_blocks,
        args.hidden_size,
        device,
    )
    donated_final, donated_aux, donated_trace = run_sequence(
        initial_hidden,
        donated_workspace,
        weights,
        layers=args.layers,
        block_size=args.block_size,
        aux_layers=aux_layers,
        donate_buffers=True,
    )
    first_trace_mismatch = next(
        (
            index + 1
            for index, (token_state, block_state) in enumerate(
                zip(token_trace, block_trace, strict=True)
            )
            if not torch.equal(token_state, block_state)
        ),
        None,
    )
    first_donated_trace_mismatch = next(
        (
            index + 1
            for index, (reference_state, donated_state) in enumerate(
                zip(block_trace, donated_trace, strict=True)
            )
            if not torch.equal(reference_state, donated_state)
        ),
        None,
    )

    # Warm all Triton/native specializations before capture.
    run_sequence(
        initial_hidden,
        token_workspace,
        weights,
        layers=args.layers,
        block_size=args.block_size,
        aux_layers=aux_layers,
    )
    run_sequence(
        initial_hidden,
        block_workspace,
        weights,
        layers=args.layers,
        block_size=args.block_size,
        aux_layers=aux_layers,
    )
    torch.cuda.synchronize()

    token_graph, token_graph_final, token_graph_aux = capture_sequence(
        initial_hidden,
        token_workspace,
        weights,
        layers=args.layers,
        block_size=args.block_size,
        aux_layers=aux_layers,
    )
    block_graph, block_graph_final, block_graph_aux = capture_sequence(
        initial_hidden,
        block_workspace,
        weights,
        layers=args.layers,
        block_size=args.block_size,
        aux_layers=aux_layers,
    )
    for _ in range(args.graph_replays):
        token_graph.replay()
        block_graph.replay()
    torch.cuda.synchronize()

    result = {
        "tokens": args.tokens,
        "capacity_tokens": args.capacity_tokens,
        "layers": args.layers,
        "num_blocks": num_blocks,
        "first_trace_mismatch_layer": first_trace_mismatch,
        "eager_final": compare(token_final, block_final),
        "eager_aux": compare(token_aux, block_aux),
        "donated_final": compare(block_final, donated_final),
        "donated_aux": compare(block_aux, donated_aux),
        "first_donated_trace_mismatch_layer": first_donated_trace_mismatch,
        "graph_final": compare(token_graph_final, block_graph_final),
        "graph_aux": compare(token_graph_aux, block_graph_aux),
        "block_strides": block_workspace.stride(),
        "token_strides": token_workspace.stride(),
    }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
