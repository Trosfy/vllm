#!/usr/bin/env python3

import argparse
import importlib
import json

import torch


def run_flashkda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    out: torch.Tensor,
    workspace: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    initial_state: torch.Tensor,
    final_state: torch.Tensor,
    cu_seqlens: torch.Tensor,
) -> None:
    torch.ops._flashkda_C.fwd(
        q,
        k,
        v,
        g,
        beta,
        128**-0.5,
        out,
        workspace,
        a_log,
        dt_bias,
        -5.0,
        initial_state,
        final_state,
        cu_seqlens,
    )


def snapshot(label: str) -> dict[str, int | str]:
    torch.cuda.synchronize()
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    reserved_bytes = torch.cuda.memory_reserved()
    return {
        "label": label,
        "device_used_bytes": total_bytes - free_bytes,
        "torch_allocated_bytes": torch.cuda.memory_allocated(),
        "torch_reserved_bytes": reserved_bytes,
        "non_torch_or_context_bytes": total_bytes - free_bytes - reserved_bytes,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--heads", type=int, default=6)
    parser.add_argument("--library")
    parser.add_argument("--random-inputs", action="store_true")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--save-output")
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda:0")
    stages: list[dict[str, int | str]] = []

    context_sentinel = torch.empty(1, dtype=torch.uint8, device=device)
    stages.append(snapshot("cuda_context"))

    if args.library:
        torch.ops.load_library(args.library)
    else:
        importlib.import_module("vllm._flashkda_C")
    stages.append(snapshot("extension_import"))

    shape = (1, args.tokens, args.heads, 128)
    generator = torch.Generator(device=device).manual_seed(args.seed)

    def input_tensor(tensor_shape: tuple[int, ...]) -> torch.Tensor:
        if args.random_inputs:
            return torch.randn(
                tensor_shape,
                dtype=torch.bfloat16,
                device=device,
                generator=generator,
            )
        return torch.zeros(tensor_shape, dtype=torch.bfloat16, device=device)

    q = input_tensor(shape)
    k = input_tensor(shape)
    v = input_tensor(shape)
    g = input_tensor(shape)
    beta = input_tensor((1, args.tokens, args.heads))
    out = torch.empty_like(q)
    initial_state = torch.zeros(
        (1, args.heads, 128, 128), dtype=torch.float32, device=device
    )
    final_state = torch.empty_like(initial_state)
    cu_seqlens = torch.tensor([0, args.tokens], dtype=torch.int32, device=device)
    a_log = torch.zeros(args.heads, dtype=torch.float32, device=device)
    dt_bias = torch.zeros((args.heads, 128), dtype=torch.float32, device=device)
    workspace_bytes = torch.ops._flashkda_C.get_workspace_size(
        args.tokens, args.heads, 1
    )
    workspace = torch.zeros(workspace_bytes, dtype=torch.uint8, device=device)
    stages.append(snapshot("inputs_and_workspace"))

    with torch.inference_mode():
        for label in ("first_launch", "second_launch"):
            run_flashkda(
                q,
                k,
                v,
                g,
                beta,
                out,
                workspace,
                a_log,
                dt_bias,
                initial_state,
                final_state,
                cu_seqlens,
            )
            stages.append(snapshot(label))

        for _ in range(args.warmup):
            run_flashkda(
                q,
                k,
                v,
                g,
                beta,
                out,
                workspace,
                a_log,
                dt_bias,
                initial_state,
                final_state,
                cu_seqlens,
            )
        elapsed_ms = None
        if args.iterations:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(args.iterations):
                run_flashkda(
                    q,
                    k,
                    v,
                    g,
                    beta,
                    out,
                    workspace,
                    a_log,
                    dt_bias,
                    initial_state,
                    final_state,
                    cu_seqlens,
                )
            end.record()
            end.synchronize()
            elapsed_ms = start.elapsed_time(end) / args.iterations

    if not args.skip_validation and (
        not torch.isfinite(out).all() or not torch.isfinite(final_state).all()
    ):
        raise RuntimeError("FlashKDA produced a non-finite output")

    fingerprints = {
        "out_sum": out.float().sum().item(),
        "out_norm": out.float().norm().item(),
        "final_state_sum": final_state.sum().item(),
        "final_state_norm": final_state.norm().item(),
    }
    if args.save_output:
        torch.save(
            {"out": out.cpu(), "final_state": final_state.cpu()},
            args.save_output,
        )

    del (
        context_sentinel,
        q,
        k,
        v,
        g,
        beta,
        out,
        initial_state,
        final_state,
        cu_seqlens,
        a_log,
        dt_bias,
        workspace,
    )
    torch.cuda.empty_cache()
    stages.append(snapshot("after_empty_cache"))

    baseline = int(stages[0]["device_used_bytes"])
    for stage in stages:
        stage["device_used_delta_bytes"] = int(stage["device_used_bytes"]) - baseline
    print(
        json.dumps(
            {
                "tokens": args.tokens,
                "heads": args.heads,
                "library": args.library,
                "workspace_bytes": workspace_bytes,
                "iterations": args.iterations,
                "mean_ms": elapsed_ms,
                "fingerprints": fingerprints,
                "stages": stages,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
