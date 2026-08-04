# SPDX-License-Identifier: Apache-2.0
"""Measure the Kimi-K3 donated shared-expert prefill tail without a model.

This reproduces the exact rank-local TP16 dimensions responsible for the
physical-1M DCP8 prefill OOM.  The old path allocates a 28 MiB Marlin output;
the donated path writes it into the dead full-width MoE input and reuses the
14 MiB latent allocation for the sharded routed-up projection.
"""

from __future__ import annotations

import argparse
import gc
import json
from types import SimpleNamespace

import torch

from vllm.model_executor.layers.fused_moe.runner import latent_moe_runner
from vllm.model_executor.layers.fused_moe.runner.latent_moe_runner import (
    _project_sharded_up_and_reduce,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
    apply_mxfp8_marlin_linear,
    prepare_mxfp8_layer_for_marlin,
)


def _mib(value: int) -> float:
    return value / (1024**2)


def _synchronize() -> None:
    torch.cuda.synchronize()


def _make_mxfp8_down(k: int, n: int) -> tuple[torch.nn.Module, torch.Tensor]:
    dtype = torch.bfloat16
    layer = torch.nn.Module()
    layer.output_size_per_partition = n
    layer.input_size_per_partition = k
    weight = (torch.randn(n, k, device="cuda", dtype=dtype) / 16).to(
        torch.float8_e4m3fn
    )
    scales = torch.full(
        (n, k // 32),
        127,
        dtype=torch.uint8,
        device="cuda",
    )
    layer.weight = torch.nn.Parameter(weight, requires_grad=False)
    layer.weight_scale = torch.nn.Parameter(scales, requires_grad=False)
    prepare_mxfp8_layer_for_marlin(layer)
    activated = torch.randn(2048, k, device="cuda", dtype=dtype)
    return layer, activated


def _peak_delta_bytes(run) -> tuple[torch.Tensor, int]:
    gc.collect()
    torch.cuda.empty_cache()
    _synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    output = run()
    _synchronize()
    peak = torch.cuda.max_memory_allocated()
    return output, peak - baseline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=2048)
    args = parser.parse_args()

    torch.manual_seed(17)
    torch.cuda.set_device(0)
    rows = args.rows
    hidden = 7168
    latent = 3584
    local_latent = latent // 16
    shared_intermediate = (3072 * 2) // 16

    down, activated_full = _make_mxfp8_down(shared_intermediate, hidden)
    activated = activated_full[:rows]
    routed_weight = torch.nn.Parameter(
        torch.randn(hidden, local_latent, device="cuda", dtype=torch.bfloat16)
        / local_latent**0.5,
        requires_grad=False,
    )
    projection = SimpleNamespace(
        input_is_parallel=False,
        input_size_per_partition=local_latent,
        tp_rank=0,
        bias=None,
        weight=routed_weight,
    )
    # The production helper only accepts an unquantized vLLM method.  A tiny
    # local class gives the harness the same contract without constructing a
    # distributed RowParallelLinear.
    projection.quant_method = type("UnquantizedLinearMethod", (), {})()

    latent_template = torch.randn(
        rows,
        latent,
        device="cuda",
        dtype=torch.bfloat16,
    )

    def old_path() -> torch.Tensor:
        latent_input = latent_template.clone()
        full_width_input = torch.empty(
            rows, hidden, device="cuda", dtype=torch.bfloat16
        )
        shared = apply_mxfp8_marlin_linear(
            activated,
            down.weight,
            down.weight_scale,
            down.workspace,
            hidden,
            shared_intermediate,
        )
        local_input = latent_input[:, :local_latent].contiguous()
        torch.mm(local_input, routed_weight.T, out=full_width_input)
        full_width_input.add_(shared)
        return full_width_input

    old_output, old_peak = _peak_delta_bytes(old_path)
    reference = old_output.cpu()
    del old_output

    def donated_path() -> torch.Tensor:
        latent_input = latent_template.clone()
        full_width_input = torch.empty(
            rows, hidden, device="cuda", dtype=torch.bfloat16
        )
        shared = apply_mxfp8_marlin_linear(
            activated,
            down.weight,
            down.weight_scale,
            down.workspace,
            hidden,
            shared_intermediate,
            output_buffer=full_width_input,
        )
        old_all_reduce = latent_moe_runner.tensor_model_parallel_all_reduce_in_place
        latent_moe_runner.tensor_model_parallel_all_reduce_in_place = lambda x: x
        try:
            return _project_sharded_up_and_reduce(
                latent_input,
                shared,
                projection,
                output_buffer=full_width_input,
            )
        finally:
            latent_moe_runner.tensor_model_parallel_all_reduce_in_place = (
                old_all_reduce
            )

    donated_output, donated_peak = _peak_delta_bytes(donated_path)
    difference = donated_output.float().cpu() - reference.float()
    result = {
        "rows": rows,
        "old_peak_delta_mib": _mib(old_peak),
        "donated_peak_delta_mib": _mib(donated_peak),
        "saved_peak_mib": _mib(old_peak - donated_peak),
        "max_abs_difference": float(difference.abs().max()),
        "mean_abs_difference": float(difference.abs().mean()),
        "exact_fraction": float((difference == 0).float().mean()),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
