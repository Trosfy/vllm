# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

import vllm.models.kimi_k3.nvidia.model as kimi_model
from vllm.models.kimi_k3.nvidia.model import (
    KimiShardedGate,
    _shard_routed_down_projection,
    _shard_router,
    _uses_native_b12x_mxfp4_intermediate_size,
)


@pytest.mark.parametrize(
    ("quantization", "backend", "use_b12x_env", "expected"),
    [
        ("mxfp4", "auto", "1", True),
        ("mxfp4", "b12x", "0", True),
        ("mxfp4", "auto", "0", False),
        ("mxfp4", "triton", "1", False),
        (None, "b12x", "1", False),
    ],
)
def test_native_b12x_mxfp4_intermediate_selection(
    monkeypatch,
    quantization: str | None,
    backend: str,
    use_b12x_env: str,
    expected: bool,
) -> None:
    monkeypatch.setenv("VLLM_USE_B12X_MOE", use_b12x_env)
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(quantization=quantization),
        kernel_config=SimpleNamespace(moe_backend=backend),
    )

    assert _uses_native_b12x_mxfp4_intermediate_size(vllm_config) is expected


@pytest.mark.parametrize(
    ("tp_size", "enabled", "expected"),
    [(16, "1", True), (16, "0", False), (1, "1", False)],
)
def test_shard_routed_down_projection_selection(
    monkeypatch,
    tp_size: int,
    enabled: str,
    expected: bool,
) -> None:
    monkeypatch.setenv("VLLM_KIMI_SHARD_ROUTED_DOWN_PROJ", enabled)

    assert _shard_routed_down_projection(tp_size) is expected


@pytest.mark.parametrize(
    ("tp_size", "enabled", "expected"),
    [(16, "1", True), (16, "0", False), (1, "1", False)],
)
def test_shard_router_selection(
    monkeypatch,
    tp_size: int,
    enabled: str,
    expected: bool,
) -> None:
    monkeypatch.setenv("VLLM_KIMI_SHARD_ROUTER", enabled)

    assert _shard_router(tp_size) is expected


def test_sharded_router_gathers_rank_ordered_fp32_logits(monkeypatch) -> None:
    gate = object.__new__(KimiShardedGate)
    torch.nn.Module.__init__(gate)
    gate.weight = torch.nn.Parameter(
        torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.bfloat16),
        requires_grad=False,
    )
    gate.tp_size = 2
    x = torch.tensor([[2.0, 1.0]], dtype=torch.bfloat16)
    expected_local = torch.nn.functional.linear(x, gate.weight).float()
    monkeypatch.setattr(
        kimi_model,
        "tensor_model_parallel_all_gather",
        lambda local: torch.cat((local, local + 10), dim=-1),
    )

    output, bias = gate(x)

    torch.testing.assert_close(
        output, torch.cat((expected_local, expected_local + 10), dim=-1)
    )
    assert output.dtype == torch.float32
    assert bias is None
