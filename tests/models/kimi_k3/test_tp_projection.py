# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.layers.fused_moe.runner import latent_moe_runner
from vllm.model_executor.layers.fused_moe.runner.latent_moe_runner import (
    _project_sharded_up_and_reduce,
)
from vllm.models.kimi_k3.nvidia.mla import _restore_merged_output_order


def test_restore_merged_output_order() -> None:
    tp_size = 4
    output_sizes = [8, 12]
    per_rank = []
    expected_q = []
    expected_kv = []
    for rank in range(tp_size):
        q = torch.full((2, output_sizes[0] // tp_size), rank + 1)
        kv = torch.full((2, output_sizes[1] // tp_size), 10 + rank)
        per_rank.append(torch.cat((q, kv), dim=-1))
        expected_q.append(q)
        expected_kv.append(kv)

    rank_major = torch.cat(per_rank, dim=-1)
    expected = torch.cat((*expected_q, *expected_kv), dim=-1)

    actual = _restore_merged_output_order(rank_major, output_sizes, tp_size)

    torch.testing.assert_close(actual, expected)


def test_restore_merged_output_order_rejects_invalid_width() -> None:
    with pytest.raises(ValueError, match="Unexpected gathered"):
        _restore_merged_output_order(torch.empty(2, 79), [32, 48], 4)


def test_project_sharded_up_reduces_after_shared_add(monkeypatch) -> None:
    fused_latent = torch.tensor([[1.0, 2.0]])
    shared_partial = torch.tensor([[3.0, 4.0]])
    weight = torch.tensor([[2.0, 0.0], [0.0, 3.0]])

    class FakeRowParallelProjection(torch.nn.Module):
        def forward(self, x: torch.Tensor):
            return x @ weight.T, None

    reduced_inputs: list[torch.Tensor] = []

    def fake_all_reduce(x: torch.Tensor) -> torch.Tensor:
        reduced_inputs.append(x.clone())
        return x + 10.0

    monkeypatch.setattr(
        latent_moe_runner,
        "tensor_model_parallel_all_reduce",
        fake_all_reduce,
    )
    actual = _project_sharded_up_and_reduce(
        fused_latent,
        shared_partial.clone(),
        FakeRowParallelProjection(),
    )

    expected_partial = torch.tensor([[5.0, 10.0]])
    torch.testing.assert_close(reduced_inputs[0], expected_partial)
    torch.testing.assert_close(actual, expected_partial + 10.0)
