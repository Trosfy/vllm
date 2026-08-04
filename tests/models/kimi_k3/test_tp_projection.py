# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.fused_moe.runner import latent_moe_runner
from vllm.model_executor.layers.fused_moe.runner.latent_moe_runner import (
    _allreduce_norm_latent_in_place,
    _project_sharded_up_and_reduce,
)
from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.models.kimi_k3.nvidia.mla import _restore_merged_output_order
from vllm.models.kimi_k3.nvidia.model import (
    _pack_aux_hidden_states_into_attn_res_workspace,
)
from vllm.models.kimi_k3.nvidia.tp_projection import (
    should_reuse_kimi_full_width_output,
)


def test_unquantized_projection_reuses_caller_output_buffer() -> None:
    torch.manual_seed(7)
    x = torch.randn(8, 16)
    weight = torch.randn(12, 16)
    expected = torch.nn.functional.linear(x, weight)
    output_buffer = torch.empty(8, 12)
    output_ptr = output_buffer.data_ptr()

    actual = UnquantizedLinearMethod().apply_with_output_buffer(
        SimpleNamespace(weight=weight),
        x,
        output_buffer,
    )

    assert actual.data_ptr() == output_ptr
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_large_kimi_output_buffer_is_reusable(monkeypatch) -> None:
    monkeypatch.setattr(
        "vllm.models.kimi_k3.nvidia.tp_projection."
        "_KIMI_OUTPUT_BUFFER_REUSE_MIN_TOKENS",
        4,
    )
    assert not should_reuse_kimi_full_width_output(torch.empty(3, 8))
    assert should_reuse_kimi_full_width_output(torch.empty(4, 8))


def test_aux_hidden_states_reuse_dead_block_major_workspace(monkeypatch) -> None:
    monkeypatch.setattr(
        "vllm.models.kimi_k3.nvidia.model._AUX_ATTN_RES_PACK_MIN_TOKENS",
        1,
    )
    tokens, blocks, hidden_size, num_aux = 5, 8, 7, 5
    storage = torch.empty(blocks, tokens, hidden_size)
    workspace = storage.permute(1, 0, 2)
    aux = [torch.full((tokens, hidden_size), float(index)) for index in range(num_aux)]
    expected = torch.cat(aux, dim=-1)

    packed_list = _pack_aux_hidden_states_into_attn_res_workspace(aux, workspace)

    assert len(packed_list) == 1
    packed = packed_list[0]
    assert packed.is_contiguous()
    assert packed.untyped_storage().data_ptr() == storage.untyped_storage().data_ptr()
    torch.testing.assert_close(packed, expected)


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


def test_project_sharded_up_reuses_full_width_input_buffer(monkeypatch) -> None:
    fused_latent = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    shared_partial = torch.tensor([[5.0, 6.0]])
    reusable_input = torch.full_like(shared_partial, -99.0)

    class FakeRowParallelProjection(torch.nn.Module):
        input_is_parallel = False
        input_size_per_partition = 2
        tp_rank = 1
        bias = None

        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(
                torch.tensor([[2.0, 0.0], [0.0, 3.0]]),
                requires_grad=False,
            )

    reduced_inputs: list[torch.Tensor] = []

    def fake_all_reduce_in_place(x: torch.Tensor) -> torch.Tensor:
        reduced_inputs.append(x.clone())
        return x + 10.0

    monkeypatch.setattr(
        latent_moe_runner,
        "_K3_PREFILL_STORAGE_REUSE_MIN_TOKENS",
        1,
    )
    monkeypatch.setattr(
        latent_moe_runner,
        "tensor_model_parallel_all_reduce_in_place",
        fake_all_reduce_in_place,
    )
    actual = _project_sharded_up_and_reduce(
        fused_latent,
        shared_partial,
        FakeRowParallelProjection(),
        output_buffer=reusable_input,
    )

    # Rank one consumes latent columns [3, 4], writes [6, 12] directly into
    # the reusable full-width input, then adds the shared [5, 6] partial.
    expected_partial = torch.tensor([[11.0, 18.0]])
    torch.testing.assert_close(reusable_input, expected_partial)
    torch.testing.assert_close(reduced_inputs[0], expected_partial)
    torch.testing.assert_close(actual, expected_partial + 10.0)


def test_project_sharded_up_reuses_latent_when_shared_aliases_input(
    monkeypatch,
) -> None:
    fused_latent = torch.arange(16, dtype=torch.float32).view(4, 4)
    shared_and_output = torch.arange(32, dtype=torch.float32).view(4, 8)
    shared_reference = shared_and_output.clone()

    class FakeRowParallelProjection(torch.nn.Module):
        input_is_parallel = False
        input_size_per_partition = 2
        tp_rank = 1
        bias = None

        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(
                torch.arange(16, dtype=torch.float32).view(8, 2),
                requires_grad=False,
            )

    projection = FakeRowParallelProjection()
    expected = shared_reference + fused_latent[:, 2:] @ projection.weight.T
    reduced_inputs: list[torch.Tensor] = []

    def fake_all_reduce_in_place(x: torch.Tensor) -> torch.Tensor:
        reduced_inputs.append(x.clone())
        return x

    monkeypatch.setattr(
        latent_moe_runner,
        "_K3_PREFILL_STORAGE_REUSE_MIN_TOKENS",
        1,
    )
    monkeypatch.setattr(
        latent_moe_runner,
        "tensor_model_parallel_all_reduce_in_place",
        fake_all_reduce_in_place,
    )
    actual = _project_sharded_up_and_reduce(
        fused_latent,
        shared_and_output,
        projection,
        output_buffer=shared_and_output,
    )

    # The 16-element latent allocation holds only two 8-wide output rows, so
    # this also exercises the bounded two-chunk projection path.
    torch.testing.assert_close(shared_and_output, expected)
    torch.testing.assert_close(reduced_inputs[0], expected)
    torch.testing.assert_close(actual, expected)


def test_allreduce_norm_latent_reuses_dead_input(monkeypatch) -> None:
    latent = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    latent_ptr = latent.data_ptr()

    def fake_all_reduce_in_place(x: torch.Tensor) -> torch.Tensor:
        x.add_(10.0)
        return x

    def fake_rms_norm_in_place(x: torch.Tensor, _norm: object) -> torch.Tensor:
        x.mul_(2.0)
        return x

    monkeypatch.setattr(
        latent_moe_runner,
        "tensor_model_parallel_all_reduce_in_place",
        fake_all_reduce_in_place,
    )
    monkeypatch.setattr(
        latent_moe_runner,
        "_rms_norm_in_place",
        fake_rms_norm_in_place,
    )
    actual = _allreduce_norm_latent_in_place(latent, object())

    assert actual.data_ptr() == latent_ptr
    torch.testing.assert_close(actual, torch.tensor([[22.0, 24.0], [26.0, 28.0]]))
