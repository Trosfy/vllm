# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

import vllm.models.kimi_k3.nvidia.tp_projection as tp_projection
from vllm.model_executor.layers.fused_moe.runner import latent_moe_runner
from vllm.model_executor.layers.fused_moe.runner.latent_moe_runner import (
    _allreduce_norm_latent_in_place,
    _project_sharded_up_and_reduce,
)
from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.models.kimi_k3.nvidia.mla import _restore_merged_output_order
from vllm.models.kimi_k3.nvidia.model import (
    KimiK3PrecomputedTopKRouter,
    _pack_aux_hidden_states_into_attn_res_workspace,
)
from vllm.models.kimi_k3.nvidia.tp_projection import (
    gather_kimi_sharded_projection,
    gather_kimi_sharded_projection_pair,
    should_reuse_kimi_full_width_output,
    try_gather_kimi_sharded_projection_pair_topk,
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
        "vllm.models.kimi_k3.nvidia.tp_projection._KIMI_OUTPUT_BUFFER_REUSE_MIN_TOKENS",
        4,
    )
    assert not should_reuse_kimi_full_width_output(torch.empty(3, 8))
    assert should_reuse_kimi_full_width_output(torch.empty(4, 8))


def _enable_fake_b12x_projection_gather(monkeypatch, gather) -> None:
    monkeypatch.setattr(
        tp_projection.envs, "VLLM_KIMI_USE_B12X_PROJECTION_GATHER", True
    )
    monkeypatch.setattr(
        tp_projection, "get_tensor_model_parallel_world_size", lambda: 4
    )
    monkeypatch.setattr(
        tp_projection, "get_dcp_group", lambda: SimpleNamespace(world_size=4)
    )
    monkeypatch.setattr(tp_projection, "dcp_b12x_all_gather_heads", gather)


def test_b12x_projection_gather_preserves_bf16_rank_order(monkeypatch) -> None:
    local = torch.arange(8, dtype=torch.bfloat16).view(1, 8)

    def fake_gather(value, _group, *, max_batch_size):
        assert value.shape == (1, 1, 8)
        assert max_batch_size == 1
        return torch.cat(tuple(value + rank for rank in range(4)), dim=1)

    _enable_fake_b12x_projection_gather(monkeypatch, fake_gather)
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda _self: True))

    actual = gather_kimi_sharded_projection(local)
    expected = torch.cat(tuple(local + rank for rank in range(4)), dim=-1)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_b12x_projection_gather_transports_unaligned_bf16_bits(monkeypatch) -> None:
    local = torch.arange(132, dtype=torch.bfloat16).view(1, 132)
    rank_values = torch.stack(tuple(local + rank for rank in range(4)), dim=1)
    padded_rank_values = torch.nn.functional.pad(rank_values, (0, 4))

    def fake_gather(value, _group, *, max_batch_size):
        assert value.dtype == torch.bfloat16
        assert value.shape == (1, 1, 136)
        assert max_batch_size == 1
        return padded_rank_values

    _enable_fake_b12x_projection_gather(monkeypatch, fake_gather)
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda _self: True))

    actual = gather_kimi_sharded_projection(local)
    expected = rank_values.flatten(1)
    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))


def test_b12x_projection_gather_transports_fp32_bits_exactly(monkeypatch) -> None:
    local = torch.tensor(
        [[0.0, -0.0, float("inf"), -3.125, 1.0 / 3.0, float("nan")]],
        dtype=torch.float32,
    )
    rank_values = torch.stack(tuple(local + rank for rank in range(4)), dim=1)

    def fake_gather(value, _group, *, max_batch_size):
        assert value.dtype == torch.float8_e4m3fn
        assert value.shape == (1, 1, local.shape[1] * 4)
        assert max_batch_size == 1
        return rank_values.view(torch.float8_e4m3fn)

    _enable_fake_b12x_projection_gather(monkeypatch, fake_gather)
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda _self: True))

    actual = gather_kimi_sharded_projection(local)
    expected = rank_values.flatten(1)
    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))


@pytest.mark.parametrize("batch", [1, 8])
def test_b12x_projection_pair_preserves_separate_rank_order(
    monkeypatch,
    batch: int,
) -> None:
    local_first = torch.arange(batch * 8, dtype=torch.bfloat16).view(batch, 8)
    local_second = torch.arange(batch * 4, dtype=torch.float32).view(batch, 4)

    _enable_fake_b12x_projection_gather(monkeypatch, lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        tp_projection.envs,
        "VLLM_KIMI_USE_B12X_PAIRED_PROJECTION_GATHER",
        True,
    )
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda _self: True))

    def fake_pair(first, second, _group, *, max_batch_size):
        assert first is local_first
        assert second is local_second
        assert max_batch_size == 8
        return (
            torch.cat(tuple(first + rank for rank in range(4)), dim=-1),
            torch.cat(tuple(second + rank for rank in range(4)), dim=-1),
        )

    monkeypatch.setattr(tp_projection, "dcp_b12x_all_gather_pair", fake_pair)

    actual_first, actual_second = gather_kimi_sharded_projection_pair(
        local_first, local_second
    )
    expected_first = torch.cat(tuple(local_first + rank for rank in range(4)), dim=-1)
    expected_second = torch.cat(tuple(local_second + rank for rank in range(4)), dim=-1)
    torch.testing.assert_close(actual_first, expected_first, rtol=0, atol=0)
    torch.testing.assert_close(actual_second, expected_second, rtol=0, atol=0)


def test_b12x_projection_pair_topk_returns_explicit_compact_payload(
    monkeypatch,
) -> None:
    local_down = torch.arange(224, dtype=torch.bfloat16).view(1, 224)
    local_router = torch.arange(56, dtype=torch.float32).view(1, 56)
    correction_bias = torch.zeros(896, dtype=torch.float32)
    expected_down = torch.arange(3584, dtype=torch.bfloat16).view(1, 3584)
    expected_payload = torch.arange(32, dtype=torch.float32).view(1, 32)

    for name in (
        "VLLM_KIMI_USE_B12X_PROJECTION_GATHER",
        "VLLM_KIMI_USE_B12X_PAIRED_PROJECTION_GATHER",
        "VLLM_KIMI_USE_B12X_PAIRED_PROJECTION_TOPK",
    ):
        monkeypatch.setattr(tp_projection.envs, name, True)
    monkeypatch.setattr(
        tp_projection, "get_tensor_model_parallel_world_size", lambda: 16
    )
    monkeypatch.setattr(
        tp_projection, "get_dcp_group", lambda: SimpleNamespace(world_size=16)
    )

    def fake_fused(down, router, bias, _group, *, max_batch_size):
        assert down is local_down
        assert router is local_router
        assert bias is correction_bias
        assert max_batch_size == 1
        return expected_down, expected_payload

    monkeypatch.setattr(
        tp_projection, "try_dcp_b12x_all_gather_pair_kimi_topk", fake_fused
    )

    actual = try_gather_kimi_sharded_projection_pair_topk(
        local_down, local_router, correction_bias
    )

    assert actual is not None
    assert actual[0] is expected_down
    assert actual[1] is expected_payload


def test_kimi_precomputed_router_decodes_payload_without_reselection() -> None:
    bias = torch.nn.Parameter(torch.zeros(896, dtype=torch.float32))
    router = KimiK3PrecomputedTopKRouter(
        top_k=16,
        global_num_experts=896,
        e_score_correction_bias=bias,
        scoring_func="sigmoid",
    )
    payload = torch.empty((1, 32), dtype=torch.float32)
    payload[:, :16] = torch.linspace(0.01, 0.16, 16)
    expected_ids = torch.arange(16, dtype=torch.int32).view(1, 16)
    payload[:, 16:].view(torch.int32).copy_(expected_ids)

    weights, ids = router._compute_routing(torch.empty(1, 3584), payload, None)

    assert weights.data_ptr() == payload.data_ptr()
    assert torch.equal(weights, payload[:, :16])
    assert torch.equal(ids, expected_ids)


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
