# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch

from vllm.compilation.b12x_capture import b12x_compile_only_warmup
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backends.mla import b12x_mla
from vllm.v1.attention.backends.mla.b12x_mla import (
    B12xMLABackend,
    B12xMLAImpl,
)
from vllm.v1.attention.backends.registry import AttentionBackendEnum


def test_b12x_mla_is_registered_with_k3_envelope() -> None:
    assert AttentionBackendEnum.B12X_MLA.get_class() is B12xMLABackend
    assert B12xMLABackend.get_name() == "B12X_MLA"
    assert B12xMLABackend.get_supported_head_sizes() == [576]
    assert B12xMLABackend.supports_block_size(944)
    assert not B12xMLABackend.supports_block_size(936)
    assert B12xMLABackend.supports_compute_capability(DeviceCapability(12, 0))
    assert not B12xMLABackend.supports_compute_capability(DeviceCapability(10, 0))


class _FakePlan:
    def shapes_and_dtypes(self):
        return (((256,), torch.uint8),)


class _FakeDenseMLA:
    def __init__(self) -> None:
        self.bindings: list[SimpleNamespace] = []
        self.compile_count = 0
        self.run_count = 0

    def bind(self, plan, **kwargs):
        binding = SimpleNamespace(plan=plan, **kwargs)
        self.bindings.append(binding)
        return binding

    def compile(self, *, binding) -> None:
        self.compile_count += 1

    def run(self, *, binding):
        self.run_count += 1
        lse = torch.zeros(
            binding.output.shape[:2], dtype=torch.float32, device=binding.output.device
        )
        return binding.output, lse


def _fake_impl(monkeypatch) -> tuple[B12xMLAImpl, _FakeDenseMLA]:
    impl = object.__new__(B12xMLAImpl)
    impl.num_heads = 8
    impl.kv_lora_rank = 512
    impl.scale = 192**-0.5
    impl.dcp_world_size = 1
    impl._dcp_comm_backend = "a2a"
    impl._dcp_max_batch_size = 64
    impl._compiled_bindings = set()
    impl._scratch_by_plan = {}
    dense_mla = _FakeDenseMLA()
    impl._dense_mla = dense_mla
    return impl, dense_mla


def test_b12x_mla_impl_keeps_configured_dcp_world_size(monkeypatch) -> None:
    def fake_common_init(
        self,
        num_heads,
        head_size,
        scale,
        num_kv_heads,
        alibi_slopes,
        sliding_window,
        kv_cache_dtype,
        logits_soft_cap,
        attn_type,
        kv_sharing_target_layer_name,
        **mla_args,
    ) -> None:
        self.num_heads = num_heads
        self.kv_lora_rank = mla_args["kv_lora_rank"]
        self.qk_nope_head_dim = mla_args["qk_nope_head_dim"]
        self.qk_rope_head_dim = mla_args["qk_rope_head_dim"]
        self.qk_head_dim = mla_args["qk_head_dim"]
        self.v_head_dim = mla_args["v_head_dim"]
        self.scale = scale
        self.dcp_world_size = -1

    monkeypatch.setattr(b12x_mla.MLACommonImpl, "__init__", fake_common_init)
    monkeypatch.setattr(b12x_mla, "_load_dense_mla", _FakeDenseMLA)
    monkeypatch.setattr(
        b12x_mla,
        "get_current_vllm_config",
        lambda: SimpleNamespace(
            parallel_config=SimpleNamespace(
                decode_context_parallel_size=8,
                prefill_context_parallel_size=1,
                dcp_comm_backend="a2a",
            ),
            scheduler_config=SimpleNamespace(max_num_batched_tokens=256),
        ),
    )

    impl = B12xMLAImpl(
        num_heads=6,
        head_size=576,
        scale=192**-0.5,
        num_kv_heads=1,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="fp8",
        logits_soft_cap=None,
        attn_type="decoder",
        kv_sharing_target_layer_name=None,
        q_lora_rank=2048,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        qk_head_dim=192,
        v_head_dim=128,
        kv_b_proj=None,
    )

    assert impl.dcp_world_size == 8


def test_b12x_mla_adapter_binds_common_decode_metadata(monkeypatch) -> None:
    impl, dense_mla = _fake_impl(monkeypatch)
    batch = 2
    q_nope = torch.randn(batch, 8, 512, dtype=torch.bfloat16)
    q_rope = torch.randn(batch, 8, 64, dtype=torch.bfloat16)
    cache = torch.randn(4, 16, 576, dtype=torch.bfloat16)
    metadata = SimpleNamespace(
        dense_mla_plan=_FakePlan(),
        query_start_loc=torch.tensor([0, 1, 2], dtype=torch.int32),
        decode=SimpleNamespace(
            block_table=torch.tensor([[0, 1], [2, 3]], dtype=torch.int32),
            seq_lens=torch.tensor([16, 32], dtype=torch.int32),
        ),
    )
    layer = SimpleNamespace(
        _q_scale=torch.tensor(0.25),
        _k_scale=torch.tensor(0.5),
    )

    output, lse = impl.forward_mqa((q_nope, q_rope), cache, metadata, layer)
    output_2, _ = impl.forward_mqa((q_nope, q_rope), cache, metadata, layer)

    assert output.shape == (batch, 8, 512)
    assert output.dtype == torch.bfloat16
    assert lse is not None and lse.dtype == torch.float32
    assert output_2.shape == output.shape
    assert dense_mla.compile_count == 1
    binding = dense_mla.bindings[0]
    assert binding.q.shape == (batch, 8, 576)
    assert binding.q.is_contiguous()
    assert binding.kv_cache is cache
    assert binding.page_table is metadata.decode.block_table
    assert binding.cache_seqlens is metadata.decode.seq_lens
    assert binding.q_scale is None
    assert binding.kv_scale is None
    assert binding.sm_scale == impl.scale


def test_b12x_mla_adapter_passes_fp8_scales(monkeypatch) -> None:
    impl, dense_mla = _fake_impl(monkeypatch)
    q = torch.empty(1, 8, 576, dtype=torch.float8_e4m3fn)
    cache = torch.empty(2, 16, 576, dtype=torch.float8_e4m3fn)
    metadata = SimpleNamespace(
        dense_mla_plan=_FakePlan(),
        query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
        decode=SimpleNamespace(
            block_table=torch.tensor([[0, 1]], dtype=torch.int32),
            seq_lens=torch.tensor([17], dtype=torch.int32),
        ),
    )
    layer = SimpleNamespace(
        _q_scale=torch.tensor(0.25),
        _k_scale=torch.tensor(0.5),
    )

    impl.forward_mqa(q, cache, metadata, layer)

    binding = dense_mla.bindings[0]
    assert binding.q_scale is layer._q_scale
    assert binding.kv_scale is layer._k_scale


def test_b12x_mla_adapter_uses_metadata_shared_scratch(monkeypatch) -> None:
    impl, dense_mla = _fake_impl(monkeypatch)
    shared_scratch = torch.empty(256, dtype=torch.uint8)
    q = torch.randn(1, 8, 576, dtype=torch.bfloat16)
    cache = torch.randn(2, 16, 576, dtype=torch.bfloat16)
    metadata = SimpleNamespace(
        dense_mla_plan=_FakePlan(),
        dense_mla_scratch=shared_scratch,
        query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
        decode=SimpleNamespace(
            block_table=torch.tensor([[0, 1]], dtype=torch.int32),
            seq_lens=torch.tensor([17], dtype=torch.int32),
        ),
    )
    layer = SimpleNamespace(
        _q_scale=torch.tensor(0.25),
        _k_scale=torch.tensor(0.5),
    )

    impl.forward_mqa(q, cache, metadata, layer)

    assert dense_mla.bindings[0].scratch is shared_scratch
    assert impl._scratch_by_plan == {}


def test_b12x_mla_compile_only_warmup_resolves_without_launch(monkeypatch) -> None:
    impl, dense_mla = _fake_impl(monkeypatch)
    q = torch.randn(1, 8, 576, dtype=torch.bfloat16)
    cache = torch.randn(2, 16, 576, dtype=torch.bfloat16)
    metadata = SimpleNamespace(
        dense_mla_plan=_FakePlan(),
        query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
        decode=SimpleNamespace(
            block_table=torch.tensor([[0, 1]], dtype=torch.int32),
            seq_lens=torch.tensor([17], dtype=torch.int32),
        ),
    )
    layer = SimpleNamespace(
        _q_scale=torch.tensor(0.25),
        _k_scale=torch.tensor(0.5),
    )

    with b12x_compile_only_warmup():
        output, lse = impl.forward_mqa(q, cache, metadata, layer)

    assert output.shape == (1, 8, 512)
    assert torch.count_nonzero(output) == 0
    assert lse is None
    assert dense_mla.compile_count == 1
    assert dense_mla.run_count == 0


def test_b12x_mla_adapter_gathers_and_reduces_dcp_heads(monkeypatch) -> None:
    impl, dense_mla = _fake_impl(monkeypatch)
    impl.num_heads = 6
    impl.dcp_world_size = 8
    group = SimpleNamespace(world_size=8)
    monkeypatch.setattr("vllm.distributed.parallel_state.get_dcp_group", lambda: group)

    gather_calls = []

    def fake_gather(q, actual_group, **kwargs):
        gather_calls.append((q.shape, actual_group, kwargs))
        return torch.cat([q] * 8, dim=1)

    reduce_calls = []

    def fake_reduce(output, lse, actual_group, **kwargs):
        reduce_calls.append((output.shape, lse.shape, actual_group, kwargs))
        return output[:, :6]

    monkeypatch.setattr(b12x_mla, "dcp_b12x_all_gather_heads", fake_gather)
    monkeypatch.setattr(b12x_mla, "dcp_a2a_lse_reduce", fake_reduce)

    q = torch.randn(1, 6, 576, dtype=torch.bfloat16)
    cache = torch.randn(2, 16, 576, dtype=torch.bfloat16)
    metadata = SimpleNamespace(
        dense_mla_plan=_FakePlan(),
        query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
        decode=SimpleNamespace(
            block_table=torch.tensor([[0, 1]], dtype=torch.int32),
            seq_lens=torch.tensor([17], dtype=torch.int32),
        ),
    )
    layer = SimpleNamespace(
        _q_scale=torch.tensor(0.25),
        _k_scale=torch.tensor(0.5),
    )

    output, lse = impl.forward_mqa(q, cache, metadata, layer)

    assert output.shape == (1, 6, 512)
    assert lse is None
    assert dense_mla.bindings[0].q.shape == (1, 48, 576)
    assert gather_calls[0][0] == (1, 6, 576)
    assert gather_calls[0][2]["output_head_dim"] == 512
    assert reduce_calls[0][0] == (1, 48, 512)
    assert reduce_calls[0][1] == (1, 48)
    assert reduce_calls[0][3]["use_b12x"] is True


def test_max_dcp_local_cache_tokens_respects_interleave() -> None:
    config = SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=1_048_576),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=8,
            cp_kv_cache_interleave_size=1,
        ),
    )
    assert b12x_mla._max_dcp_local_cache_tokens(config) == 131_072

    config.model_config.max_model_len = 1_048_577
    config.parallel_config.cp_kv_cache_interleave_size = 4
    assert b12x_mla._max_dcp_local_cache_tokens(config) == 131_076
