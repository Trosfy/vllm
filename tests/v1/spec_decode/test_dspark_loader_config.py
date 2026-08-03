# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import copy
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.worker.gpu.spec_decode.dflash import speculator as dflash_speculator
from vllm.v1.worker.gpu.spec_decode.dspark import utils as dspark_utils


def test_dspark_loader_builds_model_from_draft_parallel_config(monkeypatch) -> None:
    """An external draft must not silently inherit the target's DCP size."""

    draft_model_config = SimpleNamespace(hf_config=SimpleNamespace())
    target_config = SimpleNamespace(
        speculative_config=SimpleNamespace(
            draft_model_config=draft_model_config,
            attention_backend=AttentionBackendEnum.B12X_MLA,
            kv_cache_dtype="fp8",
        )
    )
    draft_config = SimpleNamespace(
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
        attention_config=SimpleNamespace(backend=None, use_non_causal=False),
    )
    helper_calls = []
    captured = {}

    def fake_create_draft_config(config):
        helper_calls.append(config)
        return draft_config

    def fake_replace(value, **updates):
        result = copy.copy(value)
        for key, update in updates.items():
            setattr(result, key, update)
        return result

    class ModelCaptured(RuntimeError):
        pass

    def fake_get_model(*, vllm_config, model_config):
        captured["vllm_config"] = vllm_config
        captured["model_config"] = model_config
        raise ModelCaptured

    monkeypatch.setattr(
        dspark_utils, "_create_draft_vllm_config", fake_create_draft_config
    )
    monkeypatch.setattr(dspark_utils, "replace", fake_replace)
    monkeypatch.setattr(dspark_utils, "get_model", fake_get_model)
    monkeypatch.setattr(
        "vllm.model_executor.models.qwen3_dflash.dflash_has_any_non_causal",
        lambda _config: True,
    )

    with pytest.raises(ModelCaptured):
        dspark_utils.load_dspark_model(object(), target_config)

    assert helper_calls == [target_config]
    assert captured["model_config"] is draft_model_config
    loaded_config = captured["vllm_config"]
    assert loaded_config.parallel_config.decode_context_parallel_size == 1
    assert loaded_config.attention_config.backend == AttentionBackendEnum.B12X_MLA
    assert loaded_config.attention_config.use_non_causal is True


def test_dspark_metadata_builders_use_draft_parallel_config(monkeypatch) -> None:
    """Draft metadata must not inherit the target's process-wide DCP size."""

    target_config = SimpleNamespace(
        parallel_config=SimpleNamespace(decode_context_parallel_size=8),
    )
    draft_config = SimpleNamespace(
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
        attention_config=SimpleNamespace(backend=AttentionBackendEnum.B12X_MLA),
    )
    monkeypatch.setattr(
        dflash_speculator,
        "_create_draft_vllm_config",
        lambda config: draft_config if config is target_config else None,
    )

    def fake_replace(value, **updates):
        result = copy.copy(value)
        for key, update in updates.items():
            setattr(result, key, update)
        return result

    monkeypatch.setattr(dflash_speculator, "replace", fake_replace)

    speculator = object.__new__(dflash_speculator.DFlashSpeculator)
    speculator.vllm_config = target_config
    speculator.requires_non_causal = True
    speculator.draft_kv_window = None
    speculator.draft_kv_window_block_size = None

    attn_config = speculator.attn_vllm_config

    assert attn_config.parallel_config.decode_context_parallel_size == 1
    assert attn_config.attention_config.backend == AttentionBackendEnum.B12X_MLA
    assert attn_config.attention_config.use_non_causal is True


def test_dspark_bounded_metadata_caps_only_draft_attention_plan(monkeypatch) -> None:
    """The bounded plan must copy a ModelConfig with cached runtime fields."""

    @dataclass
    class DraftModelConfig:
        max_model_len: int

    @dataclass
    class DraftAttentionConfig:
        use_non_causal: bool

    @dataclass
    class DraftConfig:
        model_config: DraftModelConfig
        attention_config: DraftAttentionConfig

    original_model_config = DraftModelConfig(max_model_len=1_048_576)
    # Mirrors ModelConfig's derived cache that is deliberately not a declared
    # dataclass field and triggered the production startup regression.
    original_model_config.model_arch_config = object()
    draft_config = DraftConfig(
        model_config=original_model_config,
        attention_config=DraftAttentionConfig(use_non_causal=False),
    )
    target_config = object()
    monkeypatch.setattr(
        dflash_speculator,
        "_create_draft_vllm_config",
        lambda config: draft_config if config is target_config else None,
    )

    speculator = object.__new__(dflash_speculator.DFlashSpeculator)
    speculator.vllm_config = target_config
    speculator.requires_non_causal = True
    speculator.draft_kv_window = 65_536
    speculator.draft_kv_window_block_size = 768

    attn_config = speculator.attn_vllm_config

    assert original_model_config.max_model_len == 1_048_576
    assert attn_config.model_config.max_model_len == 65_536 + 768 - 1
    assert attn_config.attention_config.use_non_causal is True
