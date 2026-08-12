# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import copy
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.worker.gpu.spec_decode.dflash import speculator as dflash_speculator
from vllm.v1.worker.gpu.spec_decode.dspark import utils as dspark_utils


def _speculator(target_config, *, window: int | None = None):
    speculator = object.__new__(dflash_speculator.DFlashSpeculator)
    speculator.vllm_config = target_config
    speculator.requires_non_causal = True
    speculator.draft_kv_window = window
    speculator.draft_kv_window_block_size = 768 if window is not None else None
    return speculator


def test_dspark_loader_constructs_model_with_draft_parallel_geometry(
    monkeypatch,
) -> None:
    """The draft model constructor receives its own DCP and attention config."""
    draft_model_config = SimpleNamespace(hf_config=SimpleNamespace())
    target_config = SimpleNamespace(
        speculative_config=SimpleNamespace(
            draft_model_config=draft_model_config,
            attention_backend=AttentionBackendEnum.B12X_MLA,
        )
    )
    draft_config = SimpleNamespace(
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
        attention_config=SimpleNamespace(backend=None, use_non_causal=False),
    )
    captured = {}

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
        dspark_utils,
        "_create_draft_vllm_config",
        lambda config: draft_config if config is target_config else None,
    )
    monkeypatch.setattr(dspark_utils, "replace", fake_replace)
    monkeypatch.setattr(dspark_utils, "get_model", fake_get_model)
    monkeypatch.setattr(
        "vllm.model_executor.models.qwen3_dflash.dflash_has_any_non_causal",
        lambda _config: True,
    )

    with pytest.raises(ModelCaptured):
        dspark_utils.load_dspark_model(object(), target_config)

    assert captured["model_config"] is draft_model_config
    loaded_config = captured["vllm_config"]
    assert loaded_config.parallel_config.decode_context_parallel_size == 1
    assert loaded_config.attention_config.backend == AttentionBackendEnum.B12X_MLA
    assert loaded_config.attention_config.use_non_causal


def test_dspark_attention_metadata_uses_draft_parallel_geometry(monkeypatch) -> None:
    """An external draft cache retains its configured DCP geometry."""
    target_config = SimpleNamespace(
        parallel_config=SimpleNamespace(decode_context_parallel_size=16)
    )
    draft_config = SimpleNamespace(
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
        attention_config=SimpleNamespace(
            backend=AttentionBackendEnum.B12X_MLA,
            use_non_causal=False,
        ),
    )
    monkeypatch.setattr(
        dflash_speculator,
        "_create_draft_vllm_config",
        lambda config: draft_config if config is target_config else None,
    )

    attention_config = _speculator(target_config).attn_vllm_config

    assert attention_config.parallel_config.decode_context_parallel_size == 1
    assert attention_config.attention_config.use_non_causal
    assert not draft_config.attention_config.use_non_causal


def test_bounded_draft_attention_preserves_derived_model_attributes(
    monkeypatch,
) -> None:
    """Plan sizing copies runtime objects without reconstructing dataclasses."""

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

    draft_model_config = DraftModelConfig(max_model_len=1_048_576)
    draft_model_config.model_arch_config = object()
    draft_config = DraftConfig(
        model_config=draft_model_config,
        attention_config=DraftAttentionConfig(use_non_causal=False),
    )
    target_config = object()
    monkeypatch.setattr(
        dflash_speculator,
        "_create_draft_vllm_config",
        lambda config: draft_config if config is target_config else None,
    )

    attention_config = _speculator(
        target_config,
        window=65_536,
    ).attn_vllm_config

    assert draft_model_config.max_model_len == 1_048_576
    assert attention_config.model_config.max_model_len == 65_536 + 768 - 1
    assert attention_config.model_config.model_arch_config is not None
    assert attention_config.attention_config.use_non_causal
