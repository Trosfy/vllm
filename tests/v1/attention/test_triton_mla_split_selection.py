# SPDX-License-Identifier: Apache-2.0

import pytest

import vllm.envs as envs
from vllm.v1.attention.backends.mla.triton_mla import (
    _compute_num_kv_splits,
    _select_num_kv_splits,
)


def test_adaptive_triton_mla_splits(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(envs, "VLLM_TRITON_MLA_STATIC_KV_SPLITS", None)

    assert _compute_num_kv_splits(4607, 188) == 8
    assert _compute_num_kv_splits(4608, 188) == 16
    assert _select_num_kv_splits(4607, 188) == 8
    assert _select_num_kv_splits(4608, 188) == 16


def test_static_triton_mla_splits(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(envs, "VLLM_TRITON_MLA_STATIC_KV_SPLITS", 8)

    assert _select_num_kv_splits(1, 188) == 8
    assert _select_num_kv_splits(32768, 188) == 8


@pytest.mark.parametrize("value", [0, 3, 1024])
def test_invalid_static_triton_mla_splits(
    monkeypatch: pytest.MonkeyPatch, value: int
) -> None:
    monkeypatch.setattr(envs, "VLLM_TRITON_MLA_STATIC_KV_SPLITS", value)

    with pytest.raises(ValueError, match="VLLM_TRITON_MLA_STATIC_KV_SPLITS"):
        _select_num_kv_splits(4608, 188)
