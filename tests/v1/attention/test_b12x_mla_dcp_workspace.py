# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.layers.attention.mla_attention import (
    _can_use_b12x_dcp_prefill_workspace,
)
from vllm.v1.attention.ops import common


@pytest.mark.parametrize(
    ("override", "value"),
    [
        ("enabled", False),
        ("project_before_merge", False),
        ("dcp_use_b12x", True),
        ("num_tokens", 1024),
        ("num_tokens", 3073),
        ("non_dbo_workspace", False),
        ("is_sparse_impl", False),
        ("backend_name", "FLASHINFER_MLA"),
        ("is_capturing", True),
    ],
)
def test_dcp_workspace_gate_rejects_unsupported_profiles(override, value):
    profile = {
        "enabled": True,
        "project_before_merge": True,
        "dcp_use_b12x": False,
        "num_tokens": 1025,
        "non_dbo_workspace": True,
        "is_sparse_impl": True,
        "backend_name": "B12X_MLA_SPARSE",
        "is_capturing": False,
    }
    profile[override] = value

    assert not _can_use_b12x_dcp_prefill_workspace(**profile)


@pytest.mark.parametrize("num_tokens", [1025, 2048, 3072])
def test_dcp_workspace_gate_accepts_valid_rows(num_tokens):
    assert _can_use_b12x_dcp_prefill_workspace(
        enabled=True,
        project_before_merge=True,
        dcp_use_b12x=False,
        num_tokens=num_tokens,
        non_dbo_workspace=True,
        is_sparse_impl=True,
        backend_name="B12X_MLA_SPARSE",
        is_capturing=False,
    )


def test_cp_lse_ag_out_rs_into_preserves_borrowed_output(monkeypatch):
    corrected = torch.arange(64, dtype=torch.bfloat16).view(1, 4, 16)
    corrected_lse = torch.arange(4, dtype=torch.float32).view(1, 4)
    borrowed = torch.empty((1, 1, 16), dtype=torch.bfloat16)

    monkeypatch.setattr(
        common,
        "_cp_lse_common",
        lambda *args, **kwargs: (corrected, corrected_lse),
    )

    class FakeGroup:
        world_size = 4
        rank_in_group = 2

        def reduce_scatter_into(self, input_, output, dim):
            assert input_ is corrected
            assert output is borrowed
            assert dim == 1
            output.copy_(input_[:, 2:3])
            return output

    output, lse = common.cp_lse_ag_out_rs_into(
        torch.empty_like(corrected),
        torch.empty_like(corrected_lse),
        FakeGroup(),
        output_provider=lambda value: borrowed,
        return_lse=True,
    )

    assert output is borrowed
    assert torch.equal(output, corrected[:, 2:3])
    assert torch.equal(lse, corrected_lse[:, 2:3])
