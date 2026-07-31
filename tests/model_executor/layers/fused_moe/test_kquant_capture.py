# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pathlib import Path

import torch
from safetensors.torch import load_file

from vllm.model_executor.layers.fused_moe.kquant_capture import (
    _KQuantCaptureState,
    _moe_row,
)


def test_k3_moe_layer_rows() -> None:
    assert _moe_row("language_model.model.layers.1.block_sparse_moe") == 0
    assert _moe_row("language_model.model.layers.92.block_sparse_moe") == 91


def test_pending_samples_are_batched_atomically(tmp_path: Path) -> None:
    state = _KQuantCaptureState.__new__(_KQuantCaptureState)
    state.samples_dir = tmp_path / "samples"
    state.parts = 0
    state.pending_samples = {}
    state.pending_sample_bytes = 0

    state._queue_samples(
        {
            "mid.values": torch.tensor([[1, 2]], dtype=torch.bfloat16),
            "mid.weight": torch.tensor([0.25], dtype=torch.float32),
        }
    )
    state._queue_samples(
        {
            "mid.values": torch.tensor([[3, 4]], dtype=torch.bfloat16),
            "mid.weight": torch.tensor([0.75], dtype=torch.float32),
        }
    )
    assert state.pending_sample_bytes > 0

    state._write_pending_samples()

    assert state.parts == 1
    assert state.pending_samples == {}
    assert state.pending_sample_bytes == 0
    tensors = load_file(tmp_path / "samples" / "part-00000001.safetensors")
    torch.testing.assert_close(
        tensors["mid.values"],
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16),
    )
    torch.testing.assert_close(
        tensors["mid.weight"], torch.tensor([0.25, 0.75], dtype=torch.float32)
    )
