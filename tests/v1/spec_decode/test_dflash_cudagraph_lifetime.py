# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import torch

from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.spec_decode.dflash import speculator as dflash_module
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator


def _make_speculator() -> SimpleNamespace:
    hidden_states = torch.randn(2, 8)
    return SimpleNamespace(
        _run_model=Mock(return_value=hidden_states),
        _captured_backbone_outputs=[],
        num_speculative_steps=2,
        sample_indices=torch.tensor([0, 1]),
        sample_pos=torch.tensor([1, 2]),
        sample_idx_mapping=torch.tensor([0, 0]),
        temperature=torch.ones(1),
        seeds=torch.zeros(1, dtype=torch.int64),
        sample_col=torch.tensor([0, 1]),
        draft_logits=None,
        sample_draft=Mock(return_value=torch.tensor([11, 12])),
        draft_tokens=torch.zeros(1, 2, dtype=torch.int64),
    )


def test_dflash_retains_backbone_output_during_cudagraph_capture(monkeypatch):
    speculator = _make_speculator()
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)

    DFlashSpeculator._generate_draft(
        speculator,
        num_reqs=1,
        num_tokens_padded=2,
        attn_metadata=None,
        slot_mappings=None,
        num_tokens_across_dp=None,
        cudagraph_runtime_mode=CUDAGraphMode.NONE,
    )

    assert len(speculator._captured_backbone_outputs) == 1
    assert (
        speculator._captured_backbone_outputs[0] is speculator._run_model.return_value
    )


def test_dflash_does_not_retain_eager_backbone_output(monkeypatch):
    speculator = _make_speculator()
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)

    DFlashSpeculator._generate_draft(
        speculator,
        num_reqs=1,
        num_tokens_padded=2,
        attn_metadata=None,
        slot_mappings=None,
        num_tokens_across_dp=None,
        cudagraph_runtime_mode=CUDAGraphMode.NONE,
    )

    assert speculator._captured_backbone_outputs == []


def test_dflash_input_warmup_copies_sampling_state(monkeypatch):
    temperature = torch.ones(2)
    seeds = torch.zeros(2, dtype=torch.int64)
    speculator = SimpleNamespace(
        draft_kv_cache_group_id=0,
        draft_kv_cache_group_ids=[0],
        num_query_per_req=2,
        dynamic_physical_depth=False,
        max_num_reqs=2,
        max_num_tokens=2048,
        max_model_len=4096,
        device=torch.device("cpu"),
        input_buffers=object(),
        block_tables=SimpleNamespace(
            slot_mappings=[object()],
            input_block_tables=[object()],
            kernel_block_sizes=[16],
        ),
        context_positions=object(),
        _context_slot_mappings=[object()],
        sample_indices=object(),
        sample_pos=object(),
        sample_idx_mapping=object(),
        temperature=temperature,
        seeds=seeds,
        num_cached_tokens=object(),
        parallel_drafting_token_id=1,
        sample_from_anchor=False,
        _speculative_steps_for_query_len=lambda query_len: query_len - 1,
    )
    prepare_inputs = Mock()
    monkeypatch.setattr(dflash_module, "prepare_dflash_inputs", prepare_inputs)

    DFlashSpeculator._warmup_prepare_inputs_kernel(speculator)

    assert prepare_inputs.call_count == 5
    for call in prepare_inputs.call_args_list:
        assert call.args[7] is temperature
        assert call.args[8] is seeds
        assert call.args[14] is temperature
        assert call.args[15] is seeds
