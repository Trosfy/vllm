# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import torch

from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator


def _make_speculator() -> SimpleNamespace:
    head_hidden = torch.randn(8, 16)
    model = SimpleNamespace(
        compute_local_draft_logits=Mock(side_effect=lambda hidden: hidden[:, :11] + 1)
    )
    return SimpleNamespace(
        _run_model=Mock(return_value=head_hidden),
        model=model,
        _markov_outside_cudagraph=True,
        _captured_markov_hidden=torch.empty(7, 16),
        _captured_base_logits=torch.empty(7, 11),
        sample_indices=torch.arange(7),
        num_query_per_req=7,
        _speculative_steps_for_query_len=lambda query_len: query_len,
        _sample_sequential=Mock(),
        capacity_activation_batch_size=0,
    )


def test_sharded_markov_capture_records_only_backbone_copy(monkeypatch):
    speculator = _make_speculator()
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)

    DSparkSpeculator._generate_draft(
        speculator,
        num_reqs=1,
        num_tokens_padded=8,
        attn_metadata=None,
        slot_mappings=None,
        num_tokens_across_dp=None,
        cudagraph_runtime_mode=CUDAGraphMode.FULL,
    )

    speculator._sample_sequential.assert_not_called()
    torch.testing.assert_close(
        speculator._captured_markov_hidden,
        speculator._run_model.return_value[:7],
    )
    torch.testing.assert_close(
        speculator._captured_base_logits,
        speculator._run_model.return_value[:7, :11] + 1,
    )


def test_sharded_markov_capture_warmup_skips_collectives(monkeypatch):
    speculator = _make_speculator()
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)

    DSparkSpeculator._generate_draft(
        speculator,
        num_reqs=1,
        num_tokens_padded=8,
        attn_metadata=None,
        slot_mappings=None,
        num_tokens_across_dp=None,
        cudagraph_runtime_mode=CUDAGraphMode.NONE,
        capture_only=True,
    )

    speculator._sample_sequential.assert_not_called()
    torch.testing.assert_close(
        speculator._captured_markov_hidden,
        speculator._run_model.return_value[:7],
    )
    torch.testing.assert_close(
        speculator._captured_base_logits,
        speculator._run_model.return_value[:7, :11] + 1,
    )


def test_sharded_markov_eager_path_still_samples(monkeypatch):
    speculator = _make_speculator()
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)

    DSparkSpeculator._generate_draft(
        speculator,
        num_reqs=1,
        num_tokens_padded=8,
        attn_metadata=None,
        slot_mappings=None,
        num_tokens_across_dp=None,
        cudagraph_runtime_mode=CUDAGraphMode.NONE,
    )

    speculator._sample_sequential.assert_called_once_with(
        1,
        speculator._run_model.return_value,
        7,
        7,
        is_profile=False,
        use_capacity=True,
    )


def test_sharded_markov_finishes_after_backbone_graph_replay():
    speculator = _make_speculator()

    DSparkSpeculator._finish_captured_draft(
        speculator,
        num_reqs=1,
        num_tokens_padded=8,
        num_query_per_req=7,
        is_profile=False,
    )

    args = speculator._sample_sequential.call_args
    assert args.args[0] == 1
    assert args.args[1] is None
    assert args.args[2:4] == (7, 7)
    assert args.kwargs["is_profile"] is False
    assert args.kwargs["use_capacity"] is True
    assert (
        args.kwargs["prepared_sample_hidden"].untyped_storage().data_ptr()
        == speculator._captured_markov_hidden.untyped_storage().data_ptr()
    )
    assert (
        args.kwargs["precomputed_base_logits"].untyped_storage().data_ptr()
        == speculator._captured_base_logits.untyped_storage().data_ptr()
    )
