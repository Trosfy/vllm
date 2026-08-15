# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import torch

from vllm.config.compilation import CUDAGraphMode
from vllm.distributed.device_communicators.custom_all_reduce import (
    CustomAllreduce,
    get_active_b12x_pcie_allreduce,
)
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
        _ensure_captured_markov_buffers=Mock(),
        _captured_markov_hidden=torch.empty(7, 16),
        _captured_base_logits=torch.empty(7, 11),
        sample_indices=torch.arange(7),
        num_query_per_req=7,
        _speculative_steps_for_query_len=lambda query_len: query_len,
        _sample_sequential=Mock(),
        capacity_activation_batch_size=0,
    )


def test_capture_records_backbone_output_without_markov_collectives(monkeypatch):
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
    speculator._ensure_captured_markov_buffers.assert_called_once_with()
    torch.testing.assert_close(
        speculator._captured_markov_hidden,
        speculator._run_model.return_value[:7],
    )
    torch.testing.assert_close(
        speculator._captured_base_logits,
        speculator._run_model.return_value[:7, :11] + 1,
    )


def test_capture_warmup_does_not_launch_markov_collectives(monkeypatch):
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
    speculator._ensure_captured_markov_buffers.assert_called_once_with()
    torch.testing.assert_close(
        speculator._captured_markov_hidden,
        speculator._run_model.return_value[:7],
    )
    torch.testing.assert_close(
        speculator._captured_base_logits,
        speculator._run_model.return_value[:7, :11] + 1,
    )


def test_eager_generation_samples_markov_tail(monkeypatch):
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
    speculator._ensure_captured_markov_buffers.assert_not_called()


def test_graph_replay_finishes_markov_tail_from_stable_buffers():
    speculator = _make_speculator()

    DSparkSpeculator._finish_captured_draft(
        speculator,
        num_reqs=1,
        num_tokens_padded=8,
        num_query_per_req=7,
        is_profile=False,
    )

    args = speculator._sample_sequential.call_args
    assert args.args[:4] == (1, None, 7, 7)
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


def test_captured_markov_allreduce_probe_covers_every_draft_step(monkeypatch):
    probe_shapes = []

    class CustomAllreduce:
        def should_custom_ar(self, probe):
            probe_shapes.append(tuple(probe.shape))
            return True

    model = SimpleNamespace(
        supports_local_draft_argmax=lambda: True,
        _b12x_dspark_argmax_enabled=True,
        model=SimpleNamespace(markov_head=SimpleNamespace(replicate_w1=False)),
    )
    speculator = SimpleNamespace(
        use_draft_token_capacity=False,
        draft_logits=None,
        _draft_topk=None,
        _capture_sharded_markov=True,
        _markov_outside_cudagraph=False,
        max_num_reqs=8,
        num_speculative_steps=7,
        vllm_config=object(),
        draft_model_config=SimpleNamespace(
            hf_config=SimpleNamespace(markov_rank=128)
        ),
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
    )
    monkeypatch.setenv("VLLM_DSPARK_SHARD_MARKOV_HEAD", "1")
    monkeypatch.setattr(
        "vllm.v1.worker.gpu.spec_decode.dspark.speculator.load_dspark_model",
        lambda _target_model, _config: model,
    )
    monkeypatch.setattr(
        "vllm.distributed.device_communicators.custom_all_reduce."
        "get_active_b12x_pcie_allreduce",
        lambda: CustomAllreduce(),
    )

    loaded = DSparkSpeculator.load_draft_model(
        speculator,
        target_model=object(),
        target_attn_layer_names=set(),
    )

    assert loaded is model
    assert probe_shapes == [(56, 128)]


def test_active_b12x_accessor_accepts_hierarchical_runtime(monkeypatch):
    custom_allreduce = object.__new__(CustomAllreduce)
    custom_allreduce.disabled = False
    custom_allreduce._pcie_runtime = object()
    group = SimpleNamespace(
        device_communicator=SimpleNamespace(ca_comm=custom_allreduce)
    )
    monkeypatch.setattr(
        "vllm.distributed.parallel_state.get_tp_group", lambda: group
    )

    assert get_active_b12x_pcie_allreduce() is custom_allreduce


def test_active_b12x_accessor_rejects_disabled_runtime(monkeypatch):
    custom_allreduce = object.__new__(CustomAllreduce)
    custom_allreduce.disabled = True
    custom_allreduce._pcie_runtime = object()
    group = SimpleNamespace(
        device_communicator=SimpleNamespace(ca_comm=custom_allreduce)
    )
    monkeypatch.setattr(
        "vllm.distributed.parallel_state.get_tp_group", lambda: group
    )

    assert get_active_b12x_pcie_allreduce() is None
