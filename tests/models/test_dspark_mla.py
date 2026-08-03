# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from vllm.compilation.wrapper import TorchCompileWithNoGuardsWrapper
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.models.qwen3_dspark import DSparkMarkovHead
from vllm.model_executor.models.registry import ModelRegistry
from vllm.models.kimi_k3.nvidia import dspark_mla
from vllm.models.kimi_k3.nvidia.dspark_mla import K3DSparkForCausalLM, K3DSparkModel


def test_dspark_mla_uses_compile_free_model_entrypoint():
    assert ModelRegistry._try_load_model_cls("K3DSparkModel") is K3DSparkForCausalLM
    assert not issubclass(K3DSparkModel, TorchCompileWithNoGuardsWrapper)


@pytest.mark.parametrize(
    ("checkpoint_name", "runtime_name", "shard_id"),
    [
        (
            "layers.0.self_attn.q_a_proj.weight",
            "model.layers.0.self_attn.fused_qkv_a_proj.weight",
            0,
        ),
        (
            "layers.0.self_attn.kv_a_proj_with_mqa.weight",
            "model.layers.0.self_attn.fused_qkv_a_proj.weight",
            1,
        ),
        (
            "layers.0.mlp.gate_proj.weight",
            "model.layers.0.mlp.gate_up_proj.weight",
            0,
        ),
        (
            "layers.0.mlp.up_proj.weight",
            "model.layers.0.mlp.gate_up_proj.weight",
            1,
        ),
        ("context_proj.weight", "model.context_proj.weight", None),
    ],
)
def test_dspark_mla_checkpoint_weight_mapping(checkpoint_name, runtime_name, shard_id):
    assert K3DSparkForCausalLM.hf_to_vllm_mapper._map_name_with_shard(
        checkpoint_name
    ) == (runtime_name, shard_id)


def test_dspark_mla_shares_frozen_target_weights_and_skips_training_head():
    assert not K3DSparkForCausalLM.has_own_embed_tokens
    assert not K3DSparkForCausalLM.has_own_lm_head
    assert set(K3DSparkForCausalLM.checkpoint_skip_substrs) == {
        "confidence_head",
        "embed_tokens",
        "lm_head",
    }


@pytest.mark.cpu_test
def test_dspark_markov_head_is_replicated(
    monkeypatch: pytest.MonkeyPatch,
):
    from vllm.model_executor.layers import logits_processor, vocab_parallel_embedding

    monkeypatch.setattr(
        vocab_parallel_embedding, "get_tensor_model_parallel_rank", lambda: 3
    )
    monkeypatch.setattr(
        vocab_parallel_embedding,
        "get_tensor_model_parallel_world_size",
        lambda: 8,
    )
    monkeypatch.setattr(
        logits_processor,
        "get_current_vllm_config",
        lambda: SimpleNamespace(model_config=None),
    )

    head = DSparkMarkovHead(128, 128, 8, prefix="markov_head")
    assert head.markov_w2.tp_size == 1
    assert head.markov_w1.weight.shape == (128, 8)
    assert head.markov_w2.weight.shape == (128, 8)

    def fail_collective(*args, **kwargs):
        raise AssertionError("replicated Markov head must not invoke TP collectives")

    monkeypatch.setattr(
        vocab_parallel_embedding,
        "tensor_model_parallel_all_reduce",
        fail_collective,
    )
    logits_processor = LogitsProcessor(128)
    monkeypatch.setattr(logits_processor, "_gather_logits", fail_collective)

    markov_embed = head.embed(torch.tensor([1, 2]))
    bias = head.bias(markov_embed, logits_processor)
    assert markov_embed.shape == (2, 8)
    assert bias.shape == (2, 128)


@pytest.mark.cpu_test
def test_k3_dspark_uses_replicated_markov_head(monkeypatch: pytest.MonkeyPatch):
    markov_head_calls = []

    class DummyModule(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

    def make_markov_head(*args, **kwargs):
        markov_head_calls.append((args, kwargs))
        return DummyModule()

    monkeypatch.setattr(dspark_mla, "get_draft_quant_config", lambda _: None)
    monkeypatch.setattr(dspark_mla, "ReplicatedLinear", DummyModule)
    monkeypatch.setattr(dspark_mla, "RMSNorm", DummyModule)
    monkeypatch.setattr(dspark_mla, "K3DSparkDecoderLayer", DummyModule)
    monkeypatch.setattr(dspark_mla, "DSparkMarkovHead", make_markov_head)

    config = SimpleNamespace(
        target_hidden_size=16,
        num_target_layers=2,
        hidden_size=8,
        rms_norm_eps=1e-6,
        num_hidden_layers=1,
        vocab_size=128,
        draft_vocab_size=128,
        markov_rank=4,
    )
    vllm_config = SimpleNamespace(
        speculative_config=SimpleNamespace(
            draft_model_config=SimpleNamespace(hf_config=config)
        ),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=8),
    )

    K3DSparkModel(vllm_config=vllm_config, start_layer_id=0, prefix="model")

    assert len(markov_head_calls) == 1


@pytest.mark.cpu_test
def test_dspark_context_kv_fusion_supports_sharded_qkv_a(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class DummyShardedProjection(nn.Module):
        def __init__(self, weight: torch.Tensor, tp_size: int) -> None:
            super().__init__()
            self.weight = nn.Parameter(weight)
            self.tp_size = tp_size

    monkeypatch.setattr(
        dspark_mla,
        "KimiShardedMergedColumnParallelLinear",
        DummyShardedProjection,
    )
    model = object.__new__(K3DSparkModel)
    nn.Module.__init__(model)
    model.quant_config = None
    model._max_num_context_tokens = 8
    tp_size = 4
    q_rank = 8
    kv_width = 4
    hidden = 3
    layers = []
    expected_kv_weights = []
    for layer_idx in range(2):
        weight = torch.arange(
            (q_rank // tp_size + kv_width // tp_size) * hidden,
            dtype=torch.float32,
        ).view(-1, hidden)
        weight = weight + layer_idx * 100
        expected_kv_weights.append(weight[q_rank // tp_size :])
        attn = SimpleNamespace(
            q_lora_rank=q_rank,
            kv_lora_rank=2,
            qk_rope_head_dim=2,
            fused_qkv_a_proj=DummyShardedProjection(weight, tp_size),
            kv_a_layernorm=SimpleNamespace(
                weight=torch.ones(2),
                variance_epsilon=1e-6,
            ),
        )
        layers.append(SimpleNamespace(self_attn=attn))
    model.layers = layers

    model._build_fused_context_kv_buffers()

    assert model._context_kv_fusion_available
    assert model._context_kv_sharded
    assert model._context_kv_tp_size == tp_size
    assert model._context_kv_stored_width == kv_width // tp_size
    torch.testing.assert_close(
        model._fused_context_kv_weight,
        torch.cat(expected_kv_weights, dim=0),
    )


@pytest.mark.cpu_test
def test_restore_layer_major_kv_order() -> None:
    # Gather order is rank-major; each rank contributes [layer0, layer1].
    rank_major = torch.tensor([[0, 1, 10, 11, 20, 21]])
    restored = dspark_mla._restore_layer_major_kv_order(
        rank_major,
        num_layers=2,
        local_kv_width=1,
        tp_size=3,
    )
    torch.testing.assert_close(restored, torch.tensor([[0, 10, 20, 1, 11, 21]]))
