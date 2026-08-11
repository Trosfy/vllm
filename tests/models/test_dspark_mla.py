# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from vllm.compilation.wrapper import TorchCompileWithNoGuardsWrapper
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.models import qwen3_dspark
from vllm.model_executor.models.qwen3_dspark import (
    DSparkMarkovEmbedding,
    DSparkMarkovHead,
)
from vllm.model_executor.models.registry import ModelRegistry
from vllm.models.kimi_k3.nvidia import dspark_mla
from vllm.models.kimi_k3.nvidia.dspark_mla import K3DSparkForCausalLM, K3DSparkModel
from vllm.models.kimi_k3.nvidia.mla import MultiHeadLatentAttention
from vllm.v1.kv_cache_interface import MLAAttentionSpec, SlidingWindowMLASpec


def test_dspark_mla_uses_compile_free_model_entrypoint():
    assert ModelRegistry._try_load_model_cls("K3DSparkModel") is K3DSparkForCausalLM
    assert not issubclass(K3DSparkModel, TorchCompileWithNoGuardsWrapper)


@pytest.mark.cpu_test
def test_k3_dspark_mla_uses_bounded_replicated_kv_spec(monkeypatch) -> None:
    monkeypatch.setenv("VLLM_DCP_SHARD_DRAFT", "0")
    layer = object.__new__(MultiHeadLatentAttention)
    layer.kv_cache_dtype = "auto"
    layer.head_size = 576
    layer.non_causal_multi_token_decode = True
    layer.draft_kv_window = 65536
    config = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=768),
        model_config=SimpleNamespace(dtype=torch.bfloat16),
        parallel_config=SimpleNamespace(decode_context_parallel_size=8),
    )

    spec = MultiHeadLatentAttention.get_kv_cache_spec(layer, config)

    assert isinstance(spec, SlidingWindowMLASpec)
    assert spec.sliding_window == 65536
    assert spec.block_size == 768
    assert spec.dcp_replicated is True
    assert spec.non_causal_multi_token_decode is True


@pytest.mark.cpu_test
def test_k3_dspark_full_kv_spec_can_be_explicitly_dcp_sharded(monkeypatch) -> None:
    monkeypatch.setenv("VLLM_DCP_SHARD_DRAFT", "1")
    layer = object.__new__(MultiHeadLatentAttention)
    layer.kv_cache_dtype = "auto"
    layer.head_size = 576
    layer.non_causal_multi_token_decode = True
    layer.draft_kv_window = 0
    config = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=768),
        model_config=SimpleNamespace(dtype=torch.bfloat16),
        parallel_config=SimpleNamespace(decode_context_parallel_size=8),
    )

    spec = MultiHeadLatentAttention.get_kv_cache_spec(layer, config)

    assert isinstance(spec, MLAAttentionSpec)
    assert spec.dcp_replicated is False


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
        (
            "confidence_head.proj.weight",
            "model.confidence_head.proj.weight",
            None,
        ),
    ],
)
def test_dspark_mla_checkpoint_weight_mapping(checkpoint_name, runtime_name, shard_id):
    assert K3DSparkForCausalLM.hf_to_vllm_mapper._map_name_with_shard(
        checkpoint_name
    ) == (runtime_name, shard_id)


def test_dspark_mla_shares_only_frozen_target_embedding_and_lm_head():
    assert not K3DSparkForCausalLM.has_own_embed_tokens
    assert not K3DSparkForCausalLM.has_own_lm_head
    assert set(K3DSparkForCausalLM.checkpoint_skip_substrs) == {
        "embed_tokens",
        "lm_head",
    }


@pytest.mark.cpu_test
def test_k3_dspark_compute_confidence_uses_checkpoint_head() -> None:
    class ConfidenceHead(nn.Module):
        def forward(
            self,
            hidden: torch.Tensor,
            markov_embed: torch.Tensor,
        ) -> torch.Tensor:
            return hidden.sum(dim=-1) + markov_embed.sum(dim=-1)

    model = K3DSparkForCausalLM.__new__(K3DSparkForCausalLM)
    nn.Module.__init__(model)
    model.model = nn.Module()
    model.model.confidence_head = ConfidenceHead()
    hidden = torch.randn(3, 8)
    markov_embed = torch.randn(3, 2)

    confidence = model.compute_confidence(hidden, markov_embed)

    assert confidence is not None
    torch.testing.assert_close(
        confidence,
        hidden.sum(dim=-1) + markov_embed.sum(dim=-1),
    )


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
    monkeypatch.setenv("VLLM_DSPARK_SHARD_MARKOV_HEAD", "0")
    monkeypatch.setenv("VLLM_DSPARK_REPLICATE_MARKOV_W1", "0")

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
def test_dspark_markov_head_can_be_tp_sharded(
    monkeypatch: pytest.MonkeyPatch,
):
    from vllm.model_executor.layers import logits_processor, vocab_parallel_embedding

    monkeypatch.setenv("VLLM_DSPARK_SHARD_MARKOV_HEAD", "1")
    monkeypatch.setenv("VLLM_DSPARK_REPLICATE_MARKOV_W1", "0")
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

    assert head.shard_across_tp
    assert head.markov_w1.tp_size == 8
    assert head.markov_w2.tp_size == 8
    assert head.markov_w1.weight.shape == (16, 8)
    assert head.markov_w2.weight.shape == (16, 8)

    logits_processor = LogitsProcessor(128)

    def fail_gather(*args, **kwargs):
        raise AssertionError("local Markov bias must not gather vocab logits")

    monkeypatch.setattr(logits_processor, "_gather_logits", fail_gather)
    markov_embed = torch.randn(2, 8)
    local_bias = head.local_bias(markov_embed, logits_processor)
    assert local_bias.shape == (2, 16)


@pytest.mark.cpu_test
def test_dspark_markov_head_can_replicate_only_w1(
    monkeypatch: pytest.MonkeyPatch,
):
    from vllm.model_executor.layers import logits_processor, vocab_parallel_embedding

    monkeypatch.setenv("VLLM_DSPARK_SHARD_MARKOV_HEAD", "1")
    monkeypatch.setenv("VLLM_DSPARK_REPLICATE_MARKOV_W1", "1")
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

    assert head.shard_across_tp
    assert head.replicate_w1
    assert isinstance(head.markov_w1, DSparkMarkovEmbedding)
    assert head.markov_w1.weight.shape == (128, 8)
    assert head.markov_w2.tp_size == 8
    assert head.markov_w2.weight.shape == (16, 8)

    def fail_collective(*args, **kwargs):
        raise AssertionError("replicated Markov W1 must not invoke an all-reduce")

    monkeypatch.setattr(
        vocab_parallel_embedding,
        "tensor_model_parallel_all_reduce",
        fail_collective,
    )
    assert head.embed(torch.tensor([1, 2])).shape == (2, 8)


@pytest.mark.cpu_test
def test_k3_dspark_sharded_sampling_gathers_after_local_sum(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm.model_executor.layers import vocab_parallel_embedding

    monkeypatch.setattr(
        vocab_parallel_embedding, "get_tensor_model_parallel_rank", lambda: 0
    )
    monkeypatch.setattr(
        vocab_parallel_embedding,
        "get_tensor_model_parallel_world_size",
        lambda: 1,
    )
    lm_head = ParallelLMHead(8, 4, disable_tp=True)
    model = K3DSparkForCausalLM.__new__(K3DSparkForCausalLM)
    nn.Module.__init__(model)
    model.lm_head = lm_head
    model.logits_processor = SimpleNamespace(org_vocab_size=6)

    base_logits = torch.randn(2, 4)
    markov_bias = torch.randn(2, 4)
    gathered_logits = torch.tensor(
        [
            [-3.0, 1.0, 7.0, 0.0, 2.0, 3.0, 100.0, 99.0],
            [4.0, 9.0, 0.0, 1.0, 3.0, 2.0, 100.0, 99.0],
        ]
    )
    calls = []

    def fake_all_gather(logits, dim=-1):
        calls.append((logits, dim))
        return gathered_logits

    monkeypatch.setattr(dspark_mla, "tensor_model_parallel_all_gather", fake_all_gather)

    sampled = model.sample_local_draft_logits(base_logits, markov_bias)

    assert sampled.tolist() == [2, 1]
    assert len(calls) == 1
    assert calls[0][1] == -1
    torch.testing.assert_close(calls[0][0], base_logits + markov_bias)


@pytest.mark.cpu_test
def test_k3_dspark_gathers_full_probabilistic_logits_after_local_sum(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm.model_executor.layers import vocab_parallel_embedding

    monkeypatch.setattr(
        vocab_parallel_embedding, "get_tensor_model_parallel_rank", lambda: 0
    )
    monkeypatch.setattr(
        vocab_parallel_embedding,
        "get_tensor_model_parallel_world_size",
        lambda: 1,
    )
    model = K3DSparkForCausalLM.__new__(K3DSparkForCausalLM)
    nn.Module.__init__(model)
    model.lm_head = ParallelLMHead(8, 4, disable_tp=True)
    model.logits_processor = SimpleNamespace(org_vocab_size=6)

    base_logits = torch.randn(2, 4)
    markov_bias = torch.randn(2, 4)
    gathered_logits = torch.arange(16, dtype=torch.float32).view(2, 8)
    calls = []

    def fake_all_gather(logits, dim=-1):
        calls.append((logits, dim))
        return gathered_logits

    monkeypatch.setattr(dspark_mla, "tensor_model_parallel_all_gather", fake_all_gather)

    logits = model.gather_local_draft_logits(base_logits, markov_bias)

    torch.testing.assert_close(logits, gathered_logits[:, :6])
    assert len(calls) == 1
    assert calls[0][1] == -1
    torch.testing.assert_close(calls[0][0], base_logits + markov_bias)


@pytest.mark.cpu_test
def test_k3_dspark_b12x_argmax_receives_unmaterialized_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm.model_executor.layers import vocab_parallel_embedding

    monkeypatch.setattr(
        vocab_parallel_embedding, "get_tensor_model_parallel_rank", lambda: 0
    )
    monkeypatch.setattr(
        vocab_parallel_embedding,
        "get_tensor_model_parallel_world_size",
        lambda: 1,
    )
    lm_head = ParallelLMHead(8, 4, disable_tp=True)
    lm_head.tp_size = 16
    model = K3DSparkForCausalLM.__new__(K3DSparkForCausalLM)
    nn.Module.__init__(model)
    model.lm_head = lm_head
    model.logits_processor = SimpleNamespace(org_vocab_size=6)
    model._b12x_dspark_argmax_enabled = True
    model._b12x_dspark_argmax_max_batch = 8
    model._b12x_dspark_argmax_output = torch.empty(8, dtype=torch.int64)

    calls = []

    class FakeArgmax:
        def fused_add_argmax(self, base, bias, out):
            calls.append((base, bias, out))
            out.copy_(torch.tensor([3, 5], dtype=torch.int64))
            return out

    model._b12x_dspark_argmax_runtime = FakeArgmax()

    def fail_gather(*args, **kwargs):
        raise AssertionError("B12X argmax must not gather full-vocabulary logits")

    monkeypatch.setattr(dspark_mla, "tensor_model_parallel_all_gather", fail_gather)
    storage = torch.randn(2, 3, 4, dtype=torch.bfloat16)
    base_logits = storage[:, 1]
    markov_bias = torch.randn(2, 4, dtype=torch.bfloat16)

    sampled = model.sample_local_draft_logits(base_logits, markov_bias)

    assert sampled.tolist() == [3, 5]
    assert len(calls) == 1
    assert calls[0][0] is base_logits
    assert calls[0][1] is markov_bias
    assert calls[0][2].data_ptr() == model._b12x_dspark_argmax_output.data_ptr()


@pytest.mark.cpu_test
def test_dspark_markov_embedding_finalizes_online_mxfp8(monkeypatch) -> None:
    embedding = DSparkMarkovEmbedding(16, 32, use_mxfp8=True)
    source = embedding.weight.detach().clone()
    quantized = source.to(torch.float8_e4m3fn)
    scales = torch.full((16, 1), 127, dtype=torch.uint8)
    monkeypatch.setattr(
        qwen3_dspark,
        "mxfp8_e4m3_quantize",
        lambda weight: (quantized, scales),
    )
    expected = torch.randn(2, 32, dtype=torch.bfloat16)
    calls = []

    def fake_lookup(weight, weight_scale, token_ids):
        calls.append((weight, weight_scale, token_ids))
        return expected

    monkeypatch.setattr(qwen3_dspark, "mxfp8_embedding", fake_lookup)
    embedding.process_weights_after_loading()
    result = embedding(torch.tensor([2, 7]))

    assert embedding.weight.dtype == torch.float8_e4m3fn
    assert embedding.weight_scale is scales
    assert result is expected
    assert len(calls) == 1

    # The post-load hook is reload-safe within the same materialization.
    embedding.process_weights_after_loading()
    assert embedding.weight.data_ptr() == quantized.data_ptr()


@pytest.mark.cpu_test
def test_k3_dspark_shards_context_projection_and_uses_replicated_markov_head(
    monkeypatch: pytest.MonkeyPatch,
):
    markov_head_calls = []
    context_proj_calls = []

    class DummyModule(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

    def make_markov_head(*args, **kwargs):
        markov_head_calls.append((args, kwargs))
        return DummyModule()

    def make_context_proj(*args, **kwargs):
        context_proj_calls.append((args, kwargs))
        return DummyModule()

    monkeypatch.setattr(dspark_mla, "get_draft_quant_config", lambda _: None)
    monkeypatch.setattr(dspark_mla, "ColumnParallelLinear", make_context_proj)
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
    assert len(context_proj_calls) == 1
    assert context_proj_calls[0][1]["gather_output"] is True
    assert markov_head_calls[0][1]["quant_config"] is None


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
    # A model-level quant config does not imply this particular projection is
    # quantized: selective online MXFP8 keeps qkv-a in BF16 specifically so
    # context KV fusion remains available.
    model.quant_config = object()
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
