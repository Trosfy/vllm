# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests that the auto_gptq quantization method works correctly.

Run `pytest tests/quantization/test_auto_gptq.py -v -s`.
"""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from tests.quantization.utils import is_quant_method_supported
from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.quantization import auto_gptq as auto_gptq_module
from vllm.model_executor.layers.quantization.auto_gptq import (
    AutoGPTQConfig,
    AutoGPTQLinearMethod,
    AutoGPTQMoEMethod,
)
from vllm.model_executor.layers.quantization.fp8 import Fp8LinearMethod

PROMPT = "On the surface of Mars, we found"

MODELS = [
    "TheBloke/TinyLlama-1.1B-Chat-v1.0-GPTQ",
]


@pytest.mark.skipif(
    not is_quant_method_supported("auto_gptq"),
    reason="auto_gptq is not supported on this GPU type.",
)
@pytest.mark.parametrize("model_id", MODELS)
def test_auto_gptq_quantization_method(vllm_runner, model_id: str, monkeypatch):
    """Test that quantization='auto_gptq' loads and runs correctly."""
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

    with vllm_runner(
        model_id,
        dtype=torch.float16,
        quantization="auto_gptq",
        max_model_len=2048,
        enforce_eager=True,
    ) as llm:

        def check_model(model):
            for name, submodule in model.named_modules():
                if name == "model.layers.0.self_attn.qkv_proj":
                    assert isinstance(submodule.quant_method, AutoGPTQLinearMethod)
                    break

        llm.apply_model(check_model)

        outputs = llm.generate_greedy([PROMPT], max_tokens=8)
        assert outputs
        assert len(outputs[0][1]) > 0


def test_auto_gptq_config_get_name():
    """Test that AutoGPTQConfig.get_name() returns 'auto_gptq'."""
    assert AutoGPTQConfig.get_name() == "auto_gptq"


def test_auto_gptq_moe_creates_zero_initialized_expert_biases():
    method = object.__new__(AutoGPTQMoEMethod)
    method.quant_config = AutoGPTQConfig(4, 128, False, True, False, {}, {})
    method.input_dtype = None
    method.experts_cls = None
    method.moe = SimpleNamespace(w13_num_shards=2)
    layer = torch.nn.Module()

    method.create_weights(
        layer=layer,
        num_experts=2,
        hidden_size=8,
        intermediate_size_per_partition=4,
        params_dtype=torch.float16,
        intermediate_size_full=4,
        weight_loader=lambda *args, **kwargs: None,
    )

    assert layer.w13_bias.shape == (2, 8)
    assert layer.w2_bias.shape == (2, 8)
    assert torch.count_nonzero(layer.w13_bias) == 0
    assert torch.count_nonzero(layer.w2_bias) == 0


def test_routed_experts_loads_per_expert_biases():
    class Loader:
        quant_config = None
        quant_method = object()
        moe_config = SimpleNamespace(
            is_act_and_mul=True,
            tp_rank=0,
            moe_parallel_config=SimpleNamespace(tp_size=1),
        )
        _get_hidden_dim = staticmethod(RoutedExperts._get_hidden_dim)
        _narrow_expert_data_for_padding = staticmethod(
            RoutedExperts._narrow_expert_data_for_padding
        )
        _load_w13 = RoutedExperts._load_w13
        _loaded_expert_biases = set()

        @staticmethod
        def _map_global_expert_id_to_local_expert_id(expert_id):
            return expert_id

    loader = Loader()
    w13_bias = torch.nn.Parameter(torch.zeros(1, 8), requires_grad=False)
    w2_bias = torch.nn.Parameter(torch.zeros(1, 4), requires_grad=False)

    for shard_id, loaded in (
        ("w1", torch.tensor([1.0, 2.0, 3.0, 4.0])),
        ("w3", torch.tensor([5.0, 6.0, 7.0, 8.0])),
    ):
        assert RoutedExperts.weight_loader(
            loader,
            w13_bias,
            loaded,
            weight_name="model.layers.0.mlp.experts.w13_bias",
            shard_id=shard_id,
            expert_id=0,
            return_success=True,
        )

    assert RoutedExperts.weight_loader(
        loader,
        w2_bias,
        torch.tensor([9.0, 10.0, 11.0, 12.0]),
        weight_name="model.layers.0.mlp.experts.w2_bias",
        shard_id="w2",
        expert_id=0,
        return_success=True,
    )
    assert torch.equal(w13_bias, torch.arange(1, 9, dtype=torch.float32).reshape(1, 8))
    assert torch.equal(w2_bias, torch.arange(9, 13, dtype=torch.float32).reshape(1, 4))
    assert loader._loaded_expert_biases == {"w13_bias", "w2_bias"}


# --- VLLM_FP8_HYBRID (int4+fp8 hybrid dispatch) -----------------------------
#
# CPU-only: synthetic safetensors metadata, no checkpoint download, no GPU.


def _new_config() -> AutoGPTQConfig:
    return AutoGPTQConfig(4, 128, False, True, False, {}, {})


def _fp8_metadata(prefix: str = "model.layers.0.self_attn.q_proj") -> dict:
    """One blockwise-fp8 layer (weight + weight_scale_inv) plus one bf16 layer."""
    return {
        f"{prefix}.weight": {"dtype": "F8_E4M3", "shape": [256, 256]},
        f"{prefix}.weight_scale_inv": {"dtype": "F32", "shape": [2, 2]},
        "model.layers.0.mlp.gate_up_proj.weight": {
            "dtype": "BF16",
            "shape": [512, 256],
        },
    }


def test_fp8_hybrid_detects_layers_with_scale_sibling():
    """A F8_E4M3 .weight with a .weight_scale_inv sibling is detected."""
    config = _new_config()
    config._detect_fp8_hybrid_layers(_fp8_metadata(), hf_config=None)
    assert config.fp8_layers == {"model.layers.0.self_attn.q_proj"}


def test_fp8_hybrid_ignores_fp8_weight_without_scale_sibling():
    """An F8_E4M3 weight with no .weight_scale_inv sibling is not hybrid fp8."""
    config = _new_config()
    metadata = {
        "model.layers.0.self_attn.q_proj.weight": {
            "dtype": "F8_E4M3",
            "shape": [256, 256],
        },
    }
    config._detect_fp8_hybrid_layers(metadata, hf_config=None)
    assert config.fp8_layers == set()


def test_fp8_hybrid_disabled_by_default(monkeypatch):
    """maybe_update_config leaves fp8_layers empty unless VLLM_FP8_HYBRID is set.

    Documented hazard: the fp8 weight still lands in the derived
    modules_in_block_to_quantize (any non-fp16/bf16/fp32 dtype), which is the
    known crash hazard the hybrid dispatch exists to avoid.
    """
    monkeypatch.delenv("VLLM_FP8_HYBRID", raising=False)
    monkeypatch.setattr(
        auto_gptq_module,
        "get_safetensors_params_metadata",
        lambda *args, **kwargs: _fp8_metadata(),
    )
    config = _new_config()
    config.maybe_update_config("fake-model")
    assert config.fp8_layers == set()
    assert "model.layers.0.self_attn.q_proj" in config.modules_in_block_to_quantize


@pytest.mark.parametrize("value", ["1", "true", "yes", "YES"])
def test_fp8_hybrid_enabled_detects_layers_via_maybe_update_config(monkeypatch, value):
    monkeypatch.setenv("VLLM_FP8_HYBRID", value)
    monkeypatch.setattr(
        auto_gptq_module,
        "get_safetensors_params_metadata",
        lambda *args, **kwargs: _fp8_metadata(),
    )
    config = _new_config()
    config.maybe_update_config("fake-model")
    assert config.fp8_layers == {"model.layers.0.self_attn.q_proj"}


def test_fp8_hybrid_apply_vllm_mapper_maps_fp8_layers():
    from vllm.model_executor.models.utils import WeightsMapper

    config = _new_config()
    config.fp8_layers = {"language_model.layers.0.self_attn.q_proj"}
    mapper = WeightsMapper(orig_to_new_prefix={"language_model.": "model."})
    config.apply_vllm_mapper(mapper)
    assert config.fp8_layers == {"model.layers.0.self_attn.q_proj"}


def test_fp8_hybrid_conflicts_with_dense_quant_override():
    """Fail-closed: dense_quant claims prefixes before the hybrid dispatch."""
    config = _new_config()
    hf_config = SimpleNamespace(dense_quant="fp8")
    with pytest.raises(ValueError, match="dense_quant"):
        config._detect_fp8_hybrid_layers(_fp8_metadata(), hf_config)


def test_fp8_hybrid_conflicts_with_lm_head_quant_override():
    config = _new_config()
    hf_config = SimpleNamespace(lm_head_quant="fp8")
    with pytest.raises(ValueError, match="lm_head_quant"):
        config._detect_fp8_hybrid_layers(_fp8_metadata(), hf_config)


def test_fp8_hybrid_no_conflict_without_override():
    config = _new_config()
    config._detect_fp8_hybrid_layers(_fp8_metadata(), hf_config=None)
    assert config.fp8_layers  # detected, no raise


def test_fp8_hybrid_refuses_at_config_time(monkeypatch):
    """The refusal fires inside maybe_update_config, not later in the loader."""
    monkeypatch.setenv("VLLM_FP8_HYBRID", "1")
    monkeypatch.setattr(
        auto_gptq_module,
        "get_safetensors_params_metadata",
        lambda *args, **kwargs: _fp8_metadata(),
    )
    config = _new_config()
    hf_config = SimpleNamespace(dense_quant="fp8")
    with pytest.raises(ValueError, match="VLLM_FP8_HYBRID"):
        config.maybe_update_config("fake-model", hf_config=hf_config)


def test_is_fp8_layer_false_when_no_fp8_layers_detected():
    config = _new_config()
    assert not config._is_fp8_layer("model.layers.0.self_attn.q_proj")


def test_is_fp8_layer_matches_unfused_prefix_directly():
    config = _new_config()
    config.fp8_layers = {"model.layers.0.self_attn.o_proj"}
    assert config._is_fp8_layer("model.layers.0.self_attn.o_proj")
    assert not config._is_fp8_layer("model.layers.0.self_attn.q_proj")


def test_is_fp8_layer_requires_all_fused_constituents():
    """Fused qkv_proj dispatches as one unit: all-or-nothing on its shards."""
    config = _new_config()
    config.packed_modules_mapping = {"qkv_proj": ["q_proj", "k_proj", "v_proj"]}
    config.fp8_layers = {
        "model.layers.0.self_attn.q_proj",
        "model.layers.0.self_attn.k_proj",
        "model.layers.0.self_attn.v_proj",
    }
    assert config._is_fp8_layer("model.layers.0.self_attn.qkv_proj")

    config.fp8_layers.discard("model.layers.0.self_attn.v_proj")
    assert not config._is_fp8_layer("model.layers.0.self_attn.qkv_proj")


def test_get_quant_method_dispatches_fused_layer_to_shared_fp8_config(
    default_vllm_config,
):
    """End to end through LinearBase: a fully-fp8 fused qkv_proj gets a real
    Fp8LinearMethod, and every hybrid-dispatched layer shares one Fp8Config
    instance (not a fresh one per layer)."""
    default_vllm_config.model_config = Mock(dtype=torch.bfloat16)
    config = _new_config()
    config.packed_modules_mapping = {"qkv_proj": ["q_proj", "k_proj", "v_proj"]}
    config.fp8_layers = {
        "model.layers.0.self_attn.q_proj",
        "model.layers.0.self_attn.k_proj",
        "model.layers.0.self_attn.v_proj",
        "model.layers.0.self_attn.o_proj",
    }

    fused_layer = LinearBase(
        256,
        256,
        quant_config=config,
        prefix="model.layers.0.self_attn.qkv_proj",
        disable_tp=True,
    )
    assert isinstance(fused_layer.quant_method, Fp8LinearMethod)

    shared_cfg = config._fp8_cfg
    assert shared_cfg is not None
    assert shared_cfg.is_checkpoint_fp8_serialized
    assert shared_cfg.activation_scheme == "dynamic"
    assert shared_cfg.weight_block_size == [128, 128]
    assert shared_cfg.packed_modules_mapping is config.packed_modules_mapping

    plain_layer = LinearBase(
        256,
        256,
        quant_config=config,
        prefix="model.layers.0.self_attn.o_proj",
        disable_tp=True,
    )
    assert isinstance(plain_layer.quant_method, Fp8LinearMethod)
    assert plain_layer.quant_method.quant_config is shared_cfg


def test_get_quant_method_fused_layer_requires_all_shards_fp8(default_vllm_config):
    """Partial fp8 detection on a fused module falls through to GPTQ
    unchanged -- the fused checkpoint tensor is indivisible, so a partial
    match must not be routed to Fp8Config."""
    default_vllm_config.model_config = Mock(dtype=torch.bfloat16)
    config = _new_config()
    config.packed_modules_mapping = {"qkv_proj": ["q_proj", "k_proj", "v_proj"]}
    config.modules_in_block_to_quantize = [
        "model.layers.0.self_attn.q_proj",
        "model.layers.0.self_attn.k_proj",
        "model.layers.0.self_attn.v_proj",
    ]
    config.fp8_layers = {"model.layers.0.self_attn.q_proj"}

    layer = LinearBase(
        256,
        256,
        quant_config=config,
        prefix="model.layers.0.self_attn.qkv_proj",
        disable_tp=True,
    )
    assert isinstance(layer.quant_method, AutoGPTQLinearMethod)
    assert not isinstance(layer.quant_method, Fp8LinearMethod)
