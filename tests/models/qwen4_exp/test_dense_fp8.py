# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline tests for the opt-in fp8 w8a16 dense path (dense_quant override).

No GPU, no checkpoint: small linear layers are built directly (mirroring
test_lm_head_fp8.py), synthetic bf16 weights are streamed through the
online-processing weight loader, and the CPU dequant fallback of
Qwen4ExpDenseFp8LinearMethod.apply is checked against an fp32 reference.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

import vllm.model_executor.parameter as parameter_module
import vllm.models.qwen4_exp as qwen4_exp_package
from vllm.config import CompilationConfig, set_current_vllm_config
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.quantization.utils.humming_utils import (
    convert_linear_layer_to_humming_standard,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import get_fp8_min_max
from vllm.models.qwen4_exp.nvidia.dense_fp8 import (
    DENSE_FP8_EXPECTED_MATCHES,
    DENSE_FP8_PATTERNS,
    Qwen4ExpDenseFp8Config,
    Qwen4ExpDenseFp8LinearMethod,
    dense_fp8_quant_config,
    get_dense_quant,
    maybe_dense_fp8,
)
from vllm.models.qwen4_exp.nvidia.model import Qwen4ExpForCausalLM

PACKED_MODULES_MAPPING = Qwen4ExpForCausalLM.packed_modules_mapping

HIDDEN = 32
SHARD = 16

# The deployed layer mix the expected match count is sized for: 48 decoder
# layers, 12 of them full attention (QSA), every layer carrying a shared expert.
NUM_LAYERS = 48
QSA_LAYERS = frozenset(range(3, NUM_LAYERS, 4))


@pytest.fixture(autouse=True)
def _allow_single_rank_tensor_parallel(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stand in a rank-0/size-1 TP world (mirrors test_lm_head_fp8.py)."""
    monkeypatch.setattr(parameter_module, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        parameter_module, "get_tensor_model_parallel_world_size", lambda: 1
    )


class _StubQuantConfig(QuantizationConfig):
    """Minimal stand-in for the checkpoint's own quantization config."""

    def __init__(self, name: str = "modelopt_fp4") -> None:
        super().__init__()
        self._name = name
        self.seen: list[str] = []

    def get_name(self) -> Any:
        return self._name

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        return [torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 75

    @staticmethod
    def get_config_filenames() -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> _StubQuantConfig:
        return cls()

    def get_quant_method(self, layer: torch.nn.Module, prefix: str) -> Any:
        self.seen.append(prefix)
        return UnquantizedLinearMethod()


def _vllm_config(
    dense_quant: str | None = None,
    quant_config: QuantizationConfig | None = None,
) -> SimpleNamespace:
    hf_attrs = {} if dense_quant is None else {"dense_quant": dense_quant}
    return SimpleNamespace(
        model_config=SimpleNamespace(
            dtype=torch.bfloat16,
            hf_config=SimpleNamespace(**hf_attrs),
            hf_text_config=SimpleNamespace(**hf_attrs),
        ),
        quant_config=quant_config,
        # Read by the wfp8-a16 kernel selection during create_weights.
        kernel_config=SimpleNamespace(linear_backend="auto"),
        compilation_config=CompilationConfig(custom_ops=["none"]),
    )


def _underlying_loader(param: torch.Tensor):
    """Strip the online-processing wrapper off a parameter's weight loader."""
    loader = param.weight_loader
    while loader.__name__ == "online_process_loader":
        loader = loader.__wrapped__
    return loader


def _census_prefixes() -> list[str]:
    """Every dense linear prefix the deployed decoder stack constructs."""
    prefixes: list[str] = []
    for idx in range(NUM_LAYERS):
        base = f"model.layers.{idx}"
        if idx in QSA_LAYERS:
            prefixes += [
                f"{base}.self_attn.qkv_proj",
                f"{base}.self_attn.o_proj",
                f"{base}.self_attn.indexer.index_qk_proj",
            ]
        else:
            prefixes += [
                f"{base}.linear_attn.in_proj_qkvz",
                f"{base}.linear_attn.in_proj_ba",
                f"{base}.linear_attn.conv1d",
                f"{base}.linear_attn.out_proj",
            ]
        prefixes += [
            f"{base}.mlp.gate",
            f"{base}.mlp.shared_expert_gate",
            f"{base}.mlp.shared_expert.gate_up_proj",
            f"{base}.mlp.shared_expert.down_proj",
            f"{base}.attn_hyper_connection.input_mix_weight_down_block_inject",
        ]
    prefixes.append("lm_head")
    return prefixes


# --------------------------------------------------------------------------
# Opt-in resolution / default-off inertness
# --------------------------------------------------------------------------


def test_helper_returns_none_without_override() -> None:
    assert get_dense_quant(_vllm_config()) is None


def test_helper_rejects_unknown_value() -> None:
    with pytest.raises(ValueError, match="dense_quant"):
        get_dense_quant(_vllm_config(dense_quant="int4"))


def test_helper_accepts_fp8() -> None:
    assert get_dense_quant(_vllm_config(dense_quant="fp8")) == "fp8"


def test_default_off_leaves_quant_config_untouched() -> None:
    stub = _StubQuantConfig()
    vllm_config = _vllm_config(quant_config=stub)
    with dense_fp8_quant_config(vllm_config, PACKED_MODULES_MAPPING) as wrapper:
        assert wrapper is None
        assert vllm_config.quant_config is stub
        assert type(vllm_config.quant_config) is _StubQuantConfig
    assert vllm_config.quant_config is stub


def test_scoped_swap_restores_original_config() -> None:
    stub = _StubQuantConfig()
    vllm_config = _vllm_config(dense_quant="fp8", quant_config=stub)
    with dense_fp8_quant_config(vllm_config, PACKED_MODULES_MAPPING) as wrapper:
        assert isinstance(wrapper, Qwen4ExpDenseFp8Config)
        assert vllm_config.quant_config is wrapper
    assert vllm_config.quant_config is stub


def test_scoped_swap_restores_on_error() -> None:
    stub = _StubQuantConfig()
    vllm_config = _vllm_config(dense_quant="fp8", quant_config=stub)
    with (
        pytest.raises(RuntimeError),
        dense_fp8_quant_config(vllm_config, PACKED_MODULES_MAPPING),
    ):
        raise RuntimeError("construction failed")
    assert vllm_config.quant_config is stub


def test_wrapper_reports_the_wrapped_config_name() -> None:
    # `model.without_modelopt_fp4` keys on this, so it must not change.
    wrapper = Qwen4ExpDenseFp8Config(_StubQuantConfig("modelopt_fp4"))
    assert wrapper.get_name() == "modelopt_fp4"


# --------------------------------------------------------------------------
# Allow-list routing
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "prefix",
    [
        "model.layers.0.linear_attn.in_proj_qkvz",
        "model.layers.0.linear_attn.out_proj",
        "model.layers.3.self_attn.qkv_proj",
        "model.layers.3.self_attn.o_proj",
        "model.layers.0.mlp.shared_expert.gate_up_proj",
        "model.layers.0.mlp.shared_expert.down_proj",
    ],
)
def test_allowlisted_prefixes_match(prefix: str) -> None:
    wrapper = Qwen4ExpDenseFp8Config(
        None, packed_modules_mapping=PACKED_MODULES_MAPPING
    )
    assert wrapper.matched_pattern(prefix) is not None


@pytest.mark.parametrize(
    "prefix",
    [
        "model.layers.0.linear_attn.in_proj_ba",
        "model.layers.0.linear_attn.conv1d",
        "model.layers.3.self_attn.indexer.index_qk_proj",
        "model.layers.0.mlp.gate",
        "model.layers.0.mlp.shared_expert_gate",
        "model.layers.0.mlp.gate_up_proj",
        "model.layers.0.mlp.experts.0.gate_proj",
        "model.layers.0.attn_hyper_connection.input_mix_weight_down_block_inject",
        "lm_head",
    ],
)
def test_excluded_prefixes_do_not_match(prefix: str) -> None:
    wrapper = Qwen4ExpDenseFp8Config(
        None, packed_modules_mapping=PACKED_MODULES_MAPPING
    )
    assert wrapper.matched_pattern(prefix) is None


def test_census_match_count_equals_expected() -> None:
    wrapper = Qwen4ExpDenseFp8Config(
        None, packed_modules_mapping=PACKED_MODULES_MAPPING
    )
    matched = [p for p in _census_prefixes() if wrapper.matched_pattern(p) is not None]
    assert len(matched) == DENSE_FP8_EXPECTED_MATCHES
    assert len(matched) == 192


def test_non_allowlisted_layer_delegates_to_wrapped_config() -> None:
    stub = _StubQuantConfig()
    wrapper = Qwen4ExpDenseFp8Config(
        stub, packed_modules_mapping=PACKED_MODULES_MAPPING
    )
    layer = torch.nn.Module()  # not a LinearBase: e.g. RoutedExperts / Attention
    assert wrapper.get_quant_method(layer, "model.layers.0.mlp.experts") is not None
    assert stub.seen == ["model.layers.0.mlp.experts"]
    assert sum(wrapper.match_counts.values()) == 0


def test_excluded_linear_keeps_the_wrapped_configs_method() -> None:
    stub = _StubQuantConfig()
    wrapper = Qwen4ExpDenseFp8Config(
        stub, packed_modules_mapping=PACKED_MODULES_MAPPING
    )
    vllm_config = _vllm_config(dense_quant="fp8")
    with set_current_vllm_config(vllm_config):
        layer = _build_row_linear(wrapper, "model.layers.0.linear_attn.in_proj_ba")
    assert isinstance(layer.quant_method, UnquantizedLinearMethod)
    assert stub.seen == ["model.layers.0.linear_attn.in_proj_ba"]


def test_unquantized_checkpoint_delegates_to_none() -> None:
    wrapper = Qwen4ExpDenseFp8Config(
        None, packed_modules_mapping=PACKED_MODULES_MAPPING
    )
    layer = torch.nn.Module()
    assert wrapper.get_quant_method(layer, "model.layers.0.mlp.experts") is None


# --------------------------------------------------------------------------
# Layer construction and the match-count guard
# --------------------------------------------------------------------------


def _build_row_linear(wrapper: Qwen4ExpDenseFp8Config, prefix: str):
    return RowParallelLinear(
        HIDDEN,
        HIDDEN,
        bias=False,
        quant_config=wrapper,
        prefix=prefix,
        disable_tp=True,
    )


def test_allowlisted_layer_gets_the_dense_fp8_method() -> None:
    vllm_config = _vllm_config(dense_quant="fp8")
    wrapper = Qwen4ExpDenseFp8Config(
        None, packed_modules_mapping=PACKED_MODULES_MAPPING
    )
    with set_current_vllm_config(vllm_config):
        layer = _build_row_linear(wrapper, "model.layers.0.linear_attn.out_proj")
    assert isinstance(layer.quant_method, Qwen4ExpDenseFp8LinearMethod)
    assert layer.weight.device.type == "meta"
    assert layer.logical_widths == [HIDDEN]


def test_excluded_layer_stays_unquantized() -> None:
    vllm_config = _vllm_config(dense_quant="fp8")
    wrapper = Qwen4ExpDenseFp8Config(
        None, packed_modules_mapping=PACKED_MODULES_MAPPING
    )
    with set_current_vllm_config(vllm_config):
        layer = _build_row_linear(wrapper, "model.layers.0.linear_attn.in_proj_ba")
    assert isinstance(layer.quant_method, UnquantizedLinearMethod)
    assert layer.weight.dtype == torch.get_default_dtype()


def test_match_count_guard_accepts_the_expected_count() -> None:
    vllm_config = _vllm_config(dense_quant="fp8")
    wrapper = Qwen4ExpDenseFp8Config(
        None,
        packed_modules_mapping=PACKED_MODULES_MAPPING,
        patterns=("*.linear_attn.out_proj",),
        expected_matches=2,
    )
    with set_current_vllm_config(vllm_config):
        for idx in range(2):
            _build_row_linear(wrapper, f"model.layers.{idx}.linear_attn.out_proj")
    wrapper.validate_match_count()


def test_match_count_guard_rejects_a_typo_in_the_allowlist() -> None:
    vllm_config = _vllm_config(dense_quant="fp8")
    wrapper = Qwen4ExpDenseFp8Config(
        None,
        packed_modules_mapping=PACKED_MODULES_MAPPING,
        patterns=("*.linear_attn.out_projj",),
        expected_matches=2,
    )
    with set_current_vllm_config(vllm_config):
        for idx in range(2):
            _build_row_linear(wrapper, f"model.layers.{idx}.linear_attn.out_proj")
    with pytest.raises(ValueError, match="matched 0 layers but expected 2"):
        wrapper.validate_match_count()


def test_tensor_parallel_is_rejected() -> None:
    vllm_config = _vllm_config(dense_quant="fp8")
    with set_current_vllm_config(vllm_config):
        method = Qwen4ExpDenseFp8LinearMethod()
    layer = torch.nn.Module()
    layer.tp_size = 2
    with pytest.raises(NotImplementedError, match="dense_quant"):
        method.create_weights(
            layer,
            HIDDEN,
            [HIDDEN],
            HIDDEN,
            HIDDEN,
            torch.bfloat16,
            weight_loader=lambda *args, **kwargs: None,
        )


# --------------------------------------------------------------------------
# Quantization math and merged-shard loading
# --------------------------------------------------------------------------


def _assert_quantized_like(layer, reference_weight: torch.Tensor) -> None:
    """Assert the processed layer matches an fp32 dequant reference."""
    fp8_max = get_fp8_min_max()[1]
    expected_scale = (
        reference_weight.abs().amax().to(torch.float32).reshape(1) / fp8_max
    )
    out_features, in_features = reference_weight.shape
    assert layer.weight.dtype == torch.float8_e4m3fn
    assert layer.weight.shape == (in_features, out_features)  # canonical (K, N)
    assert torch.allclose(layer.weight_scale, expected_scale)

    # Idempotent on a second call (weight-reload guard).
    layer.quant_method.process_weights_after_loading(layer)
    assert layer.weight.dtype == torch.float8_e4m3fn

    x = torch.randn(3, in_features, dtype=torch.bfloat16)
    out = layer.quant_method.apply(layer, x)
    x_f32 = x.to(torch.float32)
    weight_f32 = reference_weight.to(torch.float32)
    reference = torch.nn.functional.linear(x_f32, weight_f32)
    assert out.shape == (3, out_features)
    # fp8 e4m3 has 3 mantissa bits, so each weight carries at most 1/16
    # relative error; the bf16 dequant and output add ~2/256 more. Bound the
    # accumulated error by sum(|x| * |w|) rather than by the output magnitude,
    # which cancellation can drive arbitrarily close to zero.
    error_bound = 0.08 * (x_f32.abs() @ weight_f32.abs().t()) + 1e-2
    assert torch.all((out.to(torch.float32) - reference).abs() <= error_bound)


def test_single_shard_quant_matches_dequant_reference() -> None:
    vllm_config = _vllm_config(dense_quant="fp8")
    wrapper = Qwen4ExpDenseFp8Config(
        None, packed_modules_mapping=PACKED_MODULES_MAPPING
    )
    with set_current_vllm_config(vllm_config):
        layer = _build_row_linear(wrapper, "model.layers.0.linear_attn.out_proj")

    torch.manual_seed(7)
    checkpoint_weight = torch.randn(HIDDEN, HIDDEN, dtype=torch.bfloat16)
    layer.weight.weight_loader(layer.weight, checkpoint_weight)

    _assert_quantized_like(layer, checkpoint_weight)


def test_merged_shard_load_matches_dequant_reference() -> None:
    vllm_config = _vllm_config(dense_quant="fp8")
    wrapper = Qwen4ExpDenseFp8Config(
        None, packed_modules_mapping=PACKED_MODULES_MAPPING
    )
    with set_current_vllm_config(vllm_config):
        layer = MergedColumnParallelLinear(
            input_size=HIDDEN,
            output_sizes=[SHARD, SHARD],
            bias=False,
            quant_config=wrapper,
            prefix="model.layers.0.mlp.shared_expert.gate_up_proj",
            disable_tp=True,
        )
    assert isinstance(layer.quant_method, Qwen4ExpDenseFp8LinearMethod)
    assert layer.logical_widths == [SHARD, SHARD]
    # Merged shards only narrow correctly through the v2 loader, which
    # LinearBase picks by the method class name.
    assert _underlying_loader(layer.weight).__func__ is type(layer).weight_loader_v2

    torch.manual_seed(11)
    # Second shard deliberately larger so a per-shard scale would differ from
    # the per-tensor one and the reference check would fail.
    gate = torch.randn(SHARD, HIDDEN, dtype=torch.bfloat16)
    up = torch.randn(SHARD, HIDDEN, dtype=torch.bfloat16) * 4.0
    layer.weight.weight_loader(layer.weight, gate, 0)
    layer.weight.weight_loader(layer.weight, up, 1)

    _assert_quantized_like(layer, torch.cat([gate, up], dim=0))


def test_qkv_shard_load_matches_dequant_reference() -> None:
    vllm_config = _vllm_config(dense_quant="fp8")
    wrapper = Qwen4ExpDenseFp8Config(
        None,
        packed_modules_mapping=PACKED_MODULES_MAPPING,
        patterns=("*.self_attn.qkv_proj",),
        expected_matches=1,
    )
    head_dim = 8
    with set_current_vllm_config(vllm_config):
        layer = QKVParallelLinear(
            hidden_size=HIDDEN,
            head_size=head_dim,
            total_num_heads=2,
            total_num_kv_heads=1,
            bias=False,
            quant_config=wrapper,
            prefix="model.layers.3.self_attn.qkv_proj",
            disable_tp=True,
        )
    assert isinstance(layer.quant_method, Qwen4ExpDenseFp8LinearMethod)
    wrapper.validate_match_count()

    torch.manual_seed(13)
    shards = {
        "q": torch.randn(2 * head_dim, HIDDEN, dtype=torch.bfloat16),
        "k": torch.randn(head_dim, HIDDEN, dtype=torch.bfloat16),
        "v": torch.randn(head_dim, HIDDEN, dtype=torch.bfloat16) * 3.0,
    }
    for shard_id, shard in shards.items():
        layer.weight.weight_loader(layer.weight, shard, shard_id)

    _assert_quantized_like(layer, torch.cat(list(shards.values()), dim=0))


def _assert_survives_humming_layout_prep(layer, in_features: int) -> None:
    """Regression check for a live-serve crash.

    process_weights_after_loading() canonicalizes the weight to (K, N) via
    ``.t()``, which is a non-contiguous view (stride(-1) == in_features, not
    1) even when the pre-transpose weight is contiguous -- true here for both
    a single shard and a merged/QKV shard load, since the weight loader
    always materializes into a preallocated contiguous buffer. Humming's own
    prep, convert_linear_layer_to_humming_standard(), reads the weight's
    ``input_dim``/``output_dim`` tags to decide whether to
    transpose-and-contiguous it back before a dtype-reinterpret
    ``view(int32)`` cast, which requires a contiguous last dim; Marlin's prep
    tolerates the view directly and never caught a missing/wrong tag.
    Exercises the real prep function directly (pure tensor ops, no
    CUDA/Humming runtime needed) so this is CPU-safe.
    """
    assert not layer.weight.is_contiguous()  # a transposed view, by design
    assert layer.weight.input_dim == 0
    assert layer.weight.output_dim == 1

    convert_linear_layer_to_humming_standard(
        layer=layer, name_map={"weight": "weight", "weight_scale": "weight_scale"}
    )
    assert layer.weight.is_contiguous()
    assert layer.weight.stride(-1) == 1


def test_single_shard_survives_humming_layout_prep() -> None:
    vllm_config = _vllm_config(dense_quant="fp8")
    wrapper = Qwen4ExpDenseFp8Config(
        None, packed_modules_mapping=PACKED_MODULES_MAPPING
    )
    with set_current_vllm_config(vllm_config):
        layer = _build_row_linear(wrapper, "model.layers.0.linear_attn.out_proj")

    torch.manual_seed(17)
    checkpoint_weight = torch.randn(HIDDEN, HIDDEN, dtype=torch.bfloat16)
    layer.weight.weight_loader(layer.weight, checkpoint_weight)
    layer.quant_method.process_weights_after_loading(layer)

    _assert_survives_humming_layout_prep(layer, HIDDEN)


def test_qkv_merged_shard_survives_humming_layout_prep() -> None:
    vllm_config = _vllm_config(dense_quant="fp8")
    wrapper = Qwen4ExpDenseFp8Config(
        None,
        packed_modules_mapping=PACKED_MODULES_MAPPING,
        patterns=("*.self_attn.qkv_proj",),
        expected_matches=1,
    )
    head_dim = 8
    with set_current_vllm_config(vllm_config):
        layer = QKVParallelLinear(
            hidden_size=HIDDEN,
            head_size=head_dim,
            total_num_heads=2,
            total_num_kv_heads=1,
            bias=False,
            quant_config=wrapper,
            prefix="model.layers.3.self_attn.qkv_proj",
            disable_tp=True,
        )

    torch.manual_seed(19)
    shards = {
        "q": torch.randn(2 * head_dim, HIDDEN, dtype=torch.bfloat16),
        "k": torch.randn(head_dim, HIDDEN, dtype=torch.bfloat16),
        "v": torch.randn(head_dim, HIDDEN, dtype=torch.bfloat16) * 3.0,
    }
    for shard_id, shard in shards.items():
        layer.weight.weight_loader(layer.weight, shard, shard_id)
    layer.quant_method.process_weights_after_loading(layer)

    _assert_survives_humming_layout_prep(layer, HIDDEN)


# --------------------------------------------------------------------------
# QSA qkv_proj call site
# --------------------------------------------------------------------------

QKV_PREFIX = "model.layers.3.self_attn.qkv_proj"


def test_maybe_dense_fp8_preserves_stock_nvfp4_exclusion() -> None:
    stub = _StubQuantConfig("modelopt_fp4")
    assert maybe_dense_fp8(stub, QKV_PREFIX) is None
    assert maybe_dense_fp8(None, QKV_PREFIX) is None


def test_maybe_dense_fp8_passes_other_configs_through() -> None:
    stub = _StubQuantConfig("compressed-tensors")
    assert maybe_dense_fp8(stub, QKV_PREFIX) is stub


def test_maybe_dense_fp8_routes_qkv_to_the_wrapper() -> None:
    wrapper = Qwen4ExpDenseFp8Config(
        _StubQuantConfig(), packed_modules_mapping=PACKED_MODULES_MAPPING
    )
    assert maybe_dense_fp8(wrapper, QKV_PREFIX) is wrapper


def test_maybe_dense_fp8_falls_back_when_qkv_is_not_allowlisted() -> None:
    # A narrowed allow-list must not hand qkv_proj back to ModelOpt-FP4, whose
    # method would look for NVFP4 weights the checkpoint does not carry.
    wrapper = Qwen4ExpDenseFp8Config(
        _StubQuantConfig(),
        packed_modules_mapping=PACKED_MODULES_MAPPING,
        patterns=("*.self_attn.o_proj",),
    )
    assert maybe_dense_fp8(wrapper, QKV_PREFIX) is None


def test_both_platform_trees_route_qkv_through_the_helper() -> None:
    tree_root = Path(qwen4_exp_package.__file__).parent
    for platform in ("nvidia", "amd"):
        source = (tree_root / platform / "qsa.py").read_text()
        assert "dense_fp8.maybe_dense_fp8(" in source, platform
        assert "without_modelopt_fp4" not in source, platform


def test_shipped_patterns_are_the_planned_allowlist() -> None:
    assert DENSE_FP8_PATTERNS == (
        "*.linear_attn.in_proj_qkvz",
        "*.linear_attn.out_proj",
        "*.self_attn.qkv_proj",
        "*.self_attn.o_proj",
        "*.mlp.shared_expert.gate_up_proj",
        "*.mlp.shared_expert.down_proj",
    )
