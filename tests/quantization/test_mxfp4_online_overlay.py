# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Selection tests for the online dense overlay on MXFP4 checkpoints."""

import contextlib
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from vllm.config.quantization import QuantizationConfigArgs
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization.mxfp4 import Mxfp4Config


def _mock_linear() -> Mock:
    return Mock(spec=LinearBase)


@contextlib.contextmanager
def _patched_current_config(args: QuantizationConfigArgs):
    current = SimpleNamespace(
        model_config=SimpleNamespace(quantization_config=args, dtype=torch.bfloat16)
    )
    with (
        patch(
            "vllm.model_executor.layers.quantization.mxfp4.get_current_vllm_config",
            return_value=current,
        ),
        patch(
            "vllm.model_executor.layers.quantization.online.fp8."
            "get_current_vllm_config",
            return_value=current,
        ),
    ):
        yield


def test_mxfp4_linear_overlay_skips_shared_experts_without_spec():
    """`linear` alone must not touch shared experts (ModelOpt semantics)."""
    config = Mxfp4Config()
    args = QuantizationConfigArgs(linear="mxfp8")

    with _patched_current_config(args):
        dense = config.get_quant_method(
            _mock_linear(), "model.layers.3.self_attn.kv_b_proj"
        )
        shared = config.get_quant_method(
            _mock_linear(), "model.layers.3.mlp.shared_experts.down_proj"
        )

    assert type(dense).__name__ == "Mxfp8OnlineLinearMethod"
    assert isinstance(shared, UnquantizedLinearMethod)


def test_mxfp4_linear_overlay_quantizes_shared_experts_with_spec():
    config = Mxfp4Config()
    args = QuantizationConfigArgs(linear="mxfp8", shared_experts="mxfp8")

    with _patched_current_config(args):
        shared = config.get_quant_method(
            _mock_linear(), "model.layers.3.mlp.shared_experts.down_proj"
        )

    assert type(shared).__name__ == "Mxfp8OnlineLinearMethod"


def test_mxfp4_linear_overlay_honors_ignore():
    config = Mxfp4Config()
    args = QuantizationConfigArgs(linear="mxfp8", ignore=["re:.*kv_b_proj"])

    with _patched_current_config(args):
        ignored = config.get_quant_method(
            _mock_linear(), "model.layers.3.self_attn.kv_b_proj"
        )
        kept = config.get_quant_method(
            _mock_linear(), "model.layers.3.self_attn.q_b_proj"
        )

    assert isinstance(ignored, UnquantizedLinearMethod)
    assert type(kept).__name__ == "Mxfp8OnlineLinearMethod"


def test_mxfp4_overlay_can_select_only_shared_experts():
    config = Mxfp4Config()
    config.packed_modules_mapping = {
        "gate_up_proj": ["gate_proj", "up_proj"],
    }
    args = QuantizationConfigArgs(
        linear="mxfp8",
        shared_experts="mxfp8",
        ignore=["re:^(?!.*shared_experts).*$"],
    )

    with _patched_current_config(args):
        attention = config.get_quant_method(
            _mock_linear(), "model.layers.3.self_attn.o_proj"
        )
        shared_gate_up = config.get_quant_method(
            _mock_linear(), "model.layers.3.mlp.shared_experts.gate_up_proj"
        )
        shared_down = config.get_quant_method(
            _mock_linear(), "model.layers.3.mlp.shared_experts.down_proj"
        )

    assert isinstance(attention, UnquantizedLinearMethod)
    assert type(shared_gate_up).__name__ == "Mxfp8OnlineLinearMethod"
    assert type(shared_down).__name__ == "Mxfp8OnlineLinearMethod"


def test_mxfp4_overlay_can_select_only_kda_fused_input_projection():
    config = Mxfp4Config()
    config.packed_modules_mapping = {
        "in_proj_qkvgfab": [
            "q_proj",
            "k_proj",
            "v_proj",
            "b_proj",
            "f_a_proj",
        ],
    }
    args = QuantizationConfigArgs(
        linear="mxfp8",
        ignore=[
            "re:^(?!.*self_attn\\.(?:q_proj|k_proj|v_proj|b_proj|f_a_proj)$).*$",
        ],
    )

    with _patched_current_config(args):
        kda_input = config.get_quant_method(
            _mock_linear(), "model.layers.3.self_attn.in_proj_qkvgfab"
        )
        output = config.get_quant_method(
            _mock_linear(), "model.layers.3.self_attn.o_proj"
        )

    assert type(kda_input).__name__ == "Mxfp8OnlineLinearMethod"
    assert isinstance(output, UnquantizedLinearMethod)


def test_mxfp4_overlay_can_select_kda_input_and_attention_output():
    config = Mxfp4Config()
    config.packed_modules_mapping = {
        "in_proj_qkvgfab": [
            "q_proj",
            "k_proj",
            "v_proj",
            "b_proj",
            "f_a_proj",
        ],
    }
    args = QuantizationConfigArgs(
        linear="mxfp8",
        ignore=[
            "re:^(?!.*self_attn\\.(?:q_proj|k_proj|v_proj|b_proj|f_a_proj|o_proj)$).*$",
        ],
    )

    with _patched_current_config(args):
        kda_input = config.get_quant_method(
            _mock_linear(), "model.layers.3.self_attn.in_proj_qkvgfab"
        )
        output = config.get_quant_method(
            _mock_linear(), "model.layers.4.self_attn.o_proj"
        )
        mla_query = config.get_quant_method(
            _mock_linear(), "model.layers.4.self_attn.q_b_proj"
        )

    assert type(kda_input).__name__ == "Mxfp8OnlineLinearMethod"
    assert type(output).__name__ == "Mxfp8OnlineLinearMethod"
    assert isinstance(mla_query, UnquantizedLinearMethod)
