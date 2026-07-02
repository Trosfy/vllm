# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import importlib
import os
from typing import TYPE_CHECKING, Any

import torch

import vllm.envs as envs
from vllm.config import get_current_vllm_config_or_none
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape
from vllm.platforms import current_platform
from vllm.utils.torch_utils import (
    _USE_LAYERNAME,
    LayerName,
    _encode_layer_name,
    current_stream,
    direct_register_custom_op,
)

from .BlockScaledMMLinearKernel import (
    Fp8BlockScaledMMLinearKernel,
    FP8ScaledMMLinearLayerConfig,
)
from .deep_gemm import DeepGemmFp8BlockScaledMMKernel

_B12X_BLOCK_FP8: Any | None = None
_B12X_BLOCK_FP8_MISSING = False

# Measured crossover on RTX PRO 6000 Blackwell (DSV4-Flash TP2 dense shapes,
# b12x benchmark_dense_fp8_vs_deepgemm 2026-07-02): the b12x MXFP8 dense GEMM
# beats DeepGEMM up to M=2048 (0.73-0.95x) and loses above (1.05-1.21x at
# M=4096-8192), so prefill-sized token batches route to DeepGEMM.
_B12X_DG_PREFILL_DEFAULT_MIN_TOKENS = 2049

if TYPE_CHECKING:
    from typing import TypeAlias

    _layer_name_type: TypeAlias = str | LayerName
else:
    _layer_name_type = LayerName if _USE_LAYERNAME else str


def _import_b12x_block_fp8() -> Any | None:
    global _B12X_BLOCK_FP8, _B12X_BLOCK_FP8_MISSING
    if _B12X_BLOCK_FP8 is not None:
        return _B12X_BLOCK_FP8
    if _B12X_BLOCK_FP8_MISSING:
        return None
    try:
        _B12X_BLOCK_FP8 = importlib.import_module("b12x.gemm.block_fp8_linear")
    except ImportError:
        _B12X_BLOCK_FP8_MISSING = True
        return None
    return _B12X_BLOCK_FP8


def _current_linear_backend() -> str:
    vllm_config = get_current_vllm_config_or_none()
    if vllm_config is None:
        return "auto"
    return str(getattr(vllm_config.kernel_config, "linear_backend", "auto")).lower()


@torch.compiler.assume_constant_result
def _resolve_layer_name(layer_name: str | LayerName) -> str:
    from torch._library.fake_class_registry import FakeScriptObject

    if isinstance(layer_name, LayerName):
        return layer_name.value
    elif isinstance(layer_name, FakeScriptObject):
        return layer_name.real_obj.value
    return layer_name


def _register_b12x_fp8_block_scaled_linear_layer(layer: torch.nn.Module) -> None:
    prefix = getattr(layer, "prefix", "")
    if not prefix:
        return
    vllm_config = get_current_vllm_config_or_none()
    if vllm_config is None:
        return
    static_forward_context = vllm_config.compilation_config.static_forward_context
    existing = static_forward_context.get(prefix)
    if existing is not None and existing is not layer:
        raise ValueError(f"Duplicate B12X FP8 linear layer name: {prefix}")
    static_forward_context[prefix] = layer


def _run_b12x_fp8_block_scaled_linear(
    input_2d: torch.Tensor,
    packed_weight: Any,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    block_fp8 = _import_b12x_block_fp8()
    if block_fp8 is None:
        raise ImportError("b12x.gemm.block_fp8_linear is not importable")

    tokens = int(input_2d.shape[0])
    out_features = int(packed_weight.out_features)
    if tokens == 0:
        return input_2d.new_empty((0, out_features))
    # Functional call: block_fp8_linear_mxfp8 allocates + returns its own output,
    # so no caller-owned view is mutated by a custom op in the compile graph
    # (which inductor's decompose_auto_functionalized pass cannot remove). No
    # plan/scratch/binding needed.
    return block_fp8.block_fp8_linear_mxfp8(
        source=input_2d,
        packed_weight=packed_weight,
        bias=bias,
        expected_m=tokens,
        stream=current_stream().cuda_stream,
    )


def _b12x_dg_prefill_min_tokens() -> int:
    raw = os.getenv("VLLM_B12X_FP8_LINEAR_DG_PREFILL_MIN_TOKENS")
    if raw is None:
        return _B12X_DG_PREFILL_DEFAULT_MIN_TOKENS
    return int(raw)


def _apply_b12x_fp8_block_scaled_linear(
    layer: torch.nn.Module,
    x: torch.Tensor,
    bias: torch.Tensor | None,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    dg_kernel = getattr(layer, "b12x_dg_prefill_kernel", None)
    if (
        dg_kernel is not None
        and x.numel() // x.shape[-1] >= _b12x_dg_prefill_min_tokens()
    ):
        # Prefill-regime token counts run the DeepGEMM path on the
        # dg-processed copy of the weights; decode/small batches stay on the
        # b12x MXFP8 dense GEMM, which wins below the crossover.
        return dg_kernel.apply_weights(layer, x, bias)
    packed_weight = getattr(layer, "b12x_packed_weight", None)
    if packed_weight is None:
        raise RuntimeError(
            "b12x FP8 packed weights are missing; process_weights_after_loading "
            "did not run for this layer"
        )
    out_features = int(packed_weight.out_features)
    input_2d = x.reshape(-1, x.shape[-1]).contiguous()
    output_shape = [*x.shape[:-1], out_features]
    if input_2d.dtype != output_dtype:
        raise RuntimeError(
            "b12x FP8 linear currently expects input and output dtype to "
            f"match, got input={input_2d.dtype}, output={output_dtype}"
        )
    output = _run_b12x_fp8_block_scaled_linear(
        input_2d,
        packed_weight,
        bias,
    )
    return output.view(*output_shape)


def _b12x_fp8_block_scaled_linear(
    x: torch.Tensor,
    bias: torch.Tensor | None,
    layer_name: _layer_name_type,
    out_features: int,
) -> torch.Tensor:
    del out_features
    layer = get_forward_context().no_compile_layers[_resolve_layer_name(layer_name)]
    return _apply_b12x_fp8_block_scaled_linear(layer, x, bias, x.dtype)


def _b12x_fp8_block_scaled_linear_fake(
    x: torch.Tensor,
    bias: torch.Tensor | None,
    layer_name: _layer_name_type,
    out_features: int,
) -> torch.Tensor:
    del bias, layer_name
    return x.new_empty((*x.shape[:-1], out_features))


direct_register_custom_op(
    op_name="b12x_fp8_block_scaled_linear",
    op_func=_b12x_fp8_block_scaled_linear,
    fake_impl=_b12x_fp8_block_scaled_linear_fake,
    tags=(torch.Tag.needs_fixed_stride_order,),
)


class B12xFp8BlockScaledMMKernel(Fp8BlockScaledMMLinearKernel):
    """Block-FP8 linear through the native b12x SM120 MXFP8 GEMM path.

    When ``VLLM_B12X_FP8_LINEAR_DG_PREFILL`` is enabled (default), layers also
    keep a DeepGEMM-processed copy of the weights and route token batches at
    or above the prefill crossover to the DeepGEMM fp8 GEMM, where it is
    faster than the b12x MXFP8 dense GEMM. Decode and small batches stay on
    the b12x path. Memory cost is unchanged: the b12x pack always duplicates
    the weight, and the DeepGEMM processing replaces the original copy
    in place.
    """

    def __init__(self, config: FP8ScaledMMLinearLayerConfig):
        super().__init__(config)
        self._dg_prefill_kernel: DeepGemmFp8BlockScaledMMKernel | None = None
        if os.getenv("VLLM_B12X_FP8_LINEAR_DG_PREFILL", "1") == "1":
            dg_supported, _ = DeepGemmFp8BlockScaledMMKernel.is_supported()
            if dg_supported:
                dg_can_implement, _ = DeepGemmFp8BlockScaledMMKernel.can_implement(
                    config
                )
                if dg_can_implement:
                    self._dg_prefill_kernel = DeepGemmFp8BlockScaledMMKernel(config)

    @classmethod
    def is_supported(
        cls,
        compute_capability: int | None = None,
    ) -> tuple[bool, str | None]:
        del compute_capability
        if not current_platform.is_cuda():
            return False, "b12x FP8 kernels are only available on CUDA"
        if not current_platform.is_device_capability_family(120):
            return False, "b12x FP8 kernels require a Blackwell 12x device"
        if _import_b12x_block_fp8() is None:
            return False, "b12x.gemm.block_fp8_linear is not importable"
        return True, None

    @classmethod
    def can_implement(
        cls,
        config: FP8ScaledMMLinearLayerConfig,
    ) -> tuple[bool, str | None]:
        can_implement_base, reason = super().can_implement(config)
        if not can_implement_base:
            return can_implement_base, reason

        if _current_linear_backend() != "b12x" and not envs.VLLM_USE_B12X_FP8_GEMM:
            return False, "b12x FP8 GEMM is not enabled"

        if config.out_dtype not in (torch.bfloat16, torch.float16):
            return False, "Supports only bf16/fp16 output dtype"

        act_quant_desc = config.activation_quant_key.scale
        if act_quant_desc.group_shape != GroupShape(1, 128):
            return (
                False,
                "Supports only dynamic per-token group activation "
                "quantization with group_shape=(1,128).",
            )

        weight_group_shape = config.weight_quant_key.scale.group_shape
        if weight_group_shape != GroupShape(128, 128):
            return False, "Supports only 128x128 block-scaled FP8 weights"

        out_features, in_features = config.weight_shape
        if in_features % 128 != 0 or out_features <= 0:
            return False, "Input features must be a positive multiple of 128"
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if getattr(layer, "b12x_skip_generic_block_fp8_linear", False):
            layer.b12x_packed_weight = None
            return

        params = self._get_layer_params(layer)
        weight_scale = (
            params.weight_scale
            if params.weight_scale_inv is None
            else params.weight_scale_inv
        )
        if weight_scale is None:
            raise ValueError("b12x FP8 linear requires block weight scales")
        assert layer.weight_block_size is not None

        block_fp8 = _import_b12x_block_fp8()
        if block_fp8 is None:
            raise ImportError("b12x.gemm.block_fp8_linear is not importable")
        layer.b12x_packed_weight = block_fp8.pack_block_fp8_linear_weight_mxfp8(
            params.weight.detach(),
            weight_scale.detach(),
            block_size=tuple(layer.weight_block_size),
        )
        _register_b12x_fp8_block_scaled_linear_layer(layer)
        if self._dg_prefill_kernel is not None:
            # Must run after the b12x pack: DeepGEMM processing replaces the
            # original checkpoint-layout weight/scale parameters in place.
            self._dg_prefill_kernel.process_weights_after_loading(layer)
            layer.b12x_dg_prefill_kernel = self._dg_prefill_kernel

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        del kwargs
        if getattr(layer, "b12x_skip_generic_block_fp8_linear", False):
            raise RuntimeError(
                "b12x generic FP8 linear was called for a layer owned by a fused "
                "b12x path"
            )
        packed_weight = getattr(layer, "b12x_packed_weight", None)
        if packed_weight is None:
            raise RuntimeError(
                "b12x FP8 packed weights are missing; process_weights_after_loading "
                "did not run for this layer"
            )
        out_features = int(packed_weight.out_features)
        if x.dtype != self.config.out_dtype:
            raise RuntimeError(
                "b12x FP8 linear currently expects input and output dtype to "
                f"match, got input={x.dtype}, output={self.config.out_dtype}"
            )
        if torch.compiler.is_compiling():
            prefix = getattr(layer, "prefix", "")
            if not prefix:
                raise RuntimeError(
                    "B12X FP8 linear requires a layer prefix under torch.compile"
                )
            return torch.ops.vllm.b12x_fp8_block_scaled_linear(
                x,
                bias,
                _encode_layer_name(prefix),
                out_features,
            )
        return _apply_b12x_fp8_block_scaled_linear(
            layer,
            x,
            bias,
            self.config.out_dtype,
        )

    def apply_block_scaled_mm(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        As: torch.Tensor,
        Bs: torch.Tensor,
    ) -> torch.Tensor:
        del A, B, As, Bs
        raise NotImplementedError("b12x FP8 linear overrides apply_weights")
