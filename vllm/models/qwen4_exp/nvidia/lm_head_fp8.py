# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Online fp8 weight-only (w8a16) quantization for the qwen4_exp lm_head.

The Qwen3.8-Flash-Next checkpoint ships its lm_head in bf16 (the NVFP4
recipe exclude-lists it), so every full-vocab logits projection streams
``vocab_size x hidden_size x 2`` bytes.  With MTP speculative decoding the
shared head is read three times per decode step (two draft passes plus one
verify pass), making it the single largest weight stream of the step.
Quantizing the weight to fp8 at load time halves that traffic while keeping
bf16 activations and accumulation (w8a16).

Opt-in via ``--hf-overrides '{"lm_head_quant": "fp8"}'`` (resolved by
:func:`get_lm_head_quant_method`); default-off behavior is byte-identical to
stock.  The method follows the online-quantization house pattern
(``quantization/online/fp8.py``): the weight is created on the meta device,
materialized while loading, and quantized in ``process_weights_after_loading``
with an amax-derived per-tensor scale.  The GEMM runs through
``init_wfp8_a16_linear_kernel`` (Marlin/Humming weight-only path).
"""

from __future__ import annotations

import torch
from torch.nn import Module

from vllm import _custom_ops as ops
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.model_executor.kernels.linear import init_wfp8_a16_linear_kernel
from vllm.model_executor.kernels.linear.scaled_mm.ScaledMMLinearKernel import (
    FP8ScaledMMLinearKernel,
)
from vllm.model_executor.layers.linear import LinearMethodBase
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    get_fp8_min_max,
    kFp8DynamicTensorSym,
    kFp8StaticTensorSym,
    weight_amax,
)
from vllm.model_executor.model_loader.reload.layerwise import (
    initialize_online_processing,
)
from vllm.model_executor.parameter import ModelWeightParameter
from vllm.model_executor.utils import replace_parameter
from vllm.platforms import current_platform

LM_HEAD_QUANT_ATTR = "lm_head_quant"
_SUPPORTED_LM_HEAD_QUANT = ("fp8",)


class Qwen4ExpLMHeadFp8Method(LinearMethodBase):
    """Online per-tensor fp8 weight-only method for ``ParallelLMHead``.

    ``ParallelLMHead`` is exempt from the embedding-method requirement (its
    ``forward`` is never called), so a Linear-shaped method is a legal
    ``quant_method`` for it; ``LogitsProcessor`` reaches it through
    ``quant_method.apply``.
    """

    uses_meta_device: bool = True

    def __init__(self) -> None:
        self.out_dtype = torch.get_default_dtype()
        self.input_dtype = get_current_vllm_config().model_config.dtype
        self.linear_kernel: FP8ScaledMMLinearKernel | None = None

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        if getattr(layer, "tp_size", 1) != 1:
            raise NotImplementedError(
                "lm_head_quant='fp8' computes its per-tensor scale from the "
                "local vocab shard; with tensor parallelism each rank would "
                "derive a different scale. Run with TP=1 or drop the override."
            )
        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")
        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.orig_dtype = params_dtype
        layer.weight_block_size = None

        weight = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition,
                device="meta",  # materialized and processed during loading
                dtype=params_dtype,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)

        initialize_online_processing(layer)

        # The weight-only kernels require CUDA; without it (CPU-only unit
        # tests) `apply` falls back to dequant + F.linear.
        if current_platform.is_cuda():
            self.linear_kernel = init_wfp8_a16_linear_kernel(
                weight_quant_key=kFp8StaticTensorSym,
                activation_quant_key=kFp8DynamicTensorSym,
                weight_shape=layer.weight.shape,
                input_dtype=self.input_dtype,
                out_dtype=self.out_dtype,
                module_name=self.__class__.__name__,
            )

    def process_weights_after_loading(self, layer: Module) -> None:
        if getattr(layer, "_already_called_process_weights_after_loading", False):
            return

        layer.input_scale = None
        amax = weight_amax(layer.weight).reshape(1)
        weight_scale = amax.to(torch.float32) / get_fp8_min_max()[1]
        if layer.weight.is_cuda:
            qweight, _ = ops.scaled_fp8_quant(layer.weight, scale=weight_scale)
        else:
            fp8_min, fp8_max = get_fp8_min_max()
            qweight = (
                (layer.weight.to(torch.float32) / weight_scale)
                .clamp_(fp8_min, fp8_max)
                .to(current_platform.fp8_dtype())
            )

        # Canonicalize to (K, N); the Marlin prep asserts this layout. `.t()`
        # is a non-contiguous view (stride(-1) == input_size, not 1), so tag
        # the resulting parameter's dims the way
        # CompressedTensorsW8A16Fp8.process_weights_after_loading does: it is
        # those tags, not an eager `.contiguous()` here, that tell Humming's
        # convert_linear_layer_to_humming_standard() to transpose-and-copy
        # the weight back to a contiguous layout before its view(int32)
        # reinterpret cast, which requires a contiguous last dim. Marlin's
        # prep reads (K, N) directly and tolerates the non-contiguous view,
        # which is why only the Humming path surfaced this.
        replace_parameter(layer, "weight", qweight.t().data)
        layer.weight.input_dim = 0
        layer.weight.output_dim = 1
        replace_parameter(layer, "weight_scale", weight_scale.data)

        if self.linear_kernel is not None and layer.weight.is_cuda:
            self.linear_kernel.process_weights_after_loading(layer)
        else:
            # CPU weights (unit tests): the CUDA-only kernel cannot prep or
            # apply them; drop it so `apply` uses the dequant fallback.
            self.linear_kernel = None

        # Prevent duplicate processing (e.g., during weight reload).
        layer._already_called_process_weights_after_loading = True

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.linear_kernel is not None:
            return self.linear_kernel.apply_weights(layer, x, bias)
        # CPU/unit-test fallback: dequantize and run a plain linear. The
        # weight is stored (K, N) after process_weights_after_loading.
        weight = layer.weight.to(x.dtype) * layer.weight_scale.to(x.dtype)
        return torch.nn.functional.linear(x, weight.t(), bias)


def get_lm_head_quant_method(
    vllm_config: VllmConfig,
) -> Qwen4ExpLMHeadFp8Method | None:
    """Resolve the opt-in lm_head quantization for qwen4_exp models.

    Reads ``lm_head_quant`` from the HF config (settable via
    ``--hf-overrides '{"lm_head_quant": "fp8"}'``, mirroring ``head_dtype``).
    Returns ``None`` (stock unquantized behavior) when unset.

    Raises:
        ValueError: on an unsupported ``lm_head_quant`` value, or when a
            non-model ``head_dtype`` override is also set (the fp32-head path
            in ``LogitsProcessor`` requires an unquantized lm_head).
        NotImplementedError: with tied word embeddings (quantizing the shared
            table would also change the input embedding lookup).
    """
    model_config = vllm_config.model_config
    requested = getattr(model_config.hf_config, LM_HEAD_QUANT_ATTR, None)
    if requested is None:
        requested = getattr(model_config.hf_text_config, LM_HEAD_QUANT_ATTR, None)
    if requested is None:
        return None
    if requested not in _SUPPORTED_LM_HEAD_QUANT:
        raise ValueError(
            f"Unsupported {LM_HEAD_QUANT_ATTR}={requested!r}; supported "
            f"values: {_SUPPORTED_LM_HEAD_QUANT}."
        )
    if getattr(model_config.hf_text_config, "tie_word_embeddings", False):
        raise NotImplementedError(
            f"{LM_HEAD_QUANT_ATTR}={requested!r} is not supported with "
            "tie_word_embeddings: the lm_head shares its table with the "
            "input embedding, which must stay unquantized."
        )
    if model_config.head_dtype != model_config.dtype:
        raise ValueError(
            f"{LM_HEAD_QUANT_ATTR}={requested!r} conflicts with a "
            "head_dtype override: LogitsProcessor's non-model head_dtype "
            "path requires an unquantized lm_head. Drop one of the two "
            "overrides."
        )
    return Qwen4ExpLMHeadFp8Method()
