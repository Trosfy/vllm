# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Online fp8 weight-only (w8a16) quantization for the qwen4_exp dense path.

The Qwen3.8-Flash-Next NVFP4 recipe exclude-lists every projection that sits
outside the routed experts, so the GDN input/output projections, the QSA
output projection and the shared-expert MLP all stream bf16 weights on every
decode step (~5 GB/step from the decode shape census).  Quantizing those
weights to fp8 at load time halves that traffic while keeping bf16 activations
and accumulation (w8a16); the routed experts keep their NVFP4 method.

Opt-in via ``--hf-overrides '{"dense_quant": "fp8"}'``; default-off behavior is
byte-identical to stock because the wrapper is never constructed.
:class:`Qwen4ExpDenseFp8Config` wraps whatever quantization config the
checkpoint already selected and re-routes only the allow-listed prefixes to
:class:`Qwen4ExpDenseFp8LinearMethod`; every other layer is delegated to the
wrapped config unchanged.  The swap of ``vllm_config.quant_config`` is scoped
to model construction (:func:`dense_fp8_quant_config`) because the GDN layers
read the config off ``vllm_config`` rather than taking it as an argument.

The method follows the online-quantization house pattern
(``quantization/online/fp8.py``, ``lm_head_fp8.py``): the weight is created on
the meta device, materialized while loading, and quantized in
``process_weights_after_loading`` with an amax-derived per-tensor scale.  The
GEMM runs through ``init_wfp8_a16_linear_kernel`` (Marlin/Humming weight-only
path).

Tensor parallelism is rejected fail-closed: the per-tensor scale is derived
from the local shard, so each rank would derive a different one.  The
match-count guard likewise assumes single-worker (TP=1, PP=1) construction.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from fnmatch import fnmatch
from typing import Any

import torch
from torch.nn import Module

from vllm import _custom_ops as ops
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.model_executor.kernels.linear import init_wfp8_a16_linear_kernel
from vllm.model_executor.kernels.linear.scaled_mm.ScaledMMLinearKernel import (
    FP8ScaledMMLinearKernel,
)
from vllm.model_executor.layers.linear import (
    LinearBase,
    LinearMethodBase,
    UnquantizedLinearMethod,
    register_weight_loader_v2_supported_method,
)
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    get_fp8_min_max,
    is_layer_skipped,
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

DENSE_QUANT_ATTR = "dense_quant"
_SUPPORTED_DENSE_QUANT = ("fp8",)

# Allow-listed dense projections as fnmatch patterns over the vLLM module
# prefix, sized from the decode-step shape census. Deliberately excluded:
# ``in_proj_ba`` (17 MB/step, not worth the Marlin MIN_THREAD_N exposure),
# ``conv1d`` (never receives a quant_config), the MTP draft layer, embeddings,
# norms, the routed experts and the PLE table.
DENSE_FP8_PATTERNS: tuple[str, ...] = (
    "*.linear_attn.in_proj_qkvz",
    "*.linear_attn.out_proj",
    "*.self_attn.qkv_proj",
    "*.self_attn.o_proj",
    "*.mlp.shared_expert.gate_up_proj",
    "*.mlp.shared_expert.down_proj",
)

# 36 GDN layers x (in_proj_qkvz + out_proj) + 12 QSA layers x (qkv_proj +
# o_proj) + 48 MoE layers x (shared_expert.gate_up_proj + .down_proj).
DENSE_FP8_EXPECTED_MATCHES = 192


def _per_tensor_fp8_scale(weight: torch.Tensor) -> torch.Tensor:
    """Derive the per-tensor fp8 scale from a weight's amax.

    Args:
        weight: Materialized bf16/fp16 weight.

    Returns:
        A one-element fp32 tensor holding ``amax / fp8_max``.
    """
    amax = weight_amax(weight).reshape(1)
    return amax.to(torch.float32) / get_fp8_min_max()[1]


def _quantize_to_fp8(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Quantize ``weight`` to fp8 with a per-tensor ``scale``.

    Args:
        weight: Materialized bf16/fp16 weight.
        scale: One-element fp32 scale from :func:`_per_tensor_fp8_scale`.

    Returns:
        The fp8 weight in the same (N, K) layout as the input.
    """
    if weight.is_cuda:
        qweight, _ = ops.scaled_fp8_quant(weight, scale=scale)
        return qweight
    # CPU (unit tests): no scaled_fp8_quant kernel, so cast explicitly.
    fp8_min, fp8_max = get_fp8_min_max()
    return (
        (weight.to(torch.float32) / scale)
        .clamp_(fp8_min, fp8_max)
        .to(current_platform.fp8_dtype())
    )


def _dequant_linear(
    layer: Module,
    x: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    """Dequantize-and-linear fallback used when no weight-only kernel exists.

    Args:
        layer: Layer holding the processed ``weight`` (K, N) and
            ``weight_scale``.
        x: Activations, shape (..., K).
        bias: Optional bias.

    Returns:
        The linear output, shape (..., N).
    """
    weight = layer.weight.to(x.dtype) * layer.weight_scale.to(x.dtype)
    return torch.nn.functional.linear(x, weight.t(), bias)


@register_weight_loader_v2_supported_method
class Qwen4ExpDenseFp8LinearMethod(LinearMethodBase):
    """Online per-tensor fp8 weight-only method for dense ``LinearBase`` layers.

    Registered for ``weight_loader_v2`` so ``MergedColumnParallelLinear`` and
    ``QKVParallelLinear`` narrow each checkpoint shard into the single flat
    weight through ``BasevLLMParameter.load_merged_column_weight`` /
    ``load_qkv_weight``.  ``layer.logical_widths`` is kept plural so the Marlin
    prep can expand the per-tensor scale across the merged shards.
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
                f"{DENSE_QUANT_ATTR}='fp8' computes its per-tensor scale from "
                "the local weight shard; with tensor parallelism each rank "
                "would derive a different scale. Run with TP=1 or drop the "
                "override."
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
        weight_scale = _per_tensor_fp8_scale(layer.weight)
        qweight = _quantize_to_fp8(layer.weight, weight_scale)

        # Canonicalize to (K, N); the Marlin prep asserts this layout. `.t()`
        # is a non-contiguous view (stride(-1) == input_size, not 1) even
        # when `qweight` itself is contiguous (including after a merged- or
        # QKV-shard load), so tag the resulting parameter's dims the way
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
        return _dequant_linear(layer, x, bias)


class Qwen4ExpDenseFp8Config(QuantizationConfig):
    """Construction-scoped wrapper adding online fp8 w8a16 to dense layers.

    Every lookup that does not hit the allow-list is delegated to the wrapped
    config, so the routed experts, the KV-cache scales and every excluded
    projection behave exactly as they do without the override.  Attribute
    reads that the wrapper does not define fall through to the wrapped config
    as well, which keeps name- and attribute-driven consumers (for example
    ``model.without_modelopt_fp4`` and the block-shape check in ``LinearBase``)
    on their stock code paths.

    Args:
        inner: The config the checkpoint selected, or ``None`` for an
            unquantized checkpoint.
        packed_modules_mapping: The model's fused-module mapping, used for
            fused-name resolution while matching. Defaults to the wrapped
            config's mapping.
        patterns: fnmatch patterns naming the allow-listed projections.
        expected_matches: Number of layers the allow-list must match during
            construction; :meth:`validate_match_count` enforces it.
    """

    def __init__(
        self,
        inner: QuantizationConfig | None,
        *,
        packed_modules_mapping: Mapping[str, list[str]] | None = None,
        patterns: Sequence[str] = DENSE_FP8_PATTERNS,
        expected_matches: int = DENSE_FP8_EXPECTED_MATCHES,
    ) -> None:
        super().__init__()
        self.inner = inner
        self.patterns = list(patterns)
        self.expected_matches = expected_matches
        self.match_counts: dict[str, int] = dict.fromkeys(self.patterns, 0)
        if packed_modules_mapping is not None:
            self.packed_modules_mapping = dict(packed_modules_mapping)
        elif inner is not None:
            self.packed_modules_mapping = inner.packed_modules_mapping

    def __getattr__(self, name: str) -> Any:
        inner = self.__dict__.get("inner")
        if inner is None:
            raise AttributeError(name)
        return getattr(inner, name)

    def get_name(self) -> QuantizationMethods:
        # Delegating keeps name-driven consumers (notably
        # `model.without_modelopt_fp4`) on their stock branch.
        return self.inner.get_name() if self.inner is not None else "fp8"

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        if self.inner is not None:
            return self.inner.get_supported_act_dtypes()
        return [torch.bfloat16, torch.half]

    @classmethod
    def get_min_capability(cls) -> int:
        # Marlin weight-only fp8: Turing and up.
        return 75

    @staticmethod
    def get_config_filenames() -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Qwen4ExpDenseFp8Config:
        raise NotImplementedError(
            f"{cls.__name__} wraps an already-resolved quantization config and "
            "is only reachable through the "
            f'--hf-overrides \'{{"{DENSE_QUANT_ATTR}": "fp8"}}\' opt-in.'
        )

    def matched_pattern(self, prefix: str) -> str | None:
        """Return the allow-list pattern matching ``prefix``, if any.

        Matching mirrors ``ModelOptQuantConfigBase.is_layer_excluded``: the
        shared :func:`is_layer_skipped` first, so fused names resolve the same
        way the ModelOpt exclude list does, then fnmatch for the wildcards.

        Args:
            prefix: Full module prefix, e.g. ``model.layers.0.linear_attn.out_proj``.

        Returns:
            The matching pattern, or ``None`` when the layer is not
            allow-listed.
        """
        for pattern in self.patterns:
            if is_layer_skipped(
                prefix, [pattern], self.packed_modules_mapping
            ) or fnmatch(prefix, pattern):
                return pattern
        return None

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> QuantizeMethodBase | None:
        is_linear = isinstance(layer, LinearBase)
        if is_linear:
            pattern = self.matched_pattern(prefix)
            if pattern is not None:
                self.match_counts[pattern] += 1
                return Qwen4ExpDenseFp8LinearMethod()
        if self.inner is None:
            # Unquantized checkpoint: without the override these layers would
            # see ``quant_config=None``, and LinearBase rejects a ``None``
            # method, so the stock fallback has to be returned explicitly.
            return UnquantizedLinearMethod() if is_linear else None
        return self.inner.get_quant_method(layer, prefix)

    def validate_match_count(self) -> None:
        """Fail construction when the allow-list did not match as expected.

        Raises:
            ValueError: If the number of allow-listed layers built differs
                from ``expected_matches``. A typo in a pattern, a checkpoint
                with a different layer mix, or a multi-worker (TP/PP) split
                all land here rather than silently serving unquantized.
        """
        total = sum(self.match_counts.values())
        if total != self.expected_matches:
            breakdown = ", ".join(
                f"{pattern}={count}" for pattern, count in self.match_counts.items()
            )
            raise ValueError(
                f"{DENSE_QUANT_ATTR}='fp8' matched {total} layers but expected "
                f"{self.expected_matches} ({breakdown}). The allow-list, the "
                "checkpoint layer mix, or the TP/PP world size does not match "
                "what this override was built for."
            )


def maybe_dense_fp8(
    quant_config: QuantizationConfig | None,
    prefix: str,
) -> QuantizationConfig | None:
    """Quant config for the QSA ``qkv_proj``, which ModelOpt-FP4 excludes.

    ``qkv_proj`` is the one allow-listed projection whose call site hard-codes
    ``without_modelopt_fp4``, so it can only join the dense-fp8 path through
    this helper. One helper, both platform trees.

    Args:
        quant_config: The config in scope at the call site.
        prefix: The projection's full module prefix.

    Returns:
        The dense-fp8 wrapper when it is active *and* allow-lists ``prefix``;
        otherwise exactly what ``model.without_modelopt_fp4`` returns (kept in
        sync by hand rather than imported, since ``model`` imports this module).
    """
    if (
        isinstance(quant_config, Qwen4ExpDenseFp8Config)
        and quant_config.matched_pattern(prefix) is not None
    ):
        return quant_config
    if quant_config is not None and quant_config.get_name() == "modelopt_fp4":
        return None
    return quant_config


def get_dense_quant(vllm_config: VllmConfig) -> str | None:
    """Read and validate the opt-in ``dense_quant`` HF override.

    Args:
        vllm_config: The config under construction.

    Returns:
        The requested dense quantization, or ``None`` when unset (stock
        behavior).

    Raises:
        ValueError: On an unsupported ``dense_quant`` value.
    """
    model_config = vllm_config.model_config
    requested = getattr(model_config.hf_config, DENSE_QUANT_ATTR, None)
    if requested is None:
        requested = getattr(model_config.hf_text_config, DENSE_QUANT_ATTR, None)
    if requested is None:
        return None
    if requested not in _SUPPORTED_DENSE_QUANT:
        raise ValueError(
            f"Unsupported {DENSE_QUANT_ATTR}={requested!r}; supported values: "
            f"{_SUPPORTED_DENSE_QUANT}."
        )
    return requested


@contextmanager
def dense_fp8_quant_config(
    vllm_config: VllmConfig,
    packed_modules_mapping: Mapping[str, list[str]] | None = None,
    *,
    patterns: Sequence[str] = DENSE_FP8_PATTERNS,
    expected_matches: int = DENSE_FP8_EXPECTED_MATCHES,
) -> Iterator[Qwen4ExpDenseFp8Config | None]:
    """Swap ``vllm_config.quant_config`` for the dense-fp8 wrapper, scoped.

    The swap lasts only for the duration of the ``with`` body, because the GDN
    layers read ``vllm_config.quant_config`` at layer init while post-init
    readers (KV-scale loading, RL refit, logging) must see the original config.
    Layers constructed inside the body keep their own method instances, so
    restoring the config does not undo the quantization.

    Args:
        vllm_config: The config whose ``quant_config`` is temporarily swapped.
        packed_modules_mapping: The model's fused-module mapping.
        patterns: fnmatch patterns naming the allow-listed projections.
        expected_matches: Expected allow-list match count.

    Yields:
        The wrapper when the override is active, otherwise ``None`` (in which
        case ``vllm_config`` is left untouched).
    """
    if get_dense_quant(vllm_config) is None:
        yield None
        return

    wrapped = Qwen4ExpDenseFp8Config(
        vllm_config.quant_config,
        packed_modules_mapping=packed_modules_mapping,
        patterns=patterns,
        expected_matches=expected_matches,
    )
    original = vllm_config.quant_config
    vllm_config.quant_config = wrapped
    try:
        yield wrapped
    finally:
        vllm_config.quant_config = original
