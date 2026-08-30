# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from copy import deepcopy
from typing import Any

import torch
from safetensors.torch import _TYPES as _SAFETENSORS_TO_TORCH_DTYPE
from transformers import PretrainedConfig

import vllm.model_executor.layers.fused_moe  # noqa
from vllm.logger import init_logger
from vllm.model_executor.kernels.linear import (
    MPLinearLayerConfig,
    choose_mp_linear_kernel,
)
from vllm.model_executor.layers.fused_moe import (
    FusedMoEConfig,
    FusedMoEExpertsModular,
    FusedMoEMethodBase,
    FusedMoEQuantConfig,
    FusedMoeWeightScaleSupported,
    RoutedExperts,
    SharedExperts,
    UnquantizedFusedMoEMethod,
)
from vllm.model_executor.layers.fused_moe.oracle.int_wna16 import (
    WNA16MoEBackend,
    convert_to_wna16_moe_kernel_format,
    make_wna16_moe_kernel,
    select_wna16_moe_backend,
)
from vllm.model_executor.layers.linear import (
    LinearBase,
    LinearMethodBase,
    set_weight_attrs,
)
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.layers.quantization.fp8 import Fp8Config
from vllm.model_executor.layers.quantization.utils import replace_parameter
from vllm.model_executor.layers.quantization.utils.gptq_utils import (
    get_dynamic_override,
    get_linear_quant_method,
    override_config,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    check_moe_marlin_supports_layer,
    get_marlin_input_dtype,
    marlin_make_workspace_new,
    marlin_repeat_scales_on_all_ranks,
    verify_marlin_supported,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kInt4StaticGroupScale,
    kInt8StaticGroupScale,
)
from vllm.model_executor.parameter import (
    ChannelQuantScaleParameter,
    GroupQuantScaleParameter,
    PackedColumnParameter,
    PackedvLLMParameter,
    RowvLLMParameter,
)
from vllm.scalar_type import scalar_types
from vllm.transformers_utils.config import get_safetensors_params_metadata
from vllm.utils.collection_utils import is_list_of

logger = init_logger(__name__)

# HF override attribute names read by
# vllm/models/qwen4_exp/nvidia/dense_fp8.py (DENSE_QUANT_ATTR) and
# lm_head_fp8.py (LM_HEAD_QUANT_ATTR). Checked here by plain attribute name,
# not imported: AutoGPTQConfig is a model-agnostic quantization method and
# must not depend on a specific model package. Both override paths claim
# their allow-listed prefixes with an unquantized bf16 method
# (dense_fp8.py:360-374) before the checkpoint's own fp8 weights ever reach a
# quant_method, so they are mutually exclusive with the hybrid fp8 dispatch
# below.
_QWEN4_EXP_DENSE_FP8_OVERRIDE_ATTRS = ("dense_quant", "lm_head_quant")


def _fp8_hybrid_enabled() -> bool:
    """Whether the env-gated int4+fp8 hybrid dispatch is on (default off)."""
    return os.environ.get("VLLM_FP8_HYBRID", "0").lower() in ("1", "true", "yes")


def _active_dense_fp8_override(hf_config: PretrainedConfig | None) -> str | None:
    """Name of the first active qwen4_exp dense/lm_head fp8 override, if any."""
    for attr in _QWEN4_EXP_DENSE_FP8_OVERRIDE_ATTRS:
        if getattr(hf_config, attr, None) is not None:
            return attr
    return None


def get_moe_quant_method(
    config: "AutoGPTQConfig",
    layer: RoutedExperts,
    prefix: str,
    moe_method_cls: type,
):
    cloned_config = deepcopy(config)

    assert isinstance(layer, RoutedExperts)
    # False = skip module, None = no override, else = Positive match
    if (
        get_dynamic_override(  # noqa: E712
            cloned_config,  # noqa: E712
            layer_name=prefix,
        )
        == False
    ):  # noqa: E712
        return UnquantizedFusedMoEMethod(layer.moe_config)

    if prefix:
        # Dynamic per module/layer rules may override base config
        override_config(cloned_config, prefix=prefix)

    return moe_method_cls(cloned_config, layer.moe_config)


class AutoGPTQConfig(QuantizationConfig):
    """Config class for AutoGPTQ quantization using Marlin kernels."""

    # (num_bits, is_sym) -> quant_type
    TYPE_MAP = {
        (4, True): scalar_types.uint4b8,
        (8, True): scalar_types.uint8b128,
    }

    def __init__(
        self,
        weight_bits: int,
        group_size: int,
        desc_act: bool,
        is_sym: bool,
        lm_head_quantized: bool,
        dynamic: dict[str, dict[str, int | bool]],
        full_config: dict[str, Any],
        modules_in_block_to_quantize: list[str] | None = None,
    ) -> None:
        super().__init__()
        if desc_act and group_size == -1:
            # In this case, act_order == True is the same as act_order == False
            # (since we have only one group per output channel)
            desc_act = False

        # GPTQModel use `dynamic` config property to allow per module
        # quantization config so each module can be individually optimized.
        # Format is dict[str, dict] where key is a regex string that can
        # perform both positive ("+:" prefixed) or negative ("-:" prefixed)
        # matching of a module.
        # Default to positive match, override base quant config mode, if no
        # prefix is used. Value is in dict format of field key and override
        # value.
        # Negative matching will skip quantization init for this module
        # entirely:
        # non-quantized inference. More details and quantization examples can be
        # found at: https://github.com/ModelCloud/GPTQModel
        # Example:
        #  # last 1/2 of the layers 10-21 has 8bit vs 4bit for 0-9
        #  # last 1/4 of the layers 16-21 has 8bit and group_size 64
        # dynamic = {
        #  #`.*\.` matches the layers_node prefix
        #  # positive match layer 10-15
        #  r"+:.*\.(?:1[0-5])\..*": {"bits": 8,},
        #  # positive match layer 16-21
        #  r"+:.*\.(?:1[6-9]|20|21)\..*": {"bits": 8, "group_size": 64,},
        #  r"-:.*\.moe\..*": {}, # negative match (skip) all `moe` layers
        # }
        self.dynamic = dynamic

        self.weight_bits = weight_bits
        self.is_sym = is_sym

        self.pack_factor = 32 // weight_bits  # packed into int32
        self.group_size = group_size
        self.desc_act = desc_act
        self.lm_head_quantized = lm_head_quantized
        self.full_config = full_config

        if (weight_bits, is_sym) not in self.TYPE_MAP:
            raise ValueError(
                f"Unsupported quantization config: bits={weight_bits}, sym={is_sym}"
            )

        self.quant_type = self.TYPE_MAP[(weight_bits, is_sym)]

        self.modules_in_block_to_quantize = modules_in_block_to_quantize or []
        # used to identify GPTQ model quantized by autoround
        self.autoround_version = full_config.get("autoround_version", "")

        # VLLM_FP8_HYBRID (default off): layer names (vLLM-mapped) whose
        # checkpoint tensors are blockwise-fp8 (F8_E4M3 + .weight_scale_inv),
        # and the lazily-built shared Fp8Config those layers dispatch to.
        # See maybe_update_config/get_quant_method below.
        self.fp8_layers: set[str] = set()
        self._fp8_cfg: Fp8Config | None = None

    def __repr__(self) -> str:
        return (
            f"AutoGPTQConfig(quant_type={self.quant_type}, "
            f"group_size={self.group_size}, "
            f"desc_act={self.desc_act}, "
            f"lm_head_quantized={self.lm_head_quantized}, "
            f"dynamic={self.dynamic}, "
            f"modules_in_block_to_quantize={self.modules_in_block_to_quantize})"
        )

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return "auto_gptq"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.half, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 60

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return ["quantize_config.json"]

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "AutoGPTQConfig":
        dynamic = cls.get_from_keys_or(config, ["dynamic"], default={})
        dynamic = {} if dynamic is None else dynamic

        weight_bits = cls.get_from_keys(config, ["bits"])
        group_size = cls.get_from_keys(config, ["group_size"])
        desc_act = cls.get_from_keys(config, ["desc_act"])
        is_sym = cls.get_from_keys(config, ["sym"])
        lm_head_quantized = cls.get_from_keys_or(config, ["lm_head"], default=False)
        modules_in_block_to_quantize = cls.get_from_keys_or(
            config, ["modules_in_block_to_quantize"], default=None
        )
        return cls(
            weight_bits,
            group_size,
            desc_act,
            is_sym,
            lm_head_quantized,
            dynamic,
            config,
            modules_in_block_to_quantize,
        )

    @classmethod
    def override_quantization_method(
        cls, hf_quant_cfg, user_quant, hf_config=None
    ) -> QuantizationMethods | None:
        """Override to use AutoGPTQ for compatible GPTQ models."""
        quant_method = hf_quant_cfg.get("quant_method", "").lower()

        if quant_method != "gptq":
            return None

        is_valid_user_quant = user_quant is None or user_quant in (
            "gptq",
            "gptq_marlin",
            "auto_gptq",
            "marlin",
        )

        if is_valid_user_quant:
            return cls.get_name()

        return None

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> "QuantizeMethodBase | None":
        if isinstance(layer, LinearBase) and self._is_fp8_layer(prefix):
            logger.debug("fp8 hybrid: %s -> Fp8LinearMethod", prefix)
            return self._fp8_hybrid_config().get_quant_method(layer, prefix)

        if isinstance(layer, RoutedExperts):
            from vllm.model_executor.layers.quantization.moe_wna16 import MoeWNA16Config

            if not check_moe_marlin_supports_layer(
                layer, self.group_size, allow_tile_padding=not self.desc_act
            ):
                logger.warning_once(
                    f"Layer '{prefix}' is not supported by GPTQMoeMarlin. "
                    "Falling back to Moe WNA16 kernels."
                )
                return MoeWNA16Config.from_config(self.full_config).get_quant_method(
                    layer, prefix
                )
            moe_quant_method = get_moe_quant_method(
                self, layer, prefix, AutoGPTQMoEMethod
            )
            if moe_quant_method is None:
                return None
            moe_quant_method.input_dtype = get_marlin_input_dtype(prefix)
            return moe_quant_method

        quant_method = get_linear_quant_method(
            self, layer, prefix, AutoGPTQLinearMethod
        )
        if quant_method is None:
            return None
        quant_method.input_dtype = get_marlin_input_dtype(prefix)
        return quant_method

    def apply_vllm_mapper(self, hf_to_vllm_mapper):
        if self.modules_in_block_to_quantize is not None:
            self.modules_in_block_to_quantize = hf_to_vllm_mapper.apply_list(
                self.modules_in_block_to_quantize
            )
        if self.fp8_layers:
            self.fp8_layers = set(hf_to_vllm_mapper.apply_list(list(self.fp8_layers)))

    def maybe_update_config(
        self,
        model_name: str,
        hf_config: PretrainedConfig | None = None,
        revision: str | None = None,
    ):
        metadata: dict[str, Any] | None = None
        if self.modules_in_block_to_quantize:
            if is_list_of(self.modules_in_block_to_quantize, list):
                # original modules_in_block_to_quantize: list[list[str]]
                # flatten original modules_in_block_to_quantize
                self.modules_in_block_to_quantize = [
                    item
                    for sublist in self.modules_in_block_to_quantize
                    for item in sublist
                ]
        else:
            unquant_dtypes = [torch.float16, torch.bfloat16, torch.float32]
            metadata = get_safetensors_params_metadata(model_name, revision=revision)
            # Hazard (VLLM_FP8_HYBRID off): any checkpoint-serialized fp8
            # weight lands here too (F8_E4M3 is not in unquant_dtypes), so
            # modules_in_block_to_quantize includes it and the GPTQ path
            # below crashes trying to treat it as an int4/int8 GPTQ tensor.
            # Serving a hybrid checkpoint requires VLLM_FP8_HYBRID=1.
            quant_layers: set[str] = {
                param_name.rsplit(".", 1)[0]
                for param_name, info in metadata.items()
                if (dtype := info.get("dtype", None))
                and _SAFETENSORS_TO_TORCH_DTYPE[dtype] not in unquant_dtypes
            }
            self.modules_in_block_to_quantize = list(quant_layers)

        if _fp8_hybrid_enabled():
            if metadata is None:
                metadata = get_safetensors_params_metadata(
                    model_name, revision=revision
                )
            self._detect_fp8_hybrid_layers(metadata, hf_config)

    def _detect_fp8_hybrid_layers(
        self,
        metadata: dict[str, Any],
        hf_config: PretrainedConfig | None,
    ) -> None:
        """Record checkpoint layers stored as blockwise fp8 (VLLM_FP8_HYBRID=1).

        A layer qualifies when its ``.weight`` tensor is F8_E4M3 and it has a
        sibling ``.weight_scale_inv`` (DeepSeek-format blockwise w8a8). Names
        are hf-side at this point; ``apply_vllm_mapper`` maps them to vLLM
        module prefixes afterwards, same as ``modules_in_block_to_quantize``.

        Raises:
            ValueError: if fp8 layers are detected while a qwen4_exp
                dense_quant/lm_head_quant hf_overrides is also active. That
                override claims its allow-listed prefixes with an
                unquantized bf16 method before this dispatch ever sees them,
                so the loader would crash deep in weight loading on an
                F8_E4M3 tensor with an orphaned weight_scale_inv. Refusing
                here, at config time, turns that crash into a clear error.
        """
        self.fp8_layers = {
            name[: -len(".weight")]
            for name, info in metadata.items()
            if name.endswith(".weight")
            and info.get("dtype") == "F8_E4M3"
            and name[: -len(".weight")] + ".weight_scale_inv" in metadata
        }
        if not self.fp8_layers:
            return

        logger.info(
            "fp8 hybrid: %d blockwise-fp8 layers detected", len(self.fp8_layers)
        )
        conflict = _active_dense_fp8_override(hf_config)
        if conflict is not None:
            raise ValueError(
                f"VLLM_FP8_HYBRID=1 detected {len(self.fp8_layers)} "
                "checkpoint-serialized blockwise-fp8 layers, but the "
                f"{conflict!r} hf_overrides is also active. That override "
                "claims its allow-listed prefixes with an unquantized bf16 "
                "method before the hybrid fp8 dispatch runs, so the loader "
                "would crash on F8_E4M3 weights with an orphaned "
                "weight_scale_inv. Drop the dense_quant/lm_head_quant "
                "hf_overrides when serving a VLLM_FP8_HYBRID checkpoint, or "
                "unset VLLM_FP8_HYBRID."
            )

    def _is_fp8_layer(self, prefix: str) -> bool:
        """Whether ``prefix`` (and any fused siblings) are all blockwise-fp8.

        Fused linear modules (``packed_modules_mapping``, e.g. a merged
        ``qkv_proj``) dispatch as one indivisible unit: the checkpoint holds
        one packed weight/scale pair per fused module, so a fused layer must
        have every constituent detected as fp8 before it is safe to route
        the whole thing to Fp8Config -- a partial match falls through to the
        GPTQ path unchanged (fail closed).
        """
        if not self.fp8_layers:
            return False
        head, _, proj = prefix.rpartition(".")
        fused = self.packed_modules_mapping.get(proj)
        names = [f"{head}.{shard}" for shard in fused] if fused and head else [prefix]
        return all(
            any(fp8_name in name for fp8_name in self.fp8_layers) for name in names
        )

    def _fp8_hybrid_config(self) -> Fp8Config:
        """Shared blockwise Fp8Config for all VLLM_FP8_HYBRID dispatch."""
        if self._fp8_cfg is None:
            fp8_cfg = Fp8Config(
                is_checkpoint_fp8_serialized=True,
                activation_scheme="dynamic",
                weight_block_size=[128, 128],
            )
            fp8_cfg.packed_modules_mapping = self.packed_modules_mapping
            self._fp8_cfg = fp8_cfg
        return self._fp8_cfg


class AutoGPTQLinearMethod(LinearMethodBase):
    """Linear method for AutoGPTQ using Marlin kernels.

    Args:
        quant_config: The AutoGPTQ quantization config.
    """

    _kernel_backends_being_used: set[str] = set()

    def __init__(self, quant_config: AutoGPTQConfig) -> None:
        self.quant_config = quant_config
        self.input_dtype = None
        self.quant_type = self.quant_config.quant_type

        # Verify supported on platform.
        verify_marlin_supported(
            quant_type=self.quant_config.quant_type,
            group_size=self.quant_config.group_size,
        )

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
        output_size_per_partition = sum(output_partition_sizes)
        is_row_parallel = input_size != input_size_per_partition
        weight_loader = extra_weight_attrs.get("weight_loader")
        input_dtype = self.input_dtype

        mp_linear_kernel_config = MPLinearLayerConfig(
            full_weight_shape=(input_size, output_size),
            partition_weight_shape=(
                input_size_per_partition,
                output_size_per_partition,
            ),
            weight_type=self.quant_config.quant_type,
            act_type=params_dtype if input_dtype is None else input_dtype,
            group_size=self.quant_config.group_size,
            zero_points=False,
            has_g_idx=self.quant_config.desc_act,
        )

        kernel_type = choose_mp_linear_kernel(mp_linear_kernel_config)

        if kernel_type.__name__ not in self._kernel_backends_being_used:
            logger.info("Using %s for AutoGPTQLinearMethod", kernel_type.__name__)
            self._kernel_backends_being_used.add(kernel_type.__name__)

        # Normalize group_size
        if self.quant_config.group_size != -1:
            group_size = self.quant_config.group_size
        else:
            group_size = input_size

        # Determine sharding
        if marlin_repeat_scales_on_all_ranks(
            self.quant_config.desc_act, self.quant_config.group_size, is_row_parallel
        ):
            # By setting scale_dim == None, weight_loader will
            # repeat the scales on each GPU in TP>1 case.
            scales_and_zp_input_dim = None
            scales_and_zp_size = input_size // group_size
        else:
            # By setting scale_dim == 0, weight_loader will
            # shard the scales in TP>1 case.
            scales_and_zp_input_dim = 0
            scales_and_zp_size = input_size_per_partition // group_size

        # Quantized weights
        qweight = PackedvLLMParameter(
            data=torch.empty(
                input_size_per_partition // self.quant_config.pack_factor,
                output_size_per_partition,
                dtype=torch.int32,
            ),
            input_dim=0,
            output_dim=1,
            packed_dim=0,
            packed_factor=self.quant_config.pack_factor,
            weight_loader=weight_loader,
        )

        # Activation order
        g_idx = RowvLLMParameter(
            data=torch.empty(
                input_size_per_partition,
                dtype=torch.int32,
            ),
            input_dim=0,
            weight_loader=weight_loader,
        )

        qzeros_args = {
            "data": torch.empty(
                scales_and_zp_size,
                output_size_per_partition // self.quant_config.pack_factor,
                dtype=torch.int32,
            ),
            "weight_loader": weight_loader,
        }
        weight_scale_args = {
            "data": torch.empty(
                scales_and_zp_size,
                output_size_per_partition,
                dtype=params_dtype,
            ),
            "weight_loader": weight_loader,
        }

        if scales_and_zp_input_dim is None:
            scales = ChannelQuantScaleParameter(output_dim=1, **weight_scale_args)
            qzeros = PackedColumnParameter(
                output_dim=1,
                packed_dim=1,
                packed_factor=self.quant_config.pack_factor,
                **qzeros_args,
            )

        else:
            scales = GroupQuantScaleParameter(
                output_dim=1, input_dim=0, **weight_scale_args
            )
            qzeros = PackedvLLMParameter(
                input_dim=0,
                output_dim=1,
                packed_dim=1,
                packed_factor=self.quant_config.pack_factor,
                **qzeros_args,
            )

        layer.register_parameter("qweight", qweight)
        layer.register_parameter("g_idx", g_idx)
        layer.register_parameter("scales", scales)
        layer.register_parameter("qzeros", qzeros)

        self.kernel = kernel_type(
            mp_linear_kernel_config,
            w_q_param_name="qweight",
            w_s_param_name="scales",
            w_zp_param_name="qzeros",
            w_gidx_param_name="g_idx",
        )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        self.kernel.process_weights_after_loading(layer)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.kernel.apply_weights(layer, x, bias)


class AutoGPTQMoEMethod(FusedMoEMethodBase):
    """MoE Marlin method with quantization."""

    def __init__(
        self,
        quant_config: AutoGPTQConfig,
        moe: FusedMoEConfig,
    ) -> None:
        super().__init__(moe)
        self.quant_config = quant_config
        if self.quant_config.quant_type.size_bits == 4:
            quant_type = scalar_types.uint4b8
            scale = kInt4StaticGroupScale
        elif self.quant_config.quant_type.size_bits == 8:
            quant_type = scalar_types.uint8b128
            scale = kInt8StaticGroupScale
        else:
            raise ValueError("AutoGPTQMoEMethod only supports int4 and int8 now.")
        self.input_dtype = None
        self.use_marlin = True
        weight_key = QuantKey(quant_type, scale)

        self.wna16_moe_backend, self.experts_cls = select_wna16_moe_backend(
            moe,
            weight_key,
            quant_config=self.quant_config,
            may_have_zp=not self.quant_config.is_sym,
            may_have_bias=True,
            allow_tile_padding=not self.quant_config.desc_act,
        )

    def create_weights(
        self,
        layer: RoutedExperts,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        layer.input_dtype = self.input_dtype
        is_a_8bit = self.input_dtype is not None and self.input_dtype.itemsize == 1

        if is_a_8bit:
            assert self.quant_config.quant_type.size_bits == 8, (
                "W8A8-INT8 is not supported by marlin kernel."
            )

        intermediate_size_full = extra_weight_attrs.pop("intermediate_size_full")

        self.is_k_full = (not self.quant_config.desc_act) or (
            intermediate_size_per_partition == intermediate_size_full
        )

        if self.quant_config.group_size != -1:
            scales_size13 = hidden_size // self.quant_config.group_size
            w2_scales_size = (
                intermediate_size_full
                if self.quant_config.desc_act
                else intermediate_size_per_partition
            )
            scales_size2 = w2_scales_size // self.quant_config.group_size
            strategy = FusedMoeWeightScaleSupported.GROUP.value
        else:
            scales_size13 = 1
            scales_size2 = 1
            strategy = FusedMoeWeightScaleSupported.CHANNEL.value

        layer.num_groups_w13 = scales_size13
        layer.num_groups_w2 = scales_size2

        extra_weight_attrs.update({"quant_method": strategy, "is_transposed": True})
        # Fused gate_up_proj (column parallel)
        w13_qweight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size // self.quant_config.pack_factor,
                self.moe.w13_num_shards * intermediate_size_per_partition,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_qweight", w13_qweight)
        set_weight_attrs(w13_qweight, extra_weight_attrs)
        # down_proj (row parallel)
        w2_qweight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                intermediate_size_per_partition // self.quant_config.pack_factor,
                hidden_size,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_qweight", w2_qweight)
        set_weight_attrs(w2_qweight, extra_weight_attrs)
        # up_proj scales
        w13_scales = torch.nn.Parameter(
            torch.empty(
                num_experts,
                scales_size13,
                self.moe.w13_num_shards * intermediate_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_scales", w13_scales)
        set_weight_attrs(w13_scales, extra_weight_attrs)
        # down_proj scales
        w2_scales = torch.nn.Parameter(
            torch.empty(num_experts, scales_size2, hidden_size, dtype=params_dtype),
            requires_grad=False,
        )
        layer.register_parameter("w2_scales", w2_scales)
        set_weight_attrs(w2_scales, extra_weight_attrs)
        # don't shard the w2 scales when running act order
        set_weight_attrs(w2_scales, {"load_full_w2": self.quant_config.desc_act})
        # up_proj zero points
        w13_qzeros = torch.nn.Parameter(
            torch.empty(
                num_experts,
                scales_size13,
                self.moe.w13_num_shards
                * intermediate_size_per_partition
                // self.quant_config.pack_factor,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_qzeros", w13_qzeros)
        set_weight_attrs(w13_qzeros, extra_weight_attrs)
        # down_proj zero points
        w2_qzeros = torch.nn.Parameter(
            torch.empty(
                num_experts,
                scales_size2,
                hidden_size // self.quant_config.pack_factor,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_qzeros", w2_qzeros)
        set_weight_attrs(w2_qzeros, extra_weight_attrs)
        # don't shard the w2 scales when running act order
        set_weight_attrs(w2_qzeros, {"load_full_w2": self.quant_config.desc_act})
        w13_g_idx = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_g_idx", w13_g_idx)
        set_weight_attrs(w13_g_idx, extra_weight_attrs)
        w2_g_idx = torch.nn.Parameter(
            torch.empty(
                num_experts,
                intermediate_size_per_partition,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_g_idx", w2_g_idx)
        set_weight_attrs(w2_g_idx, extra_weight_attrs)
        w13_g_idx_sort_indices = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_g_idx_sort_indices", w13_g_idx_sort_indices)
        set_weight_attrs(w13_g_idx_sort_indices, extra_weight_attrs)
        w2_g_idx_sort_indices = torch.nn.Parameter(
            torch.empty(
                num_experts,
                intermediate_size_per_partition,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_g_idx_sort_indices", w2_g_idx_sort_indices)
        set_weight_attrs(w2_g_idx_sort_indices, extra_weight_attrs)

        # Some GPTQ checkpoints contain expert biases even when the model
        # architecture does not declare them. Zero initialization keeps
        # checkpoints without biases equivalent to the bias-free path.
        w13_bias = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                self.moe.w13_num_shards * intermediate_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_bias", w13_bias)
        set_weight_attrs(w13_bias, extra_weight_attrs)
        w2_bias = torch.nn.Parameter(
            torch.zeros(num_experts, hidden_size, dtype=params_dtype),
            requires_grad=False,
        )
        layer.register_parameter("w2_bias", w2_bias)
        set_weight_attrs(w2_bias, extra_weight_attrs)

        if self.experts_cls is not None and issubclass(
            self.experts_cls, FusedMoEExpertsModular
        ):
            device = layer.w13_qweight.device
            layer.workspace = marlin_make_workspace_new(device, 4)

    def process_weights_after_loading(self, layer: RoutedExperts) -> None:
        def replace_or_register(name: str, val: torch.Tensor | None):
            if val is None:
                return

            if hasattr(layer, name):
                replace_parameter(layer, name, val)
            else:
                layer.register_parameter(
                    name, torch.nn.Parameter(val, requires_grad=False)
                )

        is_a_8bit = self.input_dtype is not None and self.input_dtype.itemsize == 1

        assert not is_a_8bit or self.quant_config.quant_type.size_bits == 8, (
            "W8A8-INT8 is not supported by marlin kernel."
        )

        w13_bias = getattr(layer, "w13_bias", None)
        if "w13_bias" not in layer._loaded_expert_biases:
            layer.register_parameter("w13_bias", None)
            w13_bias = None
        w2_bias = getattr(layer, "w2_bias", None)
        if "w2_bias" not in layer._loaded_expert_biases:
            layer.register_parameter("w2_bias", None)
            w2_bias = None

        converted = convert_to_wna16_moe_kernel_format(
            backend=self.wna16_moe_backend,
            layer=layer,
            quant_config=self.quant_config,
            input_dtype=self.input_dtype,
            w13=layer.w13_qweight,
            w2=layer.w2_qweight,
            w13_scale=layer.w13_scales,
            w2_scale=layer.w2_scales,
            w13_g_idx=layer.w13_g_idx,
            w2_g_idx=layer.w2_g_idx,
            w13_bias=w13_bias,
            w2_bias=w2_bias,
            w13_qzeros=getattr(layer, "w13_qzeros", None),
            w2_qzeros=getattr(layer, "w2_qzeros", None),
        )

        if converted is None:
            # Backend rewrote the layer's params in place (e.g. Humming).
            self._setup_kernel(layer)
            return

        (
            w13,
            w2,
            w13_scale,
            w2_scale,
            w13_g_idx,
            w2_g_idx,
            w13_g_idx_sort_indices,
            w2_g_idx_sort_indices,
            w13_qzeros,
            w2_qzeros,
            w13_input_global_scale,
            w2_input_global_scale,
            w13_bias,
            w2_bias,
        ) = converted

        replace_parameter(layer, "w13_qweight", w13)
        replace_parameter(layer, "w2_qweight", w2)
        replace_parameter(layer, "w13_scales", w13_scale)
        replace_parameter(layer, "w2_scales", w2_scale)
        replace_parameter(layer, "w13_g_idx", w13_g_idx)
        replace_parameter(layer, "w2_g_idx", w2_g_idx)
        replace_parameter(layer, "w13_g_idx_sort_indices", w13_g_idx_sort_indices)
        replace_parameter(layer, "w2_g_idx_sort_indices", w2_g_idx_sort_indices)
        replace_or_register("w13_input_global_scale", w13_input_global_scale)
        replace_or_register("w2_input_global_scale", w2_input_global_scale)
        replace_or_register("w13_bias", w13_bias)
        replace_or_register("w2_bias", w2_bias)
        replace_or_register("w13_qzeros", w13_qzeros)
        replace_or_register("w2_qzeros", w2_qzeros)

        # The modular kernel reads w13_weight/w2_weight; marlin keeps *_qweight.
        layer.w13_weight = layer.w13_qweight
        layer.w2_weight = layer.w2_qweight

        self._setup_kernel(layer)

    def _setup_kernel(self, layer: RoutedExperts) -> None:
        """Build the FusedMoEKernel for this layer."""

        self.moe_quant_config = self.get_fused_moe_quant_config(layer)
        self.moe_kernel = make_wna16_moe_kernel(
            moe_quant_config=self.moe_quant_config,
            moe_config=self.moe,
            experts_cls=self.experts_cls,
            backend=self.wna16_moe_backend,
            is_k_full=self.is_k_full,
            w13_g_idx=getattr(layer, "w13_g_idx", None),
            w2_g_idx=getattr(layer, "w2_g_idx", None),
            w13_g_idx_sort_indices=getattr(layer, "w13_g_idx_sort_indices", None),
            w2_g_idx_sort_indices=getattr(layer, "w2_g_idx_sort_indices", None),
            routing_tables=layer._expert_routing_tables(),
        )

    def get_fused_moe_quant_config(self, layer: RoutedExperts) -> FusedMoEQuantConfig:
        if self.wna16_moe_backend == WNA16MoEBackend.HUMMING:
            from vllm.model_executor.layers.quantization.utils.humming_utils import (
                get_humming_moe_quant_config,
            )

            return get_humming_moe_quant_config(
                layer,
                gemm1_alpha=getattr(layer, "swiglu_alpha", None),
                gemm1_beta=getattr(layer, "swiglu_beta", None),
                gemm1_clamp_limit=getattr(layer, "swiglu_limit", None),
            )

        from vllm.model_executor.layers.fused_moe.config import (
            gptq_marlin_moe_quant_config,
        )

        # CPU fused_experts_cpu requires zero points even for symmetric quant
        use_zp = (
            not self.quant_config.is_sym
            or self.wna16_moe_backend == WNA16MoEBackend.CPU
        )
        return gptq_marlin_moe_quant_config(
            w1_scale=layer.w13_scales,
            w2_scale=layer.w2_scales,
            weight_bits=self.quant_config.weight_bits,
            group_size=self.quant_config.group_size,
            w1_zp=getattr(layer, "w13_qzeros", None) if use_zp else None,
            w2_zp=getattr(layer, "w2_qzeros", None) if use_zp else None,
            w1_bias=getattr(layer, "w13_bias", None),
            w2_bias=getattr(layer, "w2_bias", None),
        )

    def apply(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: SharedExperts | None,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        assert not self.is_monolithic
        assert self.moe_kernel is not None
        return self.moe_kernel.apply(
            hidden_states=x,
            w1=layer.w13_weight,
            w2=layer.w2_weight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=layer.activation,
            global_num_experts=layer.global_num_experts,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            expert_map=layer.expert_map,
            shared_experts=shared_experts,
            shared_experts_input=shared_experts_input,
        )

    def apply_monolithic(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert self.is_monolithic
        assert self.moe_kernel is not None
        return self.moe_kernel.apply_monolithic(
            hidden_states=x,
            w1=layer.w13_weight,
            w2=layer.w2_weight,
            router_logits=router_logits,
            activation=layer.activation,
            global_num_experts=layer.global_num_experts,
            expert_map=layer.expert_map,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            num_expert_group=layer.num_expert_group,
            topk_group=layer.topk_group,
            e_score_correction_bias=layer.e_score_correction_bias,
            routed_scaling_factor=layer.routed_scaling_factor,
        )
