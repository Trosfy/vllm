# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable

import regex
import torch
from torch import nn

from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import SiluAndMul, SituAndMul
from vllm.model_executor.layers.attn_res import attn_res
from vllm.model_executor.layers.fused_moe import (
    FusedMoE,
    fused_moe_make_expert_params_mapping,
)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.gdn.kimi_gdn_linear_attn import (
    KimiGatedDeltaNetAttention,
)
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFunc,
    MambaStateCopyFuncCalculator,
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
)
from vllm.model_executor.layers.mla import MLAModules, MultiHeadLatentAttentionWrapper
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs.kimi_linear import KimiLinearConfig
from vllm.utils.math_utils import cdiv

from .interfaces import HasInnerState, IsHybrid, MixtureOfExperts, SupportsPP
from .utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    init_vllm_registered_model,
    is_pp_missing_parameter,
    make_layers,
    maybe_prefix,
)

logger = init_logger(__name__)

# Matches "...experts.<id>.<w1|w2|w3>.<suffix>" checkpoint names for the O(1)
# expert-tensor dispatch in load_weights.
_EXPERT_TENSOR_RE = regex.compile(r"^(.*\.experts)\.(\d+)\.(w[123])\.(.+)$")


class KimiColumnParallelGate(ColumnParallelLinear):
    """TP-sharded K3 router with globally ordered FP32 logits.

    When num_experts does not divide the TP world size (K3: 896 at TP6/TP12),
    the weight is padded to the next multiple; ColumnParallelLinear's loader
    zero-pads the checkpoint tail via pad_or_narrow and the gathered logits
    are sliced back to the logical expert count.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        prefix: str,
    ) -> None:
        tp_size = get_tensor_model_parallel_world_size()
        self._logical_output_size = output_size
        padded_output_size = -(-output_size // tp_size) * tp_size
        super().__init__(
            input_size,
            padded_output_size,
            bias=False,
            gather_output=False,
            quant_config=None,
            prefix=prefix,
        )

    def forward(self, x: torch.Tensor):
        if x.is_cuda and x.dtype == self.weight.dtype == torch.bfloat16:
            output_parallel = torch.mm(x, self.weight.T, out_dtype=torch.float32)
        else:
            output_parallel = torch.nn.functional.linear(
                x.to(self.weight.dtype), self.weight
            ).float()
        if self.tp_size > 1:
            output = tensor_model_parallel_all_gather(output_parallel)
        else:
            output = output_parallel
        return output[..., : self._logical_output_size].contiguous(), None


class KimiPaddedColumnParallelLinear(ColumnParallelLinear):
    """ColumnParallelLinear whose output axis pads to a TP multiple.

    K3's latent size (3584) does not divide TP6/TP12; checkpoint tails
    zero-fill via pad_or_narrow and the gathered output is sliced back.
    """

    def __init__(self, input_size: int, output_size: int, prefix: str) -> None:
        tp_size = get_tensor_model_parallel_world_size()
        self._logical_output_size = output_size
        padded = -(-output_size // tp_size) * tp_size
        super().__init__(
            input_size,
            padded,
            bias=False,
            gather_output=True,
            quant_config=None,
            prefix=prefix,
        )

    def forward(self, x: torch.Tensor):
        out, bias = super().forward(x)
        out = out[..., : self._logical_output_size]
        # b12x MoE launches require contiguous inputs; the slice is a view.
        return out.contiguous(), bias


class KimiPaddedRowParallelLinear(RowParallelLinear):
    """RowParallelLinear whose input axis pads to a TP multiple.

    The input tensor is zero-extended to the padded width; the padded weight
    rows are zero-filled at load, so the product is exact.
    """

    def __init__(self, input_size: int, output_size: int, prefix: str) -> None:
        tp_size = get_tensor_model_parallel_world_size()
        padded = -(-input_size // tp_size) * tp_size
        self._input_pad = padded - input_size
        super().__init__(
            padded,
            output_size,
            bias=False,
            input_is_parallel=False,
            reduce_results=False,
            quant_config=None,
            prefix=prefix,
        )

    def forward(self, x: torch.Tensor):
        if self._input_pad:
            x = torch.nn.functional.pad(x, (0, self._input_pad))
        return super().forward(x)


class KimiMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = True,
        prefix: str = "",
        activation_situ_beta: float | None = None,
        activation_situ_linear_beta: float | None = None,
    ) -> None:
        super().__init__()

        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=f"{prefix}.down_proj",
        )
        if hidden_act == "silu":
            self.act_fn = SiluAndMul()
        elif hidden_act == "situ":
            self.act_fn = SituAndMul(
                beta=activation_situ_beta or 1.0,
                linear_beta=activation_situ_linear_beta,
            )
        else:
            raise ValueError(
                f"Unsupported activation: {hidden_act}. "
                "Only silu and situ are supported."
            )

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class KimiMoE(nn.Module):
    def __init__(
        self,
        config: KimiLinearConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        layer_idx: int = 0,
    ):
        super().__init__()
        hidden_size = config.hidden_size
        intermediate_size = config.intermediate_size
        moe_intermediate_size = config.moe_intermediate_size
        num_experts = config.num_experts
        moe_renormalize = config.moe_renormalize
        self.tp_size = get_tensor_model_parallel_world_size()
        self.routed_scaling_factor = config.routed_scaling_factor
        self.num_shared_experts = config.num_shared_experts
        self.layer_idx = layer_idx
        routed_expert_hidden_size = config.routed_expert_hidden_size
        self.moe_hidden_size = routed_expert_hidden_size or hidden_size

        if config.hidden_act not in {"silu", "situ"}:
            raise ValueError(
                f"Unsupported activation: {config.hidden_act}. "
                "Only silu and situ are supported."
            )

        self.gate = KimiColumnParallelGate(
            hidden_size,
            num_experts,
            prefix=f"{prefix}.gate",
        )

        self.gate.e_score_correction_bias = nn.Parameter(
            torch.empty(num_experts, dtype=torch.float32)
        )

        if self.num_shared_experts:
            intermediate_size = moe_intermediate_size * self.num_shared_experts
            self.shared_experts = KimiMLP(
                hidden_size=config.hidden_size,
                intermediate_size=intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                reduce_results=False,
                prefix=f"{prefix}.shared_experts",
                activation_situ_beta=config.activation_situ_beta,
                activation_situ_linear_beta=config.activation_situ_linear_beta,
            )
        else:
            self.shared_experts = None

        self.routed_expert_down_proj: ColumnParallelLinear | None = None
        self.routed_expert_norm: RMSNorm | None = None
        self.routed_expert_up_proj: RowParallelLinear | None = None
        self.routed_output_transform: KimiRoutedOutputTransform | None = None
        if routed_expert_hidden_size is not None:
            self.routed_expert_down_proj = KimiPaddedColumnParallelLinear(
                hidden_size,
                routed_expert_hidden_size,
                prefix=f"{prefix}.routed_expert_down_proj",
            )
            self.routed_expert_norm = (
                RMSNorm(routed_expert_hidden_size, eps=config.rms_norm_eps)
                if config.latent_moe_use_norm
                else None
            )
            self.routed_expert_up_proj = KimiPaddedRowParallelLinear(
                routed_expert_hidden_size,
                hidden_size,
                prefix=f"{prefix}.routed_expert_up_proj",
            )
            self.routed_output_transform = KimiRoutedOutputTransform(
                self.routed_expert_norm,
                self.routed_expert_up_proj,
            )

        self.experts = FusedMoE(
            shared_experts=self.shared_experts,
            num_experts=num_experts,
            top_k=config.num_experts_per_token,
            hidden_size=self.moe_hidden_size,
            intermediate_size=moe_intermediate_size,
            activation=config.hidden_act,
            renormalize=moe_renormalize,
            quant_config=quant_config,
            use_grouped_topk=config.use_grouped_topk,
            num_expert_group=config.num_expert_group,
            topk_group=config.topk_group,
            prefix=f"{prefix}.experts",
            scoring_func=config.moe_router_activation_func,
            e_score_correction_bias=self.gate.e_score_correction_bias,
            routed_scaling_factor=self.routed_scaling_factor,
            routed_input_transform=self.routed_expert_down_proj,
            routed_output_transform=self.routed_output_transform,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens, hidden_size = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_size)
        router_logits, _ = self.gate(hidden_states)
        final_hidden_states = self.experts(
            hidden_states=hidden_states, router_logits=router_logits
        )
        return final_hidden_states.view(num_tokens, hidden_size)


class KimiRoutedOutputTransform(nn.Module):
    def __init__(
        self,
        norm: RMSNorm | None,
        up_proj: RowParallelLinear,
    ) -> None:
        super().__init__()
        self.norm = norm
        self.up_proj = up_proj

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # The routed expert GEMM consumes TP-sharded intermediate channels and
        # therefore returns a partial latent vector. Reconstruct it before
        # latent RMSNorm, then let the row-parallel up projection emit a
        # hidden-width partial. FusedMoE's final TP all-reduce combines that
        # partial together with the shared-expert partial exactly once.
        if get_tensor_model_parallel_world_size() > 1:
            hidden_states = tensor_model_parallel_all_reduce(hidden_states)
        if self.norm is not None:
            hidden_states = self.norm(hidden_states)
        return self.up_proj(hidden_states)[0]


class KimiMLAAttention(nn.Module):
    """
    Main reference: DeepseekV2 vllm Implementation
    """

    def __init__(
        self,
        config: KimiLinearConfig,
        hidden_size: int,
        num_heads: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: int | None,
        kv_lora_rank: int,
        use_nope: bool = False,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        **kwargs,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.num_heads = num_heads
        tp_size = get_tensor_model_parallel_world_size()
        self.num_local_heads = num_heads // tp_size
        self.scaling = self.qk_head_dim**-0.5
        self.use_nope = use_nope
        assert self.use_nope is True
        assert num_heads % tp_size == 0
        if self.q_lora_rank is not None:
            self.fused_qkv_a_proj = MergedColumnParallelLinear(
                self.hidden_size,
                [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
                bias=False,
                quant_config=quant_config,
                disable_tp=True,
                prefix=f"{prefix}.fused_qkv_a_proj",
            )
            self.q_a_layernorm = RMSNorm(
                self.q_lora_rank,
                eps=config.rms_norm_eps,
            )
            self.q_b_proj = ColumnParallelLinear(
                self.q_lora_rank,
                self.num_heads * self.qk_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.q_b_proj",
            )
            self.kv_a_proj_with_mqa = None
            self.q_proj = None
        else:
            self.fused_qkv_a_proj = None
            self.q_a_layernorm = None
            self.q_b_proj = None
            self.kv_a_proj_with_mqa = ReplicatedLinear(
                self.hidden_size,
                self.kv_lora_rank + self.qk_rope_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.kv_a_proj_with_mqa",
            )
            self.q_proj = ColumnParallelLinear(
                self.hidden_size,
                self.num_heads * self.qk_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.q_proj",
            )
        self.kv_a_layernorm = RMSNorm(
            self.kv_lora_rank,
            eps=config.rms_norm_eps,
        )
        self.kv_b_proj = ColumnParallelLinear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.kv_b_proj",
        )
        self.o_proj = RowParallelLinear(
            self.num_heads * self.v_head_dim,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        self.g_proj = (
            ColumnParallelLinear(
                self.hidden_size,
                self.num_heads * self.v_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.g_proj",
            )
            if config.mla_use_output_gate
            else None
        )

        mla_modules = MLAModules(
            kv_a_layernorm=self.kv_a_layernorm,
            kv_b_proj=self.kv_b_proj,
            rotary_emb=None,
            o_proj=self.o_proj,
            fused_qkv_a_proj=self.fused_qkv_a_proj,
            kv_a_proj_with_mqa=self.kv_a_proj_with_mqa,
            q_a_layernorm=self.q_a_layernorm,
            q_b_proj=self.q_b_proj,
            q_proj=self.q_proj,
            indexer=None,
            is_sparse=False,
            topk_indices_buffer=None,
            output_gate=self.g_proj,
        )
        self.mla_attn = MultiHeadLatentAttentionWrapper(
            self.hidden_size,
            self.num_local_heads,
            self.scaling,
            self.qk_nope_head_dim,
            self.qk_rope_head_dim,
            self.v_head_dim,
            self.q_lora_rank,
            self.kv_lora_rank,
            mla_modules,
            cache_config,
            quant_config,
            prefix,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        output[:] = self.mla_attn(positions, hidden_states)


class KimiDecoderLayer(nn.Module):
    def __init__(
        self,
        config: KimiLinearConfig,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = int(prefix.rsplit(".", 1)[1])

        self.is_moe = config.is_moe
        layer_idx = self.layer_idx
        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        if config.is_kda_layer(layer_idx):
            self.self_attn = KimiGatedDeltaNetAttention(
                config,
                vllm_config,
                prefix=f"{prefix}.self_attn",
            )
        else:
            self.self_attn = KimiMLAAttention(
                layer_idx=layer_idx,
                hidden_size=self.hidden_size,
                num_heads=config.num_attention_heads,
                quant_config=quant_config,
                cache_config=cache_config,
                model_config=model_config,
                prefix=f"{prefix}.self_attn",
                config=config,
                qk_nope_head_dim=config.qk_nope_head_dim,
                qk_rope_head_dim=config.qk_rope_head_dim,
                v_head_dim=config.v_head_dim,
                q_lora_rank=config.q_lora_rank,
                kv_lora_rank=config.kv_lora_rank,
                use_nope=config.mla_use_nope,
            )

        if (
            self.is_moe
            and config.num_experts is not None
            and layer_idx >= config.first_k_dense_replace
            and layer_idx % config.moe_layer_freq == 0
        ):
            self.block_sparse_moe = KimiMoE(
                config=config,
                quant_config=quant_config,
                prefix=f"{prefix}.block_sparse_moe",
                layer_idx=layer_idx,
            )
            self.mlp = self.block_sparse_moe
        else:
            self.mlp = KimiMLP(
                hidden_size=self.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
                activation_situ_beta=config.activation_situ_beta,
                activation_situ_linear_beta=config.activation_situ_linear_beta,
            )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        self.use_attn_res = config.attn_res_block_size is not None
        if self.use_attn_res:
            assert config.attn_res_block_size is not None
            self.attn_res_block_size = config.attn_res_block_size
            self.is_block_write_layer = layer_idx % self.attn_res_block_size == 0
            self.block_write_idx = layer_idx // self.attn_res_block_size
            self.prev_valid_blocks = cdiv(layer_idx, self.attn_res_block_size)
            self.self_attention_res_norm = RMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )
            self.mlp_res_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.self_attention_res_proj = ReplicatedLinear(
                config.hidden_size,
                1,
                bias=False,
                quant_config=None,
                prefix=f"{prefix}.self_attention_res_proj",
            )
            self.mlp_res_proj = ReplicatedLinear(
                config.hidden_size,
                1,
                bias=False,
                quant_config=None,
                prefix=f"{prefix}.mlp_res_proj",
            )

    def _pre_attn_norm(
        self,
        hidden_states: torch.Tensor | None,
        residual: torch.Tensor | None,
        prefix_sum: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        if not self.use_attn_res:
            assert hidden_states is not None
            if residual is None:
                residual = hidden_states
                hidden_states = self.input_layernorm(hidden_states)
            else:
                hidden_states, residual = self.input_layernorm(hidden_states, residual)
            return hidden_states, prefix_sum, residual

        assert prefix_sum is not None
        assert residual is not None
        hidden_states = attn_res(
            prefix_sum,
            hidden_states,
            residual,
            self.self_attention_res_norm.weight,
            self.self_attention_res_proj.weight.squeeze(0),
            self.input_layernorm.weight,
            num_blocks=self.prev_valid_blocks,
            block_write_idx=(self.block_write_idx if self.is_block_write_layer else -1),
            eps=self.self_attention_res_norm.variance_epsilon,
            output_norm_eps=self.input_layernorm.variance_epsilon,
        )
        return hidden_states, prefix_sum, residual

    def _post_attn_norm(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        prefix_sum: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        if not self.use_attn_res:
            hidden_states, residual = self.post_attention_layernorm(
                hidden_states, residual
            )
            return hidden_states, prefix_sum, residual

        assert prefix_sum is not None
        if self.is_block_write_layer:
            prefix_sum = hidden_states
            prefix_delta = None
        else:
            prefix_delta = hidden_states
        mlp_valid_blocks = self.prev_valid_blocks + self.is_block_write_layer
        hidden_states = attn_res(
            prefix_sum,
            prefix_delta,
            residual,
            self.mlp_res_norm.weight,
            self.mlp_res_proj.weight.squeeze(0),
            self.post_attention_layernorm.weight,
            num_blocks=mlp_valid_blocks,
            block_write_idx=-1,
            eps=self.mlp_res_norm.variance_epsilon,
            output_norm_eps=self.post_attention_layernorm.variance_epsilon,
        )
        return hidden_states, prefix_sum, residual

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor | None,
        residual: torch.Tensor | None,
        prefix_sum: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        hidden_states, prefix_sum, residual = self._pre_attn_norm(
            hidden_states, residual, prefix_sum
        )

        attn_output = torch.empty_like(hidden_states)
        self.self_attn(
            hidden_states=hidden_states,
            positions=positions,
            output=attn_output,
        )
        hidden_states = attn_output

        hidden_states, prefix_sum, residual = self._post_attn_norm(
            hidden_states, residual, prefix_sum
        )
        hidden_states = self.mlp(hidden_states)
        return hidden_states, prefix_sum, residual


@support_torch_compile
class KimiLinearModel(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_text_config
        self.config = config
        self.attn_res_block_size = config.attn_res_block_size
        self.use_attn_res = self.attn_res_block_size is not None

        self.vocab_size = config.vocab_size

        if get_pp_group().is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                prefix=f"{prefix}.embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()

        def get_layer(prefix: str):
            return KimiDecoderLayer(
                config,
                vllm_config,
                prefix,
            )

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            get_layer,
            prefix=f"{prefix}.layers",
        )
        self.num_attn_res_blocks = (
            cdiv(self.end_layer, self.attn_res_block_size)
            if self.attn_res_block_size is not None
            else 0
        )

        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            if self.use_attn_res:
                self.output_attn_res_norm = RMSNorm(
                    config.hidden_size, eps=config.rms_norm_eps
                )
                self.output_attn_res_proj = ReplicatedLinear(
                    config.hidden_size,
                    1,
                    bias=False,
                    quant_config=None,
                    prefix=f"{prefix}.output_attn_res_proj",
                )
        else:
            self.norm = PPMissingLayer()
            if self.use_attn_res:
                self.output_attn_res_norm = PPMissingLayer()
                self.output_attn_res_proj = PPMissingLayer()

        world_size = get_tensor_model_parallel_world_size()
        assert config.num_attention_heads % world_size == 0, (
            "num_attention_heads must be divisible by world_size"
        )

    def make_empty_intermediate_tensors(
        self,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> IntermediateTensors:
        residual_shape: tuple[int, ...] = (batch_size, self.config.hidden_size)
        if self.use_attn_res:
            assert self.attn_res_block_size is not None
            residual_shape = (
                batch_size,
                cdiv(self.start_layer, self.attn_res_block_size),
                self.config.hidden_size,
            )
        return IntermediateTensors(
            {
                "hidden_states": torch.zeros(
                    (batch_size, self.config.hidden_size),
                    dtype=dtype,
                    device=device,
                ),
                "residual": torch.zeros(
                    residual_shape,
                    dtype=dtype,
                    device=device,
                ),
            }
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        prefix_sum = None
        if self.use_attn_res:
            block_residual = hidden_states.new_empty(
                hidden_states.size(0),
                self.num_attn_res_blocks,
                hidden_states.size(1),
            )
            if residual is not None:
                block_residual[:, : residual.size(1), :].copy_(residual)
            prefix_sum = hidden_states
            hidden_states = None
            residual = block_residual

        for layer in self.layers[self.start_layer : self.end_layer]:
            hidden_states, prefix_sum, residual = layer(
                positions=positions,
                hidden_states=hidden_states,
                residual=residual,
                prefix_sum=prefix_sum,
            )

        assert hidden_states is not None
        assert residual is not None
        if not get_pp_group().is_last_rank:
            if prefix_sum is not None:
                hidden_states = hidden_states + prefix_sum
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )

        if self.use_attn_res:
            assert prefix_sum is not None
            hidden_states = attn_res(
                prefix_sum,
                hidden_states,
                residual,
                self.output_attn_res_norm.weight,
                self.output_attn_res_proj.weight.squeeze(0),
                None,
                num_blocks=self.num_attn_res_blocks,
                block_write_idx=-1,
                eps=self.output_attn_res_norm.variance_epsilon,
                output_norm_eps=0.0,
            )
            hidden_states = self.norm(hidden_states)
        else:
            hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]
        if self.config.q_lora_rank is not None:
            stacked_params_mapping.extend(
                [
                    (".fused_qkv_a_proj", ".q_a_proj", 0),
                    (".fused_qkv_a_proj", ".kv_a_proj_with_mqa", 1),
                ]
            )
        if self.config.is_moe:
            # Params for weights, fp8 weight scales, fp8 activation scales
            # (param_name, weight_name, expert_id, shard_id)
            expert_params_mapping = fused_moe_make_expert_params_mapping(
                self,
                ckpt_gate_proj_name="w1",
                ckpt_down_proj_name="w2",
                ckpt_up_proj_name="w3",
                num_experts=self.config.num_experts,
            )
        else:
            expert_params_mapping = []
        params_dict = dict(self.named_parameters())
        experts_unpacked = not any(
            name.endswith("w13_weight_packed") for name in params_dict
        )
        loaded_params: set[str] = set()
        for args in weights:
            name, loaded_weight = args[:2]
            kwargs = args[2] if len(args) > 2 else {}
            if "rotary_emb.inv_freq" in name:
                continue
            if experts_unpacked and name.endswith(".weight_packed"):
                name = name.replace(".weight_packed", ".weight")
            # kquant gives NF3 payloads distinct suffixes so they can coexist
            # with untouched MXFP4 experts in one checkpoint. Normalize only
            # the logical loader name; ``loaded_weight`` already contains the
            # tensor read under the original safetensors key.
            if name.endswith(".nf3_packed"):
                name = name.removesuffix(".nf3_packed") + ".weight_packed"
            elif name.endswith(".nf3_scale"):
                name = name.removesuffix(".nf3_scale") + ".weight_scale"
            elif name.endswith(".nf3_refit_packed"):
                name = name.removesuffix(".nf3_refit_packed") + ".weight_packed"
            elif name.endswith(".nf3_refit_scale"):
                name = name.removesuffix(".nf3_refit_scale") + ".weight_scale"

            # Fast path for per-expert tensors: at K3 scale (896 experts x 92
            # layers x 3 matrices, ~500k tensors) the linear scan over the
            # expert mapping list is O(experts) substring checks per tensor
            # (~1.3e9 comparisons per worker); one regex gives the same
            # mapping in O(1).
            em = _EXPERT_TENSOR_RE.match(name)
            if em is not None and expert_params_mapping:
                eprefix, eid, wkey, esuffix = em.groups()
                mapped = (
                    f"{eprefix}."
                    f"{'w13_' if wkey in ('w1', 'w3') else 'w2_'}{esuffix}"
                )
                if is_pp_missing_parameter(mapped, self):
                    continue
                eparam = params_dict.get(mapped)
                if eparam is None:
                    continue
                eparam.weight_loader(
                    eparam,
                    loaded_weight,
                    mapped,
                    expert_id=int(eid),
                    shard_id=wkey,
                )
                loaded_params.add(mapped)
                continue

            spec_layer = get_spec_layer_idx_from_weight_name(self.config, name)
            if spec_layer is not None:
                continue  # skip spec decode layers for main model
            if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:
                # Models trained using ColossalAI may include these tensors in
                # the checkpoint. Skip them.
                continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                # We have mlp.experts[0].gate_proj in the checkpoint.
                # Since we handle the experts below in expert_params_mapping,
                # we need to skip here BEFORE we update the name, otherwise
                # name will be updated to mlp.experts[0].gate_up_proj, which
                # will then be updated below in expert_params_mapping
                # for mlp.experts[0].gate_gate_up_proj, which breaks load.
                if ("mlp.experts." in name) and name not in params_dict:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                if "fused_qkv_a_proj" in name:
                    # Replicated (disable_tp) merged linear: the stock merged
                    # loader applies a tp_rank shard offset, which is wrong
                    # for a replicated param. Copy the halves directly.
                    offset = 0 if shard_id == 0 else self.config.q_lora_rank
                    param.data.narrow(0, offset, loaded_weight.shape[0]).copy_(
                        loaded_weight.to(param.dtype)
                    )
                    break
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                for idx, (param_name, weight_name, expert_id, shard_id) in enumerate(
                    expert_params_mapping
                ):
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)
                    if is_pp_missing_parameter(name, self):
                        continue
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(
                        param,
                        loaded_weight,
                        name,
                        expert_id=expert_id,
                        shard_id=shard_id,
                    )
                    break
                else:
                    # Skip loading extra bias for GPTQ models.
                    if (
                        name.endswith(".bias")
                        and name not in params_dict
                        and not self.config.is_linear_attn
                    ):  # noqa: E501
                        continue
                    # Remapping the name of FP8 kv-scale.
                    name = maybe_remap_kv_scale_name(name, params_dict)
                    if name is None:
                        continue
                    if is_pp_missing_parameter(name, self):
                        continue

                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    try:
                        weight_loader(param, loaded_weight, **kwargs)
                    except Exception as exc:
                        raise RuntimeError(
                            f"Failed to load Kimi weight {name}: checkpoint "
                            f"shape={tuple(loaded_weight.shape)}, parameter "
                            f"shape={tuple(param.shape)}"
                        ) from exc
            loaded_params.add(name)
        return loaded_params


class KimiLinearForCausalLM(
    nn.Module, HasInnerState, SupportsPP, MixtureOfExperts, IsHybrid
):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.model_config = vllm_config.model_config
        self.vllm_config = vllm_config
        self.config = self.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.quant_config = quant_config
        self.model = KimiLinearModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )
        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                self.config.vocab_size,
                self.config.hidden_size,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
        else:
            self.lm_head = PPMissingLayer()
        logit_scale = getattr(self.config, "logit_scale", 1.0)
        self.logits_processor = LogitsProcessor(
            self.config.vocab_size, scale=logit_scale
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor | IntermediateTensors:
        hidden_states = self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds, **kwargs
        )
        return hidden_states

    @classmethod
    def get_mamba_state_dtype_from_config(
        cls,
        vllm_config: "VllmConfig",
    ) -> tuple[torch.dtype, torch.dtype]:
        return MambaStateDtypeCalculator.kda_state_dtype(
            vllm_config.model_config.dtype, vllm_config.cache_config.mamba_cache_dtype
        )

    @classmethod
    def get_mamba_state_shape_from_config(
        cls, vllm_config: "VllmConfig"
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        parallel_config = vllm_config.parallel_config
        hf_config = vllm_config.model_config.hf_config
        tp_size = parallel_config.tensor_parallel_size
        num_spec = (
            vllm_config.speculative_config.num_speculative_tokens
            if vllm_config.speculative_config
            else 0
        )
        return MambaStateShapeCalculator.kda_state_shape(
            tp_size,
            hf_config.linear_attn_config["num_heads"],
            hf_config.linear_attn_config["head_dim"],
            conv_kernel_size=hf_config.linear_attn_config["short_conv_kernel_size"],
            num_spec=num_spec,
        )

    @classmethod
    def get_mamba_state_copy_func(
        cls,
    ) -> tuple[MambaStateCopyFunc, MambaStateCopyFunc]:
        return MambaStateCopyFuncCalculator.kda_state_copy_func()

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(["lm_head."] if self.config.tie_word_embeddings else None),
        )
        return loader.load_weights(weights)


class KimiK3ForConditionalGeneration(
    nn.Module, HasInnerState, SupportsPP, MixtureOfExperts, IsHybrid
):
    """Text-only Kimi K3 model."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.config = vllm_config.model_config.hf_config
        self.language_model = init_vllm_registered_model(
            vllm_config,
            hf_config=self.config.text_config,
            architectures=["KimiLinearForCausalLM"],
            prefix=maybe_prefix(prefix, "language_model"),
        )
        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.language_model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor | IntermediateTensors:
        return self.language_model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        return self.language_model.compute_logits(hidden_states)

    @classmethod
    def get_mamba_state_dtype_from_config(
        cls,
        vllm_config: VllmConfig,
    ) -> tuple[torch.dtype, torch.dtype]:
        text_config = vllm_config.model_config.hf_config.text_config
        return KimiLinearForCausalLM.get_mamba_state_dtype_from_config(
            vllm_config.with_hf_config(text_config)
        )

    @classmethod
    def get_mamba_state_shape_from_config(
        cls,
        vllm_config: VllmConfig,
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        text_config = vllm_config.model_config.hf_config.text_config
        return KimiLinearForCausalLM.get_mamba_state_shape_from_config(
            vllm_config.with_hf_config(text_config)
        )

    @classmethod
    def get_mamba_state_copy_func(
        cls,
    ) -> tuple[MambaStateCopyFunc, MambaStateCopyFunc]:
        return KimiLinearForCausalLM.get_mamba_state_copy_func()

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=["vision_tower.", "mm_projector."],
        )
        return loader.load_weights(weights)


def get_spec_layer_idx_from_weight_name(
    config: KimiLinearConfig, weight_name: str
) -> int | None:
    if hasattr(config, "num_nextn_predict_layers") and (
        config.num_nextn_predict_layers > 0
    ):
        layer_idx = config.num_hidden_layers
        for i in range(config.num_nextn_predict_layers):
            if weight_name.startswith(f"model.layers.{layer_idx + i}."):
                return layer_idx + i
    return None
