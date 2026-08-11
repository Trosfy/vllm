# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

import vllm.envs as envs
from vllm.config import get_current_vllm_config
from vllm.distributed import (
    tensor_model_parallel_all_reduce,
    tensor_model_parallel_all_reduce_in_place,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.utils.torch_utils import aux_stream, current_stream

from .moe_runner import MoERunner, _unpack

logger = init_logger(__name__)

_K3_PREFILL_STORAGE_REUSE_MIN_TOKENS = 1024


def _rms_norm_in_place(hidden_states: torch.Tensor, norm: RMSNorm) -> torch.Tensor:
    if not hidden_states.is_contiguous():
        raise ValueError("K3 in-place latent RMSNorm requires a contiguous tensor")
    if norm.variance_size_override is not None:
        raise ValueError("K3 in-place latent RMSNorm does not support variance_size")
    weight = norm.weight.data if norm.pass_weight else None
    if weight is not None and weight.dtype != hidden_states.dtype:
        raise ValueError("K3 in-place latent RMSNorm requires matching weight dtype")
    torch.ops._C.rms_norm(
        hidden_states,
        hidden_states,
        weight,
        norm.variance_epsilon,
    )
    return hidden_states


def _allreduce_norm_latent_in_place(
    hidden_states: torch.Tensor,
    norm: RMSNorm,
) -> torch.Tensor:
    """Reduce and normalize a dead K3 latent buffer in its own storage.

    Large TP16 prefill tensors use NCCL rather than the graph-oriented custom
    all-reduce.  The normal fallback allocates one receive tensor and RMSNorm
    then allocates another tensor of the same shape.  At 2,048 K3 rows each is
    14 MiB, which is enough to OOM beside a physical 1M KV cache.

    ``hidden_states`` is dead at this point, so reuse it for both NCCL's receive
    buffer and RMSNorm's output.  The CUDA RMSNorm kernel first completes the
    row reduction and synchronizes the block before it writes any output, so
    input/output aliasing is safe and byte-identical to its out-of-place path.
    """
    reduced = tensor_model_parallel_all_reduce_in_place(hidden_states)
    return _rms_norm_in_place(reduced, norm)


def _project_sharded_up_and_reduce(
    fused_latent: torch.Tensor,
    shared_output: torch.Tensor,
    up_proj: torch.nn.Module,
    output_buffer: torch.Tensor | None = None,
) -> torch.Tensor:
    """Project a TP latent slice and reduce routed+shared partials together.

    ``output_buffer`` normally lets K3 reuse the now-dead, full-width MoE input
    for the routed projection.  When shared experts already donated that same
    input, the normalized latent buffer becomes a bounded GEMM scratch instead.
    Both variants avoid a second 28 MiB full-width tensor at 2,048 tokens.
    """
    if output_buffer is None:
        routed_output = up_proj(fused_latent)
        if isinstance(routed_output, tuple):
            routed_output = routed_output[0]
    else:
        if output_buffer.shape != shared_output.shape:
            raise ValueError(
                "The routed-up reuse buffer must match the shared output: "
                f"{output_buffer.shape=} != {shared_output.shape=}"
            )
        if output_buffer.dtype != shared_output.dtype:
            raise ValueError(
                "The routed-up reuse buffer must match the shared output "
                f"dtype: {output_buffer.dtype=} != {shared_output.dtype=}"
            )
        if getattr(up_proj, "input_is_parallel", True):
            input_parallel = fused_latent
        else:
            shard_size = up_proj.input_size_per_partition
            shard_offset = up_proj.tp_rank * shard_size
            input_parallel = fused_latent.narrow(
                -1, shard_offset, shard_size
            ).contiguous()
        if getattr(up_proj, "bias", None) is not None:
            raise ValueError("The routed-up reuse path requires a bias-free projection")
        quant_method = getattr(up_proj, "quant_method", None)
        if (
            quant_method is not None
            and quant_method.__class__.__name__ != "UnquantizedLinearMethod"
        ):
            raise ValueError(
                "The routed-up reuse path requires an unquantized projection"
            )
        output_aliases_shared = output_buffer.data_ptr() == shared_output.data_ptr()
        if output_aliases_shared:
            # Only one TP-local 224-wide latent shard is consumed by this rank.
            # Copy it before reusing the full 3,584-wide latent allocation as a
            # 7,168-wide BF16 GEMM scratch.  At K3 dimensions, the 14 MiB latent
            # buffer holds 1,024 output rows, so a 2,048-token prefill takes two
            # bounded GEMMs and never allocates a full-width output.
            input_parallel = input_parallel.contiguous()
            scratch_storage = fused_latent.view(-1)
            output_width = shared_output.shape[-1]
            max_chunk_rows = scratch_storage.numel() // output_width
            if max_chunk_rows < 1:
                raise ValueError("K3 latent storage cannot hold one output row")
            weight_t = up_proj.weight.t()
            for start in range(0, input_parallel.shape[0], max_chunk_rows):
                end = min(start + max_chunk_rows, input_parallel.shape[0])
                rows = end - start
                scratch = scratch_storage[: rows * output_width].view(
                    rows, output_width
                )
                torch.mm(input_parallel[start:end], weight_t, out=scratch)
                shared_output[start:end].add_(scratch)
            routed_output = shared_output
        else:
            torch.mm(input_parallel, up_proj.weight.t(), out=output_buffer)
            routed_output = output_buffer
    if routed_output.data_ptr() != shared_output.data_ptr():
        routed_output.add_(shared_output)
    if (
        output_buffer is not None
        and routed_output.shape[0] >= _K3_PREFILL_STORAGE_REUSE_MIN_TOKENS
    ):
        # Large prefill tensors fall through the TP16 custom-AR size limit to
        # NCCL. Reuse this dead buffer as NCCL's receive buffer instead of
        # allocating a second full hidden-width output (28 MiB at 2048 K3
        # tokens). Decode-sized tensors keep the existing custom graph path.
        return tensor_model_parallel_all_reduce_in_place(routed_output)
    return tensor_model_parallel_all_reduce(routed_output)


class LatentMoERunner(MoERunner):
    """MoE runner for latent MoE routed output projections.

    Fused path (tp>1, un-reduced combine output, shared expert, no SP):
    concatenates the un-reduced latent partial (dim d) and the un-reduced
    shared partial (dim D) into one contiguous buffer, all-reduces once, then
    splits. The latent half is normed and up-projected locally (replicated
    up-proj -> full hidden), and the shared add folds into the GEMM epilogue
    (``torch.addmm``). One collective total, no post-reduction communication.

    Native path: the replicated up-proj produces the full hidden dim on every
    rank, so the base runner combines routed + shared correctly at any TP size
    (using two collectives instead of the fused path's one).
    """

    def __init__(
        self,
        *args,
        enable_k3_latent_moe_tail_fusion: bool = False,
        up_projection_is_sharded: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.up_projection_is_sharded = up_projection_is_sharded
        self.enable_k3_latent_moe_tail_fusion = enable_k3_latent_moe_tail_fusion
        use_fused_path = self._use_fused_path()
        if self.up_projection_is_sharded and not use_fused_path:
            raise ValueError(
                "A TP-sharded latent up projection requires the fused latent "
                "MoE path (TP>1, shared experts, and no sequence parallelism)."
            )
        if self.up_projection_is_sharded:
            # The tail op consumes a full replicated up-projection weight.
            self.enable_k3_latent_moe_tail_fusion = False
        if (
            self.enable_k3_latent_moe_tail_fusion
            and use_fused_path
            and self.moe_config.tp_size not in (8, 16)
        ):
            logger.warning_once(
                "K3 latent-MoE tail fusion currently supports TP=8 and TP=16, "
                "but TP=%d is configured. Falling back to the default path.",
                self.moe_config.tp_size,
            )
            self.enable_k3_latent_moe_tail_fusion = False

        if self.enable_k3_latent_moe_tail_fusion and use_fused_path:
            vllm_config = get_current_vllm_config()
            if vllm_config.parallel_config.use_ubatching:
                raise ValueError(
                    "K3 latent-MoE tail fusion does not support DBO or ubatching."
                )
            if vllm_config.model_config.enable_sleep_mode:
                raise ValueError(
                    "K3 latent-MoE tail fusion does not support sleep mode."
                )
            transform = self.routed_output_transform
            assert transform is not None
            norm = transform.norm
            assert norm is not None
            from vllm.models.kimi_k3.nvidia.ops.latent_moe_tail import (
                KimiK3LatentMoETailOp,
            )

            op = KimiK3LatentMoETailOp.initialize(
                hidden_size=transform.up_proj.weight.shape[0],
                latent_size=norm.weight.shape[0],
                dtype=norm.weight.dtype,
                device=norm.weight.device,
                rms_eps=norm.variance_epsilon,
            )
            self._k3_latent_moe_tail_op = op

    def _get_zero_residual(
        self,
        hidden_states: torch.Tensor,
        max_token_num: int,
    ) -> torch.Tensor:
        """Read-only zero ``residual_in`` for the fused AR+RMSNorm kernel.

        flashinfer requires a residual buffer even when there is no residual to
        add.
        """
        buf = getattr(self, "_zero_residual", None)
        if buf is None:
            buf = torch.zeros(
                max_token_num * hidden_states.shape[-1],
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
            self._zero_residual = buf

        assert buf.dtype == hidden_states.dtype
        assert buf.device == hidden_states.device
        assert hidden_states.numel() <= buf.numel()

        return buf[: hidden_states.numel()].view_as(hidden_states)

    def _use_fused_path(self) -> bool:
        # The fused path merges the latent and shared reductions into one
        # all-reduce, so it needs actual TP parallelism, a shared expert (to
        # concat), an un-reduced combine output, and no sequence parallelism.
        return (
            self.moe_config.tp_size > 1
            and self._shared_experts is not None
            and not self._fused_output_is_reduced
            and not self.moe_config.is_sequence_parallel
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        input_ids: torch.Tensor | None = None,
        shared_experts_input: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self._use_fused_path():
            return self._fused_forward(
                hidden_states, router_logits, input_ids, shared_experts_input
            )
        return super().forward(
            hidden_states, router_logits, input_ids, shared_experts_input
        )

    def _fused_forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        input_ids: torch.Tensor | None = None,
        shared_experts_input: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # When the caller pre-applies the routed input transform outside the
        # runner (e.g. to overlap it on a separate stream), it passes the
        # already-transformed routed input as ``hidden_states`` and the original
        # hidden states as ``shared_experts_input``; skip the transform then.
        if shared_experts_input is None:
            hidden_states, shared_experts_input = self.apply_routed_input_transform(
                hidden_states
            )
        hidden_states, og_hidden_dim_pre_xform, og_hidden_dim_post_xform = (
            self._maybe_pad_hidden_states(
                shared_experts_input,
                hidden_states,
            )
        )

        hidden_dim_unpadded = (
            self.moe_config.hidden_dim_unpadded
            if self._quant_method.has_unpadded_output
            else 0
        )
        donate_shared_input = (
            self.up_projection_is_sharded
            and shared_experts_input is not None
            and shared_experts_input.shape[0]
            >= _K3_PREFILL_STORAGE_REUSE_MIN_TOKENS
            and self._shared_experts is not None
            and self._shared_experts.can_donate_input(shared_experts_input)
        )
        if donate_shared_input:
            fused_output = self._forward_entry_donate(
                hidden_states,
                router_logits,
                shared_experts_input,
                input_ids,
                self._encode_layer_name(),
                hidden_dim_unpadded,
            )
            shared_output = shared_experts_input
        else:
            result = self._forward_entry(
                hidden_states,
                router_logits,
                shared_experts_input,
                input_ids,
                self._encode_layer_name(),
                hidden_dim_unpadded,
            )
            shared_output, fused_output = _unpack(result)
            assert shared_output is not None

        if og_hidden_dim_pre_xform is not None:
            fused_output = fused_output[..., :og_hidden_dim_pre_xform]

        transform = self.routed_output_transform
        assert transform is not None

        if self.up_projection_is_sharded:
            fused_latent = None
            if transform.norm is not None:
                fused_latent = self.allreduce_norm_latent_out(
                    fused_output,
                    transform.norm,
                    donate_input=True,
                )
            else:
                fused_latent = tensor_model_parallel_all_reduce(fused_output)

            # RowParallelLinear slices the reconstructed latent input and
            # emits an unreduced hidden-width partial. Shared experts already
            # emit the matching TP partial, so reduce their sum only once. The
            # original shared-expert input is dead after _forward_entry and is
            # exactly full hidden width, making it a safe output buffer here.
            assert shared_experts_input is not None
            result = _project_sharded_up_and_reduce(
                fused_latent,
                shared_output,
                transform.up_proj,
                output_buffer=shared_experts_input,
            )
            result = self._maybe_reduce_final_output(
                result, og_hidden_dim_post_xform, output_is_reduced=True
            )
            return self._maybe_add_zero_expert_output(result)

        if self.enable_k3_latent_moe_tail_fusion:
            op = self._k3_latent_moe_tail_op
            if 0 < fused_output.shape[0] <= op.contract.max_num_tokens:
                norm = transform.norm
                assert norm is not None
                result = op(
                    fused_output,
                    shared_output,
                    norm.weight,
                    transform.up_proj.weight,
                )
                result = self._maybe_reduce_final_output(
                    result, og_hidden_dim_post_xform, output_is_reduced=True
                )
                return self._maybe_add_zero_expert_output(result)

        fused_latent = None
        if transform.norm is not None:
            fused_latent = self.allreduce_norm_latent_out(fused_output, transform.norm)
        else:
            fused_latent = tensor_model_parallel_all_reduce(fused_output)

        shared_expert_stream = (
            aux_stream()
            if shared_output.size(0) <= envs.VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD
            else None
        )
        if shared_expert_stream is not None:
            # overlap shared expert allreduce with latent up_proj
            main = current_stream()
            shared_output.record_stream(shared_expert_stream)
            shared_expert_stream.wait_stream(main)
            with torch.cuda.stream(shared_expert_stream):
                shared_output = tensor_model_parallel_all_reduce(shared_output)
            result = torch.mm(fused_latent, transform.up_proj.weight.t())
            main.wait_stream(shared_expert_stream)
        else:
            shared_output = tensor_model_parallel_all_reduce(shared_output)
            result = torch.mm(fused_latent, transform.up_proj.weight.t())
        result.add_(shared_output)

        # Output is already fully reduced; this only strips padding.
        result = self._maybe_reduce_final_output(
            result, og_hidden_dim_post_xform, output_is_reduced=True
        )

        return self._maybe_add_zero_expert_output(result)

    def allreduce_norm_latent_out(
        self,
        hidden_states: torch.Tensor,
        norm: RMSNorm,
        donate_input: bool = False,
    ) -> torch.Tensor:
        """All-reduce + add residual + (standard) RMSNorm, fused via flashinfer."""
        from vllm.model_executor.layers.fused_allreduce_gemma_rms_norm import (
            _AR_RESIDUAL_RMS_NORM,
            _can_use_flashinfer,
            flashinfer_trtllm_fused_allreduce_norm,
        )

        if self.moe_config.tp_size == 1:
            return norm(hidden_states)

        if flashinfer_trtllm_fused_allreduce_norm is not None:
            ok, max_token_num = _can_use_flashinfer(
                hidden_states, self.moe_config.tp_size
            )
            if ok:
                norm_out = torch.empty_like(hidden_states)
                # With norm_out provided, the kernel writes the new residual
                # (all_reduce(hidden_states) + residual) into the hidden_states
                # buffer and the normalized result into norm_out.
                flashinfer_trtllm_fused_allreduce_norm(
                    allreduce_in=hidden_states,
                    residual=self._get_zero_residual(hidden_states, max_token_num),
                    rms_gamma=norm.weight,
                    rms_eps=norm.variance_epsilon,
                    world_size=self.moe_config.tp_size,
                    weight_bias=0.0,
                    launch_with_pdl=True,
                    fp32_acc=True,
                    max_token_num=max_token_num,
                    pattern_code=_AR_RESIDUAL_RMS_NORM,
                    norm_out=norm_out,
                )
                return norm_out

        if (
            donate_input
            and hidden_states.shape[0] >= _K3_PREFILL_STORAGE_REUSE_MIN_TOKENS
        ):
            # Decode-sized tensors retain the low-latency custom collective.
            # Only eager prefill tensors large enough to fall back to NCCL
            # donate their dead input storage here.
            return _allreduce_norm_latent_in_place(hidden_states, norm)

        reduced = tensor_model_parallel_all_reduce(hidden_states)
        return norm(reduced)
