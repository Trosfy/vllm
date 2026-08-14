# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSpark speculator: semi-autoregressive parallel drafting.

DSpark drafts a block of ``num_speculative_tokens`` tokens in one parallel pass
(reusing the DFlash machinery: context-KV precompute + a query-block forward),
then injects intra-block dependency with a lightweight sequential Markov head.

Differences from DFlash:
  * Anchor-as-first-prediction: each request emits exactly ``N =
    num_speculative_tokens`` query tokens (anchor + N-1 noise), NOT ``1 + N``.
    Every query position is a prediction (the anchor predicts the first draft
    token), so we sample at all N positions and ``sample_pos = query_pos + 1``
    (standard next-token), whereas DFlash's masks sit AT the predicted position.
    This is the ``sample_from_anchor`` path in the shared prepare-inputs kernel.
    Speculators-format checkpoints instead use the DFlash ``1 + N`` fill-in
    layout (anchor is the bonus token).
  * Sequential Markov sampling: instead of DFlash's single parallel sample, we
    sample left-to-right, adding a prefix-dependent Markov bias derived from the
    previously sampled token at each step.

CUDA graphs (FULL, mirroring DFlash) cover the whole draft step: the parallel
backbone forward AND the sequential Markov sampling.
"""

import os
from pathlib import Path
from typing import Any

import torch

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.triton_utils import triton
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator
from vllm.v1.worker.gpu.spec_decode.dspark.capacity import (
    build_sps_table,
    compute_draft_token_capacity_from_confidence,
)
from vllm.v1.worker.gpu.spec_decode.dspark.online_sts import DSparkOnlineSTS
from vllm.v1.worker.gpu.spec_decode.dspark.utils import load_dspark_model


class DSparkSpeculator(DFlashSpeculator):
    _speculator_name = "DSpark"

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        super().__init__(vllm_config, device)

        # Whether to sample from the anchor position. When True, uses anchor-as-first
        # (N slots, each position predicts the next token). When False, uses 1+N
        # fill-in block (anchor is a bonus token).
        self.sample_from_anchor = getattr(
            self.draft_model_config.hf_config, "sample_from_anchor", True
        )
        if self.sample_from_anchor:
            self.num_query_per_req = self.num_speculative_steps
        else:
            self.num_query_per_req = 1 + self.num_speculative_steps

        # DSpark consumes mean-pooled target aux hidden states at the target
        # layers, combined to hidden_size via main_proj. Store that combined
        # main_x (hidden_size wide). DSpark does not use the same pre-allocated buffer
        # that DeepSeek-V4's MTP uses.
        draft_hidden = self.draft_model_config.get_hidden_size()
        self.hidden_states = torch.zeros(
            self.max_num_tokens, draft_hidden, dtype=self.dtype, device=device
        )

        self._step_cols = torch.arange(
            self.num_speculative_steps, dtype=torch.int32, device=device
        )

        # Reduced-vocab probabilistic drafting only; set in load_draft_model.
        self._d2t_scatter_index: torch.Tensor | None = None
        self._draft_scatter_buf: torch.Tensor | None = None
        self._draft_topk: int | None = getattr(
            self.draft_model_config.hf_config, "dspark_draft_topk", None
        )

        self.draft_token_confidence_logits = torch.empty(
            self.max_num_reqs,
            self.num_speculative_steps,
            dtype=torch.float32,
            device=device,
        )
        self.draft_token_survival_probs = torch.empty_like(
            self.draft_token_confidence_logits
        )
        self.draft_token_capacity = torch.full(
            (self.max_num_reqs,),
            self.num_speculative_steps,
            dtype=torch.int32,
            device=device,
        )
        self.capacity_activation_batch_size = (
            envs.VLLM_DSPARK_CAPACITY_ACTIVATION_BATCH_SIZE
        )
        if self.capacity_activation_batch_size < 0:
            raise ValueError(
                "VLLM_DSPARK_CAPACITY_ACTIVATION_BATCH_SIZE must be >= 0, got "
                f"{self.capacity_activation_batch_size}."
            )
        self._runtime_num_reqs_for_capacity = torch.zeros(
            (1,),
            dtype=torch.int32,
            device=device,
        )
        self.draft_token_valid_lengths = torch.empty(
            (self.max_num_reqs,),
            dtype=torch.int32,
            device=device,
        )
        self._last_num_speculative_steps = self.num_speculative_steps
        self._last_proposal_confidence_valid = False
        self._numeric_capture_dir = os.environ.get(
            "VLLM_DSPARK_NUMERIC_CAPTURE_DIR"
        )
        self._numeric_capture_pending = False
        self._numeric_capture_done = False
        self.min_survival_probability = (
            self.speculative_config.dspark_confidence_threshold
        )
        self.capacity_budget_frac = self.speculative_config.dspark_budget_frac
        self.confidence_temperature = (
            self.speculative_config.dspark_confidence_temperature
        )
        sps_curve = self.speculative_config.dspark_sps_curve
        self.sps_table: torch.Tensor | None = None
        self.wants_auto_sps_curve = sps_curve == "auto"
        if sps_curve is not None:
            # Sized for the pow2-padded request count the allocator kernel
            # can index under CUDA graph capture.
            padded_reqs = triton.next_power_of_2(max(self.max_num_reqs, 1))
            max_batch_tokens = padded_reqs * (1 + self.num_speculative_steps)
            if self.wants_auto_sps_curve:
                # Flat placeholder (theta argmax verifies everything) until
                # the post-capture profiling refreshes the contents in place;
                # the captured allocator kernel bakes this buffer's address.
                self.sps_table = torch.ones(
                    max_batch_tokens + 1, dtype=torch.float32, device=device
                )
            else:
                assert isinstance(sps_curve, list)
                self.sps_table = build_sps_table(
                    sps_curve,
                    max_batch_tokens,
                    device,
                )
        self.use_draft_token_capacity = (
            self.min_survival_probability > 0.0
            or self.capacity_budget_frac < 1.0
            or self.sps_table is not None
        )
        self.online_sts: DSparkOnlineSTS | None = None
        if self.use_draft_token_capacity and self.speculative_config.dspark_online_sts:
            self.online_sts = DSparkOnlineSTS(
                self.max_num_reqs, self.num_speculative_steps, device
            )
            # Calibrated survival buffer consumed by the capacity kernels
            # inside the captured draft graph.
            self.calibrated_confidence_logits = torch.zeros_like(
                self.draft_token_confidence_logits
            )

    @staticmethod
    def _capture_cpu(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.detach().to(device="cpu", copy=True).contiguous()

    def _numeric_capture_is_rank_zero(self) -> bool:
        return bool(
            self._numeric_capture_dir
            and not self._numeric_capture_done
            and torch.distributed.is_initialized()
            and torch.distributed.get_rank() == 0
        )

    def _capture_numeric_inputs(
        self,
        *,
        input_batch: InputBatch,
        last_hidden_states: torch.Tensor,
        aux_hidden_states: list[torch.Tensor] | None,
        combined_hidden_states: torch.Tensor,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        last_sampled: torch.Tensor,
        next_prefill_tokens: torch.Tensor,
        num_target_tokens: int,
        num_reqs: int,
        active_query_len: int,
        active_num_speculative_steps: int,
        dummy_run: bool,
        is_profile: bool,
    ) -> None:
        if (
            dummy_run
            or is_profile
            or not self._numeric_capture_is_rank_zero()
            or self._numeric_capture_pending
        ):
            return

        capture_dir = Path(self._numeric_capture_dir)
        capture_dir.mkdir(parents=True, exist_ok=True)
        req_state_indices = input_batch.idx_mapping[:num_reqs]
        payload: dict[str, Any] = {
            "schema": "vllm.dspark.numeric-inputs.v1",
            "num_target_tokens": num_target_tokens,
            "num_reqs": num_reqs,
            "active_query_len": active_query_len,
            "active_num_speculative_steps": active_num_speculative_steps,
            "target_positions": self._capture_cpu(
                input_batch.positions[:num_target_tokens]
            ),
            "target_query_start_loc": self._capture_cpu(
                input_batch.query_start_loc[: num_reqs + 1]
            ),
            "request_state_indices": self._capture_cpu(req_state_indices),
            "num_scheduled_tokens": torch.from_numpy(
                input_batch.num_scheduled_tokens[:num_reqs].copy()
            ),
            "num_sampled": self._capture_cpu(num_sampled[:num_reqs]),
            "num_rejected": self._capture_cpu(num_rejected[:num_reqs]),
            "last_sampled": self._capture_cpu(last_sampled[req_state_indices]),
            "next_prefill_tokens": self._capture_cpu(
                next_prefill_tokens[req_state_indices]
            ),
            "last_hidden_states": self._capture_cpu(
                last_hidden_states[:num_target_tokens]
            ),
            "combined_hidden_states": self._capture_cpu(
                combined_hidden_states[:num_target_tokens]
            ),
        }
        if aux_hidden_states is not None:
            for layer_index, hidden_state in enumerate(aux_hidden_states):
                payload[f"aux_hidden_states_{layer_index}"] = self._capture_cpu(
                    hidden_state[:num_target_tokens]
                )
        torch.save(payload, capture_dir / "inputs.pt")
        self._numeric_capture_pending = True

    def _capture_numeric_prepared_inputs(
        self,
        *,
        input_batch: InputBatch,
        num_target_tokens: int,
        num_reqs: int,
        active_query_len: int,
        active_num_speculative_steps: int,
        dummy_run: bool,
        is_profile: bool,
    ) -> None:
        if dummy_run or is_profile or not self._numeric_capture_pending:
            return

        capture_dir = Path(self._numeric_capture_dir)
        num_query_tokens = num_reqs * active_query_len
        num_sample_tokens = num_reqs * active_num_speculative_steps
        payload: dict[str, Any] = {
            "schema": "vllm.dspark.numeric-prepared-inputs.v1",
            "draft_input_ids": self._capture_cpu(
                self.input_buffers.input_ids[:num_query_tokens]
            ),
            "draft_positions": self._capture_cpu(
                self.input_buffers.positions[:num_query_tokens]
            ),
            "draft_query_start_loc": self._capture_cpu(
                self.input_buffers.query_start_loc[: num_reqs + 1]
            ),
            "draft_seq_lens": self._capture_cpu(
                self.input_buffers.seq_lens[:num_reqs]
            ),
            "context_positions": self._capture_cpu(
                self.context_positions[:num_target_tokens]
            ),
            "sample_indices": self._capture_cpu(
                self.sample_indices[:num_sample_tokens]
            ),
            "sample_positions": self._capture_cpu(
                self.sample_pos[:num_sample_tokens]
            ),
            "sample_request_state_indices": self._capture_cpu(
                self.sample_idx_mapping[:num_sample_tokens]
            ),
        }
        for group_index, group_id in enumerate(self.draft_kv_cache_group_ids):
            payload[f"context_slot_mapping_{group_index}"] = self._capture_cpu(
                self._context_slot_mappings[group_index][:num_target_tokens]
            )
            payload[f"query_slot_mapping_{group_index}"] = self._capture_cpu(
                self.block_tables.slot_mappings[group_id][:num_query_tokens]
            )
            payload[f"block_table_{group_index}"] = self._capture_cpu(
                self.block_tables.input_block_tables[group_id][:num_reqs]
            )
        torch.save(payload, capture_dir / "prepared.pt")

    def load_draft_model(
        self,
        target_model: torch.nn.Module,
        target_attn_layer_names: set[str],
    ) -> torch.nn.Module:
        model = load_dspark_model(target_model, self.vllm_config)
        confidence_head = getattr(
            getattr(model, "model", None), "confidence_head", None
        )
        if self.use_draft_token_capacity and (
            getattr(model, "compute_confidence", None) is None
            or confidence_head is None
        ):
            raise ValueError(
                "DSpark draft-token capacity requires a draft model with a "
                f"confidence head; {type(model).__name__} does not implement "
                "compute_confidence."
            )
        # Reduced draft vocab: probabilistic rejection sampling indexes draft
        # logits by target id, so precompute the draft->target column map and a
        # scratch buffer to scatter logits into target vocab before sampling.
        d2t = getattr(model, "draft_id_to_target_id", None)
        if self.draft_logits is not None and d2t is not None:
            self._d2t_scatter_index = (
                torch.arange(d2t.shape[0], device=d2t.device) + d2t
            )
            # -inf once; the per-step scatter overwrites the draft->target
            # columns. Kept separate from draft_logits to avoid aliasing.
            self._draft_scatter_buf = torch.full(
                (self.max_num_reqs, self.vocab_size),
                float("-inf"),
                dtype=self.draft_logits.dtype,
                device=self.device,
            )
        return model

    def _sample_logits(
        self,
        logits: torch.Tensor,
        idx_map: torch.Tensor,
        sample_pos: torch.Tensor,
        step: int,
    ) -> torch.Tensor:
        if self.draft_logits is None:
            return self.model.map_draft_to_target(logits.argmax(dim=-1))

        # Probabilistic sampling and rejection operate in target-vocabulary
        # space. A reduced draft vocabulary is scattered into its target rows.
        if self._d2t_scatter_index is not None:
            assert self._draft_scatter_buf is not None
            buf = self._draft_scatter_buf[: logits.shape[0]]
            buf.index_copy_(1, self._d2t_scatter_index, logits.to(buf.dtype))
            logits = buf

        # sample_pos is the predicted token's position Q. The shared sampler
        # adds one before salting, so Q-2 keys the draft stream at Q-1.
        return self._sample_probabilistic_draft(
            logits=logits,
            positions=sample_pos - 2,
            idx_mapping=idx_map,
            temperature=self.temperature,
            seeds=self.seeds,
            draft_step=self._step_cols[step],
            draft_logits=self.draft_logits,
        )

    def _sample_sequential(
        self,
        num_reqs: int,
        head_hidden: torch.Tensor,
        num_speculative_steps: int,
        num_query_per_req: int,
        is_profile: bool = False,
        use_capacity: bool = True,
    ) -> None:
        # Sequential Markov sampling over the backbone's output hidden states.
        n_spec = num_speculative_steps
        num_sample = num_reqs * n_spec
        # Per-(req, position) head hidden, ordered (req, step).
        sample_hidden = head_hidden[self.sample_indices[:num_sample]]
        sample_hidden = sample_hidden.view(num_reqs, n_spec, -1)
        # Draft-vocab logits; sampled ids are remapped to target vocab below.
        base_logits = self.model.compute_draft_logits(
            sample_hidden.reshape(num_sample, -1)
        )
        vocab_size = base_logits.shape[-1]
        base_logits = base_logits.view(num_reqs, n_spec, vocab_size)
        base_values = draft_indices = None
        if self._draft_topk is not None:
            base_values, draft_indices = base_logits.topk(self._draft_topk, dim=-1)
            base_logits.fill_(float("-inf"))

        idx_map = self.sample_idx_mapping[:num_sample].view(num_reqs, n_spec)
        sample_pos = self.sample_pos[:num_sample].view(num_reqs, n_spec)
        confidence_logits = self.draft_token_confidence_logits[:num_reqs, :n_spec]
        min_survival_probability = self.min_survival_probability
        use_confidence_capacity = self.use_draft_token_capacity and use_capacity

        # Anchor (bonus) token per request = the input id at query offset 0,
        # laid out as one row per request in the draft query block.
        prev = self.input_buffers.input_ids[
            : num_reqs * num_query_per_req : num_query_per_req
        ]
        valid_prefix = torch.ones(num_reqs, dtype=torch.bool, device=self.device)
        valid_lengths = self.draft_token_valid_lengths[:num_reqs]
        valid_lengths.zero_()

        for i in range(n_spec):
            # Sequential stage: Markov bias from the previously sampled token.
            markov_embed = self.model.markov_embed(prev)
            if use_confidence_capacity:
                confidence_i = self.model.compute_confidence(
                    sample_hidden[:, i], markov_embed
                )
                if confidence_i is None:
                    raise RuntimeError(
                        "DSpark draft-token capacity requires loaded "
                        "confidence-head weights."
                    )
                confidence_logits[:, i] = confidence_i
            if draft_indices is None:
                logits_i = base_logits[:, i] + self.model.markov_bias(markov_embed)
            else:
                assert base_values is not None
                logits_i = self.model.apply_markov_bias_gathered(
                    markov_embed,
                    base_logits[:, i],
                    base_values[:, i],
                    draft_indices[:, i],
                )
            draft_sampled_i = self._sample_logits(
                logits_i, idx_map[:, i], sample_pos[:, i], i
            )
            valid_prefix.logical_and_(
                (draft_sampled_i >= 0) & (draft_sampled_i < self.vocab_size)
            )
            draft_sampled_i = torch.where(
                valid_prefix, draft_sampled_i, torch.zeros_like(draft_sampled_i)
            )
            valid_lengths.add_(valid_prefix.to(torch.int32))
            self.draft_tokens[:num_reqs, i] = draft_sampled_i
            prev = draft_sampled_i

        if use_confidence_capacity and not is_profile:
            capacity_confidence = self.draft_token_confidence_logits
            capacity_temperature = self.confidence_temperature
            if self.online_sts is not None:
                self.online_sts.calibrate(
                    confidence_logits,
                    out=self.calibrated_confidence_logits[:num_reqs, :n_spec],
                )
                capacity_confidence = self.calibrated_confidence_logits
                capacity_temperature = 1.0
            compute_draft_token_capacity_from_confidence(
                capacity_confidence,
                self.draft_token_capacity,
                min_survival_probability,
                num_reqs,
                n_spec,
                self._runtime_num_reqs_for_capacity,
                self.draft_token_survival_probs,
                self.capacity_budget_frac,
                sps_table=self.sps_table,
                confidence_temperature=capacity_temperature,
            )
        else:
            self.draft_token_capacity[:num_reqs].fill_(n_spec)
        torch.minimum(
            self.draft_token_capacity[:num_reqs],
            valid_lengths,
            out=self.draft_token_capacity[:num_reqs],
        )

    def set_sps_curve(self, sps_curve: list[tuple[int, float]]) -> None:
        """Refresh the SPS lookup table in place (its address is baked into
        the captured allocator kernel)."""
        assert self.sps_table is not None
        dense = build_sps_table(
            sps_curve, self.sps_table.shape[0] - 1, self.sps_table.device
        )
        self.sps_table.copy_(dense)

    def compute_capacities(self, input_batch: InputBatch) -> torch.Tensor | None:
        if not self.use_draft_token_capacity:
            return None
        num_reqs = input_batch.num_reqs
        if self.online_sts is not None:
            # Join key for verification outcomes arriving next step. Staged
            # eagerly (not in the captured graph): a padded replay would
            # index_put through stale padding-row ids, and -1 sentinels wrap
            # to the last row, so neither is safe for a scatter by slot.
            n_spec = self._last_num_speculative_steps
            self.online_sts.stage_proposal(
                self.sample_idx_mapping[: num_reqs * n_spec : n_spec],
                self.draft_token_confidence_logits[:num_reqs, :n_spec],
                valid=self._last_proposal_confidence_valid,
            )
        return self.draft_token_capacity[:num_reqs]

    def warmup_capacity_kernels(self) -> None:
        self._warmup_prepare_inputs_kernel()
        if not self.use_draft_token_capacity:
            return

        self.draft_token_confidence_logits.zero_()
        sizes = {self.max_num_reqs}
        num_reqs = 1
        while num_reqs < self.max_num_reqs:
            sizes.add(num_reqs)
            num_reqs *= 2
        for num_reqs in sorted(sizes):
            self._runtime_num_reqs_for_capacity.fill_(num_reqs)
            compute_draft_token_capacity_from_confidence(
                self.draft_token_confidence_logits,
                self.draft_token_capacity,
                self.min_survival_probability,
                num_reqs,
                self.num_speculative_steps,
                self._runtime_num_reqs_for_capacity,
                self.draft_token_survival_probs,
                self.capacity_budget_frac,
                sps_table=self.sps_table,
                confidence_temperature=self.confidence_temperature,
            )

    def propose(
        self,
        input_batch: InputBatch,
        *args,
        num_speculative_tokens: int | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if self.use_draft_token_capacity:
            self._runtime_num_reqs_for_capacity.fill_(input_batch.num_reqs)
        self._last_proposal_confidence_valid = bool(
            self.use_draft_token_capacity
            and not kwargs.get("is_profile", False)
            and not kwargs.get("dummy_run", False)
            and not self._has_unaligned_cached_prefix(input_batch)
            and (
                self.capacity_activation_batch_size <= 0
                or input_batch.num_reqs >= self.capacity_activation_batch_size
            )
        )
        self._last_num_speculative_steps = (
            num_speculative_tokens
            if self.dynamic_physical_depth and num_speculative_tokens is not None
            else self.num_speculative_steps
        )
        return super().propose(
            input_batch,
            *args,
            num_speculative_tokens=num_speculative_tokens,
            **kwargs,
        )

    def _generate_draft(
        self,
        num_reqs: int,
        num_tokens_padded: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
        is_profile: bool = False,
        num_query_per_req: int | None = None,
    ) -> None:
        if num_query_per_req is None:
            num_query_per_req = self.num_query_per_req
        num_speculative_steps = self._speculative_steps_for_query_len(num_query_per_req)
        # Full draft step (captured under CUDA graph): parallel backbone forward
        # then sequential Markov sampling over its hidden state outputs.
        head_hidden = self._run_model(
            num_tokens_padded,
            attn_metadata,
            slot_mappings,
            num_tokens_across_dp,
            cudagraph_runtime_mode,
        )
        self._sample_sequential(
            num_reqs,
            head_hidden,
            num_speculative_steps,
            num_query_per_req,
            is_profile=is_profile,
            use_capacity=(
                self.capacity_activation_batch_size <= 0
                or num_reqs >= self.capacity_activation_batch_size
            ),
        )
        if self._numeric_capture_pending:
            num_sample_tokens = num_reqs * num_speculative_steps
            capture_dir = Path(self._numeric_capture_dir)
            torch.save(
                {
                    "schema": "vllm.dspark.numeric-outputs.v1",
                    "head_hidden_states": self._capture_cpu(
                        head_hidden[:num_tokens_padded]
                    ),
                    "sample_hidden_states": self._capture_cpu(
                        head_hidden[self.sample_indices[:num_sample_tokens]]
                    ),
                    "draft_tokens": self._capture_cpu(
                        self.draft_tokens[:num_reqs, :num_speculative_steps]
                    ),
                },
                capture_dir / "outputs.pt",
            )
            self._numeric_capture_pending = False
            self._numeric_capture_done = True
