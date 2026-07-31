# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Graph-safe Kimi-K3 MoE calibration capture.

The online half of the kquant calibration pipeline deliberately collects from
the official MXFP4 checkpoint running the ordinary B12X W4A16 MoE kernel.  In
that path ``intermediate_cache2`` is the canonical, route-indexed post-SiTU
activation consumed by w2.  The sidecar kernels accumulate into stable device
buffers, so the same calls can be captured and replayed by CUDA graphs.

Persistence happens after a model step, outside graph replay.  Captures are TP
rank-sharded: routing is written by rank zero, input moments are expert-sharded,
and w2-input moments/samples are channel-sharded.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import regex as re
import torch
from safetensors.torch import save_file

from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.forward_context import (
    get_forward_context,
    is_forward_context_available,
)
from vllm.logger import init_logger

logger = init_logger(__name__)

_SCHEMA_VERSION = 1
_MODEL = "moonshotai/Kimi-K3"
_REVISION = "9f62e4e9fffbd0a83ddd60e1c209d828994b3569"
_NUM_DECODER_LAYERS = 93
_NUM_MOE_LAYERS = 92
_FIRST_MOE_LAYER = 1
_NUM_EXPERTS = 896
_INPUT_SIZE = 3584
_INTERMEDIATE_SIZE = 3072
_TOP_K = 16
_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")

_state: _KQuantCaptureState | None = None


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc
    if parsed <= 0:
        raise ValueError(f"{name} must be positive, got {parsed}")
    return parsed


def kquant_capture_enabled() -> bool:
    return bool(os.getenv("VLLM_KQUANT_CAPTURE_DIR", "").strip())


def _capture_root() -> Path:
    value = os.environ["VLLM_KQUANT_CAPTURE_DIR"].strip()
    root = Path(value)
    if not root.name.endswith(".kqcapture"):
        root = root.with_name(root.name + ".kqcapture")
    return root


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=1, sort_keys=True))
    tmp.replace(path)


def _atomic_safetensors(path: Path, tensors: dict[str, torch.Tensor]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    save_file({key: value.contiguous() for key, value in tensors.items()}, str(tmp))
    tmp.replace(path)


def _decoder_layer(prefix: str) -> int:
    match = _LAYER_RE.search(prefix)
    if match is None:
        raise ValueError(f"cannot determine decoder layer from MoE prefix {prefix!r}")
    layer = int(match.group(1))
    if not _FIRST_MOE_LAYER <= layer < _NUM_DECODER_LAYERS:
        raise ValueError(f"Kimi-K3 capture received non-MoE decoder layer {layer}")
    return layer


def _moe_row(prefix: str) -> int:
    return _decoder_layer(prefix) - _FIRST_MOE_LAYER


def _current_padding(state: _KQuantCaptureState, rows: int) -> torch.Tensor:
    padding = None
    if is_forward_context_available():
        padding = get_forward_context().is_padding
    if padding is None:
        padding = state.no_padding
    if padding.device != state.device or padding.dtype != torch.bool:
        raise RuntimeError(
            "KQuant calibration requires is_padding to be a CUDA bool tensor "
            f"on {state.device}; got {padding.dtype} on {padding.device}"
        )
    if int(padding.numel()) < rows:
        raise RuntimeError(
            f"KQuant padding capacity {padding.numel()} is smaller than {rows} rows"
        )
    return padding[:rows]


class _KQuantCaptureState:
    def __init__(
        self,
        *,
        device: torch.device,
        local_intermediate_size: int,
        max_tokens: int,
    ) -> None:
        if device.type != "cuda":
            raise RuntimeError("KQuant calibration capture requires CUDA")
        self.device = device
        self.rank = int(get_tensor_model_parallel_rank())
        self.world_size = int(get_tensor_model_parallel_world_size())
        if self.world_size <= 0:
            raise RuntimeError("invalid TP world size")
        if local_intermediate_size * self.world_size != _INTERMEDIATE_SIZE:
            raise RuntimeError(
                "Kimi-K3 calibration is TP-only and requires channel-sharded w2 "
                f"input: local={local_intermediate_size}, TP={self.world_size}, "
                f"global={_INTERMEDIATE_SIZE}"
            )

        self.local_intermediate_size = int(local_intermediate_size)
        self.max_tokens = int(max_tokens)
        self.input_expert_begin = self.rank * _NUM_EXPERTS // self.world_size
        self.input_expert_end = (self.rank + 1) * _NUM_EXPERTS // self.world_size
        self.mid_channel_begin = self.rank * self.local_intermediate_size
        self.mid_channel_end = self.mid_channel_begin + self.local_intermediate_size
        self.input_experts = self.input_expert_end - self.input_expert_begin

        self.moment_sample_rate = _env_int("VLLM_KQUANT_MOMENT_SAMPLE_RATE", 16)
        self.input_hessian_sample_rate = _env_int(
            "VLLM_KQUANT_INPUT_HESSIAN_SAMPLE_RATE", 512
        )
        self.mid_hessian_sample_rate = _env_int(
            "VLLM_KQUANT_MID_HESSIAN_SAMPLE_RATE", 8192
        )
        self.sample_capacity = _env_int("VLLM_KQUANT_SAMPLE_CAPACITY", 64)
        self.stats_save_every = _env_int("VLLM_KQUANT_STATS_SAVE_EVERY", 128)
        self.sample_save_every = _env_int("VLLM_KQUANT_SAMPLE_SAVE_EVERY", 32)
        self.sample_flush_bytes = _env_int(
            "VLLM_KQUANT_SAMPLE_FLUSH_BYTES", 256 * 1024 * 1024
        )

        self.root = _capture_root()
        self.rank_dir = self.root / f"rank-{self.rank:05d}"
        self.samples_dir = self.rank_dir / "samples"
        self.run_id = os.getenv("VLLM_KQUANT_CAPTURE_RUN_ID", self.root.name)
        self.finalize_file = Path(
            os.getenv("VLLM_KQUANT_FINALIZE_FILE", str(self.root) + ".finalize")
        )
        self.registered = torch.zeros(_NUM_MOE_LAYERS, dtype=torch.bool)
        self.prefixes: dict[int, str] = {}
        self.armed = False
        self.finalized = False
        self.steps = 0
        self.parts = 0
        self.input_dropped_total = 0
        self.mid_dropped_total = 0
        self.pending_samples: dict[str, list[torch.Tensor]] = {}
        self.pending_sample_bytes = 0

        def zeros(*shape: int, dtype: torch.dtype) -> torch.Tensor:
            return torch.zeros(shape, dtype=dtype, device=device)

        self.enabled = zeros(1, dtype=torch.int32)
        self.epoch_counter = zeros(_NUM_MOE_LAYERS, dtype=torch.int64)
        self.epoch = zeros(_NUM_MOE_LAYERS, dtype=torch.int64)
        self.no_padding = zeros(self.max_tokens, dtype=torch.bool)

        self.tokens_routed = zeros(_NUM_MOE_LAYERS, _NUM_EXPERTS, dtype=torch.int64)
        self.gate_sum = zeros(_NUM_MOE_LAYERS, _NUM_EXPERTS, dtype=torch.float64)
        self.gate_sq_sum = zeros(_NUM_MOE_LAYERS, _NUM_EXPERTS, dtype=torch.float64)

        self.input_sq_sum = zeros(
            _NUM_MOE_LAYERS,
            self.input_experts,
            _INPUT_SIZE,
            dtype=torch.float32,
        )
        self.input_weight_sum = zeros(
            _NUM_MOE_LAYERS, self.input_experts, dtype=torch.float64
        )
        self.input_count = zeros(_NUM_MOE_LAYERS, self.input_experts, dtype=torch.int64)
        self.mid_sq_sum = zeros(
            _NUM_MOE_LAYERS,
            _NUM_EXPERTS,
            self.local_intermediate_size,
            dtype=torch.float32,
        )
        self.mid_weight_sum = zeros(_NUM_MOE_LAYERS, _NUM_EXPERTS, dtype=torch.float64)
        self.mid_count = zeros(_NUM_MOE_LAYERS, _NUM_EXPERTS, dtype=torch.int64)

        input_capacity = self.sample_capacity if self.rank == 0 else 0
        self.input_sample_cursor = zeros(_NUM_MOE_LAYERS, dtype=torch.int64)
        self.input_sample_dropped = zeros(_NUM_MOE_LAYERS, dtype=torch.int64)
        self.input_sample_values = zeros(
            _NUM_MOE_LAYERS, input_capacity, _INPUT_SIZE, dtype=torch.bfloat16
        )
        self.input_sample_weight = zeros(
            _NUM_MOE_LAYERS, input_capacity, dtype=torch.float32
        )
        self.input_sample_observation = zeros(
            _NUM_MOE_LAYERS, input_capacity, dtype=torch.int64
        )
        self.mid_sample_cursor = zeros(_NUM_MOE_LAYERS, dtype=torch.int64)
        self.mid_sample_dropped = zeros(_NUM_MOE_LAYERS, dtype=torch.int64)
        self.mid_sample_values = zeros(
            _NUM_MOE_LAYERS,
            self.sample_capacity,
            self.local_intermediate_size,
            dtype=torch.bfloat16,
        )
        self.mid_sample_weight = zeros(
            _NUM_MOE_LAYERS, self.sample_capacity, dtype=torch.float32
        )
        self.mid_sample_observation = zeros(
            _NUM_MOE_LAYERS, self.sample_capacity, dtype=torch.int64
        )
        self.mid_sample_expert = zeros(
            _NUM_MOE_LAYERS, self.sample_capacity, dtype=torch.int32
        )

        # These route-sized work buffers are reused sequentially by every layer.
        self.input_sample_slots = torch.full(
            (self.max_tokens,), -1, dtype=torch.int32, device=device
        )
        self.mid_sample_slots = torch.full(
            (self.max_tokens * _TOP_K,), -1, dtype=torch.int32, device=device
        )
        self._write_manifests()

    def _root_manifest(self) -> dict[str, Any]:
        executed_tokens = 0
        if self.steps and self.rank == 0:
            # Every real token produces exactly top-k routes in every MoE layer.
            executed_tokens = int(self.tokens_routed[0].sum().item()) // _TOP_K
        return {
            "schema_version": _SCHEMA_VERSION,
            "kind": "kquant_vllm_b12x_capture",
            "model": _MODEL,
            "revision": _REVISION,
            "run_id": self.run_id,
            "tp_world_size": self.world_size,
            "complete": self.finalized,
            "executed_tokens": executed_tokens,
            "corpus": os.getenv("VLLM_KQUANT_CORPUS"),
            "source": "official_mxfp4_normal_w4a16",
            "geometry": {
                "num_layers": _NUM_MOE_LAYERS,
                "num_experts": _NUM_EXPERTS,
                "input_size": _INPUT_SIZE,
                "intermediate_size": _INTERMEDIATE_SIZE,
                "top_k": _TOP_K,
            },
            "sampling": {
                "activation_moments": self.moment_sample_rate,
                "input_hessian": self.input_hessian_sample_rate,
                "mid_hessian_routes": self.mid_hessian_sample_rate,
                "ring_capacity_per_layer": self.sample_capacity,
                "sample_save_every_steps": self.sample_save_every,
                "sample_flush_bytes": self.sample_flush_bytes,
            },
        }

    def _rank_manifest(self) -> dict[str, Any]:
        return {
            "schema_version": _SCHEMA_VERSION,
            "run_id": self.run_id,
            "rank": self.rank,
            "tp_world_size": self.world_size,
            "input_expert_range": [
                self.input_expert_begin,
                self.input_expert_end,
            ],
            "intermediate_channel_range": [
                self.mid_channel_begin,
                self.mid_channel_end,
            ],
            "registered_decoder_layers": sorted(
                row + _FIRST_MOE_LAYER for row in self.prefixes
            ),
            "sample_parts": self.parts,
            "input_samples_dropped": self.input_dropped_total,
            "mid_samples_dropped": self.mid_dropped_total,
            "steps_saved": self.steps,
            "complete": self.finalized,
        }

    def _write_manifests(self) -> None:
        self.samples_dir.mkdir(parents=True, exist_ok=True)
        if self.rank == 0:
            _atomic_json(self.root / "manifest.json", self._root_manifest())
        _atomic_json(self.rank_dir / "manifest.json", self._rank_manifest())

    def register(self, prefix: str) -> None:
        row = _moe_row(prefix)
        old = self.prefixes.get(row)
        if old is not None and old != prefix:
            raise RuntimeError(
                f"KQuant capture layer row {row} collision: {old!r} vs {prefix!r}"
            )
        self.prefixes[row] = prefix
        self.registered[row] = True
        self._write_manifests()

    def _require_layer(self, prefix: str) -> int:
        row = _moe_row(prefix)
        if not bool(self.registered[row]):
            raise RuntimeError(
                f"KQuant capture layer {prefix!r} was not registered before use"
            )
        return row

    def collect_route_input(
        self,
        prefix: str,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> None:
        from sparkinfer.moe.calibration import RouteInputBuffers, collect_route_input

        row = self._require_layer(prefix)
        m = int(x.shape[0])
        if tuple(x.shape[1:]) != (_INPUT_SIZE,) or m > self.max_tokens:
            raise RuntimeError(
                f"KQuant route input shape {tuple(x.shape)} exceeds K3 contract "
                f"({self.max_tokens}, {_INPUT_SIZE})"
            )
        if tuple(topk_ids.shape) != (m, _TOP_K):
            raise RuntimeError(
                f"KQuant expected top-k shape {(m, _TOP_K)}, got {topk_ids.shape}"
            )
        if topk_weights.dtype != torch.float32:
            raise RuntimeError("KQuant capture requires applied top-k weights in FP32")
        if (
            not x.is_contiguous()
            or not topk_ids.is_contiguous()
            or not topk_weights.is_contiguous()
        ):
            raise RuntimeError("KQuant capture inputs must be contiguous")
        padding = _current_padding(self, m)
        input_capacity = int(self.input_sample_values.shape[1])
        buffers = RouteInputBuffers(
            enabled=self.enabled,
            epoch_counter=self.epoch_counter[row : row + 1],
            epoch=self.epoch[row : row + 1],
            tokens_routed=self.tokens_routed[row],
            gate_sum=self.gate_sum[row],
            gate_sq_sum=self.gate_sq_sum[row],
            input_sq_sum=self.input_sq_sum[row],
            input_weight_sum=self.input_weight_sum[row],
            input_count=self.input_count[row],
            sample_cursor=self.input_sample_cursor[row : row + 1],
            sample_dropped=self.input_sample_dropped[row : row + 1],
            sample_slots=self.input_sample_slots,
            sample_values=self.input_sample_values[row, :input_capacity],
            sample_weight=self.input_sample_weight[row, :input_capacity],
            sample_observation=self.input_sample_observation[row, :input_capacity],
        )
        collect_route_input(
            x,
            topk_weights,
            topk_ids,
            padding,
            buffers,
            num_experts=_NUM_EXPERTS,
            expert_begin=self.input_expert_begin,
            expert_end=self.input_expert_end,
            moment_sample_rate=self.moment_sample_rate,
            hessian_sample_rate=self.input_hessian_sample_rate,
            collect_routing=self.rank == 0,
        )

    def collect_mid(
        self,
        prefix: str,
        source: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> None:
        from sparkinfer.moe.calibration import MidBuffers, collect_mid

        row = self._require_layer(prefix)
        m = int(topk_ids.shape[0])
        padding = _current_padding(self, m)
        buffers = MidBuffers(
            enabled=self.enabled,
            epoch=self.epoch[row : row + 1],
            mid_sq_sum=self.mid_sq_sum[row],
            mid_weight_sum=self.mid_weight_sum[row],
            mid_count=self.mid_count[row],
            sample_cursor=self.mid_sample_cursor[row : row + 1],
            sample_dropped=self.mid_sample_dropped[row : row + 1],
            sample_slots=self.mid_sample_slots,
            sample_values=self.mid_sample_values[row],
            sample_weight=self.mid_sample_weight[row],
            sample_observation=self.mid_sample_observation[row],
            sample_expert=self.mid_sample_expert[row],
        )
        collect_mid(
            source,
            topk_weights,
            topk_ids,
            None,
            padding,
            buffers,
            num_experts=_NUM_EXPERTS,
            width=self.local_intermediate_size,
            source_stride=self.local_intermediate_size,
            moment_sample_rate=self.moment_sample_rate,
            hessian_sample_rate=self.mid_hessian_sample_rate,
        )

    def _copy_samples(self) -> dict[str, torch.Tensor]:
        input_cursors = self.input_sample_cursor.detach().cpu()
        mid_cursors = self.mid_sample_cursor.detach().cpu()
        input_dropped = self.input_sample_dropped.detach().cpu()
        mid_dropped = self.mid_sample_dropped.detach().cpu()
        self.input_dropped_total += int(input_dropped.sum().item())
        self.mid_dropped_total += int(mid_dropped.sum().item())

        values: dict[str, list[torch.Tensor]] = {
            "input.values": [],
            "input.weight": [],
            "input.observation": [],
            "input.layer": [],
            "mid.values": [],
            "mid.weight": [],
            "mid.observation": [],
            "mid.expert": [],
            "mid.layer": [],
        }
        input_capacity = int(self.input_sample_values.shape[1])
        for row in self.prefixes:
            ni = min(int(input_cursors[row]), input_capacity)
            if ni:
                values["input.values"].append(
                    self.input_sample_values[row, :ni].detach().cpu()
                )
                values["input.weight"].append(
                    self.input_sample_weight[row, :ni].detach().cpu()
                )
                values["input.observation"].append(
                    self.input_sample_observation[row, :ni].detach().cpu()
                )
                values["input.layer"].append(torch.full((ni,), row, dtype=torch.int16))
            nm = min(int(mid_cursors[row]), self.sample_capacity)
            if nm:
                values["mid.values"].append(
                    self.mid_sample_values[row, :nm].detach().cpu()
                )
                values["mid.weight"].append(
                    self.mid_sample_weight[row, :nm].detach().cpu()
                )
                values["mid.observation"].append(
                    self.mid_sample_observation[row, :nm].detach().cpu()
                )
                values["mid.expert"].append(
                    self.mid_sample_expert[row, :nm].detach().cpu()
                )
                values["mid.layer"].append(torch.full((nm,), row, dtype=torch.int16))

        self.input_sample_cursor.zero_()
        self.mid_sample_cursor.zero_()
        self.input_sample_dropped.zero_()
        self.mid_sample_dropped.zero_()
        return {key: torch.cat(parts) for key, parts in values.items() if parts}

    def _write_stats(self) -> None:
        tensors = {
            "tokens_routed": self.tokens_routed.detach().cpu(),
            "gate_sum": self.gate_sum.detach().cpu(),
            "gate_sq_sum": self.gate_sq_sum.detach().cpu(),
            "act_in_sq_sum": self.input_sq_sum.detach().cpu(),
            "act_in_weight_sum": self.input_weight_sum.detach().cpu(),
            "act_in_count": self.input_count.detach().cpu(),
            "act_mid_sq_sum": self.mid_sq_sum.detach().cpu(),
            "act_mid_weight_sum": self.mid_weight_sum.detach().cpu(),
            "act_mid_count": self.mid_count.detach().cpu(),
        }
        _atomic_safetensors(self.rank_dir / "stats.safetensors", tensors)

    def _queue_samples(self, samples: dict[str, torch.Tensor]) -> None:
        for key, value in samples.items():
            self.pending_samples.setdefault(key, []).append(value)
            self.pending_sample_bytes += value.numel() * value.element_size()

    def _write_pending_samples(self) -> None:
        if not self.pending_samples:
            return
        tensors = {
            key: torch.cat(parts)
            for key, parts in self.pending_samples.items()
            if parts
        }
        self.parts += 1
        _atomic_safetensors(
            self.samples_dir / f"part-{self.parts:08d}.safetensors",
            tensors,
        )
        self.pending_samples.clear()
        self.pending_sample_bytes = 0

    def flush_and_arm(self) -> None:
        if self.finalized:
            return
        if not self.armed:
            if len(self.prefixes) != _NUM_MOE_LAYERS:
                missing = sorted(set(range(_NUM_MOE_LAYERS)) - self.prefixes.keys())
                raise RuntimeError(
                    "KQuant capture cannot arm before all 92 K3 MoE layers are "
                    f"registered; missing rows {missing[:16]}"
                )
            self.enabled.fill_(1)
            self.armed = True
            logger.info(
                "Armed KQuant K3 capture on TP rank %d/%d at %s; the warmup "
                "request was intentionally excluded.",
                self.rank,
                self.world_size,
                self.root,
            )
            self._write_manifests()
            return

        self.steps += 1
        samples = self._copy_samples()
        if samples:
            self._queue_samples(samples)
        finalize_requested = self.finalize_file.exists()
        if (
            finalize_requested
            or self.steps % self.sample_save_every == 0
            or self.pending_sample_bytes >= self.sample_flush_bytes
        ):
            self._write_pending_samples()
        if (
            self.steps == 1
            or self.steps % self.stats_save_every == 0
            or finalize_requested
        ):
            self._write_stats()
        if finalize_requested:
            self.enabled.zero_()
            self.finalized = True
            logger.info(
                "Finalized KQuant capture on TP rank %d at %s (%d steps, "
                "input drops=%d, mid drops=%d)",
                self.rank,
                self.root,
                self.steps,
                self.input_dropped_total,
                self.mid_dropped_total,
            )
        if self.rank == 0 and (
            finalize_requested
            or self.steps == 1
            or self.steps % self.stats_save_every == 0
        ):
            _atomic_json(self.root / "manifest.json", self._root_manifest())
        self._write_manifests()


def register_kquant_capture_layer(
    *,
    prefix: str,
    device: torch.device,
    hidden_size: int,
    local_intermediate_size: int,
    num_experts: int,
    topk: int,
    quant_mode: str,
) -> None:
    """Register one official-MXFP4 W4A16 K3 MoE before graph capture."""

    if not kquant_capture_enabled():
        return
    if (hidden_size, num_experts, topk) != (_INPUT_SIZE, _NUM_EXPERTS, _TOP_K):
        raise RuntimeError(
            "VLLM_KQUANT_CAPTURE_DIR is currently a strict Kimi-K3 collector; "
            f"got hidden/experts/top-k={hidden_size}/{num_experts}/{topk}"
        )
    if quant_mode != "w4a16":
        raise RuntimeError(
            "KQuant reference capture must use the normal W4A16 MoE kernel; "
            "set B12X_MOE_FORCE_A16=1 and use the official MXFP4 checkpoint"
        )
    if os.getenv("SPARKINFER_W4A16_SMALL_M_DIRECT", "1") != "0":
        raise RuntimeError(
            "KQuant canonical mid capture requires route-major W4A16 cache2; "
            "set SPARKINFER_W4A16_SMALL_M_DIRECT=0 to bypass the micro decode "
            "kernel's private uint32/chunked scratch layout"
        )
    config = __import__("vllm.config", fromlist=["get_current_vllm_config"])
    vllm_config = config.get_current_vllm_config()
    parallel = vllm_config.parallel_config
    if (
        int(parallel.pipeline_parallel_size) != 1
        or int(parallel.data_parallel_size) != 1
        or bool(parallel.enable_expert_parallel)
    ):
        raise RuntimeError(
            "KQuant K3 capture supports TP-only execution (PP=DP=1, EP disabled)"
        )
    max_tokens = int(vllm_config.scheduler_config.max_num_batched_tokens)

    global _state
    if _state is None:
        _state = _KQuantCaptureState(
            device=device,
            local_intermediate_size=int(local_intermediate_size),
            max_tokens=max_tokens,
        )
    elif (
        _state.device != device
        or _state.local_intermediate_size != int(local_intermediate_size)
        or _state.max_tokens != max_tokens
    ):
        raise RuntimeError("inconsistent KQuant capture geometry across MoE layers")
    _state.register(prefix)


def collect_kquant_route_input(
    prefix: str,
    x: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
) -> None:
    if not kquant_capture_enabled():
        return
    if _state is None:
        raise RuntimeError("KQuant route capture ran before B12X layer registration")
    _state.collect_route_input(prefix, x, topk_weights, topk_ids)


def collect_kquant_mid(
    *,
    prefix: str,
    binding: Any,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
) -> None:
    if not kquant_capture_enabled():
        return
    if _state is None:
        raise RuntimeError("KQuant mid capture ran before B12X layer registration")
    if binding.implementation != "w4a16" or binding.quant_mode != "w4a16":
        raise RuntimeError(
            "KQuant canonical mid capture requires the ordinary W4A16 binding"
        )
    if binding.apply_router_weight_on_input:
        raise RuntimeError(
            "KQuant K3 weighting assumes router weights are applied after w2; "
            "the binding applies them on the expert input"
        )
    if binding.route_expert_map is not None:
        raise RuntimeError("KQuant reference capture does not support expert maps/EP")
    source = binding.intermediate_cache2
    if source is None:
        raise RuntimeError("B12X W4A16 binding did not expose intermediate_cache2")
    _state.collect_mid(prefix, source, topk_weights, topk_ids)


def maybe_flush_kquant_capture() -> None:
    if _state is not None:
        _state.flush_and_arm()


def _reset_kquant_capture_for_tests() -> None:
    global _state
    _state = None


__all__ = [
    "collect_kquant_mid",
    "collect_kquant_route_input",
    "kquant_capture_enabled",
    "maybe_flush_kquant_capture",
    "register_kquant_capture_layer",
]
