# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Serve the Qwen4Exp PLE n-gram table from NVMe via mmap.

Why: the n-gram (PLE) table is tens of GiB (FP8-quantized, or unquantized
BF16 for e.g. AutoRound W4A16 exports) and the stock path keeps it resident
(GPU, via ``VocabParallelEmbedding``). On a box where host and GPU share one
unified memory pool, that table cannot sit next to the rest of the model. A
token only ever touches a handful of rows out of the whole table, so it can
live on disk and be served through the page cache instead.

How, with ``VLLM_PLE_MMAP=1``:
  * ``Qwen4ExpNGramEmbedding.__init__`` swaps the GPU-resident embedding for
    :class:`MmapNgramEmbedding`, a placeholder whose ``forward`` gathers rows
    from :class:`MmapPleTable` (``np.memmap`` views over the table
    directory's safetensors shards, page-cache backed). That directory is
    the checkpoint's own by default, or ``VLLM_PLE_MMAP_DIR`` when set —
    see :func:`resolve_table_path`.
  * ``Qwen4ExpNGramEmbedding.load_weights`` drops the per-shard tensors on
    the floor (never materialized into a resident table); for a dtype whose
    :class:`_PleDtype` descriptor requires one, it also keeps the
    checkpoint's global ``weight_scale`` as a buffer on the placeholder,
    which the untouched ``Qwen4ExpPLELayer._dequantize_embeddings`` already
    knows how to consume. Unquantized dtypes (e.g. BF16) need no scale and
    pass through unscaled.
  * only the row gather is wrapped in a custom op,
    ``vllm::qwen4_exp_ple_mmap_gather``, so it runs OUTSIDE CUDA graph
    capture; the trigram hashing is symbolic-shape-safe and stays inside
    the compiled graph. (History: the original ``.numel()``-derived
    hashing specialized vLLM's dynamic dims under Dynamo —
    ``ConstraintViolationError`` on ``query_start_loc.size()[0]`` — which
    forced a temporary whole-forward op boundary until the hashing was
    rewritten with symbolic-safe constructs.) The op is listed in
    ``splitting_ops``.
  * ``MmapNgramEmbedding.forward`` tolerates out-of-range ids by
    zero-filling instead of gathering. This is not a relaxed bounds check:
    the op runs eagerly between piecewise CUDA-graph subgraph captures, and
    during capture a compiled subgraph is RECORDED, not executed, so this
    op's ``ngram_ids`` input can be read off an unexecuted intermediate
    buffer (uninitialized memory) instead of the real hashed ids. Executed
    hashing (``Qwen4ExpNGramEmbedding._hash_ngram_ids`` in ``ple_layer.py``)
    provably bounds every id via ``torch.remainder(...) + offsets``, so an
    out-of-range id can only come from such a non-semantic capture pass,
    whose output is never consumed. See :meth:`MmapPleTable.gather`, which
    keeps its own strict ``IndexError`` for every other caller (e.g.
    prewarm/tests) — the tolerance lives at the forward boundary only.

This module is imported unconditionally at ``nvidia/ple_layer.py`` module
scope so the custom op registers at import time; every behavior above is
gated on :func:`enabled` at call time. With ``VLLM_PLE_MMAP`` unset, nothing
in this module is ever invoked and the stock classes are untouched.

Knobs (env, registered in ``vllm/envs.py`` unless noted):
  VLLM_PLE_MMAP=1            enable
  VLLM_PLE_MMAP_DIR=<path>   absolute directory holding the PLE table's
                             safetensors shards, instead of the checkpoint
                             directory. Read straight from ``os.environ``
                             (NOT registered in ``vllm/envs.py``): it names
                             a data location, so it must not become a
                             torch.compile cache factor, and every consumer
                             is already behind :func:`enabled`. Lets one
                             table serve several composed checkpoints
                             without grafting a copy into each.
  VLLM_PLE_MMAP_WORKERS=32   gather threads (page faults overlap across them)
  VLLM_PLE_MMAP_CHUNK=2048   rows per gather task
  VLLM_PLE_MMAP_SERIAL=0     N > 0 = a CPU gather touching at most N
                             distinct rows runs its tasks inline on the
                             calling thread instead of dispatching them
                             through the worker pool. Read straight from
                             ``os.environ`` (NOT registered in
                             ``vllm/envs.py``) for the same reason as
                             VLLM_PLE_MMAP_DIR: it tunes the body of the
                             split-out op and must not become a
                             torch.compile cache factor, which would let a
                             threshold A/B poison its own measurement with
                             a recompile. See :func:`serial_threshold`.
  VLLM_PLE_MMAP_PREWARM=0    1 = stream the table once at load, bounded by
                             free memory, to warm the page cache
  VLLM_PLE_MMAP_GPU_GATHER=0 1 = zero-copy GPU gather: a triton kernel
                             dereferences the mmap'd table VA directly
                             (GB10 ATS/HMM: pageableMemoryAccess=1 with
                             host page tables), removing the per-step D2H
                             ids sync (a full device drain), the CPU
                             thread-pool gather, and the pageable H2D copy.
                             Probed 2026-08-28: warm rows gather in ~9 µs
                             at n=64 (vs 22 µs CPU path) and are
                             byte-identical; COLD pages are GPU-faulted
                             correctly but serially (~0.5 ms/row), so the
                             page-cache-warm assumption matters — keep
                             prewarm available and fall back to the CPU
                             path (env off) if cold-miss latency shows up.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import functools
import glob
import json
import math
import os
import re
import struct
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from safetensors.torch import _TYPES as _SAFETENSORS_TO_TORCH_DTYPE
from torch import nn

import vllm.envs as envs
from vllm.config.compilation import CompilationMode
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON, tl, triton
from vllm.utils.torch_utils import direct_register_custom_op, get_dtype_size, vllm_lib

if TYPE_CHECKING:
    from vllm.config import CompilationConfig, ModelConfig

logger = init_logger(__name__)

OP_NAME = "qwen4_exp_ple_mmap_gather"
QUALIFIED_OP_NAME = f"vllm::{OP_NAME}"


@dataclass(frozen=True)
class _PleDtype:
    """One PLE table dtype's storage type and whether it needs a scale."""

    torch_dtype: torch.dtype
    requires_scale: bool


_PLE_DTYPES: dict[str, _PleDtype] = {
    "F8_E4M3": _PleDtype(torch.float8_e4m3fn, requires_scale=True),
    "BF16": _PleDtype(torch.bfloat16, requires_scale=False),
    # F8_E5M2 deliberately excluded: is_fp8() (fp8_utils.py) only recognizes
    # float8_e4m3fn/float8_e4m3fnuz, so an e5m2 table would silently skip
    # Qwen4ExpPLELayer._dequantize_embeddings's dequant gate and fail late,
    # deep in a downstream matmul, instead of at load (invariant 4). F16 is
    # left out for want of a motivating checkpoint, not on principle.
}
_SCALE_TORCH_DTYPES: dict[str, torch.dtype] = {
    "F32": torch.float32,
    "BF16": torch.bfloat16,
    "F16": torch.float16,
}
_MAX_HEADER_BYTES = 100 << 20  # 100 MB
_PREWARM_HEADROOM_BYTES = 8 << 30  # 8 GiB
_LOG_INTERVAL_S = 60.0

_SHARD_RE = re.compile(
    r"layers\.(\d+)\.ple\.ple_embedding\.ngram_embedding\.shard_(\d+)\.weight$"
)
_SCALE_RE = re.compile(
    r"layers\.(\d+)\.ple\.ple_embedding\.ngram_embedding\.weight_scale$"
)
_LAYER_IDX_RE = re.compile(r"\.layers\.(\d+)\.")


def _itemsize(dtype_str: str) -> int:
    """Look up a safetensors dtype string's byte width.

    Raises:
        ValueError: named, in place of a bare KeyError, so a checkpoint with
            an unrecognized dtype fails with a clear message.
    """
    torch_dtype = _SAFETENSORS_TO_TORCH_DTYPE.get(dtype_str)
    if torch_dtype is None:
        raise ValueError(f"PLE mmap: unrecognized safetensors dtype {dtype_str!r}")
    return get_dtype_size(torch_dtype)


def enabled() -> bool:
    """Return True when the mmap-backed PLE path is enabled."""
    return envs.VLLM_PLE_MMAP


def serial_threshold() -> int:
    """The configured ``VLLM_PLE_MMAP_SERIAL`` row threshold, or 0 (off).

    Read straight from ``os.environ`` rather than ``vllm/envs.py`` so the
    knob stays out of the torch.compile cache factors: it only swaps how
    :meth:`MmapPleTable.gather` dispatches its tasks, never the graph, and
    a threshold sweep that forced a recompile per arm would poison its own
    A/B. Consulted once per table, from :func:`_attach_table`, which only
    runs under ``VLLM_PLE_MMAP=1``.

    Returns:
        The threshold in distinct rows; 0 (the default) always dispatches
        through the worker pool.

    Raises:
        RuntimeError: the override is set to something that is not a
            non-negative integer. Fail closed: silently reading a typo as
            0 would report a clean "serial off" arm for a run the operator
            believes measured the knob.
    """
    raw = os.environ.get("VLLM_PLE_MMAP_SERIAL", "").strip()
    if not raw:
        return 0
    try:
        value = int(raw)
    except ValueError:
        raise RuntimeError(
            f"PLE mmap: VLLM_PLE_MMAP_SERIAL={raw!r} is not an integer. Set "
            "it to the number of distinct rows at or under which a gather "
            "should run inline on the calling thread, or unset it (or set "
            "it to 0) to always dispatch through the worker pool."
        ) from None
    if value < 0:
        raise RuntimeError(
            f"PLE mmap: VLLM_PLE_MMAP_SERIAL={value} is negative. Set it to "
            "a row count, or unset it (or set it to 0) to always dispatch "
            "through the worker pool."
        )
    return value


# --------------------------------------------------------------------------- #
# safetensors header parsing. No model.safetensors.index.json exists for this
# checkpoint, so raw file offsets come from the header directly.
# --------------------------------------------------------------------------- #
def parse_safetensors_header(path: str) -> tuple[dict, int]:
    """Parse one safetensors file's header.

    Args:
        path: path to a ``.safetensors`` file.

    Returns:
        (tensor_metadata, data_start_offset): the header dict (with
        ``__metadata__`` removed) and the byte offset where tensor data
        begins.

    Raises:
        ValueError: the header is truncated, exceeds the size cap, or any
            tensor's ``data_offsets`` fall outside the file.
    """
    file_size = os.path.getsize(path)
    with open(path, "rb") as f:
        raw_len = f.read(8)
        if len(raw_len) != 8:
            raise ValueError(f"{path}: truncated safetensors header length")
        (header_len,) = struct.unpack("<Q", raw_len)
        if header_len > _MAX_HEADER_BYTES:
            raise ValueError(
                f"{path}: safetensors header is {header_len} bytes, "
                f"exceeding the {_MAX_HEADER_BYTES}-byte cap"
            )
        raw_header = f.read(header_len)
        if len(raw_header) != header_len:
            raise ValueError(f"{path}: truncated safetensors header body")
        header = json.loads(raw_header)
    header.pop("__metadata__", None)
    data_start = 8 + header_len
    for name, meta in header.items():
        try:
            start, end = meta["data_offsets"]
        except (KeyError, ValueError):
            raise ValueError(
                f"{path}: tensor {name!r} header entry has no valid "
                f"data_offsets: {meta.get('data_offsets')!r}"
            ) from None
        if start < 0 or end < start or data_start + end > file_size:
            raise ValueError(
                f"{path}: tensor {name!r} data_offsets [{start}, {end}) "
                f"fall outside the file (size {file_size})"
            )
    return header, data_start


@dataclass
class _LayerShards:
    """Discovered PLE shard/scale tensors for one PLE layer."""

    shards: dict[int, tuple[str, int, int]]  # shard_idx -> (path, offset, rows)
    cols: int
    dtype_str: str
    scale_entry: tuple[str, int, int, str] | None  # (path, offset, nbytes, dtype)


@functools.cache
def discover_shards(model_path: str) -> dict[int, _LayerShards]:
    """Parse every safetensors header under ``model_path`` for PLE tensors.

    Header-only reads (a few KB per file), never the multi-GiB tensor data;
    cheap enough to run once per load regardless of checkpoint size — and
    memoized by ``model_path`` since ``validate_shards_for`` calls this once
    per PLE layer (construction happens per-layer), which would otherwise
    re-glob and re-parse every checkpoint file's header once per layer.

    Args:
        model_path: local directory holding the checkpoint's safetensors
            shards.

    Returns:
        Mapping from (0-based) decoder layer index to its discovered shards.

    Raises:
        ValueError: a shard's on-disk size does not match its declared
            shape/dtype, or shards for one layer disagree on dtype/width.
    """
    per_layer: dict[int, dict[int, tuple[str, int, int]]] = {}
    cols_by_layer: dict[int, int] = {}
    dtype_by_layer: dict[int, str] = {}
    scale_by_layer: dict[int, tuple[str, int, int, str]] = {}

    for path in sorted(glob.glob(os.path.join(model_path, "*.safetensors"))):
        header, data_start = parse_safetensors_header(path)
        for name, meta in header.items():
            shard_match = _SHARD_RE.search(name)
            if shard_match:
                layer_idx = int(shard_match.group(1))
                shard_idx = int(shard_match.group(2))
                start, end = meta["data_offsets"]
                try:
                    rows, cols = meta["shape"]
                except (KeyError, ValueError):
                    raise ValueError(
                        f"{path}: PLE shard {name!r} has an unexpected "
                        f"shape {meta.get('shape')!r} (expected a "
                        "2-element [rows, cols])"
                    ) from None
                dtype_str = meta["dtype"]
                if end - start != rows * cols * _itemsize(dtype_str):
                    raise ValueError(
                        f"{path}: PLE shard {name!r} size does not match "
                        f"its declared shape/dtype"
                    )
                prev_dtype = dtype_by_layer.setdefault(layer_idx, dtype_str)
                if prev_dtype != dtype_str:
                    raise ValueError(
                        f"PLE layer {layer_idx}: mixed shard dtypes "
                        f"{prev_dtype!r} vs {dtype_str!r}"
                    )
                prev_cols = cols_by_layer.setdefault(layer_idx, cols)
                if prev_cols != cols:
                    raise ValueError(
                        f"PLE layer {layer_idx}: mixed shard widths "
                        f"{prev_cols} vs {cols}"
                    )
                per_layer.setdefault(layer_idx, {})[shard_idx] = (
                    path,
                    data_start + start,
                    rows,
                )
                continue
            scale_match = _SCALE_RE.search(name)
            if scale_match:
                layer_idx = int(scale_match.group(1))
                start, end = meta["data_offsets"]
                scale_by_layer[layer_idx] = (
                    path,
                    data_start + start,
                    end - start,
                    meta["dtype"],
                )

    return {
        layer_idx: _LayerShards(
            shards=shards,
            cols=cols_by_layer[layer_idx],
            dtype_str=dtype_by_layer[layer_idx],
            scale_entry=scale_by_layer.get(layer_idx),
        )
        for layer_idx, shards in per_layer.items()
    }


def _read_scale(entry: tuple[str, int, int, str]) -> torch.Tensor:
    """Read one small scalar tensor directly out of a safetensors file."""
    path, offset, nbytes, dtype_str = entry
    with open(path, "rb") as f:
        f.seek(offset)
        raw = f.read(nbytes)
    if len(raw) != nbytes:
        raise ValueError(f"{path}: truncated weight_scale read")
    torch_dtype = _SCALE_TORCH_DTYPES.get(dtype_str)
    if torch_dtype is None:
        raise ValueError(f"unsupported weight_scale dtype {dtype_str!r}")
    # A manual (u16 << 16) reconstruction overflows int32 for negative
    # (sign-bit-set) values in either 16-bit format; frombuffer with the
    # real dtype avoids the bit-manipulation entirely.
    itemsize = get_dtype_size(torch_dtype)
    raw_bytes = bytearray(raw[:itemsize])
    return torch.frombuffer(raw_bytes, dtype=torch_dtype).clone().squeeze()


_MADV_POPULATE_READ = 22  # linux 5.14+; this box runs 6.x
_PAGE_MASK = ~4095
_PREFAULT_SLOTS = 8  # > max CPU enqueue run-ahead in steps (async sched ~2)


def _load_libc() -> ctypes.CDLL | None:
    try:
        return ctypes.CDLL("libc.so.6", use_errno=True)
    except OSError:
        name = ctypes.util.find_library("c")
        return ctypes.CDLL(name, use_errno=True) if name else None


def _load_cudart() -> ctypes.CDLL | None:
    for cand in ("libcudart.so", "libcudart.so.13", "libcudart.so.12"):
        try:
            return ctypes.CDLL(cand)
        except OSError:
            continue
    import torch as _torch

    for path in glob.glob(
        os.path.join(os.path.dirname(_torch.__file__), "lib", "libcudart*")
    ):
        try:
            return ctypes.CDLL(path)
        except OSError:
            continue
    return None


@triton.jit
def _gpu_gather_kernel(
    bases_ptr,
    ids_ptr,
    out_ptr,
    shard_size,
    rows_total,
    ROW: tl.constexpr,
    ROW_POW2: tl.constexpr,
):
    """One program per row: dereference the shard's host VA directly.

    ``bases_ptr`` holds each shard memmap's data address as int64; the
    int-to-pointer cast is legal on GB10 because the GPU walks the host
    page tables (pageableMemoryAccessUsesHostPageTables=1), so file-backed
    mmap pages resolve like any other host memory. Out-of-range ids
    zero-fill IN-KERNEL — the same capture-pass tolerance the CPU forward
    boundary implements, minus the D2H inspection that forced a device
    drain there (see ``MmapNgramEmbedding.forward``).
    """
    pid = tl.program_id(0)
    rid = tl.load(ids_ptr + pid)
    valid = (rid >= 0) & (rid < rows_total)
    sid = tl.where(valid, rid // shard_size, 0)
    local = tl.where(valid, rid - sid * shard_size, 0)
    base = tl.load(bases_ptr + sid)
    src = tl.cast(base, tl.pointer_type(tl.uint8))
    offs = tl.arange(0, ROW_POW2)
    mask = (offs < ROW) & valid
    row = tl.load(src + local * ROW + offs, mask=mask, other=0)
    tl.store(out_ptr + pid * ROW + offs, row, mask=offs < ROW)


# --------------------------------------------------------------------------- #
# The mmap-backed table itself.
# --------------------------------------------------------------------------- #
class MmapPleTable:
    """Row gather over a PLE table split into shard files, served via mmap.

    Shard ``i`` holds global rows ``[i * shard_size, i * shard_size + rows)``
    — the same layout ``Qwen4ExpNGramEmbedding.load_weights``'s
    ``checkpoint_start`` math assumes, so shard/row lookup here must stay in
    lockstep with that code (see the shard-mapping contract test).

    ``model_path`` is recorded so :func:`build_tables` can detect a
    reload_weights call that repoints ``model_config`` at a different
    checkpoint on an already-attached layer (M2): silently keeping the old
    table would serve checkpoint A's mmap rows against checkpoint B's
    scale.
    """

    def __init__(
        self,
        shards: dict[int, tuple[str, int, int]],
        shard_size: int,
        row_bytes: int,
        torch_dtype: torch.dtype,
        workers: int,
        chunk: int,
        model_path: str,
        serial: int = 0,
    ) -> None:
        if not shards:
            raise ValueError("PLE mmap: no shards to build a table from")
        self.shard_size = int(shard_size)
        self.row_bytes = int(row_bytes)
        self.torch_dtype = torch_dtype
        self.model_path = model_path
        self.itemsize = get_dtype_size(torch_dtype)
        self.chunk = max(1, int(chunk))
        self.workers = max(1, int(workers))
        self.serial = max(0, int(serial))
        n_slots = max(shards) + 1
        self.mm: list[np.memmap | None] = [None] * n_slots
        self.rows_total = 0
        for idx, (path, offset, rows) in shards.items():
            self.mm[idx] = np.memmap(
                path, dtype=np.uint8, mode="r", offset=offset, shape=(rows, row_bytes)
            )
            self.rows_total += rows
        self.pool = ThreadPoolExecutor(max_workers=self.workers)
        self.gpu_gather = bool(envs.VLLM_PLE_MMAP_GPU_GATHER)
        if self.gpu_gather and not HAS_TRITON:
            raise RuntimeError(
                "VLLM_PLE_MMAP_GPU_GATHER=1 requires triton for the "
                "zero-copy gather kernel; triton is not importable. "
                "Unset VLLM_PLE_MMAP_GPU_GATHER to use the CPU gather."
            )
        self._gpu_bases: torch.Tensor | None = None
        self._base_addrs: list[int] | None = None
        self._libc: ctypes.CDLL | None = None
        self._cudart: ctypes.CDLL | None = None
        self._hostfn: Any = None  # keep the CFUNCTYPE object alive
        self._prefault_pinned: list[torch.Tensor | None] = [None] * _PREFAULT_SLOTS
        self._prefault_n: list[int] = [0] * _PREFAULT_SLOTS
        self._prefault_idx = 0
        self._prefault_errors = 0
        self._pending = 0
        self._errors = 0
        self._skipped = 0
        self._rows_since_log = 0
        self._latencies_ms: list[float] = []
        # Per-window engaged count backing the serial= log field — see
        # _record for why it is window-scoped rather than the p99 sample's
        # own flag.
        self._serial_engaged_since_log = 0
        self._last_log = time.monotonic()
        self._closed = False

    def gather(self, ids: np.ndarray) -> np.ndarray:
        """Gather table rows for a batch of global row ids.

        Args:
            ids: int64 array of global row ids, any shape.

        Returns:
            A fresh, writable ``uint8`` array shaped ``[ids.size,
            row_bytes]``, one row per input id in input order.

        Raises:
            IndexError: an id falls outside the table's row range.
        """
        start_t = time.monotonic()
        ids = np.ascontiguousarray(ids, dtype=np.int64).reshape(-1)
        if ids.size == 0:
            return np.empty((0, self.row_bytes), dtype=np.uint8)
        uniq, inverse = np.unique(ids, return_inverse=True)
        if uniq[0] < 0 or uniq[-1] >= self.rows_total:
            self._errors += 1
            raise IndexError(
                f"PLE mmap: row id out of range [{uniq[0]}, {uniq[-1]}] "
                f"for {self.rows_total} rows"
            )
        shard = uniq // self.shard_size
        local = uniq - shard * self.shard_size
        out = np.empty((uniq.size, self.row_bytes), dtype=np.uint8)

        bounds = np.flatnonzero(np.diff(shard)) + 1
        starts = np.concatenate(([0], bounds))
        ends = np.concatenate((bounds, [uniq.size]))
        tasks: list[tuple[int, int, int]] = []
        for s, e in zip(starts.tolist(), ends.tolist()):
            si = int(shard[s])
            for c in range(s, e, self.chunk):
                tasks.append((si, c, min(c + self.chunk, e)))

        def run(task: tuple[int, int, int]) -> None:
            si, a, b = task
            mm = self.mm[si]
            if mm is None:
                raise IndexError(f"PLE mmap: shard {si} missing")
            # Fancy indexing on a memmap: page faults perform the I/O, and
            # NumPy releases the GIL for the copy, so tasks overlap.
            out[a:b] = mm[local[a:b]]

        self._pending = len(tasks)
        # VLLM_PLE_MMAP_SERIAL: an opt-in bypass of the executor for small
        # gathers, keyed on uniq.size (distinct rows), NOT task count — a
        # batch-1 decode gather hashes its ~96 rows across the whole width
        # of the table by construction, so it degrades to ~96 one-row tasks
        # and pays full pool dispatch for each even though the len(tasks)
        # == 1 case below already runs inline. Measured 2026-08-29 on
        # sm_120: 0.18 ms for a direct fancy-index over 96 warm rows vs
        # 3.7 ms through the 32-worker pool, and a WORKERS=1 arm was the
        # fastest of that set — both point at dispatch, not the copy. The
        # knob has no upper bound of its own, so a high threshold
        # serializes a cold prefill-sized gather's page faults on the
        # calling thread with nothing to overlap them; accepted for an
        # opt-in knob that changes no default behavior. Only the CPU
        # gather is affected: VLLM_PLE_MMAP_GPU_GATHER=1 never routes a
        # forward through here.
        serial = self.serial > 0 and uniq.size <= self.serial
        try:
            if serial or len(tasks) == 1:
                for task in tasks:
                    run(task)
            else:
                for _ in self.pool.map(run, tasks):
                    pass
        except Exception:
            self._errors += 1
            raise
        finally:
            # Snapshot before resetting: _record's log line (fired at most
            # once per _LOG_INTERVAL_S) needs THIS call's task count, not
            # the always-zero post-reset value. It is a task count, not a
            # concurrency depth — the serial branch runs every one of them
            # at depth 1 on the calling thread.
            pending_snapshot = self._pending
            self._pending = 0
        gathered = out[inverse]
        self._record(
            int(ids.size),
            pending_snapshot,
            (time.monotonic() - start_t) * 1000.0,
            serial,
        )
        return gathered

    def _shard_base_addresses(self) -> list[int]:
        """Each shard memmap's data address, indexed by shard slot.

        Raises:
            RuntimeError: a shard slot has no memmap (a directly-constructed
                table with gaps must fail closed, never hand the GPU a null
                base address).
        """
        bases: list[int] = []
        for idx, mm in enumerate(self.mm):
            if mm is None:
                raise RuntimeError(
                    f"PLE mmap: shard {idx} has no memmap; cannot build "
                    "GPU gather base addresses"
                )
            bases.append(mm.ctypes.data)
        return bases

    def gpu_bases(self, device: torch.device) -> torch.Tensor:
        """int64 device tensor of shard base addresses, built once."""
        if self._gpu_bases is None:
            self._gpu_bases = torch.tensor(
                self._shard_base_addresses(), dtype=torch.int64, device=device
            )
        return self._gpu_bases

    def _prefault_ranges(self, ids_np: np.ndarray) -> list[tuple[int, int]]:
        """Page-aligned (addr, length) madvise ranges for a batch of row ids.

        Out-of-range ids (capture-pass garbage) are dropped — they must
        never be turned into addresses. Ranges are deduplicated by page.
        """
        ids_np = ids_np[(ids_np >= 0) & (ids_np < self.rows_total)]
        if ids_np.size == 0 or self._base_addrs is None:
            return []
        sid = ids_np // self.shard_size
        local = ids_np - sid * self.shard_size
        bases = np.asarray(self._base_addrs, dtype=np.int64)
        starts = bases[sid] + local * self.row_bytes
        pages: dict[int, int] = {}
        for start in starts.tolist():
            pg = start & _PAGE_MASK
            end = (start + self.row_bytes - 1) & _PAGE_MASK
            span = end - pg + 4096
            prev = pages.get(pg)
            if prev is None or prev < span:
                pages[pg] = span
        return sorted(pages.items())

    def _prefault_cb(self, slot_arg: int | None) -> None:
        """cudaLaunchHostFunc entry: fault this gather's pages into the
        page tables with MADV_POPULATE_READ, overlapped across the gather
        pool, BEFORE the stream reaches the gather kernel.

        Runs on a driver thread — it must never raise (a propagating
        exception would poison the stream) and must make no CUDA calls.
        Purely a performance hint: any failure falls back to the kernel's
        own (slow, serialized) ATS faults, so errors are counted and
        swallowed.
        """
        try:
            slot = int(slot_arg or 0)
            pinned = self._prefault_pinned[slot]
            n = self._prefault_n[slot]
            libc = self._libc
            if pinned is None or n == 0 or libc is None:
                return
            ranges = self._prefault_ranges(pinned[:n].numpy())
            if not ranges:
                return

            def run(chunk: list[tuple[int, int]]) -> None:
                for addr, span in chunk:
                    libc.madvise(
                        ctypes.c_void_p(addr),
                        ctypes.c_size_t(span),
                        _MADV_POPULATE_READ,
                    )

            step = max(1, len(ranges) // self.workers)
            chunks = [ranges[i : i + step] for i in range(0, len(ranges), step)]
            for _ in self.pool.map(run, chunks):
                pass
        except Exception:
            self._prefault_errors += 1

    def _prefault_enqueue(self, ids: torch.Tensor) -> None:
        """Enqueue, in-stream and fully async: D2H of ids into a pinned
        slot, then a host callback that pre-faults the pages. Stream order
        guarantees the callback runs after the copy and before the gather
        kernel launched next. Fail-open: if cudart/libc/pinned setup is
        unavailable the gather kernel simply faults on its own.
        """
        if self._cudart is None:
            self._cudart = _load_cudart()
            self._libc = _load_libc()
            if self._base_addrs is None:
                self._base_addrs = self._shard_base_addresses()
            if self._cudart is not None and self._hostfn is None:
                hostfn_t = ctypes.CFUNCTYPE(None, ctypes.c_void_p)
                self._hostfn = hostfn_t(self._prefault_cb)
        if self._cudart is None or self._libc is None or self._hostfn is None:
            return
        n = ids.numel()
        slot = self._prefault_idx % _PREFAULT_SLOTS
        self._prefault_idx += 1
        pinned = self._prefault_pinned[slot]
        if pinned is None or pinned.numel() < n:
            cap = 1 << max(10, n - 1).bit_length()
            try:
                pinned = torch.empty(cap, dtype=torch.int64, pin_memory=True)
            except RuntimeError:
                return
            self._prefault_pinned[slot] = pinned
        pinned[:n].copy_(ids, non_blocking=True)
        self._prefault_n[slot] = n
        rc = self._cudart.cudaLaunchHostFunc(
            ctypes.c_void_p(torch.cuda.current_stream(ids.device).cuda_stream),
            self._hostfn,
            ctypes.c_void_p(slot),
        )
        if rc != 0:
            self._prefault_errors += 1

    def gather_gpu(self, ids: torch.Tensor, out: torch.Tensor) -> None:
        """Gather rows into ``out`` with the zero-copy triton kernel.

        Fully asynchronous: no D2H sync, no CPU-side wait. The in-stream
        prefault (async D2H of ids -> host callback -> MADV_POPULATE_READ
        across the gather pool) populates page tables ahead of the kernel;
        without it, cold rows GPU-fault one at a time (~0.3-0.5 ms/row
        measured 2026-08-28 vs ~2-6 ms per whole all-cold batch with it).
        Out-of-range ids zero-fill in-kernel (capture-pass semantics);
        ``gather`` keeps its strict IndexError for CPU callers.

        Args:
            ids: int64 CUDA tensor of global row ids, 1-D, contiguous.
            out: uint8 CUDA tensor ``[ids.numel(), row_bytes]``.
        """
        n = ids.numel()
        if n == 0:
            return
        self._prefault_enqueue(ids)
        _gpu_gather_kernel[(n,)](
            self.gpu_bases(ids.device),
            ids,
            out,
            self.shard_size,
            self.rows_total,
            ROW=self.row_bytes,
            ROW_POW2=triton.next_power_of_2(self.row_bytes),
        )

    def record_capture_pass_skip(self) -> None:
        """Count one gather skipped by :class:`MmapNgramEmbedding` because its
        input ids were out of range.

        Never called from :meth:`gather` itself — that path stays a strict
        ``IndexError`` for every caller other than the compiled forward (see
        the module docstring). Surfaced as ``skipped=`` in the periodic
        telemetry line below, without a full warning per occurrence.
        """
        self._skipped += 1

    def _record(self, rows: int, pending: int, elapsed_ms: float, serial: bool) -> None:
        self._latencies_ms.append(elapsed_ms)
        self._rows_since_log += rows
        # serial= is a WINDOW statistic (engaged/total gathers), not the p99
        # SAMPLE's own flag: the p99 sample is by construction the window's
        # biggest gather, which is exactly the one most likely to have
        # crossed the threshold back onto the pool — keying the field on it
        # reports serial=0 for a window in which nearly every gather did
        # engage. Counting every call reflects what the window actually
        # did. The window's total is len(self._latencies_ms), appended to
        # just above.
        if serial:
            self._serial_engaged_since_log += 1
        now = time.monotonic()
        if now - self._last_log < _LOG_INTERVAL_S:
            return
        latencies = sorted(self._latencies_ms)
        p99_idx = max(0, math.ceil(len(latencies) * 0.99) - 1)
        p99 = latencies[p99_idx] if latencies else 0.0
        logger.info(
            "rows=%d p99_ms=%.2f pending=%d errors=%d skipped=%d serial=%d/%d",
            self._rows_since_log,
            p99,
            pending,
            self._errors,
            self._skipped,
            self._serial_engaged_since_log,
            len(self._latencies_ms),
        )
        self._latencies_ms.clear()
        self._rows_since_log = 0
        self._serial_engaged_since_log = 0
        self._last_log = now

    def prewarm(self, max_bytes: int) -> int:
        """Stream up to ``max_bytes`` of the table into the page cache.

        Args:
            max_bytes: byte budget; a non-positive value skips prewarm.

        Returns:
            Bytes actually read.
        """
        if max_bytes <= 0:
            return 0
        block = 64 << 20
        remaining = max_bytes
        read_total = 0
        for mm in self.mm:
            if mm is None or remaining <= 0:
                continue
            path = mm.filename
            if path is None:
                # Every memmap here was opened from a path string in
                # __init__; None only occurs for an anonymous mmap, which
                # this class never creates.
                raise RuntimeError("PLE mmap: memmap has no backing file")
            start = mm.offset
            end = start + mm.shape[0] * mm.shape[1]
            with open(path, "rb", buffering=0) as f:
                f.seek(start)
                pos = start
                while pos < end and remaining > 0:
                    chunk = f.read(min(block, end - pos, remaining))
                    if not chunk:
                        break
                    pos += len(chunk)
                    remaining -= len(chunk)
                    read_total += len(chunk)
        return read_total

    def close(self) -> None:
        """Release the gather thread pool and drop memmap references.

        Idempotent: safe to call more than once, and safe on a table that
        was never gathered from. Guards against leaking the previous
        table's ThreadPool when a layer's table is rebuilt in place (e.g. a
        weight-reload re-entering _attach_table on an already-populated
        placeholder).
        """
        if self._closed:
            return
        self._closed = True
        self.pool.shutdown(wait=False)
        self.mm = [None] * len(self.mm)
        # Base addresses point into the dropped memmaps; a stale tensor
        # (or the CPU-side address list the prefault callback reads) would
        # hand out dangling host VAs.
        self._gpu_bases = None
        self._base_addrs = None
        self._prefault_pinned = [None] * _PREFAULT_SLOTS


def _mem_available_bytes(path: str = "/proc/meminfo") -> int:
    """Read ``MemAvailable`` from a ``/proc/meminfo``-format file, in bytes."""
    with open(path) as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    raise RuntimeError(f"PLE mmap: MemAvailable not found in {path}")


def compute_prewarm_bound(table_bytes: int, mem_available_bytes: int) -> int:
    """Bound a prewarm read so it never eats into headroom (R4.10/R5.2).

    A ``mem_available_bytes`` below the headroom clamps to 0 rather than
    going negative, which would otherwise slice-read nearly the whole table
    exactly when memory is scarcest.
    """
    return min(table_bytes, max(0, mem_available_bytes - _PREWARM_HEADROOM_BYTES))


# --------------------------------------------------------------------------- #
# Placeholder that stands in for VocabParallelEmbedding.
# --------------------------------------------------------------------------- #
class MmapNgramEmbedding(nn.Module):
    """Duck-types the surface ``Qwen4ExpNGramEmbedding``/``Qwen4ExpPLELayer``
    read off ``self.ngram_embedding``: ``org_vocab_size``, ``embedding_dim``,
    ``weight_scale``, and ``__call__``. No ``.weight``/``.shard_indices`` —
    the env-gated ``load_weights`` branch intercepts shard tensors before the
    stock code would ever read those.

    ``table`` is ``None`` until the top-level model's ``load_weights``
    attaches one (never from this class's own ``load_weights`` — see
    :func:`build_tables`). While unset, ``forward``'s behavior depends on
    whether a real (non-dummy) load ever streamed weights through this
    module: ``--load-format dummy`` profiling never calls ``load_weights``
    at all, so ``weights_streamed`` stays False and zeros in the
    placeholder's still-default fp8 dtype are the correct, intentional
    stand-in — harmless regardless of which dtype eventually attaches, since
    every :class:`_PleDtype` either dequantizes a zero to zero or passes it
    through unscaled. A real load that streamed weights but never got a
    table attached (build_tables didn't run, or raised and was swallowed
    somewhere) is a bug, and must raise loudly rather than silently serve
    zeros as if they were real embeddings (invariant 4: fail closed, never
    serve garbage).

    A second, unrelated zero-fill case exists once ``table`` IS attached:
    ``forward`` zero-fills (instead of gathering) when the incoming ``ids``
    fall outside the table's row range. That is not a relaxed bounds check
    — see ``forward`` and the module docstring for why it can only happen
    during CUDA-graph capture, never for a real token.
    """

    # Declared so static type-checkers resolve this to Tensor instead of
    # falling back to nn.Module.__getattr__'s Tensor | Module return type;
    # the buffer itself is registered dynamically in __init__.
    weight_scale: torch.Tensor

    def __init__(self, num_embeddings: int, embedding_dim: int) -> None:
        super().__init__()
        self.org_vocab_size = int(num_embeddings)
        self.embedding_dim = int(embedding_dim)
        self.torch_dtype: torch.dtype = torch.float8_e4m3fn
        self.table: MmapPleTable | None = None
        self.weight_scale_loaded = False
        self.weights_streamed = False
        # (total_ms, sync_ms, gather_ms, h2d_ms) per forward call that
        # actually gathered through the CPU path — see
        # _record_forward_timing for which calls that excludes.
        self._fwd_timings_ms: list[tuple[float, float, float, float]] = []
        self._fwd_rows_since_log = 0
        self._fwd_last_log = time.monotonic()
        self.register_buffer(
            "weight_scale",
            torch.tensor(1.0, dtype=torch.bfloat16),
            persistent=False,
        )

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        table = self.table
        if table is None:
            if self.weights_streamed:
                raise RuntimeError(
                    "PLE mmap table not initialized — load_weights ran but "
                    "build_tables did not"
                )
            return torch.zeros(
                (*ids.shape, self.embedding_dim),
                dtype=self.torch_dtype,
                device=ids.device,
            )
        if table.gpu_gather and ids.is_cuda:
            return self._forward_gpu(ids, table)
        sync_t = time.monotonic()
        ids_np = ids.detach().to("cpu", non_blocking=False).numpy().reshape(-1)
        if ((ids_np < 0) | (ids_np >= table.rows_total)).any():
            # Only reachable from a CUDA-graph capture pass (module
            # docstring): the executed hash always bounds ids via
            # torch.remainder(...) + offsets (Qwen4ExpNGramEmbedding.
            # _hash_ngram_ids in ple_layer.py), so a real forward can never
            # produce an out-of-range id. Capture RECORDS this op instead of
            # running it, so `ids` here can be read off an unexecuted
            # subgraph's buffer — uninitialized memory, not real ngram ids
            # — and that capture pass's output is never consumed, so
            # zero-filling instead of gathering is correct, not merely
            # tolerated. gather() itself keeps its strict IndexError for
            # every other caller (e.g. prewarm/tests).
            logger.warning_once(
                "PLE mmap: out-of-range ngram ids reached the gather "
                "boundary; tolerating as a CUDA-graph capture-pass "
                "artifact and zero-filling the output instead of "
                "gathering. See the periodic PLE mmap telemetry line's "
                "'skipped=' count for the running total."
            )
            table.record_capture_pass_skip()
            return torch.zeros(
                (*ids.shape, self.embedding_dim),
                dtype=table.torch_dtype,
                device=ids.device,
            )
        # gather_t doubles as sync_t's end-of-window read: the only work
        # between the two is the ids D2H drain and the (numpy, one-pass)
        # capture-pass bounds check above, so one monotonic() call marks
        # both boundaries and sync_ms reads as "everything the calling
        # thread blocked on before the gather".
        gather_t = time.monotonic()
        sync_ms = (gather_t - sync_t) * 1000.0
        rows = table.gather(ids_np)  # uint8 [N, row_bytes], fresh & writable
        gather_ms = (time.monotonic() - gather_t) * 1000.0
        itemsize = table.itemsize
        if table.row_bytes != self.embedding_dim * itemsize:
            raise ValueError(
                f"PLE mmap: table row_bytes={table.row_bytes} does not "
                f"match embedding_dim={self.embedding_dim} * "
                f"itemsize={itemsize}"
            )
        h2d_t = time.monotonic()
        out = torch.from_numpy(rows).view(table.torch_dtype)
        # non_blocking=True has no effect here: `rows` (from table.gather)
        # is pageable host memory, not pinned, so this H2D copy is
        # effectively synchronous. Pinned staging (Phase-4 lever 5) is a
        # separate, not-yet-pulled lever, not something this line hides.
        out = out.to(ids.device, non_blocking=True)
        h2d_ms = (time.monotonic() - h2d_t) * 1000.0
        self._record_forward_timing(int(ids_np.size), sync_ms, gather_ms, h2d_ms)
        return out.reshape(*ids.shape, self.embedding_dim)

    def _record_forward_timing(
        self, rows: int, sync_ms: float, gather_ms: float, h2d_ms: float
    ) -> None:
        """Rate-limited log of this instance's forward CPU-blocking time.

        The gather-side line (``MmapPleTable._record``) reports only the
        middle of the three steps a CPU-path forward blocks on, so an
        operator comparing bench arms cannot tell whether the ids D2H
        drain, the mmap gather, or the H2D copy back is dominant. Each
        field here is "time this call blocked the calling thread", which is
        the number that feeds inter-token latency; ``h2d_call_ms`` is a
        copy cost, not a bandwidth figure.

        Always on, matching ``_record``'s always-on 60s-rate-limited
        posture: the cost is four ``time.monotonic()`` calls and one list
        append per forward, negligible beside the gather, and one extra log
        line per PLE layer per rank.

        Two forward paths deliberately never reach here. The table-unset
        and capture-pass zero-fills never gather, so there is nothing to
        attribute; and ``_forward_gpu`` is fully asynchronous — every step
        it enqueues completes after it returns, so a wall-clock split of it
        would measure launch overhead and read as a suspiciously fast
        gather.

        The window mixes two populations (decode: small, frequent;
        prefill: large, rare), so a p99-only line is virtually always a
        prefill sample and says nothing about what most calls cost. ``n=``
        plus a p50 split alongside p99 — the list is already sorted, so
        this is nearly free — gives both ends without guessing which
        population p99 came from.
        """
        total_ms = sync_ms + gather_ms + h2d_ms
        self._fwd_timings_ms.append((total_ms, sync_ms, gather_ms, h2d_ms))
        self._fwd_rows_since_log += rows
        now = time.monotonic()
        if now - self._fwd_last_log < _LOG_INTERVAL_S:
            return
        # Sorted by total_ms, same as _record: each percentile reports its
        # OWN sample's split, never an average across fast and slow calls.
        timings = sorted(self._fwd_timings_ms)
        n = len(timings)
        p50_idx = max(0, math.ceil(n * 0.50) - 1)
        p99_idx = max(0, math.ceil(n * 0.99) - 1)
        p50_total, p50_sync, p50_gather, p50_h2d = (
            timings[p50_idx] if timings else (0.0, 0.0, 0.0, 0.0)
        )
        p99_total, p99_sync, p99_gather, p99_h2d = (
            timings[p99_idx] if timings else (0.0, 0.0, 0.0, 0.0)
        )
        # Every key below is unique within this line and is not a substring
        # of any gather-side key: fwd_rows=/fwd_p99_ms= carry a prefix
        # because _record already owns bare rows=/p99_ms=, and each
        # percentile-scoped field spells out its own p50_/p99_ prefix
        # rather than leaving one of the pair bare — a bare sync_ms= would
        # be a substring of p50_sync_ms=, so a reader grepping for the p99
        # split would silently be handed the p50 value. The containment is
        # one-way: bare rows=/p99_ms= DO match this line's prefixed keys,
        # so a gather-side grep has to anchor on a key this line does not
        # carry (pending=, errors=, skipped=, serial=).
        logger.info(
            "PLE mmap forward: fwd_rows=%d n=%d p50_ms=%.2f p50_sync_ms=%.2f "
            "p50_gather_ms=%.2f p50_h2d_call_ms=%.2f fwd_p99_ms=%.2f "
            "p99_sync_ms=%.2f p99_gather_ms=%.2f p99_h2d_call_ms=%.2f",
            self._fwd_rows_since_log,
            n,
            p50_total,
            p50_sync,
            p50_gather,
            p50_h2d,
            p99_total,
            p99_sync,
            p99_gather,
            p99_h2d,
        )
        self._fwd_timings_ms.clear()
        self._fwd_rows_since_log = 0
        self._fwd_last_log = now

    def _forward_gpu(self, ids: torch.Tensor, table: MmapPleTable) -> torch.Tensor:
        """Zero-copy path: the GPU dereferences the mmap'd table directly.

        Everything here is async — the CPU-side out-of-range inspection
        the CPU path needs (which costs a full device drain via
        ``ids.to("cpu")``) is replaced by the kernel's in-range mask.
        """
        itemsize = table.itemsize
        if table.row_bytes != self.embedding_dim * itemsize:
            raise ValueError(
                f"PLE mmap: table row_bytes={table.row_bytes} does not "
                f"match embedding_dim={self.embedding_dim} * "
                f"itemsize={itemsize}"
            )
        ids_flat = ids.reshape(-1)
        if ids_flat.dtype != torch.int64:
            ids_flat = ids_flat.long()
        out = torch.empty(
            (ids_flat.numel(), table.row_bytes),
            dtype=torch.uint8,
            device=ids.device,
        )
        table.gather_gpu(ids_flat.contiguous(), out)
        return out.view(table.torch_dtype).reshape(*ids.shape, self.embedding_dim)


def set_weight_scale(
    embedding: MmapNgramEmbedding, weight: torch.Tensor, device: torch.device
) -> None:
    """Register the table's global scale on the placeholder.

    Quantized (``requires_scale``) dtypes only — an unquantized dtype like
    BF16 has no scale on disk and never routes through this function.

    Called from ``Qwen4ExpNGramEmbedding.load_weights`` as it intercepts
    ``ngram_embedding.weight_scale`` from the streamed weight iterator (and
    from :func:`_resolve_scale` for a header read), so ``device`` should
    match wherever the rest of the module already lives
    (e.g. an existing buffer's device) rather than being hardcoded — this is
    what lets seam tests build everything on CPU with no GPU present.
    """
    embedding.register_buffer(
        "weight_scale", weight.detach().to(device=device), persistent=False
    )
    embedding.weight_scale_loaded = True


# --------------------------------------------------------------------------- #
# Startup guard: the CPU mmap gather must never run
# inside CUDA graph capture.
# --------------------------------------------------------------------------- #
def check_cudagraph_safety(compilation_config: CompilationConfig) -> None:
    """Raise if VLLM_PLE_MMAP=1 would run the CPU gather inside a capture.

    Three independent checks (R1.1/R3.4/R4.2), any of which alone would miss
    a real route into a capture:
      * FULL cudagraph modes capture decode outside the fx graph regardless
        of splitting_ops membership.
      * enforce-eager (mode != VLLM_COMPILE) does not fully suppress capture
        on this model and leaves splitting_ops empty.
      * an operator-supplied ``-cc.splitting_ops`` list, or an attn-fusion
        reset, can silently drop our op from the split set even under
        PIECEWISE + VLLM_COMPILE.

    Raises:
        RuntimeError: any of the above conditions holds.
    """
    if compilation_config.cudagraph_mode.has_full_cudagraphs():
        raise RuntimeError(
            "VLLM_PLE_MMAP=1 requires piecewise-only CUDA graphs (the "
            "hashing+gather forward cannot run inside a capture); "
            "got cudagraph_mode="
            f"{compilation_config.cudagraph_mode}. Pass "
            "-cc.cudagraph_mode=PIECEWISE."
        )
    if compilation_config.mode != CompilationMode.VLLM_COMPILE:
        raise RuntimeError(
            "VLLM_PLE_MMAP=1 requires compilation_config.mode="
            "CompilationMode.VLLM_COMPILE; enforce-eager does not fully "
            f"suppress CUDA graph capture on this model. Got mode="
            f"{compilation_config.mode}."
        )
    if QUALIFIED_OP_NAME not in (compilation_config.splitting_ops or []):
        raise RuntimeError(
            f"VLLM_PLE_MMAP=1 requires {QUALIFIED_OP_NAME!r} in "
            "compilation_config.splitting_ops (an operator-supplied "
            "-cc.splitting_ops list, or an attn-fusion reset, can drop it). "
            f"Got splitting_ops={compilation_config.splitting_ops!r}."
        )


# --------------------------------------------------------------------------- #
# Table directory resolution (R3.2/R4.5).
# --------------------------------------------------------------------------- #
def table_dir() -> str | None:
    """The configured ``VLLM_PLE_MMAP_DIR`` override, or ``None``.

    Returns:
        The validated absolute directory, or ``None`` when the override is
        unset/empty or the mmap path is disabled altogether.

    Raises:
        RuntimeError: the override is set but is not an absolute path, or
            does not name an existing directory. Fail closed: silently
            falling back to the checkpoint's own table would serve a
            different table than the operator asked for, and the two
            differ in dtype and values.
    """
    raw = os.environ.get("VLLM_PLE_MMAP_DIR", "").strip()
    if not raw or not enabled():
        return None
    if not os.path.isabs(raw):
        raise RuntimeError(
            f"PLE mmap: VLLM_PLE_MMAP_DIR={raw!r} is not an absolute path. "
            "Point it at the absolute directory holding the PLE table's "
            "safetensors shards (the in-container path when serving under "
            "docker), or unset it to serve the table out of the checkpoint "
            "directory."
        )
    if not os.path.isdir(raw):
        raise RuntimeError(
            f"PLE mmap: VLLM_PLE_MMAP_DIR={raw!r} is not a directory. "
            "Create it (a directory of safetensors shards, or hard links to "
            "them, whose headers carry the PLE tensor names), fix the path, "
            "or unset it to serve the table out of the checkpoint directory."
        )
    return raw


def resolve_table_path(model_config: ModelConfig) -> str:
    """Resolve the directory the PLE table's shards are discovered from.

    ``VLLM_PLE_MMAP_DIR`` wins when set, which decouples the table from the
    served checkpoint: one on-disk table can back several composed
    checkpoints without grafting a copy into each. The table directory then
    carries the whole contract on its own — the PLE tensor names in its
    shard headers, plus ``ngram_embedding.weight_scale`` for a dtype that
    needs one, since nothing from it streams through the weight iterator
    (see :func:`_resolve_scale`).

    Args:
        model_config: the served model's config, used only when no override
            is set.

    Returns:
        A local directory holding the PLE table's safetensors shards.
    """
    return table_dir() or resolve_model_path(model_config)


def resolve_model_path(model_config: ModelConfig) -> str:
    """Resolve ``model_config`` to a local directory holding the checkpoint.

    Mirrors ``DefaultModelLoader._prepare_weights``, whose resolved local
    folder is a local variable never stored on any config object: verbatim
    if ``model_weights``/``model`` is already an existing directory, else an
    OFFLINE ``snapshot_download`` (never treat a repo id as a raw path).
    """
    path = model_config.model_weights or model_config.model
    if os.path.isdir(path):
        return path
    from huggingface_hub import snapshot_download

    return snapshot_download(
        model_config.model,
        revision=model_config.revision,
        allow_patterns=["*.safetensors"],
        local_files_only=True,
    )


def _extract_layer_idx(layer_name: str) -> int:
    match = _LAYER_IDX_RE.search(layer_name)
    if not match:
        raise RuntimeError(
            f"PLE mmap: cannot find a decoder layer index in {layer_name!r}"
        )
    return int(match.group(1))


def _validate_layer_shards(
    layer_shards: _LayerShards, head_dim: int, layer_idx: int, model_path: str
) -> tuple[str, int, int, str] | None:
    """Shared fail-closed checks between :func:`validate_shards_for` (cheap,
    construction-time) and :func:`_attach_table` (authoritative, attach-time)
    — the same class of validation runs at both points (M3), just at
    different times relative to the checkpoint's streamed load.

    Returns:
        The layer's validated ``scale_entry``, or ``None`` when the
        discovered dtype's descriptor has ``requires_scale=False``.
    """
    if layer_shards.cols != head_dim:
        raise RuntimeError(
            f"PLE mmap: layer {layer_idx} shard width {layer_shards.cols} "
            f"!= head_dim {head_dim}"
        )
    desc = _PLE_DTYPES.get(layer_shards.dtype_str)
    if desc is None:
        raise RuntimeError(
            f"PLE mmap: layer {layer_idx} shards have unsupported dtype "
            f"{layer_shards.dtype_str!r}; only {sorted(_PLE_DTYPES)} "
            "are supported (F8_E5M2 is refused: is_fp8() does not "
            "recognize it, so dequant would silently never fire)"
        )
    if not desc.requires_scale:
        if layer_shards.scale_entry is not None:
            raise RuntimeError(
                f"PLE mmap: layer {layer_idx} is {layer_shards.dtype_str} "
                "(unquantized, no scale required) but has an "
                f"ngram_embedding.weight_scale on disk under {model_path} — "
                "a half-finished fp8-to-bf16 conversion looks exactly like "
                "this. Drop the stray scale, or export the shards in a "
                "dtype that requires one."
            )
        return None
    if layer_shards.scale_entry is None:
        raise RuntimeError(
            f"PLE mmap: layer {layer_idx} has {layer_shards.dtype_str} "
            f"shards but no ngram_embedding.weight_scale under {model_path} "
            "— a quantized table must carry its scale alongside its rows. "
            "Add the weight_scale tensor to a shard in that directory, or "
            "point VLLM_PLE_MMAP_DIR at a table that has one."
        )
    _scale_path, _scale_offset, scale_nbytes, scale_dtype_str = layer_shards.scale_entry
    scale_torch_dtype = _SCALE_TORCH_DTYPES.get(scale_dtype_str)
    if scale_torch_dtype is not None:
        expected_nbytes = get_dtype_size(scale_torch_dtype)
        if scale_nbytes != expected_nbytes:
            raise RuntimeError(
                f"PLE mmap: layer {layer_idx} weight_scale is {scale_nbytes} "
                f"bytes, expected {expected_nbytes} for a single "
                f"{scale_dtype_str} scalar — per-channel PLE scales are "
                "unsupported (a header read would silently keep only the "
                "first element); export a single global scale for this layer"
            )
    return layer_shards.scale_entry


# --------------------------------------------------------------------------- #
# Construction-time validation (M3): cheap, header-only checks run from
# Qwen4ExpNGramEmbedding.__init__, before the ~78 GiB backbone streams.
# --------------------------------------------------------------------------- #
def validate_shards_for(
    model_config: ModelConfig, layer_name: str, head_dim: int
) -> None:
    """Refuse a bad checkpoint at construction time, not after the load.

    Header-only checks (path resolution, shard presence, dtype, width,
    weight_scale existence — the same class of fail-closed validation
    :func:`_attach_table` performs, just runnable before any weight
    streams). Row-count-per-shard and the streamed-vs-header scale
    cross-check stay exclusively in :func:`_attach_table` (R3.6/R4.4
    unchanged): those need the checkpoint's declared vocab size and the
    weights that only arrive during the real streamed load.

    Tolerates an unresolvable model path (e.g. a bare repo id with no local
    snapshot yet — common for ``--load-format dummy``/test construction):
    logs and returns rather than raising, since :func:`build_tables` still
    fail-closes at the real load if the checkpoint is genuinely broken, so
    skipping here masks nothing. A configured ``VLLM_PLE_MMAP_DIR`` is NOT
    covered by that tolerance: an operator-supplied path is either usable or
    a typo, and nothing later in the load can make it resolve.

    Raises:
        RuntimeError: ``VLLM_PLE_MMAP_DIR`` is set but unusable, or the
            table path resolves but shards are missing, wrong-dtype,
            wrong-width, or scale-less.
    """
    model_path = table_dir()
    if model_path is None:
        try:
            model_path = resolve_model_path(model_config)
        except Exception:
            logger.warning(
                "PLE mmap: %s: cannot resolve model path to pre-validate "
                "shards at construction time; deferring to load time",
                layer_name,
            )
            return
    layer_idx = _extract_layer_idx(layer_name)
    layer_shards = discover_shards(model_path).get(layer_idx)
    if layer_shards is None:
        raise RuntimeError(
            f"PLE mmap: no shard tensors for layer {layer_idx} "
            f"({layer_name!r}) under {model_path}"
        )
    _validate_layer_shards(layer_shards, head_dim, layer_idx, model_path)


# --------------------------------------------------------------------------- #
# Table construction, invoked once from the top-level model's load_weights.
# --------------------------------------------------------------------------- #
def build_tables(
    model_config: ModelConfig, compilation_config: CompilationConfig
) -> None:
    """Build and bounded-prewarm the table for every enabled PLE layer.

    Called from both ``Qwen4ExpForConditionalGeneration.load_weights`` and
    ``Qwen4ExpForCausalLM.load_weights`` after their respective streamed
    weight passes complete (R3.6/R4.4) — never from
    ``Qwen4ExpNGramEmbedding.load_weights``, which would stream the whole
    table mid-load during the tightest memory transient (R2.19). Since the
    ConditionalGeneration wrapper composes CausalLM internally, a single
    real load can reach this function twice.

    Two costs are avoided on that redundant second call: ``model_path`` is
    resolved (cheap: a directory check or a local HF cache lookup) so every
    already-attached layer can still be verified against it, but the
    ``pending`` list (layers whose table is still ``None``) is computed
    BEFORE the expensive part — ``discover_shards``' header scan of every
    checkpoint file — and that scan, plus the whole attach loop, is skipped
    entirely once ``pending`` is empty. This also restores the "prewarm on
    the return path" guarantee by construction: ``_attach_table`` only ever
    runs for a layer that has never been attached.

    On the ConditionalGeneration path, the inner ``Qwen4ExpForCausalLM``
    call reaches this function (and its prewarm) BEFORE the outer
    ``AutoWeightsLoader.load_weights`` call has fully returned — i.e.
    mid-load, not after. This is safe because ``compute_prewarm_bound``
    re-reads ``MemAvailable`` at that instant rather than assuming any
    particular point in the load timeline.

    Raises:
        RuntimeError: a PLE layer has no matching shards on disk, a
            discovered shard fails validation (invariant 4: fail closed),
            or an already-attached layer's table was built from a
            DIFFERENT table directory than the one this load resolves to
            now — reload_weights repointing ``model_config`` at a new
            checkpoint is unsupported; serving checkpoint A's mmap rows
            against checkpoint B's scale would silently corrupt output.
    """
    model_path = resolve_table_path(model_config)

    pending: list[tuple[str, Any, Any, MmapNgramEmbedding]] = []
    for layer_name, layer in compilation_config.static_forward_context.items():
        ple_embedding_module = getattr(layer, "ple_embedding", None)
        if ple_embedding_module is None:
            continue
        embedding = getattr(ple_embedding_module, "ngram_embedding", None)
        if not isinstance(embedding, MmapNgramEmbedding):
            continue
        if embedding.table is not None:
            if embedding.table.model_path != model_path:
                raise RuntimeError(
                    f"PLE mmap: layer {layer_name!r} already has a table "
                    f"built from {embedding.table.model_path!r}, but this "
                    f"load resolves to {model_path!r} — reloading weights "
                    "onto a different checkpoint is unsupported; restart "
                    "the seat"
                )
            continue
        pending.append((layer_name, layer, ple_embedding_module, embedding))
    if not pending:
        return

    shard_map = discover_shards(model_path)

    for layer_name, layer, ple_embedding_module, embedding in pending:
        layer_idx = layer.layer_idx
        layer_shards = shard_map.get(layer_idx)
        if layer_shards is None:
            raise RuntimeError(
                f"PLE mmap: no shard tensors for layer {layer_idx} "
                f"({layer_name!r}) under {model_path}"
            )
        _attach_table(
            embedding,
            layer_shards,
            split_ngram_parts=ple_embedding_module.split_ngram_parts,
            layer_idx=layer_idx,
            model_path=model_path,
        )


def _resolve_scale(
    embedding: MmapNgramEmbedding,
    scale_entry: tuple[str, int, int, str],
    layer_idx: int,
    from_table_dir: bool,
) -> None:
    """Make the placeholder carry the scale belonging to the rows about to
    be attached, for a dtype whose descriptor requires one.

    Three sources, in the order they can be trusted:

    * ``VLLM_PLE_MMAP_DIR`` mode — the table directory is the sole
      authority. Nothing in it streams through the weight iterator, so the
      scale is read from its shard headers. Any scale the CHECKPOINT
      streamed describes the checkpoint's own PLE table, which is not the
      one being attached; using it would be exactly the "checkpoint A's
      rows against checkpoint B's scale" corruption :func:`build_tables`
      guards against, so it is overridden, loudly.
    * a streamed scale (checkpoint mode) — cross-checked against an
      independent direct read of the same tensor off disk: two
      self-consistent halves (R4.13's philosophy applied to the scale, not
      just row mapping), and a mismatch means the streamed weight iterator
      silently renamed or skipped something.
    * no streamed scale and nothing streamed at all (checkpoint mode) — a
      loader topology that never routed this family here. Nothing was lost,
      so fall back to the header read rather than fail closed on it.

    Raises:
        RuntimeError: rows streamed but their scale did not (a broken or
            truncated weight iterator — the one case with something
            genuinely missing), or the streamed and on-disk scales disagree.
    """
    if from_table_dir:
        if embedding.weight_scale_loaded:
            logger.warning(
                "PLE mmap: layer %d overriding the checkpoint's streamed "
                "weight_scale with the one in VLLM_PLE_MMAP_DIR — the table "
                "served from that directory carries its own scale, and the "
                "streamed value belongs to the checkpoint's own PLE table",
                layer_idx,
            )
        # Same device the streamed path resolves to: the buffer being replaced.
        set_weight_scale(
            embedding, _read_scale(scale_entry), embedding.weight_scale.device
        )
        return
    if not embedding.weight_scale_loaded:
        if embedding.weights_streamed:
            raise RuntimeError(
                f"PLE mmap: layer {layer_idx} weight_scale was never loaded "
                "from the checkpoint's streamed weights"
            )
        logger.warning(
            "PLE mmap: layer %d weight_scale falling back to a direct header "
            "read — this layer's ngram_embedding family was never streamed "
            "through the checkpoint loader",
            layer_idx,
        )
        set_weight_scale(
            embedding, _read_scale(scale_entry), embedding.weight_scale.device
        )
        return
    header_scale = _read_scale(scale_entry).float()
    streamed_scale = embedding.weight_scale.detach().to("cpu").float()
    if not torch.allclose(header_scale, streamed_scale, atol=1e-6):
        # .tolist()[:4], not .item(): a malformed (non-scalar) streamed
        # scale must not crash the diagnostic itself with an unrelated
        # "cannot be converted to Scalar" error.
        raise RuntimeError(
            f"PLE mmap: layer {layer_idx} weight_scale mismatch between the "
            f"streamed checkpoint ({streamed_scale.flatten().tolist()[:4]}) "
            f"and the header-parsed value "
            f"({header_scale.flatten().tolist()[:4]})"
        )


def _attach_table(
    embedding: MmapNgramEmbedding,
    layer_shards: _LayerShards,
    split_ngram_parts: int,
    layer_idx: int,
    model_path: str,
) -> None:
    scale_entry = _validate_layer_shards(
        layer_shards, embedding.embedding_dim, layer_idx, model_path
    )
    desc = _PLE_DTYPES[layer_shards.dtype_str]
    if desc.requires_scale:
        assert scale_entry is not None  # guaranteed by _validate_layer_shards
        _resolve_scale(embedding, scale_entry, layer_idx, table_dir() is not None)
    # else: an unquantized dtype (e.g. BF16) has no scale to resolve.
    # _validate_layer_shards already refused a stray on-disk scale, so
    # validation is the sole authority here and this path deliberately
    # ignores weight_scale_loaded/weights_streamed: a real loader never
    # calls set_weight_scale for such a table (there is nothing to stream),
    # so weight_scale_loaded=True on an unscaled embedding only ever
    # happens in a test double, and attaching must not chase that state.

    vocab = embedding.org_vocab_size
    # Verbatim shard-placement math from Qwen4ExpNGramEmbedding.load_weights
    # (nvidia/ple_layer.py) — discovery and gather must stay two
    # self-consistent halves of the same mapping (R4.13).
    shard_size = (vocab + split_ngram_parts - 1) // split_ngram_parts
    num_expected_shards = min(
        split_ngram_parts, (vocab + shard_size - 1) // shard_size if shard_size else 0
    )
    missing = [i for i in range(num_expected_shards) if i not in layer_shards.shards]
    if missing:
        raise RuntimeError(
            f"PLE mmap: layer {layer_idx} missing shard(s) {missing} of "
            f"{num_expected_shards} under {model_path}"
        )
    for shard_index, (_path, _offset, rows) in layer_shards.shards.items():
        if shard_index >= split_ngram_parts:
            raise RuntimeError(
                f"PLE mmap: layer {layer_idx} shard {shard_index} exceeds "
                f"split_ngram_parts={split_ngram_parts}"
            )
        checkpoint_start = shard_index * shard_size
        expected_rows = max(0, min(shard_size, vocab - checkpoint_start))
        if rows != expected_rows:
            raise RuntimeError(
                f"PLE mmap: layer {layer_idx} shard {shard_index} has "
                f"{rows} rows, expected {expected_rows}"
            )

    if embedding.table is not None:
        # Defensive: build_tables' own idempotency skip (table is not None
        # -> layer skipped) should make this unreachable in practice, but a
        # direct _attach_table re-entry must not leak the old ThreadPool.
        embedding.table.close()
        embedding.table = None

    row_bytes = layer_shards.cols * _itemsize(layer_shards.dtype_str)
    table = MmapPleTable(
        layer_shards.shards,
        shard_size,
        row_bytes,
        desc.torch_dtype,
        workers=envs.VLLM_PLE_MMAP_WORKERS,
        chunk=envs.VLLM_PLE_MMAP_CHUNK,
        model_path=model_path,
        serial=serial_threshold(),
    )
    embedding.torch_dtype = table.torch_dtype
    table_bytes = table.rows_total * row_bytes

    if envs.VLLM_PLE_MMAP_PREWARM:
        bound = compute_prewarm_bound(table_bytes, _mem_available_bytes())
        read = table.prewarm(bound)
        logger.info(
            "PLE mmap: layer %d prewarm read %.2f GiB (budget %.2f GiB)",
            layer_idx,
            read / (1 << 30),
            bound / (1 << 30),
        )

    embedding.table = table
    logger.info(
        "PLE mmap: layer %d attached, %d shards, %d rows x %d B "
        "(%.2f GiB on disk), %d workers, gpu_gather=%s, serial=%d",
        layer_idx,
        len(layer_shards.shards),
        table.rows_total,
        row_bytes,
        table_bytes / (1 << 30),
        table.workers,
        table.gpu_gather,
        table.serial,
    )


# --------------------------------------------------------------------------- #
# Custom op: only the mmap gather crosses the boundary; the trigram hashing
# is symbolic-shape-safe and stays inside the compiled graph.
# --------------------------------------------------------------------------- #
def _qwen4_exp_ple_mmap_gather(
    ngram_ids: torch.Tensor, output: torch.Tensor, layer_name: str
) -> None:
    from vllm.forward_context import get_forward_context

    try:
        layer = get_forward_context().no_compile_layers[layer_name]
    except KeyError:
        raise RuntimeError(
            f"PLE mmap: {layer_name!r} is not registered in no_compile_layers"
        ) from None
    ple_embedding_module = getattr(layer, "ple_embedding", None)
    if ple_embedding_module is None:
        raise RuntimeError(f"PLE mmap: {layer_name!r} does not resolve to a PLE layer")
    result = ple_embedding_module.ngram_embedding(ngram_ids).flatten(-2)
    output.copy_(result)


def _qwen4_exp_ple_mmap_gather_fake(
    ngram_ids: torch.Tensor, output: torch.Tensor, layer_name: str
) -> None:
    return


direct_register_custom_op(
    op_name=OP_NAME,
    op_func=_qwen4_exp_ple_mmap_gather,
    mutates_args=["output"],
    fake_impl=_qwen4_exp_ple_mmap_gather_fake,
)
# The op above registers under the platform-default dispatch key (CUDA in
# production). Unit tests run without a GPU-resident model and need the same
# op reachable with plain CPU tensors, so also register it directly under
# the CPU key (R3.12) — a second direct_register_custom_op call would
# re-define the schema and raise at MODULE IMPORT, killing every serve,
# since this module is imported unconditionally (R2.5).
if current_platform.dispatch_key != "CPU":
    vllm_lib.impl(OP_NAME, _qwen4_exp_ple_mmap_gather, dispatch_key="CPU")

__all__ = [
    "OP_NAME",
    "QUALIFIED_OP_NAME",
    "MmapNgramEmbedding",
    "MmapPleTable",
    "build_tables",
    "check_cudagraph_safety",
    "compute_prewarm_bound",
    "discover_shards",
    "enabled",
    "parse_safetensors_header",
    "resolve_model_path",
    "resolve_table_path",
    "serial_threshold",
    "set_weight_scale",
    "table_dir",
    "validate_shards_for",
]
