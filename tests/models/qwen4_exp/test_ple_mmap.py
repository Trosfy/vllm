# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Phase 1 tests for the mmap-backed PLE table (VLLM_PLE_MMAP).

No GPU, no real checkpoint: synthetic fp8 safetensors fixtures stand in for
the RadixArk/Qwen3.8-Flash-Next-NVFP4 PLE shards, and the custom op is
exercised through its CPU dispatch key (R3.12).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import safetensors.torch
import torch
from torch import nn

import vllm.envs as envs
import vllm.forward_context as forward_context
import vllm.model_executor.layers.linear as linear_module
import vllm.model_executor.layers.vocab_parallel_embedding as embedding_module
import vllm.model_executor.parameter as parameter_module
import vllm.models.qwen4_exp.nvidia.model as model_module
import vllm.models.qwen4_exp.nvidia.ple_mmap as ple_mmap
from vllm.config import CompilationConfig, set_current_vllm_config
from vllm.config.compilation import CompilationMode, CUDAGraphMode
from vllm.model_executor.layers.quantization.fp8 import Fp8Config
from vllm.model_executor.layers.quantization.utils.fp8_utils import is_fp8
from vllm.models.qwen4_exp.nvidia.ple_layer import (
    Qwen4ExpNGramEmbedding,
    Qwen4ExpPLELayer,
)

# --------------------------------------------------------------------------- #
# Fixtures / helpers
# --------------------------------------------------------------------------- #


@pytest.fixture(autouse=True)
def _reset_ple_mmap_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every test starts from a clean, default-off environment."""
    for name in (
        "VLLM_PLE_MMAP",
        "VLLM_PLE_MMAP_DIR",
        "VLLM_PLE_MMAP_WORKERS",
        "VLLM_PLE_MMAP_CHUNK",
        "VLLM_PLE_MMAP_PREWARM",
        "VLLM_PLE_MMAP_GPU_GATHER",
        "VLLM_PLE_MMAP_SERIAL",
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(autouse=True)
def _allow_single_rank_tensor_parallel(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stock VocabParallelEmbedding needs a TP group; stand in a rank-0/size-1
    world without paying for real torch.distributed init (mirrors test_ple.py).
    """
    monkeypatch.setattr(embedding_module, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        embedding_module, "get_tensor_model_parallel_world_size", lambda: 1
    )
    monkeypatch.setattr(parameter_module, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        parameter_module, "get_tensor_model_parallel_world_size", lambda: 1
    )


def _make_text_config(**overrides: object) -> SimpleNamespace:
    values: dict[str, object] = dict(
        ngram_size=3,
        heads_per_ngram=2,
        eos_token_id=0,
        vocab_size=200,
        split_ngram_parts=4,
        ngram_vocab_size_base=1000,
        make_ngram_vocab_size_divisible_by=1,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _synthetic_weight(
    vocab: int,
    cols: int,
    layer_idx: int = 0,
    dtype: torch.dtype = torch.float8_e4m3fn,
) -> torch.Tensor:
    """Deterministic, layer-dependent values (never all-zero/uniform, and
    distinguishable across layers so per-layer-keying tests are meaningful).
    """
    raw = torch.arange(vocab * cols, dtype=torch.float32).reshape(vocab, cols)
    raw = torch.remainder(raw + layer_idx * 97, 6.0) - 3.0
    return raw.to(dtype)


def _write_ple_layer(
    directory: Path,
    *,
    layer_idx: int,
    vocab: int,
    parts: int,
    cols: int,
    scale: float,
    write_scale: bool = True,
    scale_dtype: torch.dtype = torch.bfloat16,
    table_dtype: torch.dtype = torch.float8_e4m3fn,
) -> torch.Tensor:
    """Write one PLE layer's shard + weight_scale tensors as synthetic
    safetensors files (no model.safetensors.index.json, matching the real
    checkpoint). Returns the full logical [vocab, cols] table in table_dtype.
    """
    prefix = (
        f"model.language_model.layers.{layer_idx}.ple.ple_embedding.ngram_embedding"
    )
    shard_size = (vocab + parts - 1) // parts
    full = _synthetic_weight(vocab, cols, layer_idx, dtype=table_dtype)
    for shard_index in range(parts):
        start = shard_index * shard_size
        rows = max(0, min(shard_size, vocab - start))
        tensors: dict[str, torch.Tensor] = {}
        if rows > 0:
            tensors[f"{prefix}.shard_{shard_index}.weight"] = full[start : start + rows]
        if write_scale and shard_index == 0:
            tensors[f"{prefix}.weight_scale"] = torch.tensor([scale], dtype=scale_dtype)
        if tensors:
            safetensors.torch.save_file(
                tensors,
                str(directory / f"model-ple-{layer_idx}-{shard_index:05d}.safetensors"),
            )
    return full


def _attached_embedding(
    directory: Path, layer_idx: int, vocab: int, parts: int, cols: int, scale: float
) -> ple_mmap.MmapNgramEmbedding:
    """Build a placeholder wired to an on-disk checkpoint via the same path
    build_tables uses, for tests that don't need the full static_forward_context
    walk.
    """
    shard_map = ple_mmap.discover_shards(str(directory))
    embedding = ple_mmap.MmapNgramEmbedding(vocab, cols)
    ple_mmap.set_weight_scale(
        embedding, torch.tensor([scale], dtype=torch.bfloat16), torch.device("cpu")
    )
    ple_mmap._attach_table(
        embedding,
        shard_map[layer_idx],
        split_ngram_parts=parts,
        layer_idx=layer_idx,
        model_path=str(directory),
    )
    return embedding


def _record_info(
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[str, tuple[Any, ...]]]:
    """Capture plain logger.info calls — the rate-limited-log-line pattern
    shared by MmapPleTable._record and
    MmapNgramEmbedding._record_forward_timing.

    Args are typed Any, not object: callers unpack these %-format args and
    do arithmetic on the numeric ones.
    """
    recorded: list[tuple[str, tuple[Any, ...]]] = []
    monkeypatch.setattr(
        ple_mmap.logger, "info", lambda msg, *args: recorded.append((msg, args))
    )
    return recorded


# --------------------------------------------------------------------------- #
# safetensors header parsing
# --------------------------------------------------------------------------- #


def test_parse_safetensors_header_returns_metadata_and_data_start(
    tmp_path: Path,
) -> None:
    path = tmp_path / "x.safetensors"
    safetensors.torch.save_file({"a": torch.zeros(3, 2)}, str(path))

    header, data_start = ple_mmap.parse_safetensors_header(str(path))

    assert header["a"]["shape"] == [3, 2]
    assert data_start == path.stat().st_size - (3 * 2 * 4)  # F32 = 4 bytes/elem


def test_parse_safetensors_header_rejects_oversized_header(tmp_path: Path) -> None:
    path = tmp_path / "big_header.safetensors"
    with open(path, "wb") as f:
        f.write((ple_mmap._MAX_HEADER_BYTES + 1).to_bytes(8, "little"))
        f.write(b"\x00" * 16)

    with pytest.raises(ValueError, match="exceeding the"):
        ple_mmap.parse_safetensors_header(str(path))


def test_parse_safetensors_header_rejects_offsets_outside_file(tmp_path: Path) -> None:
    import json
    import struct

    header = {"a": {"dtype": "F8_E4M3", "shape": [4, 4], "data_offsets": [0, 1000]}}
    body = json.dumps(header).encode()
    path = tmp_path / "bad_offsets.safetensors"
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(body)))
        f.write(body)
        f.write(b"\x00" * 16)  # far short of the declared 1000-byte tensor

    with pytest.raises(ValueError, match="fall outside the file"):
        ple_mmap.parse_safetensors_header(str(path))


def test_parse_safetensors_header_rejects_truncated_length(tmp_path: Path) -> None:
    path = tmp_path / "truncated.safetensors"
    path.write_bytes(b"\x01\x02\x03")

    with pytest.raises(ValueError, match="truncated safetensors header length"):
        ple_mmap.parse_safetensors_header(str(path))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_read_scale_round_trips_across_dtypes(
    tmp_path: Path, dtype: torch.dtype
) -> None:
    value = -1.5  # exactly representable in fp32/fp16/bf16
    path = tmp_path / "scale.safetensors"
    safetensors.torch.save_file(
        {"scale": torch.tensor([value], dtype=dtype)}, str(path)
    )
    header, data_start = ple_mmap.parse_safetensors_header(str(path))
    start, end = header["scale"]["data_offsets"]
    entry = (str(path), data_start + start, end - start, header["scale"]["dtype"])

    got = ple_mmap._read_scale(entry)

    assert got.item() == pytest.approx(value)


# --------------------------------------------------------------------------- #
# Shard discovery
# --------------------------------------------------------------------------- #


def test_discover_shards_finds_layer_shards_and_scale(tmp_path: Path) -> None:
    _write_ple_layer(tmp_path, layer_idx=1, vocab=37, parts=5, cols=4, scale=0.5)

    result = ple_mmap.discover_shards(str(tmp_path))

    assert set(result.keys()) == {1}
    layer = result[1]
    assert layer.cols == 4
    assert layer.dtype_str == "F8_E4M3"
    assert layer.scale_entry is not None
    # shard_size = ceil(37/5) = 8; last shard truncated to 5 rows.
    assert set(layer.shards.keys()) == {0, 1, 2, 3, 4}
    assert layer.shards[4][2] == 5


def test_discover_shards_separates_multiple_ple_layers(tmp_path: Path) -> None:
    """(b): two-PLE-layer synthetic case proving per-layer keying — discovery
    must not mix shard tensors across layers even when files share a
    directory."""
    full0 = _write_ple_layer(
        tmp_path, layer_idx=0, vocab=10, parts=3, cols=2, scale=0.25
    )
    full1 = _write_ple_layer(
        tmp_path, layer_idx=1, vocab=20, parts=4, cols=2, scale=0.75
    )

    result = ple_mmap.discover_shards(str(tmp_path))

    assert set(result.keys()) == {0, 1}
    assert {p for p, _o, _r in result[0].shards.values()}.isdisjoint(
        {p for p, _o, _r in result[1].shards.values()}
    )
    assert not full0.equal(full1[:10])  # fixtures are genuinely distinct


def test_discover_shards_rejects_mixed_dtype_within_a_layer(tmp_path: Path) -> None:
    prefix = "model.language_model.layers.0.ple.ple_embedding.ngram_embedding"
    safetensors.torch.save_file(
        {f"{prefix}.shard_0.weight": torch.zeros(2, 2).to(torch.float8_e4m3fn)},
        str(tmp_path / "a.safetensors"),
    )
    safetensors.torch.save_file(
        {f"{prefix}.shard_1.weight": torch.zeros(2, 2).to(torch.float8_e5m2)},
        str(tmp_path / "b.safetensors"),
    )

    with pytest.raises(ValueError, match="mixed shard dtypes"):
        ple_mmap.discover_shards(str(tmp_path))


def test_discover_shards_rejects_mixed_width_within_a_layer(tmp_path: Path) -> None:
    prefix = "model.language_model.layers.0.ple.ple_embedding.ngram_embedding"
    safetensors.torch.save_file(
        {f"{prefix}.shard_0.weight": torch.zeros(2, 4).to(torch.float8_e4m3fn)},
        str(tmp_path / "a.safetensors"),
    )
    safetensors.torch.save_file(
        {f"{prefix}.shard_1.weight": torch.zeros(2, 8).to(torch.float8_e4m3fn)},
        str(tmp_path / "b.safetensors"),
    )

    with pytest.raises(ValueError, match="mixed shard widths"):
        ple_mmap.discover_shards(str(tmp_path))


def test_discover_shards_rejects_a_header_whose_span_disagrees_with_its_shape(
    tmp_path: Path,
) -> None:
    """(F-4): a header entry whose data_offsets span doesn't match
    rows * cols * itemsize must be refused with a named error, rather than
    silently under-reading a truncated row."""
    import json
    import struct

    name = (
        "model.language_model.layers.0.ple.ple_embedding.ngram_embedding.shard_0.weight"
    )
    # shape [4, 4] of F8_E4M3 (itemsize 1) needs a 16-byte span; the header
    # declares only 12.
    header = {name: {"dtype": "F8_E4M3", "shape": [4, 4], "data_offsets": [0, 12]}}
    body = json.dumps(header).encode()
    path = tmp_path / "bad_span.safetensors"
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(body)))
        f.write(body)
        f.write(b"\x00" * 12)  # exactly the declared (too-small) span

    with pytest.raises(ValueError, match="does not match"):
        ple_mmap.discover_shards(str(tmp_path))


def test_discover_shards_rejects_a_header_with_an_unrecognized_dtype(
    tmp_path: Path,
) -> None:
    """_itemsize's None-guard: a dtype string absent from safetensors'
    _TYPES table must raise a named ValueError, not a bare KeyError."""
    import json
    import struct

    name = (
        "model.language_model.layers.0.ple.ple_embedding.ngram_embedding.shard_0.weight"
    )
    header = {
        name: {"dtype": "NOT_A_REAL_DTYPE", "shape": [4, 4], "data_offsets": [0, 16]}
    }
    body = json.dumps(header).encode()
    path = tmp_path / "bad_dtype.safetensors"
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(body)))
        f.write(body)
        f.write(b"\x00" * 16)

    with pytest.raises(ValueError, match="unrecognized safetensors dtype"):
        ple_mmap.discover_shards(str(tmp_path))


# --------------------------------------------------------------------------- #
# (d) shard-mapping contract: quotes Qwen4ExpNGramEmbedding.load_weights's
# shard-placement math verbatim.
# --------------------------------------------------------------------------- #


def _upstream_expected_rows(
    embedding: SimpleNamespace, split_ngram_parts: int, shard_index: int
) -> int:
    # Verbatim from Qwen4ExpNGramEmbedding.load_weights, including the
    # outer max(0, ...) clamp. A paraphrase dropping max(0, ...) encodes a
    # DIFFERENT function exactly at the boundary indices this test targets
    # (R4.13).
    shard_size = (embedding.org_vocab_size + split_ngram_parts - 1) // split_ngram_parts
    checkpoint_start = shard_index * shard_size
    expected_rows = max(
        0,
        min(shard_size, embedding.org_vocab_size - checkpoint_start),
    )
    return expected_rows


@pytest.mark.parametrize(
    ("org_vocab_size", "split_ngram_parts"),
    [
        (37, 5),  # last shard partially truncated (nonzero, < shard_size)
        (10, 8),  # trailing shards fully out of range (rows == 0)
    ],
)
def test_shard_mapping_matches_upstream_checkpoint_math_at_boundaries(
    tmp_path: Path, org_vocab_size: int, split_ngram_parts: int
) -> None:
    # org_vocab_size == padded_vocab_size here: VocabParallelEmbedding is
    # constructed positionally with no org_num_embeddings (R4.13).
    embedding = SimpleNamespace(org_vocab_size=org_vocab_size, embedding_dim=4)
    shard_size = (org_vocab_size + split_ngram_parts - 1) // split_ngram_parts
    cols = 4
    full = _write_ple_layer(
        tmp_path,
        layer_idx=0,
        vocab=org_vocab_size,
        parts=split_ngram_parts,
        cols=cols,
        scale=1.0,
    )
    layer_shards = ple_mmap.discover_shards(str(tmp_path))[0]
    table = ple_mmap.MmapPleTable(
        layer_shards.shards,
        shard_size,
        cols,
        torch.float8_e4m3fn,
        workers=1,
        chunk=8,
        model_path=str(tmp_path),
    )

    for shard_index in (0, 1, split_ngram_parts - 2, split_ngram_parts - 1):
        expected_rows = _upstream_expected_rows(
            embedding, split_ngram_parts, shard_index
        )
        if expected_rows == 0:
            continue
        checkpoint_start = shard_index * shard_size
        # Drive the actual boundary rows (first and last of this shard)
        # through the REAL gather path against the logical table, rather
        # than re-implementing the // and - shard/local math by hand.
        boundary_ids = np.array(
            [checkpoint_start, checkpoint_start + expected_rows - 1], dtype=np.int64
        )
        got = torch.from_numpy(table.gather(boundary_ids)).view(torch.float8_e4m3fn)
        assert torch.equal(got, full[boundary_ids])


# --------------------------------------------------------------------------- #
# MmapPleTable gather
# --------------------------------------------------------------------------- #


def test_mmap_table_gather_matches_naive_lookup(tmp_path: Path) -> None:
    full = _write_ple_layer(tmp_path, layer_idx=1, vocab=37, parts=5, cols=4, scale=0.5)
    layer_shards = ple_mmap.discover_shards(str(tmp_path))[1]
    shard_size = (37 + 5 - 1) // 5
    table = ple_mmap.MmapPleTable(
        layer_shards.shards,
        shard_size,
        4,
        torch.float8_e4m3fn,
        workers=2,
        chunk=3,
        model_path=str(tmp_path),
    )

    ids = np.array([0, 36, 5, 5, 20, 1, 31], dtype=np.int64)
    got = torch.from_numpy(table.gather(ids)).view(torch.float8_e4m3fn)

    assert torch.equal(got, full[ids])


def test_mmap_table_gather_dedupes_and_preserves_input_order(tmp_path: Path) -> None:
    full = _write_ple_layer(tmp_path, layer_idx=0, vocab=9, parts=3, cols=2, scale=1.0)
    layer_shards = ple_mmap.discover_shards(str(tmp_path))[0]
    table = ple_mmap.MmapPleTable(
        layer_shards.shards,
        3,
        2,
        torch.float8_e4m3fn,
        workers=1,
        chunk=1,
        model_path=str(tmp_path),
    )

    ids = np.array([4, 4, 0, 8, 4], dtype=np.int64)
    got = torch.from_numpy(table.gather(ids)).view(torch.float8_e4m3fn)

    assert torch.equal(got, full[ids])
    assert torch.equal(got[0], got[1])  # the duplicate resolves to the same row


def test_mmap_table_gather_rejects_out_of_range_ids(tmp_path: Path) -> None:
    full = _write_ple_layer(tmp_path, layer_idx=0, vocab=9, parts=3, cols=2, scale=1.0)
    layer_shards = ple_mmap.discover_shards(str(tmp_path))[0]
    table = ple_mmap.MmapPleTable(
        layer_shards.shards,
        3,
        2,
        torch.float8_e4m3fn,
        workers=1,
        chunk=8,
        model_path=str(tmp_path),
    )
    assert table.rows_total == 9

    with pytest.raises(IndexError, match=r"row id out of range"):
        table.gather(np.array([9_999], dtype=np.int64))

    # Exact boundary: rows_total itself is one past the last valid row.
    with pytest.raises(IndexError, match=r"row id out of range"):
        table.gather(np.array([table.rows_total], dtype=np.int64))

    # rows_total - 1 is the last valid row and must succeed.
    got = torch.from_numpy(
        table.gather(np.array([table.rows_total - 1], dtype=np.int64))
    ).view(torch.float8_e4m3fn)
    assert torch.equal(got, full[table.rows_total - 1 : table.rows_total])


def test_mmap_table_gather_empty_input_returns_empty(tmp_path: Path) -> None:
    _write_ple_layer(tmp_path, layer_idx=0, vocab=9, parts=3, cols=2, scale=1.0)
    layer_shards = ple_mmap.discover_shards(str(tmp_path))[0]
    table = ple_mmap.MmapPleTable(
        layer_shards.shards,
        3,
        2,
        torch.float8_e4m3fn,
        workers=1,
        chunk=8,
        model_path=str(tmp_path),
    )

    out = table.gather(np.empty(0, dtype=np.int64))

    assert out.shape == (0, 2)


# --------------------------------------------------------------------------- #
# Serial small-gather dispatch (VLLM_PLE_MMAP_SERIAL)
# --------------------------------------------------------------------------- #


def test_serial_threshold_defaults_to_off_and_parses_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert ple_mmap.serial_threshold() == 0  # env unset (autouse fixture)

    monkeypatch.setenv("VLLM_PLE_MMAP_SERIAL", "  256 ")
    assert ple_mmap.serial_threshold() == 256

    monkeypatch.setenv("VLLM_PLE_MMAP_SERIAL", "")
    assert ple_mmap.serial_threshold() == 0


@pytest.mark.parametrize("raw", ["yes", "256.5", "-1"])
def test_serial_threshold_refuses_a_value_that_is_not_a_row_count(
    monkeypatch: pytest.MonkeyPatch, raw: str
) -> None:
    """Fail closed rather than reading a typo as 0: a silently-off knob
    would report a clean 'serial disabled' arm for a run the operator
    believes measured it."""
    monkeypatch.setenv("VLLM_PLE_MMAP_SERIAL", raw)

    with pytest.raises(RuntimeError, match="VLLM_PLE_MMAP_SERIAL"):
        ple_mmap.serial_threshold()


def test_serial_gather_matches_pool_gather_for_the_same_ids(tmp_path: Path) -> None:
    """The knob is a dispatch swap, not a different gather: the inline and
    pooled branches must return byte-identical rows, in input order, for
    the same ids."""
    full = _write_ple_layer(tmp_path, layer_idx=0, vocab=40, parts=4, cols=8, scale=0.5)
    layer_shards = ple_mmap.discover_shards(str(tmp_path))[0]
    ids = np.array([0, 39, 12, 13, 14, 5, 5, 20, 31], dtype=np.int64)

    gathered = []
    for serial in (0, 64):  # off (pool) vs on (inline, uniq.size <= 64)
        table = ple_mmap.MmapPleTable(
            layer_shards.shards,
            10,
            8,
            torch.float8_e4m3fn,
            workers=4,
            chunk=2,
            model_path=str(tmp_path),
            serial=serial,
        )
        gathered.append(torch.from_numpy(table.gather(ids)).view(torch.float8_e4m3fn))
        table.close()

    assert torch.equal(gathered[0], full[ids])
    assert torch.equal(gathered[1], gathered[0])


def _count_pool_dispatches(
    monkeypatch: pytest.MonkeyPatch, table: ple_mmap.MmapPleTable
) -> list[int]:
    """Count table.pool.map calls without disturbing what it returns.

    One sentinel appended per call, so len(...) reads as the dispatch count.
    """
    calls: list[int] = []
    real_map = table.pool.map

    def _counting_map(fn: object, tasks: object) -> object:
        calls.append(1)
        return real_map(fn, tasks)

    monkeypatch.setattr(table.pool, "map", _counting_map)
    return calls


def test_serial_threshold_boundary_switches_inline_vs_pool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """uniq.size == N stays inline (pool.map never called); uniq.size ==
    N + 1 crosses the threshold back onto the pool."""
    _write_ple_layer(tmp_path, layer_idx=0, vocab=40, parts=4, cols=8, scale=0.5)
    layer_shards = ple_mmap.discover_shards(str(tmp_path))[0]
    table = ple_mmap.MmapPleTable(
        layer_shards.shards,
        10,
        8,
        torch.float8_e4m3fn,
        workers=4,
        chunk=1,
        model_path=str(tmp_path),
        serial=2,
    )
    calls = _count_pool_dispatches(monkeypatch, table)

    table.gather(np.array([0, 5], dtype=np.int64))  # uniq.size == 2 == N
    assert len(calls) == 0

    table.gather(np.array([0, 5, 12], dtype=np.int64))  # uniq.size == 3 > N
    assert len(calls) == 1

    table.close()


def test_serial_threshold_keys_on_distinct_rows_not_task_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A large chunk coalesces each shard's span into a single task, so a
    16-distinct-row gather spanning exactly 2 shards produces only 2 tasks
    — fewer than the serial=6 threshold. The boundary test above used
    chunk=1, where len(tasks) == uniq.size, so it cannot tell a
    uniq.size-keyed gate from a len(tasks)-keyed one; this one can. Keying
    on uniq.size is the point of the knob: hash-scattering is exactly what
    makes task count an unreliable proxy for how big a gather is."""
    _write_ple_layer(tmp_path, layer_idx=0, vocab=20, parts=2, cols=8, scale=0.5)
    layer_shards = ple_mmap.discover_shards(str(tmp_path))[0]
    table = ple_mmap.MmapPleTable(
        layer_shards.shards,
        10,
        8,
        torch.float8_e4m3fn,
        workers=4,
        chunk=2048,
        model_path=str(tmp_path),
        serial=6,
    )
    calls = _count_pool_dispatches(monkeypatch, table)

    # 8 rows from shard 0 ([0, 9]) + 8 rows from shard 1 ([10, 19]):
    # uniq.size == 16, but chunk=2048 coalesces each shard's span into one
    # task each, so len(tasks) == 2.
    ids = np.array(
        [0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15, 16, 17], dtype=np.int64
    )
    table.gather(ids)

    assert len(calls) == 1  # pooled: uniq.size (16) > serial (6)

    table.close()


def test_serial_branch_raises_the_same_named_indexerror_on_a_closed_table(
    tmp_path: Path,
) -> None:
    """The serial dispatch loop reuses run() verbatim, so a missing shard
    raises the identical named IndexError regardless of branch — exercised
    here with more than one task, which the pre-existing len(tasks) == 1
    special case never reaches. The contract under test is "a missing shard
    slot" (run()'s ``mm is None`` check), simulated cheaply by closing the
    table first, which nulls every mm slot. That is also why only the
    serial branch is exercised: gathering through the pool branch on a
    CLOSED table hits an unrelated, pre-existing divergence — pool.map
    raises RuntimeError("cannot schedule new futures after shutdown")
    straight from the shut-down executor, before run() can raise at all."""
    _write_ple_layer(tmp_path, layer_idx=0, vocab=40, parts=4, cols=8, scale=0.5)
    layer_shards = ple_mmap.discover_shards(str(tmp_path))[0]
    table = ple_mmap.MmapPleTable(
        layer_shards.shards,
        10,
        8,
        torch.float8_e4m3fn,
        workers=4,
        chunk=1,
        model_path=str(tmp_path),
        serial=64,
    )
    table.close()

    with pytest.raises(IndexError, match="shard"):
        table.gather(np.array([0, 5], dtype=np.int64))


def test_serial_field_in_the_gather_log_line_reflects_the_engaged_branch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """_record's rate-limited line gains an appended serial= field
    (append-only — rows=/p99_ms=/pending=/errors=/skipped= keep their
    names, order and meaning) reporting engaged/total gathers in the
    window."""
    _write_ple_layer(tmp_path, layer_idx=0, vocab=40, parts=4, cols=8, scale=0.5)
    layer_shards = ple_mmap.discover_shards(str(tmp_path))[0]
    table = ple_mmap.MmapPleTable(
        layer_shards.shards,
        10,
        8,
        torch.float8_e4m3fn,
        workers=4,
        chunk=2,
        model_path=str(tmp_path),
        serial=2,
    )
    logged = _record_info(monkeypatch)

    table._last_log = 0.0  # simulate the interval having elapsed
    table.gather(np.array([0, 5], dtype=np.int64))  # uniq.size == 2 <= serial
    assert len(logged) == 1
    msg, args = logged[0]
    assert "serial=" in msg
    assert args[-2:] == (1, 1)  # this window's one gather engaged serial

    logged.clear()
    table._last_log = 0.0
    table.gather(np.array([0, 5, 12], dtype=np.int64))  # uniq.size == 3 > serial
    assert len(logged) == 1
    assert logged[0][1][-2:] == (0, 1)  # this window's one gather did not

    table.close()


def test_mixed_window_serial_field_reports_engaged_over_total_gathers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The shape the field's window-keying exists for: many small
    (serial-engaged) gathers plus one large pooled one. p99 is by
    construction the window's slowest call — here the large pooled gather,
    the least representative sample of how the window actually dispatched.
    Keying serial= on that one sample's flag would report 0 for the
    19-of-20-engaged window driven below."""
    _write_ple_layer(tmp_path, layer_idx=0, vocab=600, parts=4, cols=8, scale=0.5)
    layer_shards = ple_mmap.discover_shards(str(tmp_path))[0]
    table = ple_mmap.MmapPleTable(
        layer_shards.shards,
        150,
        8,
        torch.float8_e4m3fn,
        workers=4,
        chunk=16,
        model_path=str(tmp_path),
        serial=5,
    )
    logged = _record_info(monkeypatch)

    small_ids = np.array([0, 1], dtype=np.int64)  # uniq.size == 2 <= serial
    for _ in range(19):
        table.gather(small_ids)
    large_ids = np.arange(500, dtype=np.int64)  # uniq.size == 500 > serial
    table._last_log = 0.0  # simulate the interval having elapsed on this call
    table.gather(large_ids)  # pooled, the window's slowest call by far

    assert len(logged) == 1
    msg, args = logged[0]
    assert "serial=" in msg
    assert args[-2:] == (19, 20)  # 19 of this window's 20 gathers engaged

    table.close()


def test_serial_engaged_counter_resets_with_the_log_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The engaged count is per-window, so it must be cleared alongside
    _latencies_ms when a line fires — otherwise every later window reports
    an engaged count larger than its own total."""
    _write_ple_layer(tmp_path, layer_idx=0, vocab=40, parts=4, cols=8, scale=0.5)
    layer_shards = ple_mmap.discover_shards(str(tmp_path))[0]
    table = ple_mmap.MmapPleTable(
        layer_shards.shards,
        10,
        8,
        torch.float8_e4m3fn,
        workers=4,
        chunk=2,
        model_path=str(tmp_path),
        serial=64,
    )
    logged = _record_info(monkeypatch)
    ids = np.array([0, 5], dtype=np.int64)

    table.gather(ids)
    assert table._serial_engaged_since_log == 1
    table._last_log = 0.0
    table.gather(ids)

    assert len(logged) == 1
    assert logged[0][1][-2:] == (2, 2)
    assert table._serial_engaged_since_log == 0
    assert table._latencies_ms == []

    table.close()


# --------------------------------------------------------------------------- #
# Bounded prewarm (R4.10/R5.2)
# --------------------------------------------------------------------------- #


def test_compute_prewarm_bound_caps_at_table_bytes() -> None:
    assert ple_mmap.compute_prewarm_bound(100, 200 * (1 << 30)) == 100


def test_compute_prewarm_bound_respects_headroom() -> None:
    table_bytes = 100 * (1 << 30)
    mem_available = 20 * (1 << 30)
    bound = ple_mmap.compute_prewarm_bound(table_bytes, mem_available)
    assert bound == mem_available - ple_mmap._PREWARM_HEADROOM_BYTES


def test_compute_prewarm_bound_clamps_negative_to_zero() -> None:
    """R5.2: a negative bound would slice-read nearly the whole table exactly
    when memory is scarcest."""
    table_bytes = 100 * (1 << 30)
    mem_available = 4 * (1 << 30)  # below the 8 GiB headroom
    assert ple_mmap.compute_prewarm_bound(table_bytes, mem_available) == 0


def test_mem_available_bytes_parses_meminfo_format(tmp_path: Path) -> None:
    fixture = tmp_path / "meminfo"
    fixture.write_text(
        "MemTotal:       131000000 kB\n"
        "MemFree:         20000000 kB\n"
        "MemAvailable:   109051904 kB\n"
        "Cached:          15000000 kB\n"
    )

    assert ple_mmap._mem_available_bytes(str(fixture)) == 109051904 * 1024


def test_prewarm_reads_up_to_the_bound(tmp_path: Path) -> None:
    full = _write_ple_layer(tmp_path, layer_idx=0, vocab=9, parts=1, cols=4, scale=1.0)
    layer_shards = ple_mmap.discover_shards(str(tmp_path))[0]
    table = ple_mmap.MmapPleTable(
        layer_shards.shards,
        9,
        4,
        torch.float8_e4m3fn,
        workers=1,
        chunk=8,
        model_path=str(tmp_path),
    )
    table_bytes = full.numel()

    assert table.prewarm(0) == 0
    assert table.prewarm(table_bytes // 2) <= table_bytes // 2
    assert table.prewarm(table_bytes * 10) <= table_bytes


# --------------------------------------------------------------------------- #
# Custom op registration (R3.12)
# --------------------------------------------------------------------------- #


def test_op_is_registered_under_platform_default_and_cpu_dispatch_keys() -> None:
    assert hasattr(torch.ops.vllm, ple_mmap.OP_NAME)
    assert torch._C._dispatch_has_kernel_for_dispatch_key(
        ple_mmap.QUALIFIED_OP_NAME, "CPU"
    )
    if torch.cuda.is_available():
        assert torch._C._dispatch_has_kernel_for_dispatch_key(
            ple_mmap.QUALIFIED_OP_NAME, "CUDA"
        )
    # (F-1b) The output arg's alias annotation ("(a3!)") is what tells
    # torch.compile the write to `output` must survive functionalization —
    # without it (mutates_args=[]), a compiled graph can drop the write and
    # the caller reads back its own uninitialized new_empty buffer instead
    # of the gathered rows. Registration is module-global and sticky within
    # a pytest process (a second import cannot re-register), so this pins
    # the CURRENT registration's schema string rather than re-registering.
    schema = str(getattr(torch.ops.vllm, ple_mmap.OP_NAME).default._schema)
    assert "!) output" in schema, schema
    assert schema.endswith("-> ()")
    # Exercise the CPU key directly: this is what every other test below
    # relies on to run without a GPU. The gather-only op resolves the layer
    # and calls only .ngram_embedding(...) — hashing stays in traced code.
    gather_calls: list[torch.Tensor] = []

    class _FakePleEmbeddingModule:
        def ngram_embedding(self, ngram_ids: torch.Tensor) -> torch.Tensor:
            gather_calls.append(ngram_ids)
            return torch.zeros((*ngram_ids.shape, 2), dtype=torch.float8_e4m3fn)

    fake_layer = SimpleNamespace(ple_embedding=_FakePleEmbeddingModule())
    ctx = SimpleNamespace(no_compile_layers={"layer0": fake_layer})
    ngram_ids = torch.zeros((2, 2), dtype=torch.long)
    output = torch.empty((2, 4), dtype=torch.float8_e4m3fn)

    with forward_context.override_forward_context(ctx):
        torch.ops.vllm.qwen4_exp_ple_mmap_gather(ngram_ids, output, "layer0")

    assert len(gather_calls) == 1
    assert torch.equal(output, torch.zeros_like(output))


def test_op_raises_named_error_when_layer_name_does_not_resolve() -> None:
    ctx = SimpleNamespace(no_compile_layers={"layer0": SimpleNamespace()})
    ngram_ids = torch.zeros((1, 2), dtype=torch.long)
    output = torch.empty((1, 4), dtype=torch.float8_e4m3fn)

    with (
        forward_context.override_forward_context(ctx),
        pytest.raises(RuntimeError, match="does not resolve to a PLE layer"),
    ):
        torch.ops.vllm.qwen4_exp_ple_mmap_gather(ngram_ids, output, "layer0")


# --------------------------------------------------------------------------- #
# (c) CUDAGraph startup refusal — R1.1/R3.4/R4.2, parametrized per R3.14.
# --------------------------------------------------------------------------- #


def _compilation_config(
    *,
    mode: CompilationMode,
    cudagraph_mode: CUDAGraphMode,
    splitting_ops: list[str] | None,
) -> CompilationConfig:
    return CompilationConfig(
        mode=mode, cudagraph_mode=cudagraph_mode, splitting_ops=splitting_ops
    )


@pytest.mark.parametrize(
    "cudagraph_mode",
    [
        CUDAGraphMode.FULL,
        CUDAGraphMode.FULL_DECODE_ONLY,
        CUDAGraphMode.FULL_AND_PIECEWISE,
    ],
)
def test_check_cudagraph_safety_refuses_full_cudagraph_modes(
    cudagraph_mode: CUDAGraphMode,
) -> None:
    cc = _compilation_config(
        mode=CompilationMode.VLLM_COMPILE,
        cudagraph_mode=cudagraph_mode,
        splitting_ops=[ple_mmap.QUALIFIED_OP_NAME],
    )
    # Asserted directly against the enum values, never through
    # has_full_cudagraphs() (a rebase-fragile one-liner, R3.14).
    assert cudagraph_mode in (
        CUDAGraphMode.FULL,
        CUDAGraphMode.FULL_DECODE_ONLY,
        CUDAGraphMode.FULL_AND_PIECEWISE,
    )
    with pytest.raises(RuntimeError, match="piecewise-only CUDA graphs"):
        ple_mmap.check_cudagraph_safety(cc)


def test_check_cudagraph_safety_accepts_piecewise_compiled_with_op_split() -> None:
    cc = _compilation_config(
        mode=CompilationMode.VLLM_COMPILE,
        cudagraph_mode=CUDAGraphMode.PIECEWISE,
        splitting_ops=[ple_mmap.QUALIFIED_OP_NAME],
    )
    assert cc.cudagraph_mode is CUDAGraphMode.PIECEWISE

    ple_mmap.check_cudagraph_safety(cc)  # must not raise


def test_check_cudagraph_safety_refuses_non_compile_mode() -> None:
    """mode=NONE is enforce-eager: it does not fully suppress capture on this
    model and leaves splitting_ops empty (R3.4)."""
    cc = _compilation_config(
        mode=CompilationMode.NONE,
        cudagraph_mode=CUDAGraphMode.PIECEWISE,
        splitting_ops=[ple_mmap.QUALIFIED_OP_NAME],
    )
    assert cc.mode is CompilationMode.NONE

    with pytest.raises(RuntimeError, match="VLLM_COMPILE"):
        ple_mmap.check_cudagraph_safety(cc)


def test_check_cudagraph_safety_refuses_when_op_missing_from_splitting_ops() -> None:
    """Catches an operator-supplied -cc.splitting_ops list, or an
    attn-fusion reset, that silently drops our op (R4.2)."""
    cc = _compilation_config(
        mode=CompilationMode.VLLM_COMPILE,
        cudagraph_mode=CUDAGraphMode.PIECEWISE,
        splitting_ops=["vllm::some_other_op"],
    )

    with pytest.raises(RuntimeError, match="splitting_ops"):
        ple_mmap.check_cudagraph_safety(cc)


def test_set_splitting_ops_for_v1_emits_the_new_op() -> None:
    cc = CompilationConfig(mode=CompilationMode.VLLM_COMPILE)
    cc.set_splitting_ops_for_v1(all2all_backend="naive", data_parallel_size=1)

    assert ple_mmap.QUALIFIED_OP_NAME in cc.splitting_ops


def test_set_splitting_ops_for_v1_output_satisfies_the_cudagraph_guard() -> None:
    """(Ordering, L): a CompilationConfig built through its NORMAL init
    path (set_splitting_ops_for_v1), not hand-constructed with
    splitting_ops pre-set, must both contain our op AND satisfy
    check_cudagraph_safety — the membership assertion runs BEFORE the
    guard call, proving the two checks agree on the same real object."""
    cc = CompilationConfig(
        mode=CompilationMode.VLLM_COMPILE, cudagraph_mode=CUDAGraphMode.PIECEWISE
    )
    cc.set_splitting_ops_for_v1(all2all_backend="naive", data_parallel_size=1)

    assert ple_mmap.QUALIFIED_OP_NAME in cc.splitting_ops

    ple_mmap.check_cudagraph_safety(cc)  # must not raise


# --------------------------------------------------------------------------- #
# (F-1a, HIGH) check_cudagraph_safety is unit-tested as a free function
# above, but its CALL from Qwen4ExpNGramEmbedding.__init__ (ple_layer.py)
# was never exercised — deleting that call left the whole suite green.
# Each case pins the OTHER two predicates to pass, so a failure here can
# only mean the one predicate under test.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "cudagraph_mode",
    [
        CUDAGraphMode.FULL,
        CUDAGraphMode.FULL_DECODE_ONLY,
        CUDAGraphMode.FULL_AND_PIECEWISE,
    ],
)
def test_ngram_embedding_construction_refuses_full_cudagraph_modes(
    monkeypatch: pytest.MonkeyPatch, cudagraph_mode: CUDAGraphMode
) -> None:
    monkeypatch.setenv("VLLM_PLE_MMAP", "1")
    config = _make_text_config()
    cc = CompilationConfig(
        mode=CompilationMode.VLLM_COMPILE,
        cudagraph_mode=cudagraph_mode,
        splitting_ops=[ple_mmap.QUALIFIED_OP_NAME],
    )
    vllm_config = SimpleNamespace(compilation_config=cc, model_config=SimpleNamespace())

    with (
        set_current_vllm_config(vllm_config),
        pytest.raises(RuntimeError, match="piecewise-only CUDA graphs"),
    ):
        Qwen4ExpNGramEmbedding(
            config,
            8,
            0,
            16,
            4,
            "model.layers.1.ple.ple_embedding",
            "model.layers.1.ple",
            params_dtype=torch.float32,
        )


def test_ngram_embedding_construction_refuses_non_compile_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VLLM_PLE_MMAP", "1")
    config = _make_text_config()
    cc = CompilationConfig(
        mode=CompilationMode.NONE,
        cudagraph_mode=CUDAGraphMode.PIECEWISE,
        splitting_ops=[ple_mmap.QUALIFIED_OP_NAME],
    )
    vllm_config = SimpleNamespace(compilation_config=cc, model_config=SimpleNamespace())

    with (
        set_current_vllm_config(vllm_config),
        pytest.raises(RuntimeError, match="VLLM_COMPILE"),
    ):
        Qwen4ExpNGramEmbedding(
            config,
            8,
            0,
            16,
            4,
            "model.layers.1.ple.ple_embedding",
            "model.layers.1.ple",
            params_dtype=torch.float32,
        )


def test_ngram_embedding_construction_refuses_missing_splitting_op(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VLLM_PLE_MMAP", "1")
    config = _make_text_config()
    cc = CompilationConfig(
        mode=CompilationMode.VLLM_COMPILE,
        cudagraph_mode=CUDAGraphMode.PIECEWISE,
        splitting_ops=["vllm::some_other_op"],
    )
    vllm_config = SimpleNamespace(compilation_config=cc, model_config=SimpleNamespace())

    with (
        set_current_vllm_config(vllm_config),
        pytest.raises(RuntimeError, match="splitting_ops"),
    ):
        Qwen4ExpNGramEmbedding(
            config,
            8,
            0,
            16,
            4,
            "model.layers.1.ple.ple_embedding",
            "model.layers.1.ple",
            params_dtype=torch.float32,
        )


# --------------------------------------------------------------------------- #
# Construction-time shard validation (M3): refuses a bad checkpoint BEFORE
# the ~78 GiB backbone streams, not after.
# --------------------------------------------------------------------------- #


def test_validate_shards_for_raises_when_no_shards_at_a_resolved_path(
    tmp_path: Path,
) -> None:
    model_config = _model_config(tmp_path)  # resolves; directory is empty

    with pytest.raises(RuntimeError, match="no shard tensors for layer 1"):
        ple_mmap.validate_shards_for(model_config, "model.layers.1.ple", head_dim=4)


def test_validate_shards_for_raises_on_shard_width_mismatch(tmp_path: Path) -> None:
    _write_ple_layer(tmp_path, layer_idx=1, vocab=10, parts=3, cols=2, scale=0.25)
    model_config = _model_config(tmp_path)

    with pytest.raises(RuntimeError, match="shard width"):
        ple_mmap.validate_shards_for(model_config, "model.layers.1.ple", head_dim=4)


def test_validate_shards_for_raises_when_weight_scale_missing(tmp_path: Path) -> None:
    _write_ple_layer(
        tmp_path, layer_idx=1, vocab=10, parts=3, cols=2, scale=0.25, write_scale=False
    )
    model_config = _model_config(tmp_path)

    with pytest.raises(RuntimeError, match="no ngram_embedding.weight_scale"):
        ple_mmap.validate_shards_for(model_config, "model.layers.1.ple", head_dim=2)


def test_validate_shards_for_passes_on_a_well_formed_checkpoint(
    tmp_path: Path,
) -> None:
    _write_ple_layer(tmp_path, layer_idx=1, vocab=10, parts=3, cols=2, scale=0.25)
    model_config = _model_config(tmp_path)

    ple_mmap.validate_shards_for(
        model_config, "model.layers.1.ple", head_dim=2
    )  # must not raise


def test_validate_shards_for_refuses_a_bf16_table_with_a_stray_weight_scale(
    tmp_path: Path,
) -> None:
    """BF16 (unquantized) tables are registered with requires_scale=False —
    a weight_scale present on disk anyway signals exporter confusion (e.g. a
    half fp8-to-bf16 conversion) and must be refused up front, not silently
    ignored."""
    _write_ple_layer(
        tmp_path,
        layer_idx=1,
        vocab=10,
        parts=3,
        cols=2,
        scale=0.25,
        write_scale=True,
        table_dtype=torch.bfloat16,
    )
    model_config = _model_config(tmp_path)

    with pytest.raises(RuntimeError, match="BF16"):
        ple_mmap.validate_shards_for(model_config, "model.layers.1.ple", head_dim=2)


def test_validate_shards_for_passes_on_a_well_formed_bf16_checkpoint(
    tmp_path: Path,
) -> None:
    _write_ple_layer(
        tmp_path,
        layer_idx=1,
        vocab=10,
        parts=3,
        cols=2,
        scale=0.25,
        write_scale=False,
        table_dtype=torch.bfloat16,
    )
    model_config = _model_config(tmp_path)

    ple_mmap.validate_shards_for(
        model_config, "model.layers.1.ple", head_dim=2
    )  # must not raise


def test_validate_shards_for_reads_the_mmap_dir_instead_of_the_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """VLLM_PLE_MMAP_DIR redirects construction-time discovery wholesale: a
    checkpoint with no PLE tensors at all validates fine when the table
    directory has them, and a broken table directory is refused even though
    the checkpoint is untouched."""
    checkpoint = tmp_path / "checkpoint"
    table = tmp_path / "table"
    checkpoint.mkdir()
    table.mkdir()
    _write_ple_layer(table, layer_idx=1, vocab=10, parts=3, cols=2, scale=0.25)
    monkeypatch.setenv("VLLM_PLE_MMAP", "1")
    monkeypatch.setenv("VLLM_PLE_MMAP_DIR", str(table))
    model_config = _model_config(checkpoint)

    ple_mmap.validate_shards_for(
        model_config, "model.layers.1.ple", head_dim=2
    )  # must not raise

    with pytest.raises(RuntimeError, match="shard width"):
        ple_mmap.validate_shards_for(model_config, "model.layers.1.ple", head_dim=4)


def test_validate_shards_for_never_defers_on_a_broken_mmap_dir(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The unresolvable-model-path tolerance below must not swallow a
    misconfigured override: an operator-supplied path is either usable or a
    typo, and no later stage of the load can make it resolve."""
    monkeypatch.setenv("VLLM_PLE_MMAP", "1")
    monkeypatch.setenv("VLLM_PLE_MMAP_DIR", "/nonexistent/ple-table-xyz")
    model_config = SimpleNamespace(
        model_weights="", model="nonexistent-org/nonexistent-repo-xyz", revision=None
    )

    with pytest.raises(RuntimeError, match="is not a directory"):
        ple_mmap.validate_shards_for(model_config, "model.layers.1.ple", head_dim=4)


def test_validate_shards_for_tolerates_an_unresolvable_model_path() -> None:
    """A bare repo id with no local snapshot (e.g. --load-format
    dummy/test construction, offline): validation defers to load time
    rather than raising — build_tables still fail-closes there."""
    model_config = SimpleNamespace(
        model_weights="", model="nonexistent-org/nonexistent-repo-xyz", revision=None
    )

    ple_mmap.validate_shards_for(
        model_config, "model.layers.1.ple", head_dim=4
    )  # must not raise


def test_ngram_embedding_construction_refuses_a_bad_checkpoint_before_load(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """(M3, wired): a checkpoint whose PLE shards are missing/wrong makes
    __init__ itself raise — before load_weights, before the backbone
    streams — not just at build_tables time."""
    monkeypatch.setenv("VLLM_PLE_MMAP", "1")
    config = _make_text_config()
    cc = CompilationConfig(
        mode=CompilationMode.VLLM_COMPILE,
        cudagraph_mode=CUDAGraphMode.PIECEWISE,
        splitting_ops=[ple_mmap.QUALIFIED_OP_NAME],
    )
    vllm_config = SimpleNamespace(
        compilation_config=cc, model_config=_model_config(tmp_path)
    )

    with (
        set_current_vllm_config(vllm_config),
        pytest.raises(RuntimeError, match="no shard tensors for layer 1"),
    ):
        Qwen4ExpNGramEmbedding(
            config,
            8,
            0,
            16,
            4,
            "model.layers.1.ple.ple_embedding",
            "model.layers.1.ple",
            params_dtype=torch.float32,
        )


# --------------------------------------------------------------------------- #
# (e) compile-factor assertions — R4.1/R4.12.
# --------------------------------------------------------------------------- #


def test_ple_mmap_flag_is_a_compile_factor() -> None:
    assert "VLLM_PLE_MMAP" in envs.compile_factors()


def test_ple_mmap_tuning_knobs_are_not_compile_factors() -> None:
    factors = envs.compile_factors()
    assert "VLLM_PLE_MMAP_WORKERS" not in factors
    assert "VLLM_PLE_MMAP_CHUNK" not in factors
    assert "VLLM_PLE_MMAP_PREWARM" not in factors
    # GPU gather swaps the body of the split-out op, never the graph.
    assert "VLLM_PLE_MMAP_GPU_GATHER" not in factors
    # VLLM_PLE_MMAP_DIR names where the table's bytes live, which the graph
    # never sees; it is read straight from os.environ for exactly that
    # reason, so it must not appear here even by accident.
    assert "VLLM_PLE_MMAP_DIR" not in factors
    # VLLM_PLE_MMAP_SERIAL only swaps how the gather dispatches its tasks,
    # never the graph — and a threshold sweep that forced a recompile per
    # arm would poison its own A/B. Also read straight from os.environ.
    assert "VLLM_PLE_MMAP_SERIAL" not in factors


# --------------------------------------------------------------------------- #
# Placeholder embedding
# --------------------------------------------------------------------------- #


def test_placeholder_forward_returns_fp8_zeros_when_table_unset() -> None:
    """(F1-ii) --load-format dummy: load_weights never runs at all, so
    weights_streamed stays False and the table stays unset; the placeholder
    must still produce a valid (zero) fp8 tensor against the default unit
    weight_scale — this is the ONLY case zeros are legitimate."""
    embedding = ple_mmap.MmapNgramEmbedding(16, 4)
    ids = torch.zeros((2, 3), dtype=torch.long)

    assert embedding.weights_streamed is False
    out = embedding(ids)

    assert out.shape == (2, 3, 4)
    assert out.dtype == torch.float8_e4m3fn
    assert torch.equal(out, torch.zeros_like(out))
    assert embedding.weight_scale.item() == 1.0


def test_placeholder_forward_gathers_from_attached_table(tmp_path: Path) -> None:
    full = _write_ple_layer(tmp_path, layer_idx=0, vocab=9, parts=3, cols=2, scale=0.5)
    embedding = _attached_embedding(
        tmp_path, layer_idx=0, vocab=9, parts=3, cols=2, scale=0.5
    )

    ids = torch.tensor([[0, 8], [3, 3]], dtype=torch.long)
    out = embedding(ids)

    assert out.shape == (2, 2, 2)
    assert out.dtype == torch.float8_e4m3fn
    assert torch.equal(out.reshape(-1, 2), full[ids.reshape(-1)])


# --------------------------------------------------------------------------- #
# Forward timing instrument (sync_ms / gather_ms / h2d_call_ms)
# --------------------------------------------------------------------------- #


def test_forward_timing_instrument_logs_the_rate_limited_split(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The split line fires only once per _LOG_INTERVAL_S — poked directly
    via ``_fwd_last_log``, the same way this file already pokes
    ``table._last_log``, rather than sleeping or monkeypatching
    time.monotonic — and reports the window's sample count plus a p50 and a
    p99 split of the three CPU-blocking components."""
    _write_ple_layer(tmp_path, layer_idx=0, vocab=9, parts=3, cols=2, scale=0.5)
    embedding = _attached_embedding(
        tmp_path, layer_idx=0, vocab=9, parts=3, cols=2, scale=0.5
    )
    logged = _record_info(monkeypatch)
    ids = torch.tensor([0, 8], dtype=torch.long)

    embedding(ids)  # first call: buffers a sample, does not log yet
    assert logged == []

    embedding._fwd_last_log = 0.0  # simulate the interval having elapsed
    embedding(ids)

    assert len(logged) == 1
    msg, args = logged[0]
    # rows=/p99_ms= are the gather-side line's own keys
    # (MmapPleTable._record); this line's fwd_rows=/fwd_p99_ms= are
    # namespaced so no key of this line ever matches a gather-side one. The
    # converse is one-way and not asserted here: bare rows=/p99_ms= DO
    # match the prefixed keys below, so a gather-side grep has to anchor on
    # a key this line does not carry.
    assert msg.count("rows=") == 1  # only as fwd_rows=, no bare rows= too
    assert msg.count("p99_ms=") == 1  # only as fwd_p99_ms=
    # Every percentile-scoped component spells out its own prefix: a bare
    # sync_ms= would be a substring of p50_sync_ms=, so a reader grepping
    # for the p99 split would silently get the p50 value instead.
    for key in ("sync_ms=", "gather_ms=", "h2d_call_ms="):
        assert f"p50_{key}" in msg and f"p99_{key}" in msg
        assert msg.count(key) == 2  # only ever as those two prefixed twins
    (
        fwd_rows,
        n,
        p50_ms,
        p50_sync_ms,
        p50_gather_ms,
        p50_h2d_ms,
        fwd_p99_ms,
        p99_sync_ms,
        p99_gather_ms,
        p99_h2d_ms,
    ) = args
    assert fwd_rows == 2 * ids.numel()
    assert n == 2  # both calls landed in this window
    for value in (
        p50_ms,
        p50_sync_ms,
        p50_gather_ms,
        p50_h2d_ms,
        fwd_p99_ms,
        p99_sync_ms,
        p99_gather_ms,
        p99_h2d_ms,
    ):
        assert value >= 0.0
    # Each percentile sample's own total must equal the sum of its own
    # split — an arg-order regression inside either group desyncs this.
    assert fwd_p99_ms == pytest.approx(p99_sync_ms + p99_gather_ms + p99_h2d_ms)
    assert p50_ms == pytest.approx(p50_sync_ms + p50_gather_ms + p50_h2d_ms)
    # p99 indexes at or above p50 into a sorted window, so it can never
    # come out below it — this is what a wholesale p50/p99 group swap
    # breaks, and what the two per-group sum checks above would survive.
    assert fwd_p99_ms >= p50_ms


def test_forward_timing_window_resets_after_it_logs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """n= and fwd_rows= are per-window, so both accumulators clear when a
    line fires; otherwise every later window reports the whole run."""
    _write_ple_layer(tmp_path, layer_idx=0, vocab=9, parts=3, cols=2, scale=0.5)
    embedding = _attached_embedding(
        tmp_path, layer_idx=0, vocab=9, parts=3, cols=2, scale=0.5
    )
    logged = _record_info(monkeypatch)
    ids = torch.tensor([0, 8], dtype=torch.long)

    embedding._fwd_last_log = 0.0
    embedding(ids)
    assert logged[0][1][:2] == (2, 1)  # fwd_rows=2, n=1
    assert embedding._fwd_timings_ms == []
    assert embedding._fwd_rows_since_log == 0

    embedding._fwd_last_log = 0.0
    embedding(ids)
    assert logged[1][1][:2] == (2, 1)  # the second window, not (4, 2)


def test_forward_timing_skips_the_paths_that_never_gathered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The table-unset and capture-pass zero-fills return without touching
    the table, so there is no sync/gather/H2D split to attribute and they
    must not enter the window at all — an entry of zeros would drag the
    reported p50 toward a call that never did any of the work."""
    _write_ple_layer(tmp_path, layer_idx=0, vocab=9, parts=3, cols=2, scale=0.5)
    unattached = ple_mmap.MmapNgramEmbedding(9, 2)
    unattached(torch.zeros((2, 3), dtype=torch.long))
    assert unattached._fwd_timings_ms == []

    embedding = _attached_embedding(
        tmp_path, layer_idx=0, vocab=9, parts=3, cols=2, scale=0.5
    )
    monkeypatch.setattr(ple_mmap.logger, "warning_once", lambda *a, **k: None)
    embedding(torch.tensor([0, 9], dtype=torch.long))  # 9 == rows_total

    assert embedding.table is not None
    assert embedding.table._skipped == 1
    assert embedding._fwd_timings_ms == []


# --------------------------------------------------------------------------- #
# Capture-pass tolerance: an out-of-range id reaching the gather boundary can
# only come from a CUDA-graph capture pass (module docstring) — executed
# hashing always bounds ids via torch.remainder(...) + offsets
# (Qwen4ExpNGramEmbedding._hash_ngram_ids in ple_layer.py), so a real forward
# can never produce one. forward() must zero-fill instead of gathering,
# never raise, never touch the table, and count the occurrence.
# --------------------------------------------------------------------------- #


def _raising_gather(ids: np.ndarray) -> np.ndarray:
    raise AssertionError("gather must not run for out-of-range ids")


@pytest.mark.parametrize(
    "offset",
    [0, 1],  # exactly rows_total (one past the last valid row), and beyond
)
def test_placeholder_forward_zero_fills_ids_at_or_past_rows_total(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, offset: int
) -> None:
    _write_ple_layer(tmp_path, layer_idx=0, vocab=9, parts=3, cols=2, scale=0.5)
    embedding = _attached_embedding(
        tmp_path, layer_idx=0, vocab=9, parts=3, cols=2, scale=0.5
    )
    table = embedding.table
    assert table is not None
    assert table.rows_total == 9
    monkeypatch.setattr(table, "gather", _raising_gather)
    warn_calls: list[tuple[object, object]] = []
    monkeypatch.setattr(
        ple_mmap.logger, "warning_once", lambda *a, **k: warn_calls.append((a, k))
    )

    ids = torch.tensor([0, table.rows_total + offset], dtype=torch.long)
    out = embedding(ids)

    assert out.shape == (2, 2)
    assert out.dtype == table.torch_dtype
    assert torch.equal(out, torch.zeros_like(out))
    assert table._skipped == 1
    assert len(warn_calls) == 1


def test_placeholder_forward_zero_fills_negative_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_ple_layer(tmp_path, layer_idx=0, vocab=9, parts=3, cols=2, scale=0.5)
    embedding = _attached_embedding(
        tmp_path, layer_idx=0, vocab=9, parts=3, cols=2, scale=0.5
    )
    table = embedding.table
    assert table is not None
    monkeypatch.setattr(table, "gather", _raising_gather)

    # A single bad id anywhere in the batch must zero-fill the WHOLE output
    # (the op receives one ngram_ids tensor per forward; a capture pass
    # taints all of it, not just the offending element).
    ids = torch.tensor([-1, 3], dtype=torch.long)
    out = embedding(ids)

    assert torch.equal(out, torch.zeros_like(out))
    assert table._skipped == 1


def test_placeholder_forward_boundary_ids_still_gather(tmp_path: Path) -> None:
    """(b) in-range path stays unchanged: the boundary rows (0 and
    rows_total - 1) are valid and must still take the real gather path — the
    new out-of-range check must not false-positive at either edge."""
    full = _write_ple_layer(tmp_path, layer_idx=0, vocab=9, parts=3, cols=2, scale=0.5)
    embedding = _attached_embedding(
        tmp_path, layer_idx=0, vocab=9, parts=3, cols=2, scale=0.5
    )
    table = embedding.table
    assert table is not None
    assert table.rows_total == 9

    ids = torch.tensor([0, 8], dtype=torch.long)
    out = embedding(ids)

    assert torch.equal(out, full[[0, 8]])
    assert table._skipped == 0


def test_op_zero_fills_output_for_out_of_range_ngram_ids(tmp_path: Path) -> None:
    """The production boundary is the custom op itself, not forward() in
    isolation: drives an out-of-range id (the exact magnitude from the
    measured crash) through torch.ops.vllm.qwen4_exp_ple_mmap_gather, proving
    the mutate-in-place ``output`` receives zeros instead of the op raising
    IndexError out of engine init's cudagraph memory profiling."""
    _write_ple_layer(tmp_path, layer_idx=0, vocab=9, parts=3, cols=2, scale=0.5)
    embedding = _attached_embedding(
        tmp_path, layer_idx=0, vocab=9, parts=3, cols=2, scale=0.5
    )
    fake_layer = SimpleNamespace(
        ple_embedding=SimpleNamespace(ngram_embedding=embedding)
    )
    ctx = SimpleNamespace(no_compile_layers={"layer0": fake_layer})
    ngram_ids = torch.tensor([[0, 9_223_231_297_218_904_063]], dtype=torch.long)
    output = torch.empty((1, 4), dtype=torch.float8_e4m3fn)

    with forward_context.override_forward_context(ctx):
        torch.ops.vllm.qwen4_exp_ple_mmap_gather(ngram_ids, output, "layer0")

    assert torch.equal(output, torch.zeros_like(output))
    assert embedding.table is not None
    assert embedding.table._skipped == 1


# --------------------------------------------------------------------------- #
# load_weights interception + loaded-set contract
# --------------------------------------------------------------------------- #


def _mmap_ngram_module_for_load_test(
    vocab: int = 8, cols: int = 2
) -> Qwen4ExpNGramEmbedding:
    module = Qwen4ExpNGramEmbedding.__new__(Qwen4ExpNGramEmbedding)
    torch.nn.Module.__init__(module)
    module.layer_name = "model.layers.1.ple"
    module.split_ngram_parts = 2
    module.register_buffer("layer_multipliers", torch.zeros(1, dtype=torch.long))
    module.register_buffer("ngram_heads_offsets", torch.zeros(1, dtype=torch.long))
    module.register_buffer("ngram_heads_vocab_sizes", torch.zeros(1, dtype=torch.long))
    module.ngram_embedding = ple_mmap.MmapNgramEmbedding(vocab, cols)
    return module


def test_ngram_embedding_mmap_load_weights_intercepts_shards_and_scale() -> None:
    module = _mmap_ngram_module_for_load_test(vocab=8, cols=2)
    shard_0 = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    shard_0 = shard_0.to(torch.float8_e4m3fn)
    shard_1 = torch.arange(8, 16, dtype=torch.float32).reshape(4, 2)
    shard_1 = shard_1.to(torch.float8_e4m3fn)
    weight_scale = torch.tensor([0.25], dtype=torch.bfloat16)

    loaded = module.load_weights(
        [
            ("ngram_embedding.shard_0.weight", shard_0),
            ("ngram_embedding.shard_1.weight", shard_1),
            ("ngram_embedding.weight_scale", weight_scale),
        ]
    )

    assert loaded == {"ngram_embedding.weight", "ngram_embedding.weight_scale"}
    assert torch.equal(module.ngram_embedding.weight_scale, weight_scale)
    assert module.ngram_embedding.weight_scale_loaded is True
    assert module.ngram_embedding.weights_streamed is True


def test_forward_raises_named_error_when_streamed_but_build_tables_never_ran() -> None:
    """(F1-i, HIGH): a real load_weights pass over PLE shards, with
    build_tables never called, must not silently serve fp8 zeros — that
    would be indistinguishable from a legitimate --load-format dummy probe.
    weights_streamed=True (set once load_weights sees a real shard tensor)
    is exactly the signal that distinguishes the two, and forward must
    raise the named error instead."""
    module = _mmap_ngram_module_for_load_test(vocab=8, cols=2)
    shard_0 = torch.arange(8, dtype=torch.float32).reshape(4, 2).to(torch.float8_e4m3fn)
    shard_1 = torch.arange(8, 16, dtype=torch.float32).reshape(4, 2)
    shard_1 = shard_1.to(torch.float8_e4m3fn)
    weight_scale = torch.tensor([0.25], dtype=torch.bfloat16)

    module.load_weights(
        [
            ("ngram_embedding.shard_0.weight", shard_0),
            ("ngram_embedding.shard_1.weight", shard_1),
            ("ngram_embedding.weight_scale", weight_scale),
        ]
    )

    assert module.ngram_embedding.table is None  # build_tables never ran
    with pytest.raises(
        RuntimeError,
        match="PLE mmap table not initialized",
    ):
        module.ngram_embedding(torch.zeros((2,), dtype=torch.long))


def test_ngram_embedding_mmap_load_weights_never_retains_shard_tensors() -> None:
    """Invariant 3: nothing may retain the full table, including transiently
    on the placeholder — it has no .weight attribute to retain into."""
    module = _mmap_ngram_module_for_load_test(vocab=8, cols=2)
    shard_0 = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    shard_0 = shard_0.to(torch.float8_e4m3fn)
    shard_1 = torch.arange(8, 16, dtype=torch.float32).reshape(4, 2)
    shard_1 = shard_1.to(torch.float8_e4m3fn)

    module.load_weights(
        [
            ("ngram_embedding.shard_0.weight", shard_0),
            ("ngram_embedding.shard_1.weight", shard_1),
        ]
    )

    assert not hasattr(module.ngram_embedding, "weight")
    assert module.ngram_embedding.table is None  # only build_tables ever sets it


def test_ngram_embedding_mmap_load_weights_rejects_mismatched_shard_shape() -> None:
    module = _mmap_ngram_module_for_load_test(vocab=8, cols=2)

    with pytest.raises(ValueError, match=r"Shape mismatch for PLE embedding shard 0"):
        module.load_weights([("ngram_embedding.shard_0.weight", torch.zeros(3, 2))])


# --------------------------------------------------------------------------- #
# build_tables: construction hook, fail-closed validation, per-layer keying.
# --------------------------------------------------------------------------- #


def _fake_ple_layer(
    layer_idx: int, embedding: ple_mmap.MmapNgramEmbedding, split_ngram_parts: int
) -> SimpleNamespace:
    return SimpleNamespace(
        layer_idx=layer_idx,
        ple_embedding=SimpleNamespace(
            ngram_embedding=embedding, split_ngram_parts=split_ngram_parts
        ),
    )


def _loaded_placeholder(
    vocab: int, cols: int, scale: float
) -> ple_mmap.MmapNgramEmbedding:
    embedding = ple_mmap.MmapNgramEmbedding(vocab, cols)
    ple_mmap.set_weight_scale(
        embedding, torch.tensor([scale], dtype=torch.bfloat16), torch.device("cpu")
    )
    return embedding


def _model_config(directory: Path) -> SimpleNamespace:
    return SimpleNamespace(model_weights=str(directory), model="ignored", revision=None)


def test_build_tables_wires_the_tuning_knobs_from_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """(F-5): VLLM_PLE_MMAP_WORKERS/_CHUNK/_SERIAL must reach the attached
    MmapPleTable's workers/chunk/serial in the right order — a swapped-args
    regression would still construct a table, just with the wrong
    concurrency knobs, and nothing else would notice."""
    monkeypatch.setenv("VLLM_PLE_MMAP_WORKERS", "3")
    monkeypatch.setenv("VLLM_PLE_MMAP_CHUNK", "7")
    monkeypatch.setenv("VLLM_PLE_MMAP_SERIAL", "13")
    _write_ple_layer(tmp_path, layer_idx=0, vocab=10, parts=3, cols=2, scale=0.25)
    emb = _loaded_placeholder(10, 2, 0.25)
    cc = SimpleNamespace(static_forward_context={"a.ple": _fake_ple_layer(0, emb, 3)})

    ple_mmap.build_tables(_model_config(tmp_path), cc)

    assert emb.table is not None
    assert emb.table.workers == 3
    assert emb.table.chunk == 7
    assert emb.table.serial == 13


def test_build_tables_leaves_serial_off_when_the_environment_is_unset(
    tmp_path: Path,
) -> None:
    """Default-off is the whole safety story for this knob: an attached
    table must dispatch through the pool exactly as it did before
    VLLM_PLE_MMAP_SERIAL existed unless an operator asks otherwise."""
    _write_ple_layer(tmp_path, layer_idx=0, vocab=10, parts=3, cols=2, scale=0.25)
    emb = _loaded_placeholder(10, 2, 0.25)
    cc = SimpleNamespace(static_forward_context={"a.ple": _fake_ple_layer(0, emb, 3)})

    ple_mmap.build_tables(_model_config(tmp_path), cc)

    assert emb.table is not None
    assert emb.table.serial == 0


def test_build_tables_attaches_a_table_per_ple_layer_without_cross_contamination(
    tmp_path: Path,
) -> None:
    """(b): build_tables must key tables per layer prefix, never a module
    global — attaching layer 0's table must not affect layer 1's."""
    full0 = _write_ple_layer(
        tmp_path, layer_idx=0, vocab=10, parts=3, cols=2, scale=0.25
    )
    full1 = _write_ple_layer(
        tmp_path, layer_idx=1, vocab=20, parts=4, cols=2, scale=0.75
    )

    emb0 = _loaded_placeholder(10, 2, 0.25)
    emb1 = _loaded_placeholder(20, 2, 0.75)
    cc = SimpleNamespace(
        static_forward_context={
            "a.ple": _fake_ple_layer(0, emb0, 3),
            "b.ple": _fake_ple_layer(1, emb1, 4),
        }
    )
    model_config = _model_config(tmp_path)

    ple_mmap.build_tables(model_config, cc)

    assert emb0.table is not None and emb1.table is not None
    assert emb0.table is not emb1.table
    out0 = emb0(torch.tensor([0, 9], dtype=torch.long))
    out1 = emb1(torch.tensor([0, 19], dtype=torch.long))
    assert torch.equal(out0, full0[[0, 9]])
    assert torch.equal(out1, full1[[0, 19]])
    # No cross-wiring: layer 0's output must not equal layer 1's data.
    assert not torch.equal(out0.reshape(-1), full1[[0, 9]].reshape(-1))


def test_build_tables_ignores_layers_without_an_mmap_placeholder(
    tmp_path: Path,
) -> None:
    """A layer whose ngram_embedding is not our placeholder (env-off) must be
    skipped, never mistaken for a PLE layer needing a table."""
    _write_ple_layer(tmp_path, layer_idx=0, vocab=10, parts=3, cols=2, scale=0.25)
    cc = SimpleNamespace(
        static_forward_context={
            "not_a_ple_layer": SimpleNamespace(),
            "no_embedding_attr": SimpleNamespace(ple_embedding=None),
        }
    )
    model_config = _model_config(tmp_path)

    ple_mmap.build_tables(model_config, cc)  # must not raise


def test_build_tables_raises_when_a_ple_layer_has_no_shards_on_disk(
    tmp_path: Path,
) -> None:
    emb = _loaded_placeholder(10, 2, 0.25)
    cc = SimpleNamespace(static_forward_context={"a.ple": _fake_ple_layer(0, emb, 3)})
    model_config = _model_config(tmp_path)

    with pytest.raises(RuntimeError, match="no shard tensors for layer 0"):
        ple_mmap.build_tables(model_config, cc)


def test_build_tables_raises_on_shard_width_mismatch(tmp_path: Path) -> None:
    _write_ple_layer(tmp_path, layer_idx=0, vocab=10, parts=3, cols=2, scale=0.25)
    emb = _loaded_placeholder(10, 4, 0.25)  # embedding_dim disagrees with on-disk cols
    cc = SimpleNamespace(static_forward_context={"a.ple": _fake_ple_layer(0, emb, 3)})
    model_config = _model_config(tmp_path)

    with pytest.raises(RuntimeError, match="shard width"):
        ple_mmap.build_tables(model_config, cc)


def test_build_tables_refuses_a_uniformly_e5m2_ple_table(tmp_path: Path) -> None:
    """(F-3): F8_E5M2 was dropped from _FP8_DTYPES because is_fp8() does not
    recognize it (dequant would silently never fire) — a UNIFORMLY-e5m2
    checkpoint (not a mixed-dtype one, already covered by discover_shards'
    own check) must still be refused, from BOTH validate_shards_for
    (construction-time) and build_tables (load-time), with the same
    dtype diagnosis.

    (F-9, folded in): after a SEPARATE successful e4m3 attach, the
    placeholder's own torch_dtype tracks the attached table's dtype
    exactly.
    """
    assert not is_fp8(torch.float8_e5m2)

    vocab, parts, cols = 10, 3, 2
    shard_size = (vocab + parts - 1) // parts
    prefix = "model.language_model.layers.0.ple.ple_embedding.ngram_embedding"
    for shard_index in range(parts):
        start = shard_index * shard_size
        rows = max(0, min(shard_size, vocab - start))
        tensors = {
            f"{prefix}.shard_{shard_index}.weight": torch.zeros(rows, cols).to(
                torch.float8_e5m2
            )
        }
        if shard_index == 0:
            tensors[f"{prefix}.weight_scale"] = torch.tensor(
                [0.5], dtype=torch.bfloat16
            )
        safetensors.torch.save_file(
            tensors, str(tmp_path / f"e5m2-{shard_index:05d}.safetensors")
        )

    model_config = _model_config(tmp_path)
    with pytest.raises(RuntimeError, match=r"F8_E5M2"):
        ple_mmap.validate_shards_for(
            model_config, "model.language_model.layers.0.ple", head_dim=cols
        )

    emb = _loaded_placeholder(vocab, cols, 0.5)
    cc = SimpleNamespace(
        static_forward_context={"a.ple": _fake_ple_layer(0, emb, parts)}
    )
    with pytest.raises(RuntimeError, match=r"F8_E5M2"):
        ple_mmap.build_tables(model_config, cc)

    # (F-9) a fully separate, successful e4m3 attach.
    e4m3_dir = tmp_path / "e4m3"
    e4m3_dir.mkdir()
    _write_ple_layer(
        e4m3_dir, layer_idx=0, vocab=vocab, parts=parts, cols=cols, scale=0.5
    )
    emb2 = _loaded_placeholder(vocab, cols, 0.5)
    cc2 = SimpleNamespace(
        static_forward_context={"a.ple": _fake_ple_layer(0, emb2, parts)}
    )
    ple_mmap.build_tables(_model_config(e4m3_dir), cc2)

    assert emb2.table is not None
    assert emb2.table.torch_dtype is emb2.torch_dtype


def test_build_tables_attaches_a_bf16_table_and_forward_gathers_it_value_exact(
    tmp_path: Path,
) -> None:
    """Intel AutoRound W4A16 exports pass the PLE table through as
    unquantized BF16 with no weight_scale on disk: a real streamed load sets
    weights_streamed True but weight_scale_loaded stays False (there is no
    scale tensor to stream) — the True/False quadrant the streamed-loader
    error would otherwise refuse. For a requires_scale=False dtype this must
    still attach and serve real values through the full load_weights ->
    build_tables -> forward chain, not raise."""
    full = _write_ple_layer(
        tmp_path,
        layer_idx=0,
        vocab=8,
        parts=2,
        cols=2,
        scale=0.0,
        write_scale=False,
        table_dtype=torch.bfloat16,
    )
    module = _mmap_ngram_module_for_load_test(vocab=8, cols=2)
    module.load_weights(
        [
            ("ngram_embedding.shard_0.weight", full[0:4]),
            ("ngram_embedding.shard_1.weight", full[4:8]),
        ]
    )
    assert module.ngram_embedding.weights_streamed is True
    assert module.ngram_embedding.weight_scale_loaded is False

    cc = SimpleNamespace(
        static_forward_context={"a.ple": _fake_ple_layer(0, module.ngram_embedding, 2)}
    )
    ple_mmap.build_tables(_model_config(tmp_path), cc)

    ids = torch.tensor([[0, 7], [3, 3]], dtype=torch.long)
    out = module.ngram_embedding(ids)

    assert out.dtype == torch.bfloat16
    assert torch.equal(out.reshape(-1, 2), full[ids.reshape(-1)])


def test_build_tables_raises_on_missing_shard_file(tmp_path: Path) -> None:
    prefix = "model.language_model.layers.0.ple.ple_embedding.ngram_embedding"
    full = _synthetic_weight(10, 2, layer_idx=0)
    # shard_size = ceil(10/3) = 4; write shards 0 and 2, skip shard 1 entirely.
    safetensors.torch.save_file(
        {
            f"{prefix}.shard_0.weight": full[0:4],
            f"{prefix}.weight_scale": torch.tensor([0.25], dtype=torch.bfloat16),
        },
        str(tmp_path / "shard0.safetensors"),
    )
    safetensors.torch.save_file(
        {f"{prefix}.shard_2.weight": full[8:10]},
        str(tmp_path / "shard2.safetensors"),
    )
    emb = _loaded_placeholder(10, 2, 0.25)
    cc = SimpleNamespace(static_forward_context={"a.ple": _fake_ple_layer(0, emb, 3)})
    model_config = _model_config(tmp_path)

    with pytest.raises(RuntimeError, match=r"missing shard\(s\) \[1\]"):
        ple_mmap.build_tables(model_config, cc)


def test_build_tables_raises_when_weight_scale_was_never_loaded(tmp_path: Path) -> None:
    """Rows streamed but weight_scale absent from the same iterable is a
    broken or truncated weight iterator, not an unstreamed family — this
    must stay in the fail-closed True/False quadrant (weights_streamed
    True, weight_scale_loaded False), never fall back to a header read."""
    _write_ple_layer(
        tmp_path, layer_idx=0, vocab=8, parts=2, cols=2, scale=0.25, write_scale=True
    )
    module = _mmap_ngram_module_for_load_test(vocab=8, cols=2)
    shard_0 = torch.arange(8, dtype=torch.float32).reshape(4, 2).to(torch.float8_e4m3fn)
    shard_1 = torch.arange(8, 16, dtype=torch.float32).reshape(4, 2)
    shard_1 = shard_1.to(torch.float8_e4m3fn)

    module.load_weights(
        [
            ("ngram_embedding.shard_0.weight", shard_0),
            ("ngram_embedding.shard_1.weight", shard_1),
        ]
    )

    assert module.ngram_embedding.weights_streamed is True
    assert module.ngram_embedding.weight_scale_loaded is False

    cc = SimpleNamespace(
        static_forward_context={"a.ple": _fake_ple_layer(0, module.ngram_embedding, 2)}
    )
    model_config = _model_config(tmp_path)

    with pytest.raises(RuntimeError, match="weight_scale was never loaded"):
        ple_mmap.build_tables(model_config, cc)


def test_build_tables_falls_back_to_header_scale_when_family_never_streamed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A layer whose ngram_embedding family was never routed to this worker
    never streams anything, so weights_streamed stays False along with
    weight_scale_loaded (the False/False quadrant) — that must attach off a
    direct header read and warn, not raise the streamed-loader's fail-closed
    error. The on-disk scale is F32 (0.1) to exercise the no-cast rule:
    casting to the placeholder's default bf16 would silently rewrite 0.1 to
    a different float and trip on nothing, masking the bug.
    """
    _write_ple_layer(
        tmp_path,
        layer_idx=0,
        vocab=8,
        parts=2,
        cols=2,
        scale=0.1,
        write_scale=True,
        scale_dtype=torch.float32,
    )
    embedding = ple_mmap.MmapNgramEmbedding(8, 2)
    assert embedding.weights_streamed is False
    assert embedding.weight_scale_loaded is False
    cc = SimpleNamespace(
        static_forward_context={"a.ple": _fake_ple_layer(0, embedding, 2)}
    )
    warnings: list[tuple[str, tuple[object, ...]]] = []
    monkeypatch.setattr(
        ple_mmap.logger,
        "warning",
        lambda msg, *args: warnings.append((msg, args)),
    )

    ple_mmap.build_tables(_model_config(tmp_path), cc)

    assert embedding.weight_scale_loaded is True
    assert embedding.weight_scale.dtype is torch.float32
    assert torch.equal(
        embedding.weight_scale, torch.tensor([0.1], dtype=torch.float32).squeeze()
    )
    assert len(warnings) == 1
    assert warnings[0][1] == (0,)  # layer_idx


def test_build_tables_raises_on_a_non_scalar_weight_scale(tmp_path: Path) -> None:
    """A per-channel (multi-element) weight_scale would silently truncate to
    its first element in _read_scale: _validate_layer_shards must refuse it
    up front, before the header-read path (or any other quadrant) ever gets
    a chance to attach off a truncated value.
    """
    prefix = "model.language_model.layers.0.ple.ple_embedding.ngram_embedding"
    vocab, parts, cols = 8, 2, 2
    shard_size = (vocab + parts - 1) // parts
    full = _synthetic_weight(vocab, cols)
    for shard_index in range(parts):
        start = shard_index * shard_size
        rows = max(0, min(shard_size, vocab - start))
        tensors: dict[str, torch.Tensor] = {
            f"{prefix}.shard_{shard_index}.weight": full[start : start + rows]
        }
        if shard_index == 0:
            tensors[f"{prefix}.weight_scale"] = torch.tensor(
                [0.1, 0.2, 0.3, 0.4], dtype=torch.float32
            )
        safetensors.torch.save_file(
            tensors, str(tmp_path / f"model-ple-0-{shard_index:05d}.safetensors")
        )
    emb = _loaded_placeholder(vocab, cols, 0.1)
    cc = SimpleNamespace(
        static_forward_context={"a.ple": _fake_ple_layer(0, emb, parts)}
    )

    with pytest.raises(RuntimeError, match=r"per-channel"):
        ple_mmap.build_tables(_model_config(tmp_path), cc)


def test_build_tables_raises_when_no_weight_scale_tensor_exists_on_disk(
    tmp_path: Path,
) -> None:
    _write_ple_layer(
        tmp_path, layer_idx=0, vocab=10, parts=3, cols=2, scale=0.25, write_scale=False
    )
    emb = _loaded_placeholder(10, 2, 0.25)
    cc = SimpleNamespace(static_forward_context={"a.ple": _fake_ple_layer(0, emb, 3)})
    model_config = _model_config(tmp_path)

    with pytest.raises(RuntimeError, match="no ngram_embedding.weight_scale"):
        ple_mmap.build_tables(model_config, cc)


def test_build_tables_raises_on_scale_mismatch_between_streamed_and_header(
    tmp_path: Path,
) -> None:
    _write_ple_layer(tmp_path, layer_idx=0, vocab=10, parts=3, cols=2, scale=0.25)
    emb = _loaded_placeholder(10, 2, scale=0.5)  # disagrees with the on-disk 0.25
    cc = SimpleNamespace(static_forward_context={"a.ple": _fake_ple_layer(0, emb, 3)})
    model_config = _model_config(tmp_path)

    with pytest.raises(RuntimeError, match="weight_scale mismatch"):
        ple_mmap.build_tables(model_config, cc)


def test_build_tables_prewarm_is_bounded_by_available_memory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """(F-6): PREWARM=1 must call MmapPleTable.prewarm exactly once, with
    the clamped (R4.10/R5.2) bound; PREWARM=0 must never call it at all —
    the env gate, not just the bound math, is under test."""
    prewarm_calls: list[int] = []
    real_prewarm = ple_mmap.MmapPleTable.prewarm

    def spy_prewarm(self: ple_mmap.MmapPleTable, max_bytes: int) -> int:
        prewarm_calls.append(max_bytes)
        return real_prewarm(self, max_bytes)

    monkeypatch.setattr(ple_mmap.MmapPleTable, "prewarm", spy_prewarm)
    # Pretend memory is scarce: bound must clamp to 0, not raise.
    monkeypatch.setattr(ple_mmap, "_mem_available_bytes", lambda: 1 << 20)

    monkeypatch.setenv("VLLM_PLE_MMAP_PREWARM", "1")
    on_dir = tmp_path / "on"
    on_dir.mkdir()
    _write_ple_layer(on_dir, layer_idx=0, vocab=10, parts=1, cols=2, scale=0.25)
    emb_on = _loaded_placeholder(10, 2, 0.25)
    cc_on = SimpleNamespace(
        static_forward_context={"a.ple": _fake_ple_layer(0, emb_on, 1)}
    )

    ple_mmap.build_tables(_model_config(on_dir), cc_on)

    assert emb_on.table is not None
    assert prewarm_calls == [0]  # 1 MiB available - 8 GiB headroom, clamped to 0

    monkeypatch.setenv("VLLM_PLE_MMAP_PREWARM", "0")
    off_dir = tmp_path / "off"
    off_dir.mkdir()
    _write_ple_layer(off_dir, layer_idx=0, vocab=10, parts=1, cols=2, scale=0.25)
    emb_off = _loaded_placeholder(10, 2, 0.25)
    cc_off = SimpleNamespace(
        static_forward_context={"a.ple": _fake_ple_layer(0, emb_off, 1)}
    )

    ple_mmap.build_tables(_model_config(off_dir), cc_off)

    assert emb_off.table is not None
    assert prewarm_calls == [0]  # unchanged: prewarm was not called again


def test_build_tables_is_idempotent_and_reuses_an_already_attached_table(
    tmp_path: Path,
) -> None:
    """(F1-iii/M7): Qwen4ExpForCausalLM.load_weights and
    Qwen4ExpForConditionalGeneration.load_weights both call build_tables on
    a real ConditionalGeneration load (the wrapper composes CausalLM
    internally), so a second call for the same layer must not re-attach —
    or leak the first table's ThreadPool."""
    _write_ple_layer(tmp_path, layer_idx=0, vocab=10, parts=3, cols=2, scale=0.25)
    emb = _loaded_placeholder(10, 2, 0.25)
    cc = SimpleNamespace(static_forward_context={"a.ple": _fake_ple_layer(0, emb, 3)})
    model_config = _model_config(tmp_path)

    ple_mmap.build_tables(model_config, cc)
    first_table = emb.table
    assert first_table is not None

    ple_mmap.build_tables(model_config, cc)  # second call: must not raise

    assert emb.table is first_table  # skipped, not rebuilt


def test_build_tables_second_call_skips_the_discover_shards_header_scan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """(M1): with everything already attached, the redundant second call
    (CausalLM's build_tables call, then ConditionalGeneration's) must not
    re-scan every checkpoint file's header — only resolve_model_path (cheap:
    a directory check) runs on the empty-pending path."""
    _write_ple_layer(tmp_path, layer_idx=0, vocab=10, parts=3, cols=2, scale=0.25)
    emb = _loaded_placeholder(10, 2, 0.25)
    cc = SimpleNamespace(static_forward_context={"a.ple": _fake_ple_layer(0, emb, 3)})
    model_config = _model_config(tmp_path)

    calls = 0
    real_discover_shards = ple_mmap.discover_shards

    def counting_discover_shards(path: str):
        nonlocal calls
        calls += 1
        return real_discover_shards(path)

    monkeypatch.setattr(ple_mmap, "discover_shards", counting_discover_shards)

    ple_mmap.build_tables(model_config, cc)
    ple_mmap.build_tables(model_config, cc)

    assert calls == 1


def test_build_tables_raises_when_reloaded_against_a_different_checkpoint(
    tmp_path: Path,
) -> None:
    """(M2): gpu_model_runner reload_weights can repoint model_config.model
    at a new checkpoint and re-call load_weights on the SAME live model.
    Silently keeping the already-attached table would serve checkpoint A's
    mmap rows against checkpoint B's scale — fail closed instead."""
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    dir_a.mkdir()
    dir_b.mkdir()
    _write_ple_layer(dir_a, layer_idx=0, vocab=10, parts=3, cols=2, scale=0.25)
    _write_ple_layer(dir_b, layer_idx=0, vocab=10, parts=3, cols=2, scale=0.75)

    emb = _loaded_placeholder(10, 2, 0.25)
    cc = SimpleNamespace(static_forward_context={"a.ple": _fake_ple_layer(0, emb, 3)})

    ple_mmap.build_tables(_model_config(dir_a), cc)
    assert emb.table is not None
    assert emb.table.model_path == str(dir_a)

    with pytest.raises(RuntimeError, match="different checkpoint|reloading"):
        ple_mmap.build_tables(_model_config(dir_b), cc)


def test_mmap_ple_table_close_drops_memmaps_and_is_idempotent(tmp_path: Path) -> None:
    _write_ple_layer(tmp_path, layer_idx=0, vocab=9, parts=3, cols=2, scale=1.0)
    layer_shards = ple_mmap.discover_shards(str(tmp_path))[0]
    table = ple_mmap.MmapPleTable(
        layer_shards.shards,
        3,
        2,
        torch.float8_e4m3fn,
        workers=1,
        chunk=8,
        model_path=str(tmp_path),
    )

    table.close()
    table.close()  # idempotent: must not raise

    assert all(mm is None for mm in table.mm)
    with pytest.raises(IndexError, match="shard"):
        table.gather(np.array([0], dtype=np.int64))


def test_attach_table_closes_a_stale_table_before_building_the_new_one(
    tmp_path: Path,
) -> None:
    """(M7): a direct _attach_table re-entry on an already-populated
    placeholder must close the old table (ThreadPool + memmaps) rather
    than leaking it, even though build_tables' own idempotency skip makes
    this unreachable through the normal path."""
    full = _write_ple_layer(tmp_path, layer_idx=0, vocab=9, parts=3, cols=2, scale=0.5)
    embedding = _attached_embedding(
        tmp_path, layer_idx=0, vocab=9, parts=3, cols=2, scale=0.5
    )
    stale_table = embedding.table
    assert stale_table is not None
    layer_shards = ple_mmap.discover_shards(str(tmp_path))[0]

    ple_mmap._attach_table(
        embedding,
        layer_shards,
        split_ngram_parts=3,
        layer_idx=0,
        model_path=str(tmp_path),
    )

    assert embedding.table is not stale_table
    assert all(mm is None for mm in stale_table.mm)  # the old table was closed
    assert stale_table.pool._shutdown  # (F-10) the old ThreadPool was shut down
    ids = torch.tensor([0, 8], dtype=torch.long)
    assert torch.equal(embedding(ids), full[ids])  # the new table still works


# --------------------------------------------------------------------------- #
# Directory resolution (R3.2/R4.5)
# --------------------------------------------------------------------------- #


def test_resolve_model_path_uses_existing_directory_verbatim(tmp_path: Path) -> None:
    model_config = SimpleNamespace(model_weights=str(tmp_path), model="ignored")

    assert ple_mmap.resolve_model_path(model_config) == str(tmp_path)


def test_resolve_model_path_falls_back_to_offline_snapshot_download(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, object] = {}

    def fake_snapshot_download(repo_id, revision, allow_patterns, local_files_only):
        calls["repo_id"] = repo_id
        calls["revision"] = revision
        calls["allow_patterns"] = allow_patterns
        calls["local_files_only"] = local_files_only
        return "/resolved/snapshot/path"

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)
    model_config = SimpleNamespace(
        model_weights="", model="RadixArk/Qwen3.8-Flash-Next-NVFP4", revision="deadbeef"
    )

    path = ple_mmap.resolve_model_path(model_config)

    assert path == "/resolved/snapshot/path"
    assert calls == {
        "repo_id": "RadixArk/Qwen3.8-Flash-Next-NVFP4",
        "revision": "deadbeef",
        "allow_patterns": ["*.safetensors"],
        "local_files_only": True,
    }


# --------------------------------------------------------------------------- #
# VLLM_PLE_MMAP_DIR: the table decoupled from the checkpoint it serves.
# --------------------------------------------------------------------------- #


def _use_mmap_dir(monkeypatch: pytest.MonkeyPatch, directory: Path) -> None:
    """Enable the mmap path and point it at a standalone table directory."""
    monkeypatch.setenv("VLLM_PLE_MMAP", "1")
    monkeypatch.setenv("VLLM_PLE_MMAP_DIR", str(directory))


def test_resolve_table_path_prefers_the_mmap_dir_over_the_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VLLM_PLE_MMAP", "1")
    monkeypatch.setenv("VLLM_PLE_MMAP_DIR", str(tmp_path))
    model_config = SimpleNamespace(model_weights="/some/checkpoint", model="ignored")

    assert ple_mmap.table_dir() == str(tmp_path)
    assert ple_mmap.resolve_table_path(model_config) == str(tmp_path)


def test_resolve_table_path_ignores_the_override_when_the_mmap_path_is_off(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The override is documented as consulted only under VLLM_PLE_MMAP=1 —
    with the mmap path off there is no table to redirect, and honouring it
    would make a stale export in the operator's shell change behavior."""
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    monkeypatch.setenv("VLLM_PLE_MMAP_DIR", str(tmp_path))
    model_config = SimpleNamespace(model_weights=str(checkpoint), model="ignored")

    assert ple_mmap.table_dir() is None
    assert ple_mmap.resolve_table_path(model_config) == str(checkpoint)


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ("relative/table/dir", "not an absolute path"),
        ("/nonexistent/ple-table-xyz", "is not a directory"),
    ],
)
def test_table_dir_refuses_an_unusable_override(
    value: str, message: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Falling back to the checkpoint's own table on a typo'd override would
    serve a DIFFERENT table than the operator asked for — one that differs
    in dtype and values — so an unusable override fails closed."""
    monkeypatch.setenv("VLLM_PLE_MMAP", "1")
    monkeypatch.setenv("VLLM_PLE_MMAP_DIR", value)

    with pytest.raises(RuntimeError, match=message):
        ple_mmap.table_dir()


def test_build_tables_serves_an_fp8_table_from_the_dir_off_a_header_scale(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The headline DIR contract: the checkpoint holds no PLE tensors at
    all, so nothing streams and weight_scale_loaded stays False. The table
    directory's headers carry both the rows and the scale, and the layer
    must attach off them and gather value-exact."""
    checkpoint = tmp_path / "checkpoint"
    table = tmp_path / "table"
    checkpoint.mkdir()
    table.mkdir()
    full = _write_ple_layer(table, layer_idx=0, vocab=10, parts=3, cols=2, scale=0.25)
    embedding = ple_mmap.MmapNgramEmbedding(10, 2)
    cc = SimpleNamespace(
        static_forward_context={"a.ple": _fake_ple_layer(0, embedding, 3)}
    )

    _use_mmap_dir(monkeypatch, table)
    ple_mmap.build_tables(_model_config(checkpoint), cc)

    assert embedding.table is not None
    assert embedding.table.model_path == str(table)
    assert embedding.table.torch_dtype is torch.float8_e4m3fn
    assert embedding.weight_scale_loaded is True
    assert embedding.weight_scale.float().item() == 0.25
    ids = torch.tensor([0, 9, 4], dtype=torch.long)
    assert torch.equal(embedding(ids), full[ids])


def test_build_tables_serves_a_bf16_table_from_the_dir_with_no_scale(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """DIR mode over an unquantized table: no scale exists anywhere, and
    none is required — the requires_scale=False descriptor must keep the
    whole scale block (header read included) from running."""
    checkpoint = tmp_path / "checkpoint"
    table = tmp_path / "table"
    checkpoint.mkdir()
    table.mkdir()
    full = _write_ple_layer(
        table,
        layer_idx=0,
        vocab=10,
        parts=3,
        cols=2,
        scale=0.0,
        write_scale=False,
        table_dtype=torch.bfloat16,
    )
    embedding = ple_mmap.MmapNgramEmbedding(10, 2)
    cc = SimpleNamespace(
        static_forward_context={"a.ple": _fake_ple_layer(0, embedding, 3)}
    )

    _use_mmap_dir(monkeypatch, table)
    ple_mmap.build_tables(_model_config(checkpoint), cc)

    assert embedding.table is not None
    assert embedding.table.torch_dtype is torch.bfloat16
    assert embedding.weight_scale_loaded is False
    ids = torch.tensor([0, 9, 4], dtype=torch.long)
    assert torch.equal(embedding(ids), full[ids])


def test_build_tables_dir_mode_overrides_a_scale_streamed_by_the_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A checkpoint that still ships its own PLE family streams a scale that
    describes ITS table, not the one in VLLM_PLE_MMAP_DIR. Cross-checking
    the two would refuse a legitimate config, and keeping the streamed one
    would serve the dir's rows against the checkpoint's scale — precisely
    the mismatch build_tables guards against elsewhere. The dir wins."""
    checkpoint = tmp_path / "checkpoint"
    table = tmp_path / "table"
    checkpoint.mkdir()
    table.mkdir()
    _write_ple_layer(checkpoint, layer_idx=0, vocab=10, parts=3, cols=2, scale=0.25)
    _write_ple_layer(table, layer_idx=0, vocab=10, parts=3, cols=2, scale=0.75)
    embedding = _loaded_placeholder(10, 2, 0.25)  # as the checkpoint streamed it
    cc = SimpleNamespace(
        static_forward_context={"a.ple": _fake_ple_layer(0, embedding, 3)}
    )

    _use_mmap_dir(monkeypatch, table)
    ple_mmap.build_tables(_model_config(checkpoint), cc)

    assert embedding.table is not None
    assert embedding.table.model_path == str(table)
    assert embedding.weight_scale.float().item() == 0.75


def test_build_tables_dir_mode_refuses_an_fp8_table_with_no_scale_anywhere(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fail closed on the one combination that could silently corrupt: an
    fp8 table directory with no weight_scale in its headers. A scale the
    CHECKPOINT streamed must not stand in for it — it belongs to a
    different table."""
    checkpoint = tmp_path / "checkpoint"
    table = tmp_path / "table"
    checkpoint.mkdir()
    table.mkdir()
    _write_ple_layer(
        table, layer_idx=0, vocab=10, parts=3, cols=2, scale=0.25, write_scale=False
    )
    embedding = _loaded_placeholder(10, 2, 0.25)  # a streamed scale exists
    cc = SimpleNamespace(
        static_forward_context={"a.ple": _fake_ple_layer(0, embedding, 3)}
    )

    _use_mmap_dir(monkeypatch, table)
    with pytest.raises(RuntimeError, match="no ngram_embedding"):
        ple_mmap.build_tables(_model_config(checkpoint), cc)

    assert embedding.table is None


# --------------------------------------------------------------------------- #
# (a) env-on vs env-off FORWARD equivalence, through the CPU dispatch key.
# Widened per the R2.18 fallback: env-on now hashes AND gathers inside the
# op. Both arms call the SAME Qwen4ExpNGramEmbedding._hash_ngram_ids, so
# this test proves the env-on path loads the RIGHT weights and gathers and
# dequantizes them the same way the stock VocabParallelEmbedding path
# does — it does NOT independently verify the hashing math itself: a bug
# in _hash_ngram_ids would move both arms identically and cancel out here.
# Hashing correctness is pinned separately by
# test_hash_ngram_ids_matches_golden_ids below.
# --------------------------------------------------------------------------- #


def test_env_on_off_forward_equivalence_fp8_and_dequantized(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """env-off: a real stock VocabParallelEmbedding under
    Qwen4ExpPLEFp8EmbeddingMethod (real FP8 weight + weight_scale
    Parameters, mirrors test_ple.py's _make_fp8_embedding_layer). env-on:
    an MmapNgramEmbedding placeholder attached to shard files holding the
    IDENTICAL weight values, driven through the REGISTERED widened op via
    its CPU dispatch key. Same input_ids/query_start_loc/ngram_context on
    both sides; compared byte-equal at fp8 AND through
    _dequantize_embeddings to bf16. Proves weight-loading/gather/dequant
    equivalence between the two paths, not hashing correctness (both
    arms share the same _hash_ngram_ids call, see module comment above).
    """
    config = _make_text_config()  # ngram_size=3, heads_per_ngram=2 -> 4 heads
    embedding_dim = 8  # head_dim = 2
    scale = 0.5

    # --- env-off: real stock FP8 VocabParallelEmbedding. ---
    quant_config = Fp8Config(
        is_checkpoint_fp8_serialized=True,
        ignored_layers=[],
        weight_block_size=[128, 128],
    )
    stock = Qwen4ExpNGramEmbedding(
        config,
        embedding_dim,
        0,
        16,
        4,
        "model.layers.1.ple.ple_embedding",
        "model.layers.1.ple",
        quant_config=quant_config,
        params_dtype=torch.bfloat16,
    )
    assert isinstance(stock.ngram_embedding, embedding_module.VocabParallelEmbedding)
    vocab = stock.ngram_embedding.org_vocab_size
    head_dim = stock.head_dim
    parts = stock.split_ngram_parts
    weight = _synthetic_weight(vocab, head_dim, layer_idx=1)
    stock.ngram_embedding.weight.data.copy_(weight)
    stock.ngram_embedding.weight_scale.data.copy_(
        torch.tensor([scale], dtype=torch.bfloat16)
    )

    input_ids = torch.tensor([1, 2], dtype=torch.long)
    query_start_loc = torch.tensor([0, 2], dtype=torch.long)
    ngram_context = torch.zeros((1, 4), dtype=torch.long)

    reference = stock.forward(input_ids, query_start_loc, ngram_context)
    assert reference.dtype == torch.float8_e4m3fn

    # --- env-on: mmap placeholder backed by shards holding the SAME
    # weight values, driven through the registered custom op. ---
    _write_ple_layer(
        tmp_path, layer_idx=1, vocab=vocab, parts=parts, cols=head_dim, scale=scale
    )
    monkeypatch.setenv("VLLM_PLE_MMAP", "1")
    cc = CompilationConfig(
        mode=CompilationMode.VLLM_COMPILE,
        cudagraph_mode=CUDAGraphMode.PIECEWISE,
        splitting_ops=[ple_mmap.QUALIFIED_OP_NAME],
    )
    vllm_config = SimpleNamespace(
        compilation_config=cc, model_config=_model_config(tmp_path)
    )
    with set_current_vllm_config(vllm_config):
        mmap_module = Qwen4ExpNGramEmbedding(
            config,
            embedding_dim,
            0,
            16,
            4,
            "model.layers.1.ple.ple_embedding",
            "model.layers.1.ple",
            params_dtype=torch.bfloat16,
        )
    assert isinstance(mmap_module.ngram_embedding, ple_mmap.MmapNgramEmbedding)
    ple_mmap.set_weight_scale(
        mmap_module.ngram_embedding,
        torch.tensor([scale], dtype=torch.bfloat16),
        torch.device("cpu"),
    )
    layer_shards = ple_mmap.discover_shards(str(tmp_path))[1]
    ple_mmap._attach_table(
        mmap_module.ngram_embedding,
        layer_shards,
        split_ngram_parts=parts,
        layer_idx=1,
        model_path=str(tmp_path),
    )

    fake_ple_layer = SimpleNamespace(ple_embedding=mmap_module)
    ctx = SimpleNamespace(no_compile_layers={mmap_module.layer_name: fake_ple_layer})
    with forward_context.override_forward_context(ctx):
        got = mmap_module.forward(input_ids, query_start_loc, ngram_context)

    assert torch.equal(got, reference)

    # Real nn.Module chain (mirrors test_ple.py's
    # test_ple_fp8_embedding_dequantizes_in_ple_layer): __new__ + a manual
    # nn.Module.__init__ skips the heavy real __init__, but
    # _get_embedding_weight_scale/_dequantize_embeddings stay the REAL
    # bound methods, exercising the actual getattr chain — no lambda stub.
    stock_ple_layer = Qwen4ExpPLELayer.__new__(Qwen4ExpPLELayer)
    nn.Module.__init__(stock_ple_layer)
    stock_ple_layer.ple_embedding = stock
    dequant_off = stock_ple_layer._dequantize_embeddings(reference, torch.bfloat16)

    mmap_ple_layer = Qwen4ExpPLELayer.__new__(Qwen4ExpPLELayer)
    nn.Module.__init__(mmap_ple_layer)
    mmap_ple_layer.ple_embedding = mmap_module
    dequant_on = mmap_ple_layer._dequantize_embeddings(got, torch.bfloat16)

    assert torch.equal(dequant_on, dequant_off)


def test_dequantize_embeddings_casts_a_bf16_table_to_the_output_dtype() -> None:
    """A BF16 (unquantized) PLE table carries no scale to apply: the
    non-fp8 branch of _dequantize_embeddings must still cast to
    output_dtype, mirroring the fp8 branch's final cast — without it, a
    bf16 table served under e.g. ``--dtype float16`` reaches key_proj with
    a stale bf16 dtype and fails there, unattributably, instead of here."""
    layer = Qwen4ExpPLELayer.__new__(Qwen4ExpPLELayer)
    nn.Module.__init__(layer)
    embeddings = torch.tensor([[1.5, -2.25], [0.5, 3.0]], dtype=torch.bfloat16)
    assert not is_fp8(embeddings)

    out = layer._dequantize_embeddings(embeddings, torch.float16)

    assert out.dtype == torch.float16
    assert torch.equal(out, embeddings.to(torch.float16))


# --------------------------------------------------------------------------- #
# (M1) _hash_ngram_ids golden pin. The equivalence test above drives BOTH
# arms through the same _hash_ngram_ids call, so it cannot catch a bug in
# the hashing math itself (xor chain / remainder / offset) — a mutation
# there moves both arms identically and cancels out. This test freezes the
# exact output of a fixed, small, real Qwen4ExpNGramEmbedding on fixed
# inputs, so a hashing regression has to change these hardcoded numbers.
# --------------------------------------------------------------------------- #


def test_hash_ngram_ids_matches_golden_ids() -> None:
    """Golden values captured by running this exact scenario once and
    hardcoding the result — they pin the xor-chain/remainder/offset math
    in _hash_ngram_ids (ngram_size=3, heads_per_ngram=2, seed=1234,
    ple_dense_layer_id=0), not merely its shape.
    """
    config = _make_text_config()  # ngram_size=3, heads_per_ngram=2 -> 4 heads
    module = Qwen4ExpNGramEmbedding(
        config,
        8,
        0,
        8,
        2,
        "model.layers.1.ple.ple_embedding",
        "model.layers.1.ple",
        params_dtype=torch.float32,
    )

    input_ids = torch.tensor([11, 22, 33], dtype=torch.long)
    query_start_loc = torch.tensor([0, 2, 3], dtype=torch.long)
    ngram_context = torch.tensor([[44, 55], [66, 77]], dtype=torch.long)

    ngram_ids = module._hash_ngram_ids(input_ids, query_start_loc, ngram_context)

    assert ngram_ids.shape == (3, 4)
    golden = torch.tensor(
        [
            [647, 1359, 2559, 3257],
            [128, 1518, 2612, 3993],
            [891, 1118, 2768, 3902],
        ],
        dtype=torch.long,
    )
    assert torch.equal(ngram_ids, golden)


# --------------------------------------------------------------------------- #
# (M3) R1.2 hinge: Qwen4ExpNGramEmbedding.forward's mmap branch allocates
# the output buffer in the TABLE's dtype, not params_dtype — zero prior
# coverage exercised this exact allocation through the real forward().
# --------------------------------------------------------------------------- #


def test_mmap_forward_allocates_an_fp8_output_buffer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A regression here (e.g. back to params_dtype bf16) would leave the
    model serving unscaled embeddings — is_fp8() would stop firing and
    Qwen4ExpPLELayer._dequantize_embeddings would silently skip
    dequantization — while every test that exercises only the custom op or
    the placeholder in isolation stays green."""
    monkeypatch.setenv("VLLM_PLE_MMAP", "1")
    config = _make_text_config()  # ngram_size=3, heads_per_ngram=2 -> 4 heads
    layer_name = "model.language_model.layers.1.ple"
    cc = CompilationConfig(
        mode=CompilationMode.VLLM_COMPILE,
        cudagraph_mode=CUDAGraphMode.PIECEWISE,
        splitting_ops=[ple_mmap.QUALIFIED_OP_NAME],
    )
    # Nonexistent repo: __init__'s validate_shards_for tolerates an
    # unresolvable path and defers — org_vocab_size (needed to write a
    # matching shard fixture) is only known once the module is built.
    unresolvable_config = SimpleNamespace(
        dtype=torch.bfloat16,
        model_weights="",
        model="nonexistent-org/nonexistent-repo-xyz",
        revision=None,
    )
    vllm_config = SimpleNamespace(
        compilation_config=cc, model_config=unresolvable_config
    )
    with set_current_vllm_config(vllm_config):
        module = Qwen4ExpNGramEmbedding(
            config,
            8,
            0,
            16,
            4,
            f"{layer_name}.ple_embedding",
            layer_name,
            params_dtype=torch.bfloat16,
        )

    embedding = module.ngram_embedding
    assert isinstance(embedding, ple_mmap.MmapNgramEmbedding)
    vocab = embedding.org_vocab_size
    head_dim = module.head_dim
    parts = module.split_ngram_parts
    _write_ple_layer(
        tmp_path, layer_idx=1, vocab=vocab, parts=parts, cols=head_dim, scale=0.5
    )
    ple_mmap.set_weight_scale(
        embedding, torch.tensor([0.5], dtype=torch.bfloat16), torch.device("cpu")
    )
    layer_shards = ple_mmap.discover_shards(str(tmp_path))[1]
    ple_mmap._attach_table(
        embedding,
        layer_shards,
        split_ngram_parts=parts,
        layer_idx=1,
        model_path=str(tmp_path),
    )

    # ple_embedding must be the REAL module (not a bare embedding wrapper):
    # the widened (R2.18) op calls ple_embedding_module._hash_ngram_ids(...)
    # before the gather, and only Qwen4ExpNGramEmbedding provides that.
    fake_ple_layer = SimpleNamespace(ple_embedding=module)
    ctx = SimpleNamespace(no_compile_layers={layer_name: fake_ple_layer})

    input_ids = torch.tensor([1, 2], dtype=torch.long)
    query_start_loc = torch.tensor([0, 2], dtype=torch.long)
    ngram_context = torch.zeros((1, 4), dtype=torch.long)

    with forward_context.override_forward_context(ctx):
        out = module.forward(input_ids, query_start_loc, ngram_context)

    assert out.dtype == torch.float8_e4m3fn
    assert out.shape == (2, 8)
    assert is_fp8(out)


# --------------------------------------------------------------------------- #
# layer_name plumbing (R2.6)
# --------------------------------------------------------------------------- #


def test_ngram_embedding_stores_the_layer_name_it_is_constructed_with() -> None:
    config = _make_text_config()
    module = Qwen4ExpNGramEmbedding(
        config,
        8,
        0,
        16,
        4,
        "model.layers.1.ple.ple_embedding",
        "model.layers.1.ple",
        params_dtype=torch.float32,
    )

    assert module.layer_name == "model.layers.1.ple"


def test_ple_layer_registers_its_own_prefix_as_the_static_forward_context_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """(R2.6, dynamic): Qwen4ExpPLELayer.__init__ passes its OWN prefix as
    layer_name — the exact key it registers into
    compilation_config.static_forward_context — never
    f"{prefix}.ple_embedding". Constructs a REAL Qwen4ExpPLELayer (not just
    the embedding in isolation) using the suite's TP rank/size
    monkeypatches, extended to vllm.model_executor.layers.linear for
    ReplicatedLinear's construction.
    """
    monkeypatch.setenv("VLLM_PLE_MMAP", "1")
    monkeypatch.setattr(linear_module, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        linear_module, "get_tensor_model_parallel_world_size", lambda: 1
    )

    config = _make_text_config(
        hidden_size=8,
        hc_count=2,
        ple_conv_kernel_size=3,
        ple_embed_dim=8,
        rms_norm_eps=1e-5,
    )
    cc = CompilationConfig(
        mode=CompilationMode.VLLM_COMPILE,
        cudagraph_mode=CUDAGraphMode.PIECEWISE,
        splitting_ops=[ple_mmap.QUALIFIED_OP_NAME],
    )
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            dtype=torch.float32, model_weights="", model="ignored/repo", revision=None
        ),
        cache_config=SimpleNamespace(mamba_cache_dtype="auto"),
        quant_config=None,
        scheduler_config=SimpleNamespace(max_num_batched_tokens=16, max_num_seqs=4),
        num_speculative_tokens=0,
        compilation_config=cc,
    )

    with set_current_vllm_config(vllm_config):
        layer = Qwen4ExpPLELayer(
            config,
            vllm_config=vllm_config,
            layer_idx=1,
            ple_dense_layer_id=0,
            prefix="model.layers.1.ple",
        )

    assert layer.ple_embedding.layer_name == layer.prefix
    assert layer.ple_embedding.layer_name in cc.static_forward_context
    assert cc.static_forward_context[layer.ple_embedding.layer_name] is layer
    # (Seam gap) build_tables' _extract_layer_idx(layer_name) must recover
    # the SAME layer_idx the real layer was constructed with — the two
    # halves (registry key -> layer_idx string parsing, and the layer's own
    # int attribute) must agree on a real, not synthetic, prefix.
    assert ple_mmap._extract_layer_idx(layer.prefix) == layer.layer_idx


# --------------------------------------------------------------------------- #
# model.py load_weights hook (F1-iii/M6): both Qwen4ExpForCausalLM and
# Qwen4ExpForConditionalGeneration must call build_tables when enabled —
# ForCausalLM's call was the missing HIGH-severity gap (F1), since a
# text-only checkpoint served through that class alone previously left its
# PLE layer silently serving fp8 zeros forever.
# --------------------------------------------------------------------------- #


class _FakeAutoWeightsLoader:
    """Stands in for AutoWeightsLoader so the stub `self` below never needs
    to be a real nn.Module (AutoWeightsLoader introspects named_parameters/
    named_buffers/named_modules on construction)."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs

    def load_weights(self, weights: object, mapper: object = None) -> set[str]:
        del weights, mapper
        return {"dummy"}


@pytest.mark.parametrize(
    "cls_name", ["Qwen4ExpForCausalLM", "Qwen4ExpForConditionalGeneration"]
)
def test_model_load_weights_calls_build_tables_exactly_once_when_enabled(
    monkeypatch: pytest.MonkeyPatch, cls_name: str, tmp_path: Path
) -> None:
    monkeypatch.setenv("VLLM_PLE_MMAP", "1")
    monkeypatch.setattr(model_module, "AutoWeightsLoader", _FakeAutoWeightsLoader)
    calls: list[tuple[object, object]] = []
    monkeypatch.setattr(ple_mmap, "build_tables", lambda mc, cc: calls.append((mc, cc)))

    cls = getattr(model_module, cls_name)
    model_config = _model_config(tmp_path)
    stub_self = SimpleNamespace(
        hf_to_vllm_mapper=cls.hf_to_vllm_mapper,
        model_config=model_config,
        language_model_only=False,
    )
    cc = SimpleNamespace(static_forward_context={})
    vllm_config = SimpleNamespace(compilation_config=cc)

    with set_current_vllm_config(vllm_config):
        result = cls.load_weights(stub_self, iter([]))

    assert result == {"dummy"}
    assert calls == [(model_config, cc)]


def test_model_load_weights_never_calls_build_tables_when_disabled(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(model_module, "AutoWeightsLoader", _FakeAutoWeightsLoader)
    calls: list[tuple[object, object]] = []
    monkeypatch.setattr(ple_mmap, "build_tables", lambda mc, cc: calls.append((mc, cc)))

    stub_self = SimpleNamespace(
        hf_to_vllm_mapper=model_module.Qwen4ExpForCausalLM.hf_to_vllm_mapper,
        model_config=_model_config(tmp_path),
    )
    cc = SimpleNamespace(static_forward_context={})
    vllm_config = SimpleNamespace(compilation_config=cc)

    with set_current_vllm_config(vllm_config):
        result = model_module.Qwen4ExpForCausalLM.load_weights(stub_self, iter([]))

    assert result == {"dummy"}
    assert calls == []


# --------------------------------------------------------------------------- #
# Default-off inertness (invariant 2)
# --------------------------------------------------------------------------- #


def test_default_off_uses_the_stock_vocab_parallel_embedding() -> None:
    assert ple_mmap.enabled() is False
    config = _make_text_config()

    module = Qwen4ExpNGramEmbedding(
        config,
        8,
        0,
        16,
        4,
        "model.layers.1.ple.ple_embedding",
        "model.layers.1.ple",
        params_dtype=torch.float32,
    )

    assert isinstance(module.ngram_embedding, embedding_module.VocabParallelEmbedding)
    assert not isinstance(module.ngram_embedding, ple_mmap.MmapNgramEmbedding)


def test_default_off_forward_never_calls_the_mmap_gather_op(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With VLLM_PLE_MMAP unset, forward() must take the exact stock branch
    (direct call + .flatten(-2)), never the custom op."""
    config = _make_text_config()
    module = Qwen4ExpNGramEmbedding(
        config,
        8,
        0,
        16,
        4,
        "model.layers.1.ple.ple_embedding",
        "model.layers.1.ple",
        params_dtype=torch.float32,
    )
    sentinel = torch.arange(2 * 4 * 2, dtype=torch.bfloat16).reshape(2, 4, 2)
    calls: list[torch.Tensor] = []

    def spy_forward(ids: torch.Tensor) -> torch.Tensor:
        calls.append(ids)
        return sentinel

    monkeypatch.setattr(module.ngram_embedding, "forward", spy_forward)
    op_calls: list[object] = []
    monkeypatch.setattr(
        torch.ops.vllm,
        ple_mmap.OP_NAME,
        lambda *a, **k: op_calls.append((a, k)),
        raising=False,
    )

    input_ids = torch.tensor([1, 2], dtype=torch.long)
    query_start_loc = torch.tensor([0, 2], dtype=torch.long)
    ngram_context = torch.zeros((1, 4), dtype=torch.long)

    output = module.forward(input_ids, query_start_loc, ngram_context)

    assert len(calls) == 1  # the stock embedding was called directly
    assert not op_calls  # the custom op was never reached
    assert torch.equal(output, sentinel.flatten(-2))


def test_default_off_load_weights_matches_the_stock_contract() -> None:
    module = Qwen4ExpNGramEmbedding.__new__(Qwen4ExpNGramEmbedding)
    torch.nn.Module.__init__(module)
    module.split_ngram_parts = 2
    module.register_buffer("layer_multipliers", torch.zeros(1, dtype=torch.long))
    module.register_buffer("ngram_heads_offsets", torch.zeros(1, dtype=torch.long))
    module.register_buffer("ngram_heads_vocab_sizes", torch.zeros(1, dtype=torch.long))
    embedding = SimpleNamespace(
        org_vocab_size=8,
        embedding_dim=2,
        weight=torch.nn.Parameter(torch.full((4, 2), -1.0)),
        shard_indices=SimpleNamespace(org_vocab_start_index=2, org_vocab_end_index=6),
    )
    module.ngram_embedding = embedding  # not MmapNgramEmbedding -> mmap_enabled=False

    shard_0 = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    shard_1 = torch.arange(8, 16, dtype=torch.float32).reshape(4, 2)
    loaded = module.load_weights(
        [
            ("ngram_embedding.shard_0.weight", shard_0),
            ("ngram_embedding.shard_1.weight", shard_1),
        ]
    )

    assert loaded == {"ngram_embedding.weight"}
    expected = torch.cat((shard_0[2:4], shard_1[0:2]))
    torch.testing.assert_close(embedding.weight, expected)


# --------------------------------------------------------------------------- #
# Dynamic-shape compile safety. The engine compiles the model with
# query_start_loc/ngram_context/input_ids marked dynamic in dim 0
# (support_torch_compile dynamic_arg_dims); the original hashing specialized
# those dims (ConstraintViolationError at engine init, R2.18). This traces
# _hash_ngram_ids exactly as production does and pins: one graph, no
# recompile, eager-identical outputs across two different
# (num_tokens, num_reqs) shapes.
# --------------------------------------------------------------------------- #


def test_hash_ngram_ids_traces_with_dynamic_shapes() -> None:
    config = _make_text_config()
    module = Qwen4ExpNGramEmbedding(
        config,
        8,
        0,
        8,
        4,
        "model.layers.1.ple.ple_embedding",
        "model.layers.1.ple",
        params_dtype=torch.float32,
    )

    shapes_a = (
        torch.tensor([11, 22, 33], dtype=torch.long),
        torch.tensor([0, 2, 3], dtype=torch.long),
        torch.tensor([[44, 55], [66, 77]], dtype=torch.long),
    )
    shapes_b = (
        torch.tensor([3, 14, 15, 92, 65], dtype=torch.long),
        torch.tensor([0, 1, 3, 5], dtype=torch.long),
        torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.long),
    )
    eager_a = module._hash_ngram_ids(*shapes_a)
    eager_b = module._hash_ngram_ids(*shapes_b)

    torch._dynamo.reset()
    counter = torch._dynamo.testing.CompileCounter()
    compiled = torch.compile(module._hash_ngram_ids, backend=counter, fullgraph=True)

    for tensors in (shapes_a, shapes_b):
        for tensor in tensors:
            torch._dynamo.mark_dynamic(tensor, 0)

    traced_a = compiled(*shapes_a)
    traced_b = compiled(*shapes_b)

    assert torch.equal(traced_a, eager_a)
    assert torch.equal(traced_b, eager_b)
    assert counter.frame_count == 1, (
        f"expected one dynamic graph, got {counter.frame_count} "
        "(shape-specialized recompile)"
    )


# --------------------------------------------------------------------------- #
# Zero-copy GPU gather (VLLM_PLE_MMAP_GPU_GATHER). CPU-runnable pieces are
# tested unconditionally; kernel behavior needs a CUDA device (GB10 ATS) and
# skips elsewhere.
# --------------------------------------------------------------------------- #

_needs_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="zero-copy gather needs a CUDA device"
)


def test_gpu_gather_defaults_off_and_table_records_the_flag(
    tmp_path: Path,
) -> None:
    _write_ple_layer(tmp_path, layer_idx=0, vocab=50, parts=4, cols=8, scale=0.5)
    embedding = _attached_embedding(tmp_path, 0, 50, 4, 8, 0.5)
    assert embedding.table is not None
    assert embedding.table.gpu_gather is False


def test_gpu_gather_env_flag_enables_on_the_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VLLM_PLE_MMAP_GPU_GATHER", "1")
    _write_ple_layer(tmp_path, layer_idx=0, vocab=50, parts=4, cols=8, scale=0.5)
    embedding = _attached_embedding(tmp_path, 0, 50, 4, 8, 0.5)
    assert embedding.table is not None
    assert embedding.table.gpu_gather is True


def test_gpu_gather_without_triton_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VLLM_PLE_MMAP_GPU_GATHER", "1")
    monkeypatch.setattr(ple_mmap, "HAS_TRITON", False)
    _write_ple_layer(tmp_path, layer_idx=0, vocab=50, parts=4, cols=8, scale=0.5)
    with pytest.raises(RuntimeError, match="triton"):
        _attached_embedding(tmp_path, 0, 50, 4, 8, 0.5)


def test_shard_base_addresses_match_memmaps(tmp_path: Path) -> None:
    _write_ple_layer(tmp_path, layer_idx=0, vocab=50, parts=4, cols=8, scale=0.5)
    embedding = _attached_embedding(tmp_path, 0, 50, 4, 8, 0.5)
    table = embedding.table
    assert table is not None
    bases = table._shard_base_addresses()
    assert bases == [mm.ctypes.data for mm in table.mm]


def test_shard_base_addresses_fail_closed_on_a_missing_shard(
    tmp_path: Path,
) -> None:
    _write_ple_layer(tmp_path, layer_idx=0, vocab=50, parts=4, cols=8, scale=0.5)
    embedding = _attached_embedding(tmp_path, 0, 50, 4, 8, 0.5)
    table = embedding.table
    assert table is not None
    table.mm[1] = None
    with pytest.raises(RuntimeError, match="shard 1"):
        table._shard_base_addresses()


@_needs_cuda
def test_gpu_gather_matches_cpu_gather_across_shard_boundaries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VLLM_PLE_MMAP_GPU_GATHER", "1")
    vocab, parts, cols = 50, 4, 8
    _write_ple_layer(
        tmp_path, layer_idx=0, vocab=vocab, parts=parts, cols=cols, scale=0.5
    )
    embedding = _attached_embedding(tmp_path, 0, vocab, parts, cols, 0.5)
    table = embedding.table
    assert table is not None
    # Boundary rows of every shard plus interior rows, unsorted, with repeats.
    ids = torch.tensor([0, 12, 13, 25, 26, 38, 39, 49, 7, 7, 30], dtype=torch.int64)
    ref = table.gather(ids.numpy())
    out = torch.empty((ids.numel(), table.row_bytes), dtype=torch.uint8, device="cuda")
    table.gather_gpu(ids.cuda(), out)
    torch.cuda.synchronize()
    assert (out.cpu().numpy() == ref).all()


@_needs_cuda
def test_gpu_gather_zero_fills_out_of_range_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VLLM_PLE_MMAP_GPU_GATHER", "1")
    _write_ple_layer(tmp_path, layer_idx=0, vocab=50, parts=4, cols=8, scale=0.5)
    embedding = _attached_embedding(tmp_path, 0, 50, 4, 8, 0.5)
    table = embedding.table
    assert table is not None
    ids = torch.tensor([-3, 50, 1], dtype=torch.int64, device="cuda")
    out = torch.full((3, table.row_bytes), 7, dtype=torch.uint8, device="cuda")
    table.gather_gpu(ids, out)
    torch.cuda.synchronize()
    got = out.cpu()
    assert (got[0] == 0).all()
    assert (got[1] == 0).all()
    assert (got[2].numpy() == table.gather(np.array([1]))[0]).all()


@_needs_cuda
def test_forward_gpu_path_matches_cpu_path_byte_for_byte(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VLLM_PLE_MMAP_GPU_GATHER", "1")
    vocab, parts, cols = 50, 4, 8
    _write_ple_layer(
        tmp_path, layer_idx=0, vocab=vocab, parts=parts, cols=cols, scale=0.5
    )
    embedding = _attached_embedding(tmp_path, 0, vocab, parts, cols, 0.5)
    ids = torch.tensor([[0, 12, 49], [26, 7, 39]], dtype=torch.int64)
    # CPU ids -> CPU gather path (gpu_gather requires ids.is_cuda).
    cpu_out = embedding.forward(ids)
    gpu_out = embedding.forward(ids.cuda())
    assert gpu_out.is_cuda
    assert gpu_out.dtype == cpu_out.dtype
    assert gpu_out.shape == cpu_out.shape
    assert (
        gpu_out.view(torch.uint8).cpu().numpy()
        == cpu_out.view(torch.uint8).cpu().numpy()
    ).all()


def test_prefault_ranges_page_math_and_oob_filtering(tmp_path: Path) -> None:
    _write_ple_layer(tmp_path, layer_idx=0, vocab=50, parts=4, cols=8, scale=0.5)
    embedding = _attached_embedding(tmp_path, 0, 50, 4, 8, 0.5)
    table = embedding.table
    assert table is not None
    table._base_addrs = table._shard_base_addresses()
    ids = np.array([-1, 0, 5, 50, 5], dtype=np.int64)  # OOB dropped, dupes merge
    ranges = table._prefault_ranges(ids)
    for addr, span in ranges:
        assert addr % 4096 == 0
        assert span % 4096 == 0 and span >= 4096
    # every in-range row's bytes are covered by some range
    for rid in (0, 5):
        sid, local = divmod(rid, table.shard_size)
        start = table._base_addrs[sid] + local * table.row_bytes
        end = start + table.row_bytes - 1
        assert any(a <= start and end < a + s for a, s in ranges)
    assert table._prefault_ranges(np.array([-7, 999], dtype=np.int64)) == []


def test_prefault_cb_never_raises_on_garbage_state(tmp_path: Path) -> None:
    _write_ple_layer(tmp_path, layer_idx=0, vocab=50, parts=4, cols=8, scale=0.5)
    embedding = _attached_embedding(tmp_path, 0, 50, 4, 8, 0.5)
    table = embedding.table
    assert table is not None
    table._libc = ple_mmap._load_libc()  # get past the fail-open early return
    table._prefault_pinned[0] = "not a tensor"  # type: ignore[list-item]
    table._prefault_n[0] = 3
    table._prefault_cb(0)  # driver-thread entry: must swallow, never raise
    assert table._prefault_errors == 1


@_needs_cuda
def test_gather_gpu_prefault_path_runs_clean(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VLLM_PLE_MMAP_GPU_GATHER", "1")
    _write_ple_layer(tmp_path, layer_idx=0, vocab=50, parts=4, cols=8, scale=0.5)
    embedding = _attached_embedding(tmp_path, 0, 50, 4, 8, 0.5)
    table = embedding.table
    assert table is not None
    ids = torch.tensor([0, 12, 26, 49], dtype=torch.int64)
    out = torch.empty((4, table.row_bytes), dtype=torch.uint8, device="cuda")
    table.gather_gpu(ids.cuda(), out)
    torch.cuda.synchronize()
    assert (out.cpu().numpy() == table.gather(ids.numpy())).all()
    assert table._prefault_errors == 0
