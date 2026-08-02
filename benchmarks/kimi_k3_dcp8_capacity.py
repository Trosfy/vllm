#!/usr/bin/env python3
"""Weight-free KV-capacity proof for the Kimi K3 TP16/DCP8 1M profile."""

from __future__ import annotations

import argparse
import json
from types import SimpleNamespace

import torch

from vllm.v1.core.kv_cache_utils import (
    get_kv_cache_capacity,
    get_kv_cache_configs,
)
from vllm.v1.kv_cache_interface import (
    KVQuantMode,
    MambaSpec,
    MLAAttentionSpec,
)

MAX_MODEL_LEN = 1_048_576
TP_SIZE = 16
DCP_SIZE = 8
KV_CACHE_MEMORY_BYTES = 1_879_048_192
MLA_LAYERS = 24
KDA_LAYERS = 69
BLOCK_SIZE = 720


def make_config(dcp_size: int, max_model_len: int) -> SimpleNamespace:
    return SimpleNamespace(
        model_config=SimpleNamespace(
            max_model_len=max_model_len,
            original_max_model_len=None,
        ),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=TP_SIZE,
            decode_context_parallel_size=dcp_size,
            prefill_context_parallel_size=1,
        ),
        cache_config=SimpleNamespace(
            mamba_cache_mode="none",
            num_gpu_blocks_override=None,
        ),
        scheduler_config=SimpleNamespace(
            disable_hybrid_kv_cache_manager=False,
        ),
        kv_transfer_config=None,
        speculative_config=None,
    )


def make_specs() -> dict[str, MLAAttentionSpec | MambaSpec]:
    # K3 has one 576-byte FP8 latent record per token and MLA layer.
    mla = MLAAttentionSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=1,
        head_size=576,
        head_size_v=512,
        dtype=torch.float8_e4m3fn,
        kv_quant_mode=KVQuantMode.FP8_PER_TENSOR,
        indexes_kv_by_block_stride=True,
    )

    # TP16-local KDA state: conv=(2304, 3) BF16 and recurrent=(6,128,128)
    # FP32.  vLLM pads it to the 720*576-byte MLA page.
    kda = MambaSpec(
        block_size=BLOCK_SIZE,
        shapes=((2304, 3), (6, 128, 128)),
        dtypes=(torch.bfloat16, torch.float32),
        page_size_padded=mla.page_size_bytes,
        mamba_cache_mode="none",
    )
    specs: dict[str, MLAAttentionSpec | MambaSpec] = {}
    specs.update({f"mla.{i}": mla for i in range(MLA_LAYERS)})
    specs.update({f"kda.{i}": kda for i in range(KDA_LAYERS)})
    return specs


def plan(
    *,
    dcp_size: int = DCP_SIZE,
    max_model_len: int = MAX_MODEL_LEN,
    available_memory: int = KV_CACHE_MEMORY_BYTES,
) -> dict[str, int | float | list[int]]:
    if TP_SIZE % dcp_size:
        raise ValueError(f"TP{TP_SIZE} is not divisible by DCP{dcp_size}")
    config = make_config(dcp_size, max_model_len)
    cache = get_kv_cache_configs(config, [make_specs()], [available_memory])[0]
    capacity, concurrency = get_kv_cache_capacity(config, cache)
    return {
        "tp_size": TP_SIZE,
        "dcp_size": dcp_size,
        "max_model_len": max_model_len,
        "kv_cache_memory_bytes_per_rank": available_memory,
        "num_blocks": cache.num_blocks,
        "block_size": BLOCK_SIZE,
        "capacity_tokens": capacity,
        "max_concurrency": concurrency,
        "group_layer_counts": [
            len(group.layer_names) for group in cache.kv_cache_groups
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dcp-size", type=int, default=DCP_SIZE)
    parser.add_argument("--max-model-len", type=int, default=MAX_MODEL_LEN)
    parser.add_argument(
        "--kv-cache-memory-bytes", type=int, default=KV_CACHE_MEMORY_BYTES
    )
    args = parser.parse_args()
    result = plan(
        dcp_size=args.dcp_size,
        max_model_len=args.max_model_len,
        available_memory=args.kv_cache_memory_bytes,
    )
    print(json.dumps(result, indent=2))
    if result["capacity_tokens"] < args.max_model_len:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
