from benchmarks.kimi_k3_dcp8_capacity import (
    KV_CACHE_MEMORY_BYTES,
    MAX_MODEL_LEN,
    plan,
)


def test_kimi_k3_full_mxfp4_dcp8_cache_holds_one_million_tokens() -> None:
    result = plan()

    assert result["group_layer_counts"] == [24, 23, 23, 23]
    assert result["num_blocks"] == 188
    assert result["capacity_tokens"] == 1_059_851
    assert result["capacity_tokens"] >= MAX_MODEL_LEN
    assert result["max_concurrency"] > 1.0


def test_kimi_k3_dcp8_cache_is_exact_one_eighth_dcp1_budget() -> None:
    dcp1 = plan(dcp_size=1, available_memory=KV_CACHE_MEMORY_BYTES * 8)
    dcp8 = plan()

    assert dcp1["capacity_tokens"] == 1_084_486
    assert (
        dcp8["kv_cache_memory_bytes_per_rank"] * 8
        == (dcp1["kv_cache_memory_bytes_per_rank"])
    )
