#!/usr/bin/env bash
# Memory-tight full-MXFP4 Kimi K3 profile: physical 1M KV cache plus a 2048
# token scheduler chunk.  This is kept separate from the validated chunk-256
# profile so the latter remains an exact rollback point.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# TP-sharding KDA f_a saves 120.75 MiB/rank. Dense MLA layers share one
# capture-stable scratch arena, rather than retaining one arena per layer.
export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-2048}"
export KV_CACHE_MEMORY_BYTES="${KV_CACHE_MEMORY_BYTES:-1860000000}"
# A 2048-token scheduler chunk reaches at most 6144 cached context rows during
# the 8k validation request. An 8k gather arena therefore keeps that path
# single-segment while returning 30.38 MiB/rank to BF16 projection transients.
export VLLM_MLA_CHUNKED_PREFILL_WORKSPACE_SIZE="${VLLM_MLA_CHUNKED_PREFILL_WORKSPACE_SIZE:-8192}"

KIMI_ADDITIONAL_CONFIG="${KIMI_ADDITIONAL_CONFIG:-{\"kda_shard_f_a\":true}}"

exec "${SCRIPT_DIR}/serve-kimi-k3-full-mxfp4-dcp8-1m.sh" \
  --additional-config "${KIMI_ADDITIONAL_CONFIG}" \
  "$@"
