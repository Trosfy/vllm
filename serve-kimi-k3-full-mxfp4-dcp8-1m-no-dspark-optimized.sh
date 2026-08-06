#!/usr/bin/env bash
# Full stock Kimi-K3 MXFP4 target without speculative decoding. This profile
# applies the validated TP16 projection, hierarchical all-reduce, and MoE
# decode optimizations to the physical-1M DCP8 layout.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export PYTHON_BIN="${PYTHON_BIN:-/opt/venv/bin/python}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-1048576}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-256}"
export KV_CACHE_MEMORY_BYTES="${KV_CACHE_MEMORY_BYTES:-1860000000}"
export VLLM_MLA_CHUNKED_PREFILL_WORKSPACE_SIZE="${VLLM_MLA_CHUNKED_PREFILL_WORKSPACE_SIZE:-2048}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Kimi-K3-MXFP4-HH-DCP8-1M-NoDSpark-Optimized}"

export VLLM_KIMI_USE_B12X_PROJECTION_GATHER=1
export VLLM_KIMI_USE_B12X_PAIRED_PROJECTION_GATHER=1
export VLLM_KIMI_USE_B12X_PAIRED_PROJECTION_TOPK=1

export KIMI_PCIE_ALLREDUCE_BACKEND=b12x
export KIMI_PCIE_ONESHOT_SINGLE_CHANNEL=1
export B12X_PCIE_HIERARCHICAL_DEFERRED_CONSUMPTION=1
export B12X_PCIE_HIERARCHICAL_DOUBLE_BUFFER=0
export B12X_PCIE_HIERARCHICAL_THREADS=224
export B12X_PCIE_HIERARCHICAL_NANOSLEEP_CYCLES=24
export B12X_PCIE_HIERARCHICAL_BF16X2=1
export B12X_W4A16_SMALL_M_HOST_BARRIER_RESET=0

export VLLM_DCP_A2A_MAX_TOKENS=1
export B12X_PCIE_DCP_THREADS=512
export B12X_PCIE_DCP_BLOCK_LIMIT=8
export VLLM_MEMORY_PROFILE_INCLUDE_ATTN=0
export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export COMPILATION_CONFIG='{"mode":0,"cudagraph_mode":"PIECEWISE","cudagraph_capture_sizes":[1],"pass_config":{"fuse_allreduce_rms":true}}'

# Replicating f_a removes 69 decode gathers at a measured cost of about
# 121 MiB/rank. The CC1 profile keeps a 256-token scheduler chunk and a 2K
# prefill arena so the physical 1M cache still leaves Triton autotune headroom.
# Callers that need larger prefill chunks can set KIMI_SHARD_F_A=1 and spend
# the recovered memory on MAX_NUM_BATCHED_TOKENS instead.
KIMI_SHARD_F_A="${KIMI_SHARD_F_A:-0}"
case "${KIMI_SHARD_F_A}" in
  0) KIMI_ADDITIONAL_CONFIG='{}' ;;
  1) KIMI_ADDITIONAL_CONFIG='{"kda_shard_f_a":true}' ;;
  *)
    echo "KIMI_SHARD_F_A must be 0 or 1" >&2
    exit 2
    ;;
esac

exec "${SCRIPT_DIR}/serve-kimi-k3-full-mxfp4-dcp8-1m.sh" \
  --additional-config "${KIMI_ADDITIONAL_CONFIG}" \
  "$@"
