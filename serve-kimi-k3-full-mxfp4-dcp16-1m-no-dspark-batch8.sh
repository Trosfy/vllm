#!/usr/bin/env bash
# Batch-8 version of the validated 47 tok/s no-DSpark DCP16 profile.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export VLLM_SOURCE_DIR="${VLLM_SOURCE_DIR:-${SCRIPT_DIR}}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-1048576}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-2048}"
export KV_CACHE_MEMORY_BYTES="${KV_CACHE_MEMORY_BYTES:-1325000000}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Kimi-K3-MXFP4-HH-DCP16-1M-NoDSpark-Batch8}"

# Preserve the size-1 winning no-DSpark path.  Only extend graph and DCP
# communication geometry to cover one token from each of eight requests.
export KIMI_SHARD_F_A=0
export B12X_PCIE_HIERARCHICAL_DEFERRED_CONSUMPTION=1
export B12X_PCIE_HIERARCHICAL_DOUBLE_BUFFER=0
export B12X_PCIE_HIERARCHICAL_THREADS=224
export B12X_PCIE_HIERARCHICAL_NANOSLEEP_CYCLES=24
export B12X_PCIE_HIERARCHICAL_BF16X2=1
export VLLM_DCP_A2A_MAX_TOKENS=8
export B12X_PCIE_DCP_THREADS=512
export B12X_PCIE_DCP_BLOCK_LIMIT=8
export VLLM_MEMORY_PROFILE_INCLUDE_ATTN=0
export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0
export COMPILATION_CONFIG='{"mode":0,"cudagraph_mode":"PIECEWISE","cudagraph_capture_sizes":[1,8],"pass_config":{"fuse_allreduce_rms":true}}'

exec "${SCRIPT_DIR}/serve-kimi-k3-full-mxfp4-dcp16-1m-no-dspark.sh" "$@"
