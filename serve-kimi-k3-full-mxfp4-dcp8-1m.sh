#!/usr/bin/env bash
# Full stock Kimi K3 MXFP4 on TP16/DCP8 with a physical 1M-token FP8 cache.
#
# This deliberately uses the original compressed-tensors MXFP4 checkpoint,
# not an EXL3 or NF3 overlay.  KDA layers keep one replicated recurrent state
# per sequence; only the 24 dense MLA histories are token-sharded by DCP.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

TP_SIZE="${TP_SIZE:-16}"
DCP_SIZE="${DCP_SIZE:-8}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1048576}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-256}"

# 14 GiB produced 1,084,486 tokens at DCP1 with this exact hybrid
# MLA/KDA layout.  DCP8 needs one eighth of that allocation and the planner
# still reports 1,059,851 tokens (1.01075x at a 1,048,576-token request).
KV_CACHE_MEMORY_BYTES="${KV_CACHE_MEMORY_BYTES:-1879048192}"

DCP_COMM_BACKEND="${DCP_COMM_BACKEND:-a2a}"
DCP_A2A_MAX_TOKENS="${DCP_A2A_MAX_TOKENS:-64}"
DCP_A2A_LARGE_BACKEND="${DCP_A2A_LARGE_BACKEND:-ag_rs}"

if (( TP_SIZE != 16 )); then
  echo "This full-MXFP4 profile is validated only for TP_SIZE=16, got ${TP_SIZE}" >&2
  exit 2
fi
if (( DCP_SIZE != 8 )); then
  echo "This profile is validated only for DCP_SIZE=8, got ${DCP_SIZE}" >&2
  exit 2
fi
if (( TP_SIZE % DCP_SIZE != 0 )); then
  echo "TP_SIZE (${TP_SIZE}) must be divisible by DCP_SIZE (${DCP_SIZE})" >&2
  exit 2
fi
if [[ "${DCP_COMM_BACKEND}" != "a2a" ]]; then
  echo "This profile requires DCP_COMM_BACKEND=a2a" >&2
  exit 2
fi
case "${DCP_A2A_LARGE_BACKEND}" in
  ag_rs | a2a) ;;
  *)
    echo "DCP_A2A_LARGE_BACKEND must be ag_rs or a2a" >&2
    exit 2
    ;;
esac
if [[ ! "${DCP_A2A_MAX_TOKENS}" =~ ^[0-9]+$ ]]; then
  echo "DCP_A2A_MAX_TOKENS must be a non-negative integer" >&2
  exit 2
fi

# K3 dense MLA: use the low-latency SparkInfer PCIe A2A+LSE reduction for
# decode/small batches and the bounded NCCL AG+RS path for larger prefill.
export VLLM_USE_B12X_DCP_A2A="${VLLM_USE_B12X_DCP_A2A:-1}"
export VLLM_DCP_A2A_MAX_TOKENS="${VLLM_DCP_A2A_MAX_TOKENS:-${DCP_A2A_MAX_TOKENS}}"
export VLLM_DCP_A2A_LARGE_BACKEND="${VLLM_DCP_A2A_LARGE_BACKEND:-${DCP_A2A_LARGE_BACKEND}}"

# Keep the numerically validated K3 reduction topology across all context
# lengths.  Dynamic 8->16 split transitions caused the old token-4608 collapse.
export VLLM_TRITON_MLA_STATIC_KV_SPLITS="${VLLM_TRITON_MLA_STATIC_KV_SPLITS:-8}"

# GLM's sparse-indexer/CKV policies do not apply to K3.  Pin them off instead
# of paying setup memory or accidentally taking an unvalidated sparse path.
export VLLM_DCP_INDEXER_SHARDS="${VLLM_DCP_INDEXER_SHARDS:-0}"
export VLLM_DCP_QUERY_SPLIT="${VLLM_DCP_QUERY_SPLIT:-0}"
export VLLM_DCP_GLOBAL_TOPK="${VLLM_DCP_GLOBAL_TOPK:-0}"
export VLLM_DCP_PROJECT_BEFORE_MERGE="${VLLM_DCP_PROJECT_BEFORE_MERGE:-0}"
export VLLM_B12X_MLA_DCP_GATHER_IN_WORKSPACE="${VLLM_B12X_MLA_DCP_GATHER_IN_WORKSPACE:-0}"

# The source branch carries the TP16 SparkInfer one-shot all-reduce.  Keep the
# lossless path enabled for TP projections/MoE while DCP shards attention.
export VLLM_ENABLE_PCIE_ALLREDUCE="${VLLM_ENABLE_PCIE_ALLREDUCE:-1}"
export VLLM_PCIE_ALLREDUCE_BACKEND="${VLLM_PCIE_ALLREDUCE_BACKEND:-cpp}"

# K3 needs piecewise graphs with KDA and MLA eager breaks.  Capture only the
# decode batch used by this max_num_seqs=1 profile and fuse TP all-reduce+RMS.
if [[ -z "${COMPILATION_CONFIG:-}" ]]; then
  export COMPILATION_CONFIG='{"mode":0,"cudagraph_mode":"PIECEWISE","cudagraph_capture_sizes":[1],"pass_config":{"fuse_allreduce_rms":true}}'
fi

export MODEL="${MODEL:-/root/.cache/huggingface/hub/models--moonshotai--Kimi-K3/snapshots/2496450e92e425c886db095102a52a6682ca3970}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Kimi-K3-MXFP4-DCP8-1M}"
export TP_SIZE DCP_SIZE MAX_MODEL_LEN MAX_NUM_SEQS MAX_NUM_BATCHED_TOKENS
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.985}"

# An empty NCCL graph filename makes NCCL try to open an empty path.
unset NCCL_GRAPH_FILE NCCL_GRAPH_DUMP_FILE

exec "${SCRIPT_DIR}/serve-kimi-k3-instanttensor.sh" \
  --decode-context-parallel-size "${DCP_SIZE}" \
  --dcp-comm-backend "${DCP_COMM_BACKEND}" \
  --dcp-kv-cache-interleave-size 1 \
  --kv-cache-dtype fp8 \
  --kv-cache-memory-bytes "${KV_CACHE_MEMORY_BYTES}" \
  --no-enable-prefix-caching \
  "$@"
