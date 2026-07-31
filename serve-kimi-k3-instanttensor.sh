#!/usr/bin/env bash
# Fast-loading stock Kimi K3 TP16 profile. Requires the consumer-stream-safe
# InstantTensor wheel from voipmonitor/InstantTensor:dev/gg-k3-consumer-event.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# The stock checkpoint contains two 2.19-GiB BF16 tensors. The vLLM loader
# streams all smaller weights through this ring and loads only those two via
# CPU safetensors, avoiding a loader-time GPU OOM.
export INSTANTTENSOR_BUFFER_SIZE="${INSTANTTENSOR_BUFFER_SIZE:-536870912}"

# InstantTensor's distributed NCCL path retains about 154 MiB/rank. This
# measured budget serves 40,960 tokens with 1.24 GiB of KV cache on 96-GiB
# RTX PRO 6000 Blackwell ranks.
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.988}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-40960}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"

# Keep decode on the stable MLA topology validated with this checkpoint.
export VLLM_TRITON_MLA_STATIC_KV_SPLITS="${VLLM_TRITON_MLA_STATIC_KV_SPLITS:-8}"
export VLLM_DCP_INDEXER_SHARDS="${VLLM_DCP_INDEXER_SHARDS:-0}"

exec "${SCRIPT_DIR}/serve-kimi-k3.sh" --load-format instanttensor "$@"
