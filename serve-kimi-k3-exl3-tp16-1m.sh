#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/opt/venv/bin/python}"
MODEL="${MODEL:-/mnt/luke/models/Kimi-K3-EXL3-3p09-MXFP4-44p0-TP16-1M-20260801-serve}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Kimi-K3-EXL3-3p09-MXFP4-44p0}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1048576}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-256}"
# FP8 Kimi K3 cache. 14 GiB/rank covers at least 1,048,576 tokens while
# retaining headroom for the 44% MXFP4 / 56% EXL3 expert allocation.
KV_CACHE_MEMORY_BYTES="${KV_CACHE_MEMORY_BYTES:-15032385536}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}"
export PYTHONPATH="/root/sparkinfer${PYTHONPATH:+:${PYTHONPATH}}"
export INSTANTTENSOR_BUFFER_SIZE="${INSTANTTENSOR_BUFFER_SIZE:-536870912}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export CUTE_DSL_ARCH="${CUTE_DSL_ARCH:-sm_120a}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-forkserver}"
export VLLM_USE_B12X_MOE="${VLLM_USE_B12X_MOE:-1}"
export B12X_MOE_FORCE_A16="${B12X_MOE_FORCE_A16:-1}"
export SPARKINFER_MOE_FORCE_A16="${SPARKINFER_MOE_FORCE_A16:-1}"
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS="${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS:-1800}"
export VLLM_TRITON_MLA_STATIC_KV_SPLITS="${VLLM_TRITON_MLA_STATIC_KV_SPLITS:-8}"
export VLLM_DCP_INDEXER_SHARDS="${VLLM_DCP_INDEXER_SHARDS:-0}"
export KDA_DISABLE_AUTOTUNE="${KDA_DISABLE_AUTOTUNE:-1}"

if [[ -n "${KLD_CAPTURE_DIR:-}" ]]; then
  mkdir -p "${KLD_CAPTURE_DIR}"
  export VLLM_KLD_CAPTURE_DIR="${KLD_CAPTURE_DIR}"
fi

exec "${PYTHON_BIN}" -m vllm.entrypoints.cli.main serve "${MODEL}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --trust-remote-code \
  --host "${HOST}" \
  --port "${PORT}" \
  --tensor-parallel-size 16 \
  --enable-expert-parallel \
  --disable-custom-all-reduce \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-num-seqs 1 \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
  --kv-cache-memory-bytes "${KV_CACHE_MEMORY_BYTES}" \
  --kv-cache-dtype fp8 \
  --load-format instanttensor \
  --reasoning-parser kimi_k3 \
  --tool-call-parser kimi_k3 \
  --enable-auto-tool-choice \
  --enforce-eager \
  "$@"
