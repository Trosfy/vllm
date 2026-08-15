#!/usr/bin/env bash
# Serve the stock Kimi-K3 MXFP4 checkpoint on 16 SM120 GPUs. Routed experts
# retain their checkpoint MXFP4 weights; dense tensors remain BF16. TP16 and
# DCP16 provide a physical one-million-token FP8 KV cache.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/opt/venv/bin/python}"
VLLM_SOURCE_DIR="${VLLM_SOURCE_DIR:-${SCRIPT_DIR}}"
B12X_SOURCE_DIR="${B12X_SOURCE_DIR:-/opt/kimi-k3/b12x}"
MODEL="${MODEL:-/root/.cache/huggingface/hub/models--moonshotai--Kimi-K3/snapshots/2496450e92e425c886db095102a52a6682ca3970}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python interpreter is missing: ${PYTHON_BIN}" >&2
  exit 1
fi
if [[ ! -f "${VLLM_SOURCE_DIR}/vllm/__init__.py" ]]; then
  echo "vLLM source tree is missing: ${VLLM_SOURCE_DIR}" >&2
  exit 1
fi
if [[ ! -f "${B12X_SOURCE_DIR}/b12x/attention/dense_mla/__init__.py" ]]; then
  echo "B12X dense MLA source is missing: ${B12X_SOURCE_DIR}" >&2
  exit 1
fi
if [[ ! -f "${MODEL}/model.safetensors.index.json" ]]; then
  echo "Kimi-K3 checkpoint is incomplete: ${MODEL}" >&2
  exit 1
fi

TP_SIZE="${TP_SIZE:-16}"
DCP_SIZE="${DCP_SIZE:-16}"
if (( TP_SIZE != 16 || DCP_SIZE != 16 )); then
  echo "This serving profile requires TP_SIZE=16 and DCP_SIZE=16" >&2
  exit 2
fi

export PYTHONPATH="${VLLM_SOURCE_DIR}:${B12X_SOURCE_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export CUTE_DSL_ARCH="${CUTE_DSL_ARCH:-sm_120a}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_MODULE_LOADING="${CUDA_MODULE_LOADING:-LAZY}"
export CUDA_MODULE_DATA_LOADING="${CUDA_MODULE_DATA_LOADING:-LAZY}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export VLLM_USE_V2_MODEL_RUNNER="${VLLM_USE_V2_MODEL_RUNNER:-1}"
export VLLM_USE_BREAKABLE_CUDAGRAPH="${VLLM_USE_BREAKABLE_CUDAGRAPH:-1}"
export VLLM_USE_B12X_MOE="${VLLM_USE_B12X_MOE:-1}"
export B12X_MOE_FORCE_A16="${B12X_MOE_FORCE_A16:-1}"
export VLLM_KIMI_SHARD_QKV_A="${VLLM_KIMI_SHARD_QKV_A:-1}"
export VLLM_KIMI_FUSED_TOPK16="${VLLM_KIMI_FUSED_TOPK16:-1}"
export VLLM_KIMI_USE_B12X_PROJECTION_GATHER="${VLLM_KIMI_USE_B12X_PROJECTION_GATHER:-1}"
export VLLM_KIMI_USE_B12X_PAIRED_PROJECTION_GATHER="${VLLM_KIMI_USE_B12X_PAIRED_PROJECTION_GATHER:-1}"
export VLLM_KIMI_USE_B12X_PAIRED_PROJECTION_TOPK="${VLLM_KIMI_USE_B12X_PAIRED_PROJECTION_TOPK:-1}"
export VLLM_KIMI_USE_B12X_BATCHED_PROJECTION_TOPK="${VLLM_KIMI_USE_B12X_BATCHED_PROJECTION_TOPK:-0}"
export VLLM_USE_B12X_DCP_A2A="${VLLM_USE_B12X_DCP_A2A:-1}"
export VLLM_DCP_A2A_MAX_TOKENS="${VLLM_DCP_A2A_MAX_TOKENS:-1}"
export VLLM_DCP_A2A_LARGE_BACKEND="${VLLM_DCP_A2A_LARGE_BACKEND:-ag_rs}"
export VLLM_ENABLE_PCIE_ALLREDUCE="${VLLM_ENABLE_PCIE_ALLREDUCE:-1}"
export VLLM_PCIE_ALLREDUCE_BACKEND="${VLLM_PCIE_ALLREDUCE_BACKEND:-b12x}"
export VLLM_PCIE_ONESHOT_SINGLE_CHANNEL="${VLLM_PCIE_ONESHOT_SINGLE_CHANNEL:-1}"
export B12X_PCIE_HIERARCHICAL_DEFERRED_CONSUMPTION="${B12X_PCIE_HIERARCHICAL_DEFERRED_CONSUMPTION:-0}"
export B12X_PCIE_HIERARCHICAL_DOUBLE_BUFFER="${B12X_PCIE_HIERARCHICAL_DOUBLE_BUFFER:-1}"
export B12X_PCIE_HIERARCHICAL_THREADS="${B12X_PCIE_HIERARCHICAL_THREADS:-224}"
export B12X_PCIE_HIERARCHICAL_NANOSLEEP_CYCLES="${B12X_PCIE_HIERARCHICAL_NANOSLEEP_CYCLES:-24}"
export B12X_PCIE_HIERARCHICAL_BF16X2="${B12X_PCIE_HIERARCHICAL_BF16X2:-1}"
export B12X_PCIE_DCP_THREADS="${B12X_PCIE_DCP_THREADS:-512}"
export B12X_PCIE_DCP_BLOCK_LIMIT="${B12X_PCIE_DCP_BLOCK_LIMIT:-4}"
export B12X_PCIE_KIMI_TOPK_THREADS="${B12X_PCIE_KIMI_TOPK_THREADS:-384}"
export B12X_MOE_WORKSPACE_TOKEN_LIMIT="${B12X_MOE_WORKSPACE_TOKEN_LIMIT:-2048}"
export VLLM_DISABLE_SHARED_EXPERTS_STREAM="${VLLM_DISABLE_SHARED_EXPERTS_STREAM:-0}"
export VLLM_K3_KV_GROUP_SIZE="${VLLM_K3_KV_GROUP_SIZE:-6}"
export VLLM_MLA_CHUNKED_PREFILL_WORKSPACE_SIZE="${VLLM_MLA_CHUNKED_PREFILL_WORKSPACE_SIZE:-4096}"
export VLLM_MEMORY_PROFILE_INCLUDE_ATTN="${VLLM_MEMORY_PROFILE_INCLUDE_ATTN:-0}"
export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS="${VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS:-0}"
export INSTANTTENSOR_COPY="${INSTANTTENSOR_COPY:-0}"
export INSTANTTENSOR_BUFFER_SIZE="${INSTANTTENSOR_BUFFER_SIZE:-536870912}"
export INSTANTTENSOR_BACKEND="${INSTANTTENSOR_BACKEND:-AIO}"
export INSTANTTENSOR_MAX_FREE_MEM_USAGE="${INSTANTTENSOR_MAX_FREE_MEM_USAGE:-0.6}"
export SAFETENSORS_FAST_GPU="${SAFETENSORS_FAST_GPU:-1}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
unset NCCL_GRAPH_FILE

if [[ -z "${COMPILATION_CONFIG:-}" ]]; then
  COMPILATION_CONFIG='{"mode":0,"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[1],"pass_config":{"fuse_allreduce_rms":true}}'
fi

cd "${VLLM_SOURCE_DIR}"
exec "${PYTHON_BIN}" -m vllm.entrypoints.cli.main serve "${MODEL}" \
  --served-model-name "${SERVED_MODEL_NAME:-Kimi-K3-MXFP4-DCP16-1M}" \
  --trust-remote-code \
  --language-model-only \
  --host "${HOST:-0.0.0.0}" \
  --port "${PORT:-8000}" \
  --tensor-parallel-size "${TP_SIZE}" \
  --decode-context-parallel-size "${DCP_SIZE}" \
  --dcp-comm-backend a2a \
  --dcp-kv-cache-interleave-size 1 \
  --load-format instanttensor \
  --moe-backend b12x \
  --linear-backend b12x \
  --attention-backend B12X_MLA \
  --kda-prefill-backend triton \
  --kv-cache-dtype fp8 \
  --kv-cache-memory-bytes "${KV_CACHE_MEMORY_BYTES:-1325000000}" \
  --max-model-len "${MAX_MODEL_LEN:-1048576}" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-2048}" \
  --max-num-seqs "${MAX_NUM_SEQS:-1}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.985}" \
  --enable-chunked-prefill \
  --no-enable-prefix-caching \
  --compilation-config "${COMPILATION_CONFIG}" \
  --reasoning-parser kimi_k3 \
  --tool-call-parser kimi_k3 \
  --enable-auto-tool-choice \
  "$@"
