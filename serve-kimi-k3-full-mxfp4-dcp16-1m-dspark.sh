#!/usr/bin/env bash
# Serve the stock Kimi-K3 MXFP4 target with the Inferact five-layer DSpark
# draft on TP16/DCP16. Target KDA input projections and draft linear weights
# use weight-only MXFP8/Marlin; target verification preserves model semantics.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/opt/venv/bin/python}"
VLLM_SOURCE_DIR="${VLLM_SOURCE_DIR:-${SCRIPT_DIR}}"
B12X_SOURCE_DIR="${B12X_SOURCE_DIR:-/opt/kimi-k3/b12x}"
MODEL="${MODEL:-/root/.cache/huggingface/hub/models--moonshotai--Kimi-K3/snapshots/2496450e92e425c886db095102a52a6682ca3970}"
DRAFT_MODEL="${DRAFT_MODEL:-/root/.cache/huggingface/hub/models--Inferact--Kimi-K3-DSpark/snapshots/cf6b8244620e7ea4b0651d214f28e89eac75bed6}"

for required in \
  "${PYTHON_BIN}" \
  "${VLLM_SOURCE_DIR}/vllm/__init__.py" \
  "${B12X_SOURCE_DIR}/b12x/attention/dense_mla/__init__.py" \
  "${MODEL}/model.safetensors.index.json" \
  "${DRAFT_MODEL}/model.safetensors"; do
  if [[ ! -e "${required}" ]]; then
    echo "Required Kimi-K3 serving artifact is missing: ${required}" >&2
    exit 1
  fi
done

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
export B12X_MOE_WORKSPACE_TOKEN_LIMIT="${B12X_MOE_WORKSPACE_TOKEN_LIMIT:-1024}"
export B12X_W4A16_SMALL_M_DIRECT="${B12X_W4A16_SMALL_M_DIRECT:-1}"
export B12X_W4A16_SMALL_M_HOST_BARRIER_RESET="${B12X_W4A16_SMALL_M_HOST_BARRIER_RESET:-0}"
export VLLM_DISABLE_SHARED_EXPERTS_STREAM="${VLLM_DISABLE_SHARED_EXPERTS_STREAM:-1}"

export VLLM_KIMI_SHARD_QKV_A="${VLLM_KIMI_SHARD_QKV_A:-1}"
export VLLM_KIMI_USE_B12X_PROJECTION_GATHER="${VLLM_KIMI_USE_B12X_PROJECTION_GATHER:-1}"
export VLLM_KIMI_USE_B12X_PAIRED_PROJECTION_GATHER="${VLLM_KIMI_USE_B12X_PAIRED_PROJECTION_GATHER:-1}"
export VLLM_KIMI_USE_B12X_PAIRED_PROJECTION_TOPK="${VLLM_KIMI_USE_B12X_PAIRED_PROJECTION_TOPK:-1}"
export VLLM_KIMI_USE_B12X_BATCHED_PROJECTION_TOPK="${VLLM_KIMI_USE_B12X_BATCHED_PROJECTION_TOPK:-0}"

export VLLM_USE_B12X_DCP_A2A="${VLLM_USE_B12X_DCP_A2A:-1}"
export VLLM_DCP_A2A_MAX_TOKENS="${VLLM_DCP_A2A_MAX_TOKENS:-8}"
export VLLM_DCP_A2A_LARGE_BACKEND="${VLLM_DCP_A2A_LARGE_BACKEND:-ag_rs}"
export VLLM_DCP_SHARD_DRAFT="${VLLM_DCP_SHARD_DRAFT:-0}"
export VLLM_ENABLE_PCIE_ALLREDUCE="${VLLM_ENABLE_PCIE_ALLREDUCE:-1}"
export VLLM_PCIE_ALLREDUCE_BACKEND="${VLLM_PCIE_ALLREDUCE_BACKEND:-b12x}"
export VLLM_PCIE_ONESHOT_SINGLE_CHANNEL="${VLLM_PCIE_ONESHOT_SINGLE_CHANNEL:-1}"
export B12X_PCIE_HIERARCHICAL_DEFERRED_CONSUMPTION="${B12X_PCIE_HIERARCHICAL_DEFERRED_CONSUMPTION:-1}"
export B12X_PCIE_HIERARCHICAL_DOUBLE_BUFFER="${B12X_PCIE_HIERARCHICAL_DOUBLE_BUFFER:-0}"
export B12X_PCIE_HIERARCHICAL_THREADS="${B12X_PCIE_HIERARCHICAL_THREADS:-256}"
export B12X_PCIE_HIERARCHICAL_NANOSLEEP_CYCLES="${B12X_PCIE_HIERARCHICAL_NANOSLEEP_CYCLES:-24}"
export B12X_PCIE_HIERARCHICAL_BF16X2="${B12X_PCIE_HIERARCHICAL_BF16X2:-1}"
export B12X_PCIE_HIERARCHICAL_BF16X2_MAX_ELEMENTS="${B12X_PCIE_HIERARCHICAL_BF16X2_MAX_ELEMENTS:-7168}"
export B12X_PCIE_DCP_THREADS="${B12X_PCIE_DCP_THREADS:-512}"
export B12X_PCIE_DCP_BLOCK_LIMIT="${B12X_PCIE_DCP_BLOCK_LIMIT:-8}"
export B12X_PCIE_KIMI_TOPK_THREADS="${B12X_PCIE_KIMI_TOPK_THREADS:-384}"

export VLLM_DSPARK_DRAFT_KV_WINDOW="${VLLM_DSPARK_DRAFT_KV_WINDOW:-32768}"
export VLLM_DSPARK_COMPACT_ROPE="${VLLM_DSPARK_COMPACT_ROPE:-1}"
export VLLM_DSPARK_SHARD_MARKOV_HEAD="${VLLM_DSPARK_SHARD_MARKOV_HEAD:-1}"
export VLLM_DSPARK_CAPTURE_SHARDED_MARKOV="${VLLM_DSPARK_CAPTURE_SHARDED_MARKOV:-1}"
# Replicating the 80 MiB W1 table avoids a graph-captured TP all-reduce while
# preserving TP sharding for the vocabulary-sized W2 projection.
export VLLM_DSPARK_REPLICATE_MARKOV_W1="${VLLM_DSPARK_REPLICATE_MARKOV_W1:-1}"
export VLLM_KIMI_FUSED_TOPK16="${VLLM_KIMI_FUSED_TOPK16:-1}"
export VLLM_KIMI_K3_B12X_DSPARK_ARGMAX="${VLLM_KIMI_K3_B12X_DSPARK_ARGMAX:-1}"
export VLLM_K3_KV_GROUP_SIZE="${VLLM_K3_KV_GROUP_SIZE:-6}"

export VLLM_MEMORY_PROFILE_INCLUDE_ATTN="${VLLM_MEMORY_PROFILE_INCLUDE_ATTN:-0}"
export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS="${VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS:-0}"
export VLLM_MLA_CHUNKED_PREFILL_WORKSPACE_SIZE="${VLLM_MLA_CHUNKED_PREFILL_WORKSPACE_SIZE:-4096}"
export INSTANTTENSOR_COPY="${INSTANTTENSOR_COPY:-0}"
export INSTANTTENSOR_BUFFER_SIZE="${INSTANTTENSOR_BUFFER_SIZE:-536870912}"
export INSTANTTENSOR_BACKEND="${INSTANTTENSOR_BACKEND:-AIO}"
export INSTANTTENSOR_MAX_FREE_MEM_USAGE="${INSTANTTENSOR_MAX_FREE_MEM_USAGE:-0.6}"
export SAFETENSORS_FAST_GPU="${SAFETENSORS_FAST_GPU:-1}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
unset NCCL_GRAPH_FILE

for kernel in \
  B12xMxfp8LinearKernel \
  FlashInferCutedslMxfp8LinearKernel \
  FlashInferCutlassMxfp8LinearKernel; do
  case ",${VLLM_DISABLED_KERNELS:-}," in
    *,"${kernel}",*) ;;
    *) VLLM_DISABLED_KERNELS="${VLLM_DISABLED_KERNELS:+${VLLM_DISABLED_KERNELS},}${kernel}" ;;
  esac
done
export VLLM_DISABLED_KERNELS

printf -v SPECULATIVE_CONFIG \
  '{"method":"dspark","model":"%s","num_speculative_tokens":7,"attention_backend":"B12X_MLA","kv_cache_dtype":"fp8","draft_sample_method":"greedy","rejection_sample_method":"block","quantization":"mxfp8","quantization_config":{"linear":"mxfp8","ignore":["re:.*fused_qkv_a_proj$"]}}' \
  "${DRAFT_MODEL}"
TARGET_QUANT_CONFIG='{"linear":"mxfp8","ignore":["re:^(?!.*self_attn\\.(?:q_proj|k_proj|v_proj|b_proj|f_a_proj)$).*$"]}'
if [[ -z "${COMPILATION_CONFIG:-}" ]]; then
  COMPILATION_CONFIG='{"mode":0,"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[8],"pass_config":{"fuse_allreduce_rms":true}}'
fi

cd "${VLLM_SOURCE_DIR}"
exec "${PYTHON_BIN}" -m vllm.entrypoints.cli.main serve "${MODEL}" \
  --served-model-name "${SERVED_MODEL_NAME:-Kimi-K3-MXFP4-DSpark7-DCP16-1M}" \
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
  --linear-backend auto \
  --attention-backend B12X_MLA \
  --kda-prefill-backend triton \
  --additional-config '{"kda_shard_f_a":false}' \
  --kv-cache-dtype fp8 \
  --kv-cache-memory-bytes "${KV_CACHE_MEMORY_BYTES:-1325000000}" \
  --max-model-len "${MAX_MODEL_LEN:-1048576}" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-4096}" \
  --max-num-seqs "${MAX_NUM_SEQS:-1}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.985}" \
  --enable-chunked-prefill \
  --no-enable-prefix-caching \
  --quantization-config "${TARGET_QUANT_CONFIG}" \
  --speculative-config "${SPECULATIVE_CONFIG}" \
  --compilation-config "${COMPILATION_CONFIG}" \
  --reasoning-parser kimi_k3 \
  --tool-call-parser kimi_k3 \
  --enable-auto-tool-choice \
  "$@"
