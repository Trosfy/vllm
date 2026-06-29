#!/usr/bin/env bash
set -euo pipefail

# Opinionated DeepSeek-V4-Flash launcher for the shared Eldritch image.
# It keeps the compose surface small and derives the graph sizing from the
# concurrency and speculative mode. Use EXTRA_VLLM_ARGS for one-off experiments.

unset NCCL_GRAPH_FILE NCCL_GRAPH_DUMP_FILE VLLM_B12X_MLA_EXTEND_MAX_CHUNKS

model_path="${MODEL_PATH:-deepseek-ai/DeepSeek-V4-Flash-DSpark}"
served_model_name="${SERVED_MODEL_NAME:-DeepSeek-V4-Flash-DSpark}"
port="${PORT:-8000}"
tp_size="${TP_SIZE:-${TP:-2}}"
dcp_size="${DCP_SIZE:-1}"
backend="${BACKEND:-b12x}"
spec_mode="${SPEC_MODE:-}"
max_num_seqs="${MAX_NUM_SEQS:-${CONCURRENCY:-128}}"
max_model_len="${MAX_MODEL_LEN:-262144}"
max_num_batched_tokens="${MAX_NUM_BATCHED_TOKENS:-8192}"
gpu_memory_utilization="${GPU_MEMORY_UTILIZATION:-0.90}"
block_size="${BLOCK_SIZE:-256}"
load_format="${LOAD_FORMAT:-auto}"
prefix_cache="${PREFIX_CACHE:-1}"
mtp_tokens="${MTP_TOKENS:-2}"
dspark_tokens="${DSPARK_TOKENS:-5}"
draft_sample_method="${DRAFT_SAMPLE_METHOD:-probabilistic}"
spec_model_path="${SPEC_MODEL_PATH:-${model_path}}"

if [[ -z "${spec_mode}" ]]; then
  if [[ "${DSPARK:-0}" != "0" ]]; then
    spec_mode="dspark"
  elif [[ "${MTP:-0}" != "0" ]]; then
    spec_mode="mtp"
  elif [[ "${MTP_TOKENS:-0}" != "0" && -n "${MTP_TOKENS:-}" ]]; then
    spec_mode="mtp"
  else
    spec_mode="off"
  fi
fi

case "${spec_mode}" in
  off|none|0)
    spec_mode="off"
    spec_tokens=0
    graph_multiplier=1
    spec_args=()
    ;;
  mtp)
    if (( mtp_tokens < 1 )); then
      echo "MTP_TOKENS must be >= 1 for SPEC_MODE=mtp" >&2
      exit 2
    fi
    spec_tokens="${mtp_tokens}"
    graph_multiplier=$((mtp_tokens + 1))
    spec_args=(
      --speculative-config
      "$(printf '{"method":"mtp","num_speculative_tokens":%s,"draft_sample_method":"%s","moe_backend":"b12x"}' \
        "${mtp_tokens}" "${draft_sample_method}")"
    )
    ;;
  dspark)
    if (( dspark_tokens < 1 )); then
      echo "DSPARK_TOKENS must be >= 1 for SPEC_MODE=dspark" >&2
      exit 2
    fi
    spec_tokens="${dspark_tokens}"
    graph_multiplier=$((dspark_tokens + 1))
    spec_args=(
      --speculative-config
      "$(printf '{"model":"%s","method":"dspark","num_speculative_tokens":%s,"draft_sample_method":"%s"}' \
        "${spec_model_path}" "${dspark_tokens}" "${draft_sample_method}")"
    )
    ;;
  *)
    echo "Unknown SPEC_MODE=${spec_mode}; expected off, mtp, or dspark" >&2
    exit 2
    ;;
esac

case "${backend}" in
  b12x)
    export VLLM_USE_B12X_WO_PROJECTION="${VLLM_USE_B12X_WO_PROJECTION:-1}"
    export VLLM_USE_B12X_MHC="${VLLM_USE_B12X_MHC:-1}"
    export VLLM_USE_B12X_FP8_GEMM="${VLLM_USE_B12X_FP8_GEMM:-1}"
    export VLLM_USE_B12X_MOE="${VLLM_USE_B12X_MOE:-1}"
    export VLLM_USE_B12X_SPARSE_INDEXER="${VLLM_USE_B12X_SPARSE_INDEXER:-1}"
    export VLLM_ENABLE_PCIE_ALLREDUCE="${VLLM_ENABLE_PCIE_ALLREDUCE:-1}"
    export VLLM_PCIE_ALLREDUCE_BACKEND="${VLLM_PCIE_ALLREDUCE_BACKEND:-b12x}"
    export B12X_MLA_SM120_UNIFIED="${B12X_MLA_SM120_UNIFIED:-1}"
    export B12X_MHC_MAX_TOKENS="${B12X_MHC_MAX_TOKENS:-16384}"
    export B12X_DENSE_SPLITK_TURBO="${B12X_DENSE_SPLITK_TURBO:-1}"
    export B12X_W4A16_TC_DECODE="${B12X_W4A16_TC_DECODE:-1}"
    export B12X_MOE_FORCE_A16="${B12X_MOE_FORCE_A16:-1}"
    backend_args=(
      --attention-backend B12X_MLA_SPARSE
      --moe-backend b12x
      --linear-backend b12x
    )
    ;;
  lucifer-cutlass)
    export VLLM_ENABLE_PCIE_ALLREDUCE="${VLLM_ENABLE_PCIE_ALLREDUCE:-0}"
    export VLLM_PCIE_ALLREDUCE_BACKEND="${VLLM_PCIE_ALLREDUCE_BACKEND:-cpp}"
    unset VLLM_USE_B12X_WO_PROJECTION VLLM_USE_B12X_MHC VLLM_USE_B12X_FP8_GEMM
    unset VLLM_USE_B12X_MOE VLLM_USE_B12X_SPARSE_INDEXER B12X_MLA_SM120_UNIFIED
    backend_args=(
      --attention-backend FLASHINFER_MLA_SPARSE_DSV4
      --kernel-config.moe_backend flashinfer_cutlass
      --disable-custom-all-reduce
    )
    ;;
  lucifer-default)
    export VLLM_ENABLE_PCIE_ALLREDUCE="${VLLM_ENABLE_PCIE_ALLREDUCE:-0}"
    export VLLM_PCIE_ALLREDUCE_BACKEND="${VLLM_PCIE_ALLREDUCE_BACKEND:-cpp}"
    unset VLLM_USE_B12X_WO_PROJECTION VLLM_USE_B12X_MHC VLLM_USE_B12X_FP8_GEMM
    unset VLLM_USE_B12X_MOE VLLM_USE_B12X_SPARSE_INDEXER B12X_MLA_SM120_UNIFIED
    backend_args=(
      --attention-backend FLASHINFER_MLA_SPARSE_DSV4
      --disable-custom-all-reduce
    )
    ;;
  *)
    echo "Unknown BACKEND=${backend}; expected b12x, lucifer-cutlass, or lucifer-default" >&2
    exit 2
    ;;
esac

export CUTE_DSL_ARCH="${CUTE_DSL_ARCH:-sm_120a}"
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_P2P_LEVEL="${NCCL_P2P_LEVEL:-SYS}"
export NCCL_PROTO="${NCCL_PROTO:-LL,LL128,Simple}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export VLLM_USE_AOT_COMPILE="${VLLM_USE_AOT_COMPILE:-1}"
export VLLM_USE_BREAKABLE_CUDAGRAPH="${VLLM_USE_BREAKABLE_CUDAGRAPH:-0}"
export VLLM_USE_MEGA_AOT_ARTIFACT="${VLLM_USE_MEGA_AOT_ARTIFACT:-1}"
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-1}"
export VLLM_USE_V2_MODEL_RUNNER="${VLLM_USE_V2_MODEL_RUNNER:-1}"
export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS="${VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS:-1}"
export SAFETENSORS_FAST_GPU="${SAFETENSORS_FAST_GPU:-1}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/cache}"
export VLLM_CACHE_DIR="${VLLM_CACHE_DIR:-/cache/vllm}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/cache/triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/cache/torchinductor}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-/cache/torch_extensions}"
export FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-/cache/flashinfer}"
export TILELANG_CACHE_DIR="${TILELANG_CACHE_DIR:-/cache/tilelang}"
export TILELANG_TMP_DIR="${TILELANG_TMP_DIR:-/cache/tilelang/tmp}"
export TVM_CACHE_DIR="${TVM_CACHE_DIR:-/cache/tvm}"

mkdir -p \
  "${VLLM_CACHE_DIR}" \
  "${TRITON_CACHE_DIR}" \
  "${TORCHINDUCTOR_CACHE_DIR}" \
  "${TORCH_EXTENSIONS_DIR}" \
  "${FLASHINFER_WORKSPACE_BASE}" \
  "${TILELANG_CACHE_DIR}" \
  "${TILELANG_TMP_DIR}" \
  "${TVM_CACHE_DIR}"

graph_cap="${MAX_CUDAGRAPH_CAPTURE_SIZE:-${GRAPH_CAP:-}}"
if [[ -z "${graph_cap}" ]]; then
  graph_cap=$((max_num_seqs * graph_multiplier))
  if (( graph_cap < 4 )); then
    graph_cap=4
  fi
fi

capture_args=()
capture_sizes="${CUDAGRAPH_CAPTURE_SIZES:-${CAPTURE_SIZES:-auto}}"
if [[ "${capture_sizes}" != "none" && "${capture_sizes}" != "0" ]]; then
  if [[ "${capture_sizes}" == "auto" ]]; then
    sizes=(1)
    n=2
    while (( n < graph_cap )); do
      sizes+=("${n}")
      n=$((n * 2))
    done
    sizes+=("${max_num_seqs}" "${graph_cap}")
    mapfile -t sizes < <(printf '%s\n' "${sizes[@]}" | awk '$1 ~ /^[0-9]+$/ && $1 > 0' | sort -n -u)
  else
    # shellcheck disable=SC2206
    sizes=( ${capture_sizes} )
  fi
  capture_args=(--cudagraph-capture-sizes "${sizes[@]}")
fi

prefix_args=(--enable-prefix-caching)
if [[ "${prefix_cache}" == "0" || "${prefix_cache}" == "false" ]]; then
  prefix_args=(--no-enable-prefix-caching)
fi

extra_args=()
if [[ -n "${EXTRA_VLLM_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  extra_args=( ${EXTRA_VLLM_ARGS} )
fi

echo "DeepSeek-V4-Flash launch:"
echo "  model=${model_path}"
echo "  backend=${backend} tp=${tp_size} dcp=${dcp_size} spec=${spec_mode}:${spec_tokens}"
echo "  max_num_seqs=${max_num_seqs} max_batched=${max_num_batched_tokens} graph_cap=${graph_cap} capture_sizes=${sizes[*]:-${capture_sizes}}"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  exit 0
fi

exec vllm serve "${model_path}" \
  --served-model-name "${served_model_name}" \
  --host 0.0.0.0 \
  --port "${port}" \
  --trust-remote-code \
  --kv-cache-dtype fp8 \
  --block-size "${block_size}" \
  --load-format "${load_format}" \
  --tensor-parallel-size "${tp_size}" \
  --decode-context-parallel-size "${dcp_size}" \
  --gpu-memory-utilization "${gpu_memory_utilization}" \
  --max-model-len "${max_model_len}" \
  --max-num-seqs "${max_num_seqs}" \
  --max-num-batched-tokens "${max_num_batched_tokens}" \
  --max-cudagraph-capture-size "${graph_cap}" \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}' \
  --async-scheduling \
  --no-scheduler-reserve-full-isl \
  --enable-chunked-prefill \
  --enable-flashinfer-autotune \
  --tokenizer-mode deepseek_v4 \
  --tool-call-parser deepseek_v4 \
  --reasoning-parser deepseek_v4 \
  --enable-auto-tool-choice \
  --default-chat-template-kwargs.thinking=true \
  --default-chat-template-kwargs.reasoning_effort=high \
  "${prefix_args[@]}" \
  "${capture_args[@]}" \
  "${spec_args[@]}" \
  "${backend_args[@]}" \
  "${extra_args[@]}" \
  "$@"
