#!/usr/bin/env bash
# Correctness-validated Kimi-K3 EXL3-3p09 launcher.
#
# Defaults reproduce the full 93-layer TP12 run that generated the same next
# token as the streamed PyTorch reference. The checkpoint contains serialized
# MXFP8 non-expert weights, so InstantTensor only copies prepared tensors; no
# online weight conversion is involved.
set -euo pipefail

K3_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
K3_PYTHON_BIN="${K3_PYTHON_BIN:-${K3_SCRIPT_DIR}/.venv/bin/python}"
K3_DEFAULT_GPU_UUIDS="GPU-ac6fcbb2-ae5f-231d-cc3e-e843c305baff,GPU-d3f30e71-0df9-2fcc-add2-977c3893288f,GPU-673fde42-acea-c0a9-efb4-04ddc5a5952a,GPU-901b8a05-1c0c-61f6-260b-6a949135ae8f,GPU-a0816187-68b2-b679-587f-0e56bac804f5,GPU-9c204557-77b4-7ffb-c9f2-effcb51d054a,GPU-48d28d14-08f3-f3d1-cb99-20c3fa5eca41,GPU-cfe1f792-1907-1f21-64b7-fdeeb9056425,GPU-f0121aa7-a898-82be-f537-a099d50ef7d8,GPU-afd4b1ad-8a64-7057-4bde-241822724c7f,GPU-4e6952e1-d0fc-03ec-320c-5f76db1275ce,GPU-c7dc46e0-30bb-08e8-2ebb-f164ec57ce31"

json_bool() {
  local name="$1"
  local value="$2"

  case "${value,,}" in
    1|true|yes|on)
      echo true
      ;;
    0|false|no|off)
      echo false
      ;;
    *)
      echo "ERROR: ${name} must be one of 1/0, true/false, yes/no, on/off; got '${value}'" >&2
      exit 1
      ;;
  esac
}

# Overlay this checkout only when its compiled extension is present. Otherwise,
# leave an installed wheel intact instead of importing a sourceless worktree.
if [[ -e "${K3_SCRIPT_DIR}/vllm/_C_stable_libtorch.abi3.so" ]]; then
  export PYTHONPATH="${K3_SCRIPT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${K3_DEFAULT_GPU_UUIDS}}"
export CUTE_DSL_ARCH="${CUTE_DSL_ARCH:-sm_120a}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-32}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_P2P_LEVEL="${NCCL_P2P_LEVEL:-SYS}"
export NCCL_PROTO="${NCCL_PROTO:-LL,LL128,Simple}"
export NCCL_BUFFSIZE="${NCCL_BUFFSIZE:-2097152}"
export NCCL_MAX_NCHANNELS="${NCCL_MAX_NCHANNELS:-8}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export VLLM_USE_V2_MODEL_RUNNER="${VLLM_USE_V2_MODEL_RUNNER:-1}"
export VLLM_ENABLE_PCIE_ALLREDUCE="${VLLM_ENABLE_PCIE_ALLREDUCE:-1}"
export VLLM_PCIE_ALLREDUCE_BACKEND="${VLLM_PCIE_ALLREDUCE_BACKEND:-b12x}"
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS="${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS:-1800}"
export VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE="${VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE:-134217728}"

# These select the production B12X dense MXFP8 and normal W4A16 MoE kernels.
export VLLM_USE_B12X_FP8_GEMM="${VLLM_USE_B12X_FP8_GEMM:-1}"
export VLLM_USE_B12X_MOE="${VLLM_USE_B12X_MOE:-1}"
export B12X_MOE_FORCE_A16="${B12X_MOE_FORCE_A16:-1}"
export KDA_DISABLE_AUTOTUNE="${KDA_DISABLE_AUTOTUNE:-1}"

export INSTANTTENSOR_BACKEND="${INSTANTTENSOR_BACKEND:-AIO}"
export INSTANTTENSOR_MAX_FREE_MEM_USAGE="${INSTANTTENSOR_MAX_FREE_MEM_USAGE:-0.6}"

K3_PROFILE="${K3_PROFILE:-0}"
K3_PROFILER_ARGS=()
case "${K3_PROFILE,,}" in
  0|false|no|off|"")
    ;;
  1|true|yes|on|torch)
    K3_PROFILE_DIR="${K3_PROFILE_DIR:-/tmp/vllm-profile/kimi-k3-$(date +%Y%m%d-%H%M%S)}"
    K3_TORCH_PROFILER_WITH_STACK_JSON="$(
      json_bool K3_TORCH_PROFILER_WITH_STACK \
        "${K3_TORCH_PROFILER_WITH_STACK:-1}"
    )"
    K3_TORCH_PROFILER_RECORD_SHAPES_JSON="$(
      json_bool K3_TORCH_PROFILER_RECORD_SHAPES \
        "${K3_TORCH_PROFILER_RECORD_SHAPES:-0}"
    )"
    K3_TORCH_PROFILER_WITH_MEMORY_JSON="$(
      json_bool K3_TORCH_PROFILER_WITH_MEMORY \
        "${K3_TORCH_PROFILER_WITH_MEMORY:-0}"
    )"
    K3_TORCH_PROFILER_WITH_FLOPS_JSON="$(
      json_bool K3_TORCH_PROFILER_WITH_FLOPS \
        "${K3_TORCH_PROFILER_WITH_FLOPS:-0}"
    )"
    K3_TORCH_PROFILER_USE_GZIP_JSON="$(
      json_bool K3_TORCH_PROFILER_USE_GZIP \
        "${K3_TORCH_PROFILER_USE_GZIP:-1}"
    )"
    K3_TORCH_PROFILER_DUMP_CUDA_TIME_TOTAL_JSON="$(
      json_bool K3_TORCH_PROFILER_DUMP_CUDA_TIME_TOTAL \
        "${K3_TORCH_PROFILER_DUMP_CUDA_TIME_TOTAL:-0}"
    )"
    K3_PROFILE_IGNORE_FRONTEND_JSON="$(
      json_bool K3_PROFILE_IGNORE_FRONTEND \
        "${K3_PROFILE_IGNORE_FRONTEND:-1}"
    )"

    if [[ "${K3_PROFILE_DIR}" != *"://"* ]]; then
      mkdir -p "${K3_PROFILE_DIR}"
    fi
    export VLLM_RPC_TIMEOUT="${VLLM_RPC_TIMEOUT:-1800000}"
    K3_PROFILER_ARGS+=(
      --profiler-config.profiler=torch
      --profiler-config.torch_profiler_dir="${K3_PROFILE_DIR}"
      --profiler-config.torch_profiler_with_stack="${K3_TORCH_PROFILER_WITH_STACK_JSON}"
      --profiler-config.torch_profiler_record_shapes="${K3_TORCH_PROFILER_RECORD_SHAPES_JSON}"
      --profiler-config.torch_profiler_with_memory="${K3_TORCH_PROFILER_WITH_MEMORY_JSON}"
      --profiler-config.torch_profiler_with_flops="${K3_TORCH_PROFILER_WITH_FLOPS_JSON}"
      --profiler-config.torch_profiler_use_gzip="${K3_TORCH_PROFILER_USE_GZIP_JSON}"
      --profiler-config.torch_profiler_dump_cuda_time_total="${K3_TORCH_PROFILER_DUMP_CUDA_TIME_TOTAL_JSON}"
      --profiler-config.ignore_frontend="${K3_PROFILE_IGNORE_FRONTEND_JSON}"
      --profiler-config.delay_iterations="${K3_PROFILE_DELAY_ITERATIONS:-0}"
      --profiler-config.max_iterations="${K3_PROFILE_MAX_ITERATIONS:-4}"
      --profiler-config.warmup_iterations="${K3_PROFILE_WARMUP_ITERATIONS:-0}"
      --profiler-config.active_iterations="${K3_PROFILE_ACTIVE_ITERATIONS:-5}"
      --profiler-config.wait_iterations="${K3_PROFILE_WAIT_ITERATIONS:-0}"
    )
    echo "Torch profiling enabled. Traces will be written under: ${K3_PROFILE_DIR}" >&2
    ;;
  cuda|nsys|nsight)
    export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
    K3_PROFILER_ARGS+=(--profiler-config.profiler=cuda)
    echo "CUDA profiler enabled. Use nsys with --capture-range=cudaProfilerApi and drive /start_profile + /stop_profile." >&2
    ;;
  *)
    echo "ERROR: K3_PROFILE must be one of 1/0, true/false, torch, cuda, nsys, or nsight; got '${K3_PROFILE}'" >&2
    exit 1
    ;;
esac

# K3's GDN layers support full CUDA graphs for decode; prefill stays eager.
# FP8 MLA plus the KDA state currently resolves to a 944-token hybrid cache
# block. Keep the scheduler budget at the next power of two so one entire cache
# block always fits in a step. CUDA-graph profiling reserves about 0.11% of an
# RTX PRO 6000; 0.9711 preserves the effective KV budget of the old 0.9700
# setting while still accounting for captured graphs.
exec "${K3_PYTHON_BIN}" -m vllm.entrypoints.cli.main serve \
  "${K3_MODEL_DIR:-/models/Kimi-K3-EXL3-3p09-serve}" \
  --served-model-name "${K3_SERVED_MODEL_NAME:-kimi-k3-exl3}" \
  --trust-remote-code \
  --host "${K3_HOST:-0.0.0.0}" \
  --port "${K3_PORT:-8000}" \
  --tensor-parallel-size 12 \
  --load-format instanttensor \
  --linear-backend b12x \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --enable-prefix-caching \
  --enable-auto-tool-choice \
  --tool-call-parser kimi_k3 \
  --reasoning-parser kimi_k3 \
  --max-model-len auto \
  --kv-cache-dtype fp8 \
  --gpu-memory-utilization "${K3_GPU_MEMORY_UTILIZATION:-0.9711}" \
  --max-num-batched-tokens "${K3_MAX_NUM_BATCHED_TOKENS:-1024}" \
  --max-num-seqs "${K3_MAX_NUM_SEQS:-1}" \
  "${K3_PROFILER_ARGS[@]}" \
  "$@"
