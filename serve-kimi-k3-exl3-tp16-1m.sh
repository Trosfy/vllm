#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/opt/venv/bin/python}"
SPARKINFER_SRC="${SPARKINFER_SRC:-/root/sparkinfer}"
VLLM_INSTALLED_DIR="${VLLM_INSTALLED_DIR:-/opt/venv/lib/python3.12/site-packages/vllm}"
MODEL="${MODEL:-/mnt/luke/models/Kimi-K3-EXL3-3p09-MXFP4-44p0-TP16-1M-20260801-serve}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Kimi-K3-EXL3-3p09-MXFP4-44p0}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1048576}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-256}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.986}"
KV_CACHE_MEMORY_BYTES="${KV_CACHE_MEMORY_BYTES:-15032385536}"
COMPILATION_CONFIG="${COMPILATION_CONFIG:-}"
if [[ -z "${COMPILATION_CONFIG}" ]]; then
  COMPILATION_CONFIG='{"mode":0,"cudagraph_mode":"PIECEWISE","cudagraph_capture_sizes":[1],"pass_config":{"fuse_allreduce_rms":true}}'
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}"
export PYTHONPATH="${SCRIPT_DIR}:${SPARKINFER_SRC}${PYTHONPATH:+:${PYTHONPATH}}"
export INSTANTTENSOR_BUFFER_SIZE="${INSTANTTENSOR_BUFFER_SIZE:-536870912}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export CUTE_DSL_ARCH="${CUTE_DSL_ARCH:-sm_120a}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-forkserver}"
export VLLM_K3_EXL3_ONEGRID="${VLLM_K3_EXL3_ONEGRID:-1}"
export VLLM_USE_B12X_MOE="${VLLM_USE_B12X_MOE:-1}"
export VLLM_USE_BREAKABLE_CUDAGRAPH="${VLLM_USE_BREAKABLE_CUDAGRAPH:-1}"
# Do not inherit the image's generic defaults here: the GG image currently
# sets graph estimation to 1 and the PCIe backend to cpp, neither of which is
# the validated K3 TP16 profile.  K3-specific knobs still allow an explicit
# experiment without editing this launcher.
export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS="${K3_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS:-0}"
export VLLM_ENABLE_PCIE_ALLREDUCE="${K3_ENABLE_PCIE_ALLREDUCE:-1}"
export VLLM_PCIE_ALLREDUCE_BACKEND="${K3_PCIE_ALLREDUCE_BACKEND:-b12x}"
export VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE="${K3_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE:-64KB}"
export VLLM_PCIE_ONESHOT_FUSED_ADD_RMS_NORM_MAX_SIZE="${K3_PCIE_ONESHOT_FUSED_ADD_RMS_NORM_MAX_SIZE:-84KB}"
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

# A source-tree PYTHONPATH shadows the wheel's vllm package, including its
# separately installed FlashAttention ABI modules.  MLA decode uses Triton,
# but Kimi-K3 prefill still needs these modules.  Reuse the exact wheel
# binaries instead of copying another ~1.15 GB into every source overlay.
SOURCE_FA_DIR="${SCRIPT_DIR}/vllm/vllm_flash_attn"
INSTALLED_FA_DIR="${VLLM_INSTALLED_DIR}/vllm_flash_attn"
for fa_module in _vllm_fa2_C.abi3.so _vllm_fa3_C.abi3.so; do
  if [[ ! -e "${SOURCE_FA_DIR}/${fa_module}" ]]; then
    if [[ ! -r "${INSTALLED_FA_DIR}/${fa_module}" ]]; then
      echo "Missing required FlashAttention module: ${INSTALLED_FA_DIR}/${fa_module}" >&2
      exit 1
    fi
    ln -s "${INSTALLED_FA_DIR}/${fa_module}" "${SOURCE_FA_DIR}/${fa_module}"
  fi
done

# Fail before the multi-minute weight load if vLLM and SparkInfer overlays are
# from incompatible revisions.
"${PYTHON_BIN}" - <<'PY'
import inspect
from sparkinfer.moe import fused_moe

required = "tp_local_intermediate_hadamard_tail"
parameters = inspect.signature(fused_moe.plan_weights).parameters
if required not in parameters:
    raise RuntimeError(
        f"SparkInfer {fused_moe.__file__} is too old: plan_weights() lacks "
        f"{required!r}"
    )
PY

exec "${PYTHON_BIN}" -m vllm.entrypoints.cli.main serve "${MODEL}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --trust-remote-code \
  --host "${HOST}" \
  --port "${PORT}" \
  --tensor-parallel-size 16 \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-num-seqs 1 \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --kv-cache-memory-bytes "${KV_CACHE_MEMORY_BYTES}" \
  --kv-cache-dtype fp8 \
  --load-format instanttensor \
  --reasoning-parser kimi_k3 \
  --tool-call-parser kimi_k3 \
  --enable-auto-tool-choice \
  --compilation-config "${COMPILATION_CONFIG}" \
  "$@"
