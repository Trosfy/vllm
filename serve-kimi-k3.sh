#!/usr/bin/env bash
# Kimi K3 on 16x RTX PRO 6000 Blackwell (SM120), native TP16 W4A16 profile.
#
# Serves the stock moonshotai/Kimi-K3 checkpoint as-is: vLLM auto-detects the
# compressed-tensors mxfp4-pack-quantized config and routes the SiTU experts
# through the B12X W4A16 MoE; every non-expert tensor stays BF16. The fit is
# tight by design (~91.7 GiB weights per 95.6 GiB rank): utilization, capture
# sizes, and concurrency below are the measured maximums, and the memory
# profiler's cudagraph reservation is disabled because it over-reserves ~4x
# what the piecewise-breakable capture actually uses.
#
# Chat/tool/reasoning parsing is native (KimiK3Renderer via
# tokenizer_mode=kimi_k3); do not pass --chat-template.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${SCRIPT_DIR}/.venv/bin/python}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python interpreter not found or not executable: ${PYTHON_BIN}" >&2
  echo "Create the venv with: uv venv --python 3.12" >&2
  exit 1
fi

# Only shadow the installed vLLM when the source tree carries the compiled
# extensions; wheel-based deployments run site-packages with this repo's
# python files overlaid instead.
if [[ -e "${SCRIPT_DIR}/vllm/_C_stable_libtorch.abi3.so" ]]; then
  export PYTHONPATH="${SCRIPT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export CUTE_DSL_ARCH="${CUTE_DSL_ARCH:-sm_120a}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-forkserver}"
# Breakable cudagraphs auto-enable for the K3 architecture; the B12X MoE env
# keeps the capture prewarm path active for the auto-selected SiTU experts.
export VLLM_USE_B12X_MOE="${VLLM_USE_B12X_MOE:-1}"
# The measured reservation is ~4x actual capture use; K3's fit needs it back.
export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS="${VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS:-0}"
export KDA_DISABLE_AUTOTUNE="${KDA_DISABLE_AUTOTUNE:-1}"

MODEL="${MODEL:-moonshotai/Kimi-K3}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Kimi-K3}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
TP_SIZE="${TP_SIZE:-16}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.985}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-2}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-512}"
# Decode-only size-1 graphs: capture memory competes with the last GiB of
# weights; batch sizes above the capture list run eagerly.
COMPILATION_CONFIG="${COMPILATION_CONFIG:-{\"mode\": 0, \"cudagraph_mode\": \"PIECEWISE\", \"cudagraph_capture_sizes\": [1]}}"


exec "${PYTHON_BIN}" -m vllm.entrypoints.cli.main serve "${MODEL}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --trust-remote-code \
  --host "${HOST}" \
  --port "${PORT}" \
  --tensor-parallel-size "${TP_SIZE}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --compilation-config "${COMPILATION_CONFIG}" \
  --reasoning-parser kimi_k3 \
  --tool-call-parser kimi_k3 \
  --enable-auto-tool-choice \
  "$@"
