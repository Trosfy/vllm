#!/usr/bin/env bash
# Run one fixed physical DSpark-depth measurement on the linked 5-layer target.
set -euo pipefail

if (( $# != 2 )); then
  echo "usage: $0 DEPTH RUN_LABEL" >&2
  exit 2
fi

DEPTH="$1"
RUN_LABEL="$2"
if [[ ! "${DEPTH}" =~ ^[1-7]$ ]]; then
  echo "DEPTH must be in [1, 7], got ${DEPTH}" >&2
  exit 2
fi
if [[ ! "${RUN_LABEL}" =~ ^[a-zA-Z0-9._-]+$ ]]; then
  echo "RUN_LABEL contains unsupported characters: ${RUN_LABEL}" >&2
  exit 2
fi

VLLM_ROOT="${VLLM_ROOT:-/mnt/luke/vllm-k3-hh-unified}"
IMAGE="${IMAGE:-voipmonitor/vllm:kimi-k3-hh-unified-v90-control-20260806}"
ENV_FILE="${ENV_FILE:-/mnt/luke/kimi-k3-runs/dspark-dcp16-unified-v90-restored-20260806-a/container-env-after-control.txt}"
TARGET_MODEL="${TARGET_MODEL:-/mnt/luke/Kimi-K3-MXFP4-5L-DSparkHarness-linked}"
DRAFT_MODEL="${DRAFT_MODEL:-/mnt/luke/Kimi-K3-DSpark-5L-Harness-linked}"
HUMMING_CACHE="${HUMMING_CACHE:-/mnt/luke/kimi-k3-cache/humming}"
TOKEN_FILE="${TOKEN_FILE:-/root/vllm/kimi/rtx6kpro-kimi-k3-kld-docs/models/kimi-k3/tools/decode-baseline-256-token-ids.json}"
RUN_ROOT="${RUN_ROOT:-/mnt/luke/kimi-k3-runs}"
PORT="${PORT:-8001}"
BATCHED_PROJECTION_TOPK="${BATCHED_PROJECTION_TOPK:-0}"
KIMI_TOPK_THREADS="${KIMI_TOPK_THREADS:-384}"
PROFILE="${PROFILE:-0}"
REJECTION_METHOD="${REJECTION_METHOD:-block}"
DRAFT_SAMPLE_METHOD="${DRAFT_SAMPLE_METHOD:-greedy}"
PRELAUNCH_SHARED_EXPERTS="${PRELAUNCH_SHARED_EXPERTS:-0}"
BENCH_MAX_TOKENS="${BENCH_MAX_TOKENS:-512}"
BENCH_RUNS="${BENCH_RUNS:-4}"
BENCH_WARMUPS="${BENCH_WARMUPS:-2}"
BENCH_TEMPERATURE="${BENCH_TEMPERATURE:-0}"
DSPARK_SPS_CURVE="${DSPARK_SPS_CURVE:-}"
DSPARK_SPS_OVERHEAD_MS="${DSPARK_SPS_OVERHEAD_MS:-0.0}"
DSPARK_CAPACITY_ACTIVATION_BATCH_SIZE="${DSPARK_CAPACITY_ACTIVATION_BATCH_SIZE:-0}"
CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-0}"
CONTAINER="kimi-k3-hh-dspark-5l-fixed-k${DEPTH}-${RUN_LABEL}"
MODEL_NAME="Kimi-K3-MXFP4-5L-DSpark-K${DEPTH}-${RUN_LABEL}"
RUN_DIR="${RUN_ROOT}/dspark-5l-fixed-k${DEPTH}-${RUN_LABEL}"

if docker inspect "${CONTAINER}" >/dev/null 2>&1; then
  echo "container already exists: ${CONTAINER}" >&2
  exit 1
fi
if [[ -e "${RUN_DIR}" ]]; then
  echo "run directory already exists: ${RUN_DIR}" >&2
  exit 1
fi
if [[ "${PROFILE}" != 0 && "${PROFILE}" != 1 ]]; then
  echo "PROFILE must be 0 or 1, got ${PROFILE}" >&2
  exit 2
fi
for value_name in BENCH_MAX_TOKENS BENCH_RUNS BENCH_WARMUPS; do
  value="${!value_name}"
  if [[ ! "${value}" =~ ^[0-9]+$ ]] || (( value < 1 )); then
    echo "${value_name} must be a positive integer, got ${value}" >&2
    exit 2
  fi
done
mkdir -p "${RUN_DIR}"

PROFILE_ARGS=""
if [[ "${PROFILE}" == 1 ]]; then
  mkdir -p "${RUN_DIR}/torch"
  PROFILE_ARGS=" \
--profiler-config.profiler=torch \
--profiler-config.torch_profiler_dir=${RUN_DIR}/torch \
--profiler-config.torch_profiler_with_stack=false \
--profiler-config.torch_profiler_record_shapes=false \
--profiler-config.torch_profiler_with_memory=false \
--profiler-config.torch_profiler_with_flops=false \
--profiler-config.torch_profiler_use_gzip=true \
--profiler-config.torch_profiler_dump_cuda_time_total=false \
--profiler-config.ignore_frontend=true \
--profiler-config.delay_iterations=0 \
--profiler-config.max_iterations=4 \
--profiler-config.warmup_iterations=0 \
--profiler-config.active_iterations=5 \
--profiler-config.wait_iterations=0"
fi

cleanup() {
  if [[ "$(docker inspect -f '{{.State.Running}}' "${CONTAINER}" 2>/dev/null || true)" == true ]]; then
    docker stop -t 45 "${CONTAINER}" >/dev/null
  fi
}
trap cleanup EXIT

docker run -d \
  --name "${CONTAINER}" \
  --hostname aiserver \
  --gpus all \
  --network host \
  --ipc host \
  --env-file "${ENV_FILE}" \
  -e HUMMING_CACHE_DIR=/cache/humming \
  -e MODEL="${TARGET_MODEL}" \
  -e DRAFT_MODEL="${DRAFT_MODEL}" \
  -e SERVED_MODEL_NAME="${MODEL_NAME}" \
  -e MAX_MODEL_LEN=8192 \
  -e MAX_NUM_SEQS=1 \
  -e MAX_NUM_BATCHED_TOKENS=512 \
  -e KV_CACHE_MEMORY_BYTES=500000000 \
  -e DSPARK_DRAFT_KV_WINDOW=4096 \
  -e DSPARK_BATCH_SIZE_SPECULATIVE_SCHEDULE="[[1,1,${DEPTH}]]" \
  -e DRAFT_SAMPLE_METHOD="${DRAFT_SAMPLE_METHOD}" \
  -e REJECTION_SAMPLE_METHOD="${REJECTION_METHOD}" \
  -e DSPARK_SPS_CURVE="${DSPARK_SPS_CURVE}" \
  -e DSPARK_SPS_OVERHEAD_MS="${DSPARK_SPS_OVERHEAD_MS}" \
  -e DSPARK_CAPACITY_ACTIVATION_BATCH_SIZE="${DSPARK_CAPACITY_ACTIVATION_BATCH_SIZE}" \
  -e CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING}" \
  -e VLLM_KIMI_PRELAUNCH_SHARED_EXPERTS="${PRELAUNCH_SHARED_EXPERTS}" \
  -e DSPARK_EXPECTED_TARGET_LAYER_IDS=0,1,2,3,4 \
  -e VLLM_DSPARK_DYNAMIC_DRAFT_DEPTH=1 \
  -e VLLM_KIMI_USE_B12X_BATCHED_PROJECTION_TOPK="${BATCHED_PROJECTION_TOPK}" \
  -e B12X_PCIE_KIMI_TOPK_THREADS="${KIMI_TOPK_THREADS}" \
  -e VLLM_MEMORY_PROFILE_INCLUDE_ATTN=0 \
  -e VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0 \
  -e KIMI_TARGET_MXFP8_PROFILE=kda_in_proj \
  -v /root/vllm/kimi/source-overlay:/source-overlay:ro \
  -v /mnt/luke:/mnt/luke \
  -v /root/.cache/huggingface:/root/.cache/huggingface \
  -v "${HUMMING_CACHE}:/cache/humming" \
  -w "${VLLM_ROOT}" \
  --entrypoint bash \
  "${IMAGE}" \
  -lc "unset NCCL_GRAPH_FILE NCCL_GRAPH_DUMP_FILE; exec ./serve-kimi-k3-full-mxfp4-dspark7-dcp16-1m-kda-mxfp8.sh --skip-tokenizer-init --port ${PORT}${PROFILE_ARGS} > ${RUN_DIR}/server.log 2>&1" \
  >/dev/null

for _ in $(seq 1 240); do
  if curl -fsS --max-time 2 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    break
  fi
  if [[ "$(docker inspect -f '{{.State.Running}}' "${CONTAINER}")" != true ]]; then
    tail -200 "${RUN_DIR}/server.log" >&2
    exit 1
  fi
  sleep 1
done
if ! curl -fsS --max-time 2 "http://127.0.0.1:${PORT}/health" >/dev/null; then
  tail -200 "${RUN_DIR}/server.log" >&2
  exit 1
fi

if [[ "${PROFILE}" == 1 ]]; then
  curl -fsS -X POST "http://127.0.0.1:${PORT}/start_profile" \
    -H 'Content-Type: application/json' \
    -d '{}' \
    > "${RUN_DIR}/start-profile.json"
fi

/root/.local/bin/uv run --no-project --python 3.12 \
  "${VLLM_ROOT}/tools/kimi_k3/benchmark_dspark_normalized.py" \
  --url "http://127.0.0.1:${PORT}" \
  --model "${MODEL_NAME}" \
  --token-file "${TOKEN_FILE}" \
  --prompt-tokens 256 \
  --max-tokens "${BENCH_MAX_TOKENS}" \
  --temperature "${BENCH_TEMPERATURE}" \
  --warmups "${BENCH_WARMUPS}" \
  --runs "${BENCH_RUNS}" \
  --output-dir "${RUN_DIR}/bench-normalized"

if [[ "${PROFILE}" == 1 ]]; then
  for _ in $(seq 1 120); do
    if find "${RUN_DIR}/torch" -type f -name '*.pt.trace.json.gz' -print -quit \
      | rg -q .; then
      break
    fi
    sleep 1
  done
  if ! find "${RUN_DIR}/torch" -type f -name '*.pt.trace.json.gz' -print -quit \
    | rg -q .; then
    echo "Torch profiler did not write a trace" >&2
    exit 1
  fi
fi

echo "summary: ${RUN_DIR}/bench-normalized/summary.json"
