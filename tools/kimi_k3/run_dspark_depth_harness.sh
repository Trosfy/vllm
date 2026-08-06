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
mkdir -p "${RUN_DIR}"

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
  -e DSPARK_EXPECTED_TARGET_LAYER_IDS=0,1,2,3,4 \
  -e VLLM_DSPARK_DYNAMIC_DRAFT_DEPTH=1 \
  -e VLLM_KIMI_USE_B12X_BATCHED_PROJECTION_TOPK=0 \
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
  -lc "unset NCCL_GRAPH_FILE NCCL_GRAPH_DUMP_FILE; exec ./serve-kimi-k3-full-mxfp4-dspark7-dcp16-1m-kda-mxfp8.sh --skip-tokenizer-init --port ${PORT} > ${RUN_DIR}/server.log 2>&1" \
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

/root/.local/bin/uv run --no-project --python 3.12 \
  "${VLLM_ROOT}/tools/kimi_k3/benchmark_dspark_normalized.py" \
  --url "http://127.0.0.1:${PORT}" \
  --model "${MODEL_NAME}" \
  --token-file "${TOKEN_FILE}" \
  --prompt-tokens 256 \
  --max-tokens 512 \
  --warmups 2 \
  --runs 4 \
  --output-dir "${RUN_DIR}/bench-normalized"

echo "summary: ${RUN_DIR}/bench-normalized/summary.json"
