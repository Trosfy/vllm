#!/usr/bin/env bash
# KLD capture server for Kimi-K3 TARGET arithmetic.
#
# Wraps the stock no-DSpark launcher (the daily-validated memory profile).
# KLD captures must NOT run on a DSpark server: with aux-hidden-state taps
# active, >=1024-row prompt-logprob requests return corrupted logits while
# decode stays coherent (see the KLD tooling notes).
#
# Capture-sized overrides: max_model_len 4096, KV 500 MB, MNBT 256 — the same
# shape as the accepted 2026-08-04 capture servers. These do not change the
# model's arithmetic (their 4096-length capture matches the canonical 1M-era
# reference at KL~0.003).
#
# usage: run_kld_capture_server.sh RUN_LABEL CAPTURE_DIR
# env: KIMI_SHARD_F_A (default 0), VLLM_ROOT, B12X_ROOT
set -euo pipefail

if (( $# != 2 )); then
  echo "usage: $0 RUN_LABEL CAPTURE_DIR" >&2
  exit 2
fi
RUN_LABEL="$1"
CAPTURE_DIR="$2"

VLLM_ROOT="${VLLM_ROOT:-/mnt/luke/vllm-k3-hh-unified}"
B12X_ROOT="${B12X_ROOT:-/mnt/luke/b12x-k3-hh-unified}"
IMAGE="${IMAGE:-voipmonitor/vllm:kimi-k3-hh-runtime-pr238-pr118-r5-20260805}"
ENV_FILE="${ENV_FILE:-/mnt/luke/kimi-k3-runs/dspark-dcp16-unified-v90-restored-20260806-a/container-env-after-control.txt}"
KIMI_SHARD_F_A="${KIMI_SHARD_F_A:-0}"
PORT="${PORT:-8001}"
CONTAINER="kimi-k3-cx-kldcap-${RUN_LABEL}"
RUN_DIR="/mnt/luke/kimi-k3-runs/cx-kldcap-${RUN_LABEL}"

if [[ "${KIMI_SHARD_F_A}" != 0 && "${KIMI_SHARD_F_A}" != 1 ]]; then
  echo "KIMI_SHARD_F_A must be 0 or 1" >&2
  exit 2
fi
if docker inspect "${CONTAINER}" >/dev/null 2>&1; then
  echo "container already exists: ${CONTAINER}" >&2
  exit 1
fi
mkdir -p "${RUN_DIR}" "${CAPTURE_DIR}"

docker run -d \
  --name "${CONTAINER}" \
  --hostname aiserver --gpus all --network host --ipc host \
  --env-file "${ENV_FILE}" \
  -e PYTHONPATH="/source-overlay:${VLLM_ROOT}:${B12X_ROOT}" \
  -e VLLM_SOURCE_OVERLAY_ROOT="${VLLM_ROOT}" \
  -e VLLM_SOURCE_DIR="${VLLM_ROOT}" \
  -e VLLM_BINARY_PACKAGE_DIR=/opt/kimi-k3-hh/vllm/vllm \
  -e SPARKINFER_DIR="${B12X_ROOT}" \
  -e TORCH_EXTENSIONS_DIR=/cache/torch-ext-cx \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e NCCL_P2P_DISABLE=0 -e KIMI_NCCL_P2P_DISABLE=0 \
  -e VLLM_MEMORY_PROFILE_INCLUDE_ATTN=0 \
  -e VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0 \
  -e KIMI_SHARD_F_A="${KIMI_SHARD_F_A}" \
  -e MAX_MODEL_LEN=4096 \
  -e MAX_NUM_SEQS=1 \
  -e MAX_NUM_BATCHED_TOKENS=256 \
  -e KV_CACHE_MEMORY_BYTES=500000000 \
  -e VLLM_PROMPT_LOGPROBS_CHUNK_SIZE=128 \
  -e VLLM_KLD_CAPTURE_DIR="${CAPTURE_DIR}" \
  -e SERVED_MODEL_NAME=Kimi-K3 \
  -v /root/vllm/kimi/source-overlay:/source-overlay:ro \
  -v /mnt/luke:/mnt/luke \
  -v /root/.cache/huggingface:/root/.cache/huggingface \
  -v /mnt/luke/kimi-k3-cache/torch-ext-cx:/cache/torch-ext-cx \
  -w "${VLLM_ROOT}" \
  --entrypoint bash \
  "${IMAGE}" \
  -lc "unset NCCL_GRAPH_FILE NCCL_GRAPH_DUMP_FILE; exec ./serve-kimi-k3-full-mxfp4-dcp16-1m-no-dspark.sh --port ${PORT} > ${RUN_DIR}/server.log 2>&1" \
  >/dev/null

for _ in $(seq 1 900); do
  if curl -fsS --max-time 2 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    break
  fi
  if [[ "$(docker inspect -f '{{.State.Running}}' "${CONTAINER}")" != true ]]; then
    tail -100 "${RUN_DIR}/server.log" >&2
    exit 1
  fi
  sleep 1
done
curl -fsS --max-time 2 "http://127.0.0.1:${PORT}/health" >/dev/null
echo "capture server up: ${CONTAINER} (shard_f_a=${KIMI_SHARD_F_A})"
