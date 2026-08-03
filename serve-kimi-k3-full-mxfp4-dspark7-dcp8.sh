#!/usr/bin/env bash
# DCP8 profile for the stock Kimi-K3 MXFP4 target and the BF16 Inferact
# five-layer DSpark draft with its trained seven-token speculative block.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export DCP_SIZE="${DCP_SIZE:-8}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
export KV_CACHE_MEMORY_BYTES="${KV_CACHE_MEMORY_BYTES:-650000000}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Kimi-K3-MXFP4-HH-DSpark7-DCP8}"

exec "${SCRIPT_DIR}/serve-kimi-k3-full-mxfp4-dspark7.sh" "$@"
