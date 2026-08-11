#!/usr/bin/env bash
# Full Kimi-K3 MXFP4 target with a 1M target context and a bounded 4K
# replicated DSpark tail.  The scheduler may host up to eight short requests;
# KV admission naturally reduces concurrency as their aggregate target cache
# approaches the single-request 1M capacity.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export MAX_MODEL_LEN="${MAX_MODEL_LEN:-1048576}"
export DSPARK_DRAFT_KV_WINDOW="${DSPARK_DRAFT_KV_WINDOW:-4096}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Kimi-K3-MXFP4-HH-DSpark7-AdaptiveK-BF16-DCP16-1M-Batch8-W4K}"

exec "${SCRIPT_DIR}/serve-kimi-k3-full-mxfp4-dspark7-dcp16-batch8-hierarchical-bf16x2.sh" "$@"
