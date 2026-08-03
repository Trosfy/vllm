#!/usr/bin/env bash
# Full stock Kimi-K3 MXFP4 target with a selectively online-MXFP8 Inferact
# DSpark7 draft, TP16/DCP8, and a physical 1M-token target KV cache.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export DCP_SIZE="${DCP_SIZE:-8}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-1048576}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-2048}"

# Exact model-free lower bound for one 1M request with DCP8 target MLA,
# current+7 KDA rollback states, and a replicated 65,536-token draft tail is
# 2,342,338,560 bytes/rank. The draft tail includes two 2,048-token in-flight
# scheduler batches (async scheduling). Round up slightly while retaining
# one-request concurrency and leaving the rest for weights/workspaces.
export KV_CACHE_MEMORY_BYTES="${KV_CACHE_MEMORY_BYTES:-2343000000}"
export DSPARK_DRAFT_KV_WINDOW="${DSPARK_DRAFT_KV_WINDOW:-65536}"
export DSPARK_DRAFT_WEIGHT_FORMAT="${DSPARK_DRAFT_WEIGHT_FORMAT:-mxfp8}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Kimi-K3-MXFP4-HH-DSpark7-MXFP8-DCP8-1M}"

exec "${SCRIPT_DIR}/serve-kimi-k3-full-mxfp4-dspark7.sh" "$@"
