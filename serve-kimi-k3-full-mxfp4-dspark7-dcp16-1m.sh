#!/usr/bin/env bash
# Full stock Kimi-K3 MXFP4 target with an Inferact DSpark7 draft, TP16/DCP16,
# and a physical 1M-token target KV cache.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export DCP_SIZE="${DCP_SIZE:-16}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-1048576}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-2048}"

# Exact model-free lower bound for one 1M request with DCP16 target MLA,
# current+7 replicated KDA rollback states, and a replicated 32,768-token
# DSpark tail is 1,297,907,712 bytes/rank. Round up slightly. This is
# 860.625 MiB/rank smaller than the equivalent DCP8 cache.
export KV_CACHE_MEMORY_BYTES="${KV_CACHE_MEMORY_BYTES:-1299000000}"
export DSPARK_DRAFT_KV_WINDOW="${DSPARK_DRAFT_KV_WINDOW:-32768}"
export DSPARK_DRAFT_WEIGHT_FORMAT="${DSPARK_DRAFT_WEIGHT_FORMAT:-bf16}"
export DSPARK_SHARD_MARKOV_HEAD="${DSPARK_SHARD_MARKOV_HEAD:-1}"

# DCP16 recovers enough memory to retain the target shared experts in their
# source BF16 format. DCP8 needs the online shared-expert MXFP8 overlay.
export KIMI_TARGET_MXFP8_PROFILE="${KIMI_TARGET_MXFP8_PROFILE:-none}"
export VLLM_DSPARK_COMPACT_ROPE="${VLLM_DSPARK_COMPACT_ROPE:-1}"
export VLLM_K3_KV_GROUP_SIZE="${VLLM_K3_KV_GROUP_SIZE:-6}"
# The production world-16 sweep selected 512 threads / eight CTAs. The exact
# vLLM graph pair falls from 117.88 to 98.30 us/layer; both the eight- and
# sixteen-rank settings pass SparkInfer's eager and CUDA-graph oracle.
export SPARKINFER_PCIE_DCP_THREADS="${SPARKINFER_PCIE_DCP_THREADS:-512}"
export SPARKINFER_PCIE_DCP_BLOCK_LIMIT="${SPARKINFER_PCIE_DCP_BLOCK_LIMIT:-8}"
export KDA_PREFILL_BACKEND="${KDA_PREFILL_BACKEND:-triton}"
export VLLM_MLA_CHUNKED_PREFILL_WORKSPACE_SIZE="${VLLM_MLA_CHUNKED_PREFILL_WORKSPACE_SIZE:-2048}"
if [[ -z "${COMPILATION_CONFIG:-}" ]]; then
  export COMPILATION_CONFIG='{"mode":0,"cudagraph_mode":"PIECEWISE","cudagraph_capture_sizes":[8],"pass_config":{"fuse_allreduce_rms":true}}'
fi
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Kimi-K3-MXFP4-HH-DSpark7-BF16-DCP16-1M-W32K}"

exec "${SCRIPT_DIR}/serve-kimi-k3-full-mxfp4-dspark7.sh" "$@"
