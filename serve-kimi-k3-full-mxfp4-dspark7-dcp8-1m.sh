#!/usr/bin/env bash
# Full stock Kimi-K3 MXFP4 target with an Inferact DSpark7 draft, TP16/DCP8,
# and a physical 1M-token target KV cache.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export DCP_SIZE="${DCP_SIZE:-8}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-1048576}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-2048}"

# Exact model-free lower bound for one 1M request with DCP8 target MLA,
# current+7 KDA rollback states, and a replicated 32,768-token draft tail is
# 2,200,338,432 bytes/rank. The draft tail includes two 2,048-token in-flight
# scheduler batches (async scheduling). Group size six uses only 17 block
# tables (three fewer than the default) while removing 44.7 MiB/rank of cache
# padding at this window. Round up slightly.
export KV_CACHE_MEMORY_BYTES="${KV_CACHE_MEMORY_BYTES:-2201000000}"
export DSPARK_DRAFT_KV_WINDOW="${DSPARK_DRAFT_KV_WINDOW:-32768}"
export DSPARK_DRAFT_WEIGHT_FORMAT="${DSPARK_DRAFT_WEIGHT_FORMAT:-bf16}"
# BF16 restores the trained draft numerics.  TP-sharding the two 163840x256
# Markov matrices reduces them from 160 MiB to 10 MiB/rank at TP16; the exact
# seven-step harness measured a 0.353 ms median penalty when base and Markov
# logits are added locally before one eager vocabulary gather per step.
export DSPARK_SHARD_MARKOV_HEAD="${DSPARK_SHARD_MARKOV_HEAD:-1}"
export KIMI_TARGET_MXFP8_PROFILE="${KIMI_TARGET_MXFP8_PROFILE:-shared_experts}"
export VLLM_DSPARK_COMPACT_ROPE="${VLLM_DSPARK_COMPACT_ROPE:-1}"
export VLLM_K3_KV_GROUP_SIZE="${VLLM_K3_KV_GROUP_SIZE:-6}"
# Exact 16-GPU/two-DCP8-group sweep: 512 threads and eight CTAs reduce the
# production FP8-query gather + BF16 LSE-reduce graph from 74.19 to 69.74 us.
export SPARKINFER_PCIE_DCP_THREADS="${SPARKINFER_PCIE_DCP_THREADS:-512}"
export SPARKINFER_PCIE_DCP_BLOCK_LIMIT="${SPARKINFER_PCIE_DCP_BLOCK_LIMIT:-8}"
# FlashKDA's SM120 prefill cubin materializes several GiB/rank on first use.
# Keep the 1M profile on the bounded-memory Triton prefill path; this does not
# affect KDA decode, DSpark acceptance, or target verification throughput.
export KDA_PREFILL_BACKEND="${KDA_PREFILL_BACKEND:-triton}"
# A single request can schedule at most 2,048 prompt rows here.  Avoid keeping
# four times that capacity in every dense-MLA metadata workspace.
export VLLM_MLA_CHUNKED_PREFILL_WORKSPACE_SIZE="${VLLM_MLA_CHUNKED_PREFILL_WORKSPACE_SIZE:-2048}"
if [[ -z "${COMPILATION_CONFIG:-}" ]]; then
  export COMPILATION_CONFIG='{"mode":0,"cudagraph_mode":"PIECEWISE","cudagraph_capture_sizes":[8],"pass_config":{"fuse_allreduce_rms":true}}'
fi
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Kimi-K3-MXFP4-HH-DSpark7-BF16-DCP8-1M-W32K}"

exec "${SCRIPT_DIR}/serve-kimi-k3-full-mxfp4-dspark7.sh" "$@"
