#!/usr/bin/env bash
# Full stock Kimi-K3 MXFP4 target plus the BF16 Inferact DSpark7 draft.
# This profile combines the validated DCP16/physical-1M DSpark layout with the
# bit-exact TP16 hierarchical all-reduce and Kimi projection optimizations.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export PYTHON_BIN="${PYTHON_BIN:-/opt/venv/bin/python}"
export SPARKINFER_DIR="${SPARKINFER_DIR:-/opt/kimi-k3-hh/sparkinfer}"

export DCP_SIZE=16
export MAX_MODEL_LEN=1048576
export MAX_NUM_SEQS=1
export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-2048}"

# Exact validated lower bound for a physical 1M target cache with DCP16 and a
# replicated 32,768-token DSpark DCP1 tail, including scheduler overlap.
export KV_CACHE_MEMORY_BYTES="${KV_CACHE_MEMORY_BYTES:-1299000000}"
export DSPARK_DRAFT_KV_WINDOW="${DSPARK_DRAFT_KV_WINDOW:-32768}"
export DSPARK_DRAFT_WEIGHT_FORMAT=bf16
export DSPARK_SHARD_MARKOV_HEAD=1
export KIMI_TARGET_MXFP8_PROFILE=none
export VLLM_DSPARK_COMPACT_ROPE=1
export VLLM_K3_KV_GROUP_SIZE=6

export KDA_PREFILL_BACKEND=triton
export VLLM_MLA_CHUNKED_PREFILL_WORKSPACE_SIZE=2048
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Retain the lossless target projection shards needed by the tight target +
# draft memory profile, then use the low-latency B12X transport and fused K3
# router path for every supported decode shape.
export VLLM_KIMI_USE_B12X_PROJECTION_GATHER=1
export VLLM_KIMI_USE_B12X_PAIRED_PROJECTION_GATHER=1
export VLLM_KIMI_USE_B12X_PAIRED_PROJECTION_TOPK=1

# Bit-exact TP16 hierarchical all-reduce settings selected by the long
# model-free A/B/A and the full no-DSpark E2E control.
export KIMI_DSPARK_PCIE_ALLREDUCE_BACKEND=b12x
export KIMI_DSPARK_PCIE_ONESHOT_SINGLE_CHANNEL=1
export SPARKINFER_PCIE_HIERARCHICAL_DEFERRED_CONSUMPTION=1
export SPARKINFER_PCIE_HIERARCHICAL_DOUBLE_BUFFER=0
export SPARKINFER_PCIE_HIERARCHICAL_THREADS=256
export SPARKINFER_PCIE_HIERARCHICAL_NANOSLEEP_CYCLES=24
export SPARKINFER_PCIE_HIERARCHICAL_BF16X2=1
# Preserve the M=1 vector path and select the measured scalar-256 path for
# DSpark's eight-row target verifier collectives.
export SPARKINFER_PCIE_HIERARCHICAL_BF16X2_MAX_ELEMENTS=7168
export SPARKINFER_W4A16_SMALL_M_HOST_BARRIER_RESET=0

# DSpark verifies one current token plus seven draft tokens. DCP16 attention
# was validated at 512 threads and an eight-CTA cap for this shape.
export SPARKINFER_PCIE_DCP_THREADS=512
export SPARKINFER_PCIE_DCP_BLOCK_LIMIT=8
export COMPILATION_CONFIG='{"mode":0,"cudagraph_mode":"PIECEWISE","cudagraph_capture_sizes":[8],"pass_config":{"fuse_allreduce_rms":true}}'
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Kimi-K3-MXFP4-HH-DSpark7-BF16-DCP16-1M-AdaptiveAR-PairB8}"

exec "${SCRIPT_DIR}/serve-kimi-k3-full-mxfp4-dspark7.sh" "$@"
