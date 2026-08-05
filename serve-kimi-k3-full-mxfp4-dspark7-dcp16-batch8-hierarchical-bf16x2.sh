#!/usr/bin/env bash
# Full stock Kimi-K3 MXFP4 target plus BF16 Inferact DSpark7, configured for
# true continuous batching of eight simultaneous requests on TP16/DCP16.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export PYTHON_BIN="${PYTHON_BIN:-/opt/venv/bin/python}"
export SPARKINFER_DIR="${SPARKINFER_DIR:-/opt/kimi-k3-hh/sparkinfer}"

export DCP_SIZE=16
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-1048576}"
export MAX_NUM_SEQS=8
export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-2048}"

# Reduce proposal width as concurrency grows so every target verification
# shape is at most M=8. The maximum resident recurrent state-page count is
# exactly eight: 1x(K7+1), 2x(K3+1), 4x(K1+1), or 8x(K0+1).
unset VLLM_DSPARK_MAX_VERIFICATION_TOKENS
export DSPARK_BATCH_SIZE_SPECULATIVE_SCHEDULE='[[1,1,7],[2,2,3],[3,4,1],[5,8,0]]'
export KV_CACHE_MEMORY_BYTES="${KV_CACHE_MEMORY_BYTES:-1299000000}"
# Keep the replicated draft history bounded while allowing the target to use
# its full 1M context.  With DCP16 and the 1.299 GB/rank pool, the model-free
# lower bound is 1,199,702,016 bytes/rank and leaves roughly 99 MB of margin.
export DSPARK_DRAFT_KV_WINDOW="${DSPARK_DRAFT_KV_WINDOW:-4096}"
export DSPARK_DRAFT_WEIGHT_FORMAT=bf16
export DSPARK_SHARD_MARKOV_HEAD=1
export KIMI_TARGET_MXFP8_PROFILE=none
export VLLM_DSPARK_COMPACT_ROPE=1
export VLLM_K3_KV_GROUP_SIZE=6

export KDA_PREFILL_BACKEND=triton
export VLLM_MLA_CHUNKED_PREFILL_WORKSPACE_SIZE=2048
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

export VLLM_KIMI_USE_B12X_PROJECTION_GATHER=1
export VLLM_KIMI_USE_B12X_PAIRED_PROJECTION_GATHER=1
export VLLM_KIMI_USE_B12X_PAIRED_PROJECTION_TOPK=1

# M=16 exhausts the remaining memory; the adaptive grid above keeps every
# supported concurrency on the measured M=8 PIECEWISE graph instead.
export VLLM_USE_BREAKABLE_CUDAGRAPH=1
export VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD=256

export KIMI_DSPARK_PCIE_ALLREDUCE_BACKEND=b12x
export KIMI_DSPARK_PCIE_ONESHOT_SINGLE_CHANNEL=1
export SPARKINFER_PCIE_HIERARCHICAL_DEFERRED_CONSUMPTION=1
export SPARKINFER_PCIE_HIERARCHICAL_DOUBLE_BUFFER=0
export SPARKINFER_PCIE_HIERARCHICAL_THREADS=256
export SPARKINFER_PCIE_HIERARCHICAL_NANOSLEEP_CYCLES=24
export SPARKINFER_PCIE_HIERARCHICAL_BF16X2=1
export SPARKINFER_PCIE_HIERARCHICAL_BF16X2_MAX_ELEMENTS=7168
export SPARKINFER_W4A16_SMALL_M_HOST_BARRIER_RESET=0

# Let the SparkInfer DCP channel cover the complete eight-row target batch.
export VLLM_DCP_A2A_MAX_TOKENS=64
export SPARKINFER_PCIE_DCP_THREADS=512
export SPARKINFER_PCIE_DCP_BLOCK_LIMIT=8

# Runtime widths map concurrency 1/2/4/8 to M=8 in every case.
export COMPILATION_CONFIG='{"mode":0,"cudagraph_mode":"PIECEWISE","cudagraph_capture_sizes":[8],"pass_config":{"fuse_allreduce_rms":true}}'
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Kimi-K3-MXFP4-HH-DSpark7-AdaptiveK-BF16-DCP16-1M-Batch8-W4K}"

exec "${SCRIPT_DIR}/serve-kimi-k3-full-mxfp4-dspark7.sh" "$@"
