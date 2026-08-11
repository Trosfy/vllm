#!/usr/bin/env bash
# Kimi-K3 production profile: stock MXFP4 routed experts, a weight-only MXFP8
# DSpark7 draft, DCP16, a physical 1M-token cache, and low-sensitivity KDA
# projections quantized online to MXFP8/Marlin.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# The 69 KDA input projections save about 1.36 GiB/rank.  The 32-window KLD
# campaign measured KL(reference || candidate)=0.003183 over 65,504 positions,
# within the runtime repeat floor.  Marlin remains W8A16, so activations stay
# BF16 and the measured decode cycle does not regress.
export KIMI_TARGET_MXFP8_PROFILE="${KIMI_TARGET_MXFP8_PROFILE:-kda_in_proj}"

# The five-layer draft uses weight-only MXFP8 through Marlin W8A16. The target
# launcher disables W8A8 kernels whenever an MXFP8 draft or target overlay is
# selected, so activations remain BF16 in both models.
export DSPARK_DRAFT_WEIGHT_FORMAT="${DSPARK_DRAFT_WEIGHT_FORMAT:-mxfp8}"
export DSPARK_DRAFT_MXFP8_BACKEND="${DSPARK_DRAFT_MXFP8_BACKEND:-marlin}"

# A 4096-token prefill shape needs the logical output for all rows but only a
# 1024-row routed-expert scratch arena. Chunking preserves token-independent
# MoE results while keeping CUDA-graph capture within the physical 1M profile.
export B12X_MOE_WORKSPACE_TOKEN_LIMIT="${B12X_MOE_WORKSPACE_TOKEN_LIMIT:-1024}"
export B12X_W4A16_SMALL_M_DIRECT="${B12X_W4A16_SMALL_M_DIRECT:-1}"
export VLLM_DISABLE_SHARED_EXPERTS_STREAM="${VLLM_DISABLE_SHARED_EXPERTS_STREAM:-1}"

# B12X kernel resolution traverses both graph shapes without launching a
# second eager model execution. The graph manager captures separate M=8
# speculative and M=1 no-draft FULL graphs before freezing kernel resolution.
export VLLM_B12X_CUDAGRAPH_COMPILE_ONLY_PREWARM="${VLLM_B12X_CUDAGRAPH_COMPILE_ONLY_PREWARM:-1}"

# At a 4096 scheduler budget, speculative slots require two additional hybrid
# cache blocks compared with a 2048-token scheduler profile. This allocation
# provides 1,057,049 physical tokens: enough for max_model_len=1,048,576
# without using logical oversubscription.
export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}"
export VLLM_MLA_CHUNKED_PREFILL_WORKSPACE_SIZE="${VLLM_MLA_CHUNKED_PREFILL_WORKSPACE_SIZE:-4096}"
export KV_CACHE_MEMORY_BYTES="${KV_CACHE_MEMORY_BYTES:-1325000000}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Kimi-K3-MXFP4-HH-DSpark7-DraftMXFP8-DCP16-1M-KDA-MXFP8-P4096}"

exec "${SCRIPT_DIR}/serve-kimi-k3-full-mxfp4-dspark7-dcp16-1m.sh" "$@"
