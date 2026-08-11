#!/usr/bin/env bash
# Kimi-K3 production profile: stock MXFP4 routed experts, BF16 DSpark7,
# DCP16, a physical 1M-token cache, and low-sensitivity KDA projections
# quantized online to MXFP8/Marlin.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# The 69 KDA input projections save about 1.36 GiB/rank.  The 32-window KLD
# campaign measured KL(reference || candidate)=0.003183 over 65,504 positions,
# within the runtime repeat floor.  Marlin remains W8A16, so activations stay
# BF16 and the measured decode cycle does not regress.
export KIMI_TARGET_MXFP8_PROFILE="${KIMI_TARGET_MXFP8_PROFILE:-kda_in_proj}"

# At a 4096 scheduler budget, speculative slots require two additional hybrid
# cache blocks compared with the old 2048 profile.  This allocation provides
# 1,057,049 physical tokens: enough for max_model_len=1,048,576 without using
# logical oversubscription.
export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}"
export VLLM_MLA_CHUNKED_PREFILL_WORKSPACE_SIZE="${VLLM_MLA_CHUNKED_PREFILL_WORKSPACE_SIZE:-4096}"
export KV_CACHE_MEMORY_BYTES="${KV_CACHE_MEMORY_BYTES:-1325000000}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Kimi-K3-MXFP4-HH-DSpark7-BF16-DCP16-1M-KDA-MXFP8-P4096}"

exec "${SCRIPT_DIR}/serve-kimi-k3-full-mxfp4-dspark7-dcp16-1m.sh" "$@"
