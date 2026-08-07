#!/usr/bin/env bash
# Full stock Kimi K3 target with the DFlash draft (modal-labs/Kimi-K3-DFlash),
# TP16, DCP16, physical one-million-token cache.
#
# The draft is a six-layer qwen3-style GQA model with a 4096 sliding window and
# dflash_config.target_layer_ids [19,37,54,66,78,90]. Three of the defaults
# below are load-bearing and were each found the hard way:
#
#   * FULL decode graphs. DFlashSpeculator supports only full graphs and
#     silently runs the draft eagerly otherwise, which costs 5.3x
#     (5.05 -> 26.96 target cycles/s). The capture size must be 1 + K.
#   * draft_load_config load_format=auto. InstantTensor demands
#     chunk x concurrency x io_depth x world_size free bytes when it opens the
#     draft, which a ~90 GiB/rank target does not leave; the plain loader needs
#     none of it.
#   * VLLM_USE_B12X_FP8_GEMM=1 with the kda_in_proj MXFP8 profile. The profile
#     frees the ~1.36 GiB/rank that makes room for the draft, but one K3 KDA
#     projection is narrower than flashinfer's mm_mxfp8 n>=128 / k>=128 bound;
#     the B12X kernel has no such bound.
#
# KV budget: 1.20 GB yields 1,039,043 tokens. DSpark's 1.325 GB budget reports
# more (1,151,050) but leaves so little device memory that CUDA module loads
# spin in the driver during startup warmup, so do not raise it blindly.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

DFLASH_DRAFT="${DFLASH_DRAFT:-/root/.cache/huggingface/hub/models--modal-labs--Kimi-K3-DFlash/snapshots/c192d15a43407bf758b5ae0880d5c72052fef1de}"
DFLASH_TOKENS="${DFLASH_TOKENS:-7}"
DFLASH_ATTENTION_BACKEND="${DFLASH_ATTENTION_BACKEND:-TRITON_ATTN}"
# Quantize the draft's linears online. qkv_proj must stay BF16: the fused
# context-KV precompute consumes those weights through a raw F.linear and
# rejects an FP8 operand.
DFLASH_DRAFT_MXFP8="${DFLASH_DRAFT_MXFP8:-1}"

if [[ ! -f "${DFLASH_DRAFT}/config.json" ]]; then
  echo "DFlash draft checkpoint is missing: ${DFLASH_DRAFT}" >&2
  exit 1
fi

export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Kimi-K3-MXFP4-HH-DFlash-DCP16-1M}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-1048576}"
export KV_CACHE_MEMORY_BYTES="${KV_CACHE_MEMORY_BYTES:-1200000000}"
export KIMI_TARGET_MXFP8_PROFILE="${KIMI_TARGET_MXFP8_PROFILE:-kda_in_proj}"
export VLLM_USE_B12X_FP8_GEMM="${VLLM_USE_B12X_FP8_GEMM:-1}"
export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS="${VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS:-1}"
# Draft graph capture runs long enough for gloo's TCP rendezvous to pick a
# routable-but-unreachable global address on this box and abort the barrier.
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-lo}"
export TP_SOCKET_IFNAME="${TP_SOCKET_IFNAME:-lo}"

if [[ -z "${COMPILATION_CONFIG:-}" ]]; then
  export COMPILATION_CONFIG="{\"mode\":0,\"cudagraph_mode\":\"FULL_AND_PIECEWISE\",\"cudagraph_capture_sizes\":[$((DFLASH_TOKENS + 1))],\"pass_config\":{\"fuse_allreduce_rms\":true}}"
fi

DRAFT_QUANT_JSON=""
if [[ "${DFLASH_DRAFT_MXFP8}" == 1 ]]; then
  DRAFT_QUANT_JSON=',"quantization":"mxfp8","quantization_config":{"linear":"mxfp8","ignore":["re:.*qkv_proj$"]}'
fi

printf -v SPECULATIVE_CONFIG \
  '{"method":"dflash","model":"%s","num_speculative_tokens":%s,"attention_backend":"%s","draft_load_config":{"load_format":"auto"}%s}' \
  "${DFLASH_DRAFT}" "${DFLASH_TOKENS}" "${DFLASH_ATTENTION_BACKEND}" \
  "${DRAFT_QUANT_JSON}"

exec "${SCRIPT_DIR}/serve-kimi-k3-full-mxfp4-dcp16-1m-no-dspark.sh" \
  --speculative-config "${SPECULATIVE_CONFIG}" \
  "$@"
