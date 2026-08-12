#!/usr/bin/env bash
# Serve the stock Kimi-K3 MXFP4 target with the modal-labs Kimi-K3-DFlash
# draft on TP16/DCP16. The target retains a physical one-million-token FP8
# cache; KDA input projections and eligible draft linears use weight-only
# MXFP8 to fit both models without changing target activation precision.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DFLASH_DRAFT="${DFLASH_DRAFT:-/root/.cache/huggingface/hub/models--modal-labs--Kimi-K3-DFlash/snapshots/c192d15a43407bf758b5ae0880d5c72052fef1de}"
DFLASH_TOKENS="${DFLASH_TOKENS:-7}"
DFLASH_ATTENTION_BACKEND="${DFLASH_ATTENTION_BACKEND:-TRITON_ATTN}"
DFLASH_DRAFT_MXFP8="${DFLASH_DRAFT_MXFP8:-1}"

if [[ ! -f "${DFLASH_DRAFT}/config.json" || ! -f "${DFLASH_DRAFT}/model.safetensors" ]]; then
  echo "Kimi-K3 DFlash checkpoint is incomplete: ${DFLASH_DRAFT}" >&2
  exit 1
fi
if (( DFLASH_TOKENS != 7 )); then
  echo "The Kimi-K3 DFlash profile requires DFLASH_TOKENS=7" >&2
  exit 2
fi
case "${DFLASH_DRAFT_MXFP8}" in
  0 | 1) ;;
  *)
    echo "DFLASH_DRAFT_MXFP8 must be 0 or 1" >&2
    exit 2
    ;;
esac

export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Kimi-K3-MXFP4-DFlash-DCP16-1M}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-1048576}"
export KV_CACHE_MEMORY_BYTES="${KV_CACHE_MEMORY_BYTES:-1200000000}"
export VLLM_USE_B12X_FP8_GEMM="${VLLM_USE_B12X_FP8_GEMM:-1}"
export VLLM_KIMI_FUSED_TOPK16="${VLLM_KIMI_FUSED_TOPK16:-1}"
export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS="${VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS:-1}"
# The target and DFlash capture sequences require distinct MoE output storage.
export VLLM_DISABLE_FUSED_MOE_OUTPUT_ALIAS="${VLLM_DISABLE_FUSED_MOE_OUTPUT_ALIAS:-1}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-lo}"
export TP_SOCKET_IFNAME="${TP_SOCKET_IFNAME:-lo}"

if [[ -z "${COMPILATION_CONFIG:-}" ]]; then
  export COMPILATION_CONFIG="{\"mode\":0,\"cudagraph_mode\":\"FULL_AND_PIECEWISE\",\"cudagraph_capture_sizes\":[$((DFLASH_TOKENS + 1))],\"pass_config\":{\"fuse_allreduce_rms\":true}}"
fi

DRAFT_QUANT_JSON=""
if [[ "${DFLASH_DRAFT_MXFP8}" == 1 ]]; then
  # qkv_proj remains BF16 because context-KV precompute consumes the raw
  # projection weights through torch.nn.functional.linear.
  DRAFT_QUANT_JSON=',"quantization":"mxfp8","quantization_config":{"linear":"mxfp8","ignore":["re:.*qkv_proj$"]}'
fi

printf -v SPECULATIVE_CONFIG \
  '{"method":"dflash","model":"%s","num_speculative_tokens":%s,"attention_backend":"%s","draft_load_config":{"load_format":"auto"}%s}' \
  "${DFLASH_DRAFT}" "${DFLASH_TOKENS}" "${DFLASH_ATTENTION_BACKEND}" \
  "${DRAFT_QUANT_JSON}"

TARGET_QUANT_CONFIG='{"linear":"mxfp8","ignore":["re:^(?!.*self_attn\\.(?:q_proj|k_proj|v_proj|b_proj|f_a_proj)$).*$"]}'

NO_SPEC_LAUNCHER="${KIMI_NO_SPEC_LAUNCHER:-${SCRIPT_DIR}/serve-kimi-k3-full-mxfp4-dcp16-1m-no-spec.sh}"
if [[ ! -x "${NO_SPEC_LAUNCHER}" ]]; then
  echo "Kimi-K3 no-spec launcher is missing: ${NO_SPEC_LAUNCHER}" >&2
  exit 1
fi

exec "${NO_SPEC_LAUNCHER}" \
  --additional-config '{"kda_shard_f_a":false}' \
  --quantization-config "${TARGET_QUANT_CONFIG}" \
  --speculative-config "${SPECULATIVE_CONFIG}" \
  "$@"
