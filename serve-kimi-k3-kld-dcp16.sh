#!/usr/bin/env bash
# Target-only Kimi-K3 KLD capture profile matching the production DCP16 math.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

: "${VLLM_KLD_CAPTURE_DIR:?set VLLM_KLD_CAPTURE_DIR to a fresh directory}"
export PYTHON_BIN="${PYTHON_BIN:-/opt/venv/bin/python}"
export MODEL="${MODEL:-/root/.cache/huggingface/hub/models--moonshotai--Kimi-K3/snapshots/2496450e92e425c886db095102a52a6682ca3970}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Kimi-K3}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-256}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.982}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="${SCRIPT_DIR}:/mnt/luke/b12x-k3-hh-dense-mla-dcp8-latest${PYTHONPATH:+:${PYTHONPATH}}"

export VLLM_KIMI_SHARD_QKV_A=1
export VLLM_KIMI_SHARD_ROUTED_DOWN_PROJ=1
export VLLM_KIMI_SHARD_ROUTED_UP_PROJ=1
export VLLM_KIMI_SHARD_ROUTER=1
export VLLM_ENABLE_PCIE_ALLREDUCE=1
export VLLM_PCIE_ALLREDUCE_BACKEND=b12x
export VLLM_PCIE_ONESHOT_SINGLE_CHANNEL=1
export VLLM_USE_B12X_DCP_A2A=1
export VLLM_DCP_A2A_MAX_TOKENS=8
export VLLM_DCP_A2A_LARGE_BACKEND=ag_rs
export VLLM_DCP_INDEXER_SHARDS=0
export VLLM_DCP_QUERY_SPLIT=0
export VLLM_DCP_GLOBAL_TOPK=0
export VLLM_DCP_PROJECT_BEFORE_MERGE=0
export VLLM_MEMORY_PROFILE_INCLUDE_ATTN=0
export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0
export KDA_PREFILL_BACKEND=triton
export B12X_PCIE_DCP_THREADS=512
export B12X_PCIE_DCP_BLOCK_LIMIT=8
export VLLM_K3_KV_GROUP_SIZE=6
export CUDA_MODULE_LOADING=LAZY
export CUDA_MODULE_DATA_LOADING=LAZY
if [[ -z "${COMPILATION_CONFIG:-}" ]]; then
  export COMPILATION_CONFIG='{"mode":0,"cudagraph_mode":"PIECEWISE","cudagraph_capture_sizes":[1],"pass_config":{"fuse_allreduce_rms":true}}'
fi

profile="${KIMI_TARGET_MXFP8_PROFILE:-none}"
quant_args=()
case "${profile}" in
  none) ;;
  kda_in_proj)
    export VLLM_DISABLED_KERNELS="${VLLM_DISABLED_KERNELS:+${VLLM_DISABLED_KERNELS},}B12xMxfp8LinearKernel,FlashInferCutedslMxfp8LinearKernel,FlashInferCutlassMxfp8LinearKernel"
    quant_args+=(
      --quantization-config
      '{"linear":"mxfp8","ignore":["re:^(?!.*self_attn\\.(?:q_proj|k_proj|v_proj|b_proj|f_a_proj)$).*$"]}'
    )
    ;;
  attention_o_proj)
    export VLLM_DISABLED_KERNELS="${VLLM_DISABLED_KERNELS:+${VLLM_DISABLED_KERNELS},}B12xMxfp8LinearKernel,FlashInferCutedslMxfp8LinearKernel,FlashInferCutlassMxfp8LinearKernel"
    quant_args+=(
      --quantization-config
      '{"linear":"mxfp8","ignore":["re:^(?!.*self_attn\\.o_proj$).*$"]}'
    )
    ;;
  kda_in_and_o_proj)
    export VLLM_DISABLED_KERNELS="${VLLM_DISABLED_KERNELS:+${VLLM_DISABLED_KERNELS},}B12xMxfp8LinearKernel,FlashInferCutedslMxfp8LinearKernel,FlashInferCutlassMxfp8LinearKernel"
    quant_args+=(
      --quantization-config
      '{"linear":"mxfp8","ignore":["re:^(?!.*self_attn\\.(?:q_proj|k_proj|v_proj|b_proj|f_a_proj|o_proj)$).*$"]}'
    )
    ;;
  *)
    echo "Unsupported KIMI_TARGET_MXFP8_PROFILE=${profile}" >&2
    exit 2
    ;;
esac

exec "${SCRIPT_DIR}/serve-kimi-k3-instanttensor.sh" \
  --language-model-only \
  --attention-backend B12X_MLA \
  --decode-context-parallel-size 16 \
  --dcp-comm-backend a2a \
  --dcp-kv-cache-interleave-size 1 \
  --kda-prefill-backend triton \
  --kv-cache-dtype fp8 \
  --kv-cache-memory-bytes 500000000 \
  --no-enable-prefix-caching \
  --additional-config '{"kda_shard_f_a":true}' \
  "${quant_args[@]}" \
  "$@"
