#!/usr/bin/env bash
# Kimi-K3 keep+EXL3-3.0 (kquant Phase B) at TP=12, eager bring-up.
# Mirrors Martin's proven TP16 config where possible: KDA projections stay
# BF16 (overlay covers MLA attention + shared experts only), which the
# 3.19 bpw budget affords at TP12.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${SCRIPT_DIR}/.venv/bin/python}"

export PYTHONPATH="${SCRIPT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7,8,9,10,11}"
export CUTE_DSL_ARCH="${CUTE_DSL_ARCH:-sm_120a}"
export CUDA_DEVICE_MAX_CONNECTIONS=32
export NCCL_IB_DISABLE=1
export NCCL_P2P_LEVEL=SYS
export NCCL_PROTO=LL,LL128,Simple
export NCCL_BUFFSIZE=2097152
export NCCL_MAX_NCHANNELS=8
export OMP_NUM_THREADS=16
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_B12X_MOE=1
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_ENABLE_PCIE_ALLREDUCE=0
export KDA_DISABLE_AUTOTUNE=1
export VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE=134217728
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# EXL3 skeleton replicates suh/svh sign vectors per rank (~1.4 GiB), leaving
# little free memory at load start; staging buffers are freed after load.
export INSTANTTENSOR_MAX_FREE_MEM_USAGE=0.9
export INSTANTTENSOR_BACKEND=AIO

OVERLAY='{"linear":{"weight":"mxfp8"},"shared_experts":{"weight":"mxfp8"},"ignore":["re:.*kv_b_proj","re:.*conv1d","re:.*\\.b_proj","re:.*\\.q_proj","re:.*\\.k_proj","re:.*\\.v_proj","re:.*\\.g_proj","re:.*f_a_proj","re:.*f_b_proj","re:.*o_proj","re:.*lm_head","re:.*attn_res"]}'

exec "${PYTHON_BIN}" -m vllm.entrypoints.cli.main serve \
  "${MODEL_DIR:-/models/Kimi-K3-EXL3-3p14-serve}" \
  --served-model-name kimi-k3-exl3 \
  --trust-remote-code \
  --host 127.0.0.1 \
  --port "${PORT:-8011}" \
  --max-model-len "${MAX_MODEL_LEN:-4096}" \
  --tensor-parallel-size 12 \
  --enforce-eager \
  --load-format instanttensor \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.97}" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-2048}" \
  --max-num-seqs "${MAX_NUM_SEQS:-4}" \
  --quantization-config "${OVERLAY}" \
  "$@"
