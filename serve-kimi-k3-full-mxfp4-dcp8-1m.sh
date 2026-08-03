#!/usr/bin/env bash
# Full stock Kimi K3 MXFP4 on HH, TP16/DCP8, native SparkInfer dense MLA,
# and a physical 1M-token FP8 KV cache.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${SCRIPT_DIR}/.venv/bin/python}"

# This wrapper runs its compatibility preflight before serve-kimi-k3.sh gets a
# chance to prepend the checkout. Make that preflight use the same HH sources.
if [[ -e "${SCRIPT_DIR}/vllm/_C_stable_libtorch.abi3.so" ]]; then
  export PYTHONPATH="${SCRIPT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
fi

TP_SIZE="${TP_SIZE:-16}"
DCP_SIZE="${DCP_SIZE:-8}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1048576}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-256}"
KV_CACHE_MEMORY_BYTES="${KV_CACHE_MEMORY_BYTES:-1860000000}"

DCP_COMM_BACKEND="${DCP_COMM_BACKEND:-a2a}"
# This profile serves one sequence at a time.  Sizing SparkInfer's two
# graph/eager PCIe staging channels for 64 rows wastes about 13 MiB/rank at
# the tight 1M-cache fit without accelerating a size-1 decode.
DCP_A2A_MAX_TOKENS="${DCP_A2A_MAX_TOKENS:-1}"
DCP_A2A_LARGE_BACKEND="${DCP_A2A_LARGE_BACKEND:-ag_rs}"
KDA_PREFILL_BACKEND="${KDA_PREFILL_BACKEND:-triton}"

if (( TP_SIZE != 16 )); then
  echo "This profile is validated only for TP_SIZE=16, got ${TP_SIZE}" >&2
  exit 2
fi
if (( DCP_SIZE != 8 )); then
  echo "This profile is validated only for DCP_SIZE=8, got ${DCP_SIZE}" >&2
  exit 2
fi
if (( TP_SIZE % DCP_SIZE != 0 )); then
  echo "TP_SIZE (${TP_SIZE}) must be divisible by DCP_SIZE (${DCP_SIZE})" >&2
  exit 2
fi
if [[ "${DCP_COMM_BACKEND}" != "a2a" ]]; then
  echo "This profile requires DCP_COMM_BACKEND=a2a" >&2
  exit 2
fi
case "${DCP_A2A_LARGE_BACKEND}" in
  ag_rs | a2a) ;;
  *)
    echo "DCP_A2A_LARGE_BACKEND must be ag_rs or a2a" >&2
    exit 2
    ;;
esac
if [[ ! "${DCP_A2A_MAX_TOKENS}" =~ ^[0-9]+$ ]]; then
  echo "DCP_A2A_MAX_TOKENS must be a non-negative integer" >&2
  exit 2
fi
case "${KDA_PREFILL_BACKEND}" in
  triton | flashkda) ;;
  *)
    echo "KDA_PREFILL_BACKEND must be triton or flashkda" >&2
    exit 2
    ;;
esac
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python interpreter not found or not executable: ${PYTHON_BIN}" >&2
  exit 1
fi

# Fail before loading 1.4 TiB of weights when the requested Luke/SparkInfer
# dense MLA package is missing from PYTHONPATH.
export KDA_PREFILL_BACKEND
"${PYTHON_BIN}" - <<'PY'
import os

from sparkinfer.attention import dense_mla
from vllm.model_executor.layers.activation import ensure_kimi_k3_activation_ops
from vllm.models.kimi_k3.nvidia.kda import ensure_fused_kda_decode_op
from vllm.models.kimi_k3.nvidia.ops.fused_mla_key_concat_kv_cache import (
    ensure_kimi_k3_cache_ops,
)

required = ("Caps", "plan", "bind", "compile", "run")
missing = [name for name in required if not hasattr(dense_mla, name)]
if missing:
    raise RuntimeError(f"incomplete sparkinfer.attention.dense_mla: {missing}")
print(f"SparkInfer dense MLA preflight: {dense_mla.__file__}", flush=True)
ensure_kimi_k3_cache_ops()
print("HH Kimi-K3 fused cache-op preflight: OK", flush=True)
if not ensure_fused_kda_decode_op():
    raise RuntimeError("HH Kimi-K3 fused KDA decode op is unavailable")
print("HH Kimi-K3 fused KDA decode preflight: OK", flush=True)
if os.environ["KDA_PREFILL_BACKEND"] == "flashkda":
    import vllm._flashkda_C  # noqa: F401, E402

    print("HH FlashKDA prefill preflight: OK", flush=True)
else:
    print("HH Triton KDA prefill selected: OK", flush=True)
if not ensure_kimi_k3_activation_ops():
    raise RuntimeError("HH Kimi-K3 fused SiTU activation ops are unavailable")
print("HH Kimi-K3 fused SiTU activation preflight: OK", flush=True)
PY

# Native dense MLA gathers the six local TP16 query heads over DCP8, executes
# 48 effective heads against the local 1/8 KV shard, and LSE-reduces the eight
# partial outputs. Decode/small batches use SparkInfer's PCIe A2A path.
export VLLM_USE_B12X_DCP_A2A="${VLLM_USE_B12X_DCP_A2A:-1}"
export VLLM_B12X_CUDAGRAPH_COMPILE_ONLY_PREWARM="${VLLM_B12X_CUDAGRAPH_COMPILE_ONLY_PREWARM:-1}"
export VLLM_DCP_A2A_MAX_TOKENS="${VLLM_DCP_A2A_MAX_TOKENS:-${DCP_A2A_MAX_TOKENS}}"
export VLLM_DCP_A2A_LARGE_BACKEND="${VLLM_DCP_A2A_LARGE_BACKEND:-${DCP_A2A_LARGE_BACKEND}}"

# K3 has no sparse indexer. Keep all GLM sparse/CKV policies disabled.
export VLLM_DCP_INDEXER_SHARDS="${VLLM_DCP_INDEXER_SHARDS:-0}"
export VLLM_DCP_QUERY_SPLIT="${VLLM_DCP_QUERY_SPLIT:-0}"
export VLLM_DCP_GLOBAL_TOPK="${VLLM_DCP_GLOBAL_TOPK:-0}"
export VLLM_DCP_PROJECT_BEFORE_MERGE="${VLLM_DCP_PROJECT_BEFORE_MERGE:-0}"
export VLLM_B12X_MLA_DCP_GATHER_IN_WORKSPACE="${VLLM_B12X_MLA_DCP_GATHER_IN_WORKSPACE:-0}"

# FlashKDA's first SM120 launch commits about 3.74 GiB of non-PyTorch CUDA
# module state. The full stock checkpoint plus a physical 1M cache has only
# about 0.45 GiB free after graph capture. Triton KDA prefill commits about
# 0.12 GiB for the same 69-layer warmup, while decode still uses the separate
# fused conv+KDA+norm companion. FlashKDA remains an explicit override for
# smaller-cache experiments, but it cannot fit this profile as currently built.

# Manual KV sizing and a measured size-1 graph make both conservative memory
# reservations unnecessary on this very tight full-MXFP4 fit.
export VLLM_MEMORY_PROFILE_INCLUDE_ATTN="${KIMI_MEMORY_PROFILE_INCLUDE_ATTN:-0}"
export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS="${KIMI_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS:-0}"

# Bound context-gather and expanded Q/K/V transients while retaining the full
# physical 1M cache. Long contexts are merged from additional exact chunks.
export VLLM_MLA_CHUNKED_PREFILL_WORKSPACE_SIZE="${VLLM_MLA_CHUNKED_PREFILL_WORKSPACE_SIZE:-32768}"

# The 24 dense MLA q_a/kv_a projections are BF16 in the checkpoint. Shard
# their output rows across TP16 and gather the 2112-element latent result,
# saving about 0.63 GiB/rank without changing checkpoint precision.
export VLLM_KIMI_SHARD_QKV_A="${VLLM_KIMI_SHARD_QKV_A:-1}"

# Keep HH's replicated routed up-projection and fused latent-MoE tail, but
# TP-shard the 49-MiB BF16 routed down-projection in each of 92 MoE layers.
# Gathering the 3584-element latent output saves about 4.13 GiB/rank.
export VLLM_KIMI_SHARD_ROUTED_DOWN_PROJ="${VLLM_KIMI_SHARD_ROUTED_DOWN_PROJ:-1}"

# HH otherwise replicates another 49-MiB BF16 up projection in every MoE
# layer. Row-shard it and reduce the routed+shared hidden partial together.
# This saves another 4.13 GiB/rank and makes the full 93-layer model fit.
export VLLM_KIMI_SHARD_ROUTED_UP_PROJ="${VLLM_KIMI_SHARD_ROUTED_UP_PROJ:-1}"

# The 896x7168 BF16 router is otherwise replicated in all 92 MoE layers.
# Shard it and gather the 896 FP32 logits, saving about 1.03 GiB/rank.
export VLLM_KIMI_SHARD_ROUTER="${VLLM_KIMI_SHARD_ROUTER:-1}"

export VLLM_ENABLE_PCIE_ALLREDUCE="${VLLM_ENABLE_PCIE_ALLREDUCE:-1}"
export VLLM_PCIE_ALLREDUCE_BACKEND="${KIMI_PCIE_ALLREDUCE_BACKEND:-b12x}"
export VLLM_PCIE_ONESHOT_SINGLE_CHANNEL="${KIMI_PCIE_ONESHOT_SINGLE_CHANNEL:-1}"

if [[ -z "${COMPILATION_CONFIG:-}" ]]; then
  export COMPILATION_CONFIG='{"mode":0,"cudagraph_mode":"PIECEWISE","cudagraph_capture_sizes":[1],"pass_config":{"fuse_allreduce_rms":true}}'
fi

export MODEL="${MODEL:-/root/.cache/huggingface/hub/models--moonshotai--Kimi-K3/snapshots/2496450e92e425c886db095102a52a6682ca3970}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Kimi-K3-MXFP4-HH-DenseMLA-DCP8-1M}"
export TP_SIZE DCP_SIZE MAX_MODEL_LEN MAX_NUM_SEQS MAX_NUM_BATCHED_TOKENS
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.985}"

unset NCCL_GRAPH_FILE NCCL_GRAPH_DUMP_FILE

exec "${SCRIPT_DIR}/serve-kimi-k3-instanttensor.sh" \
  --language-model-only \
  --attention-backend B12X_MLA \
  --decode-context-parallel-size "${DCP_SIZE}" \
  --dcp-comm-backend "${DCP_COMM_BACKEND}" \
  --dcp-kv-cache-interleave-size 1 \
  --kda-prefill-backend "${KDA_PREFILL_BACKEND}" \
  --kv-cache-dtype fp8 \
  --kv-cache-memory-bytes "${KV_CACHE_MEMORY_BYTES}" \
  --no-enable-prefix-caching \
  "$@"
