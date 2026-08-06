#!/usr/bin/env bash
# Full stock Kimi K3 target without speculative decoding.  This profile keeps
# routed experts in source MXFP4, dense weights in BF16, FP8 target KV, TP16,
# DCP16, and a physical one-million-token cache.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/opt/venv/bin/python}"
VLLM_SOURCE_DIR="${VLLM_SOURCE_DIR:-/opt/kimi-k3-hh/vllm}"
SPARKINFER_DIR="${SPARKINFER_DIR:-/opt/kimi-k3-hh/b12x}"

MODEL="${MODEL:-/root/.cache/huggingface/hub/models--moonshotai--Kimi-K3/snapshots/2496450e92e425c886db095102a52a6682ca3970}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Kimi-K3-MXFP4-HH-DCP16-1M-NoDSpark}"
TP_SIZE="${TP_SIZE:-16}"
DCP_SIZE="${DCP_SIZE:-16}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1048576}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
# Full-BF16 dense/KDA plus the physical 1M cache does not leave Triton's
# one-time 256-MiB autotune scratch above the 448-MiB AttnRes workspace needed
# by a 4096-token chunk.  The validated 2048-token chunk retains the complete
# cache and does not affect size-1 decode throughput.
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-2048}"
KV_CACHE_MEMORY_BYTES="${KV_CACHE_MEMORY_BYTES:-1325000000}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python interpreter is missing: ${PYTHON_BIN}" >&2
  exit 1
fi
if [[ ! -f "${VLLM_SOURCE_DIR}/vllm/__init__.py" ]]; then
  echo "vLLM source tree is missing: ${VLLM_SOURCE_DIR}" >&2
  exit 1
fi
if [[ ! -f "${MODEL}/model.safetensors.index.json" ]]; then
  echo "Kimi K3 target checkpoint is incomplete: ${MODEL}" >&2
  exit 1
fi
if [[ ! -f "${SPARKINFER_DIR}/b12x/attention/dense_mla/__init__.py" ]]; then
  echo "B12X dense MLA source is missing: ${SPARKINFER_DIR}" >&2
  exit 1
fi
if (( TP_SIZE != 16 || DCP_SIZE != 16 )); then
  echo "This profile requires TP_SIZE=16 and DCP_SIZE=16" >&2
  exit 2
fi
if [[ ! "${KV_CACHE_MEMORY_BYTES}" =~ ^[1-9][0-9]*$ ]]; then
  echo "KV_CACHE_MEMORY_BYTES must be a positive integer" >&2
  exit 2
fi

export PYTHON_BIN MODEL SERVED_MODEL_NAME TP_SIZE DCP_SIZE
export MAX_MODEL_LEN MAX_NUM_SEQS MAX_NUM_BATCHED_TOKENS
export PYTHONPATH="${VLLM_SOURCE_DIR}:${SPARKINFER_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.985}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# Decode uses many small TP16 NCCL reductions. Override the image's generic
# host-staging fallback for the validated Kimi-K3 topology while retaining an
# explicit operator escape hatch.
export NCCL_P2P_DISABLE="${KIMI_NCCL_P2P_DISABLE:-0}"
unset NCCL_GRAPH_FILE
export CUDA_MODULE_LOADING="${CUDA_MODULE_LOADING:-LAZY}"
export CUDA_MODULE_DATA_LOADING="${CUDA_MODULE_DATA_LOADING:-LAZY}"

# An image workdir containing another vLLM checkout appears as sys.path[0] and
# takes precedence over PYTHONPATH. Enter the selected tree so the preflight
# and server cannot silently import the image's older source instead.
cd "${VLLM_SOURCE_DIR}"

# Lossless projection sharding is retained for the initial matched baseline.
# Each switch can be disabled independently after a model-free communication
# screen; disabling spends VRAM to remove its corresponding TP collective.
export VLLM_KIMI_SHARD_QKV_A="${KIMI_SHARD_QKV_A:-1}"
export VLLM_KIMI_SHARD_ROUTED_DOWN_PROJ="${KIMI_SHARD_ROUTED_DOWN_PROJ:-1}"
export VLLM_KIMI_SHARD_ROUTED_UP_PROJ="${KIMI_SHARD_ROUTED_UP_PROJ:-1}"
export VLLM_KIMI_SHARD_ROUTER="${KIMI_SHARD_ROUTER:-1}"
export VLLM_KIMI_USE_B12X_PROJECTION_GATHER="${KIMI_B12X_PROJECTION_GATHER:-1}"
export VLLM_KIMI_USE_B12X_PAIRED_PROJECTION_GATHER="${KIMI_B12X_PAIRED_PROJECTION_GATHER:-1}"
export VLLM_KIMI_USE_B12X_PAIRED_PROJECTION_TOPK="${KIMI_B12X_PAIRED_PROJECTION_TOPK:-1}"
# The full 128-wide BF16 projection is bit-identical to 16 local 8-wide GEMVs
# and avoids 69 decode gathers. It costs about 121 MiB/rank, which fits this
# fixed-cache profile and wins the TP16 model-free screen.
KIMI_SHARD_F_A="${KIMI_SHARD_F_A:-0}"
case "${KIMI_SHARD_F_A}" in
  0) KIMI_ADDITIONAL_CONFIG='{}' ;;
  1) KIMI_ADDITIONAL_CONFIG='{"kda_shard_f_a":true}' ;;
  *)
    echo "KIMI_SHARD_F_A must be 0 or 1" >&2
    exit 2
    ;;
esac

export VLLM_ENABLE_PCIE_ALLREDUCE="${VLLM_ENABLE_PCIE_ALLREDUCE:-1}"
# The base image exports backend=cpp, whose legacy custom-allreduce runtime is
# limited to world size <= 8.  Do not inherit that image-wide default for this
# TP16 profile; select the validated hierarchical B12X implementation unless
# the profile-specific override explicitly requests otherwise.
export VLLM_PCIE_ALLREDUCE_BACKEND="${KIMI_NO_DSPARK_PCIE_ALLREDUCE_BACKEND:-b12x}"
export VLLM_PCIE_ONESHOT_SINGLE_CHANNEL="${VLLM_PCIE_ONESHOT_SINGLE_CHANNEL:-1}"
# The optional two-generation staging path is exact but was 1.2% slower in
# full-model decode, so retain the single-generation serving default.
export B12X_PCIE_HIERARCHICAL_DOUBLE_BUFFER="${B12X_PCIE_HIERARCHICAL_DOUBLE_BUFFER:-0}"
# The mixed 32/16/32-grid TP16 harness is bit-exact through 48,000 collectives
# and cuts the measured all-reduce sequence latency by 6.87%.  Keep the switch
# independently overridable so serving A/B tests can select the legacy path.
export B12X_PCIE_HIERARCHICAL_DEFERRED_CONSUMPTION="${B12X_PCIE_HIERARCHICAL_DEFERRED_CONSUMPTION:-1}"
# K3's 7168/32 and 3584/16 grids both contain exactly 224 BF16 values per
# block.  The vectorized path processes 112 BF16 pairs with four warps,
# preserves the scalar FP32 accumulation order, and is bit-exact in the TP16
# mixed-grid and odd-tail stress harnesses.  A long model-free A/B/A measured
# 338.21 us/graph versus 356.21 us for the original deferred configuration.
export B12X_PCIE_HIERARCHICAL_THREADS="${B12X_PCIE_HIERARCHICAL_THREADS:-224}"
export B12X_PCIE_HIERARCHICAL_NANOSLEEP_CYCLES="${B12X_PCIE_HIERARCHICAL_NANOSLEEP_CYCLES:-24}"
export B12X_PCIE_HIERARCHICAL_BF16X2="${B12X_PCIE_HIERARCHICAL_BF16X2:-1}"
export VLLM_DISABLE_SHARED_EXPERTS_STREAM="${VLLM_DISABLE_SHARED_EXPERTS_STREAM:-0}"
export VLLM_USE_B12X_DCP_A2A=1
export VLLM_DCP_A2A_MAX_TOKENS="${VLLM_DCP_A2A_MAX_TOKENS:-1}"
export VLLM_DCP_A2A_LARGE_BACKEND="${VLLM_DCP_A2A_LARGE_BACKEND:-ag_rs}"
export B12X_PCIE_DCP_THREADS="${B12X_PCIE_DCP_THREADS:-512}"
# The same exact fused projection/router kernel serves M=1 and M=8; 384
# threads is faster than its former 512-thread launch at both shapes.
export B12X_PCIE_KIMI_TOPK_THREADS="${B12X_PCIE_KIMI_TOPK_THREADS:-384}"
# Size-1 K3 DCP16 query gather measured best at four CTAs; LSE reduction and
# projection gathers already select fewer CTAs from their row counts.
export B12X_PCIE_DCP_BLOCK_LIMIT="${B12X_PCIE_DCP_BLOCK_LIMIT:-4}"
# The small-M MoE grid barrier resets its counter and advances its epoch before
# releasing the grid, so the next completed launch can reuse both scalars.
# This removes two 1x-int32 fill kernels per sparse layer (184 launches/token).
export B12X_W4A16_SMALL_M_HOST_BARRIER_RESET="${B12X_W4A16_SMALL_M_HOST_BARRIER_RESET:-0}"

# K3 has dense MLA and no GLM sparse indexer/selected-CKV decode path.
export VLLM_DCP_INDEXER_SHARDS=0
export VLLM_DCP_QUERY_SPLIT=0
export VLLM_DCP_GLOBAL_TOPK=0
export VLLM_DCP_PROJECT_BEFORE_MERGE=0

export KDA_PREFILL_BACKEND="${KDA_PREFILL_BACKEND:-triton}"
export VLLM_K3_KV_GROUP_SIZE="${VLLM_K3_KV_GROUP_SIZE:-6}"
export VLLM_MLA_CHUNKED_PREFILL_WORKSPACE_SIZE="${VLLM_MLA_CHUNKED_PREFILL_WORKSPACE_SIZE:-4096}"
export VLLM_MEMORY_PROFILE_INCLUDE_ATTN="${VLLM_MEMORY_PROFILE_INCLUDE_ATTN:-0}"
export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS="${VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS:-0}"
if [[ -z "${COMPILATION_CONFIG:-}" ]]; then
  export COMPILATION_CONFIG='{"mode":0,"cudagraph_mode":"PIECEWISE","cudagraph_capture_sizes":[1],"pass_config":{"fuse_allreduce_rms":true}}'
fi

"${PYTHON_BIN}" - <<'PY'
from b12x.attention import dense_mla
from vllm.model_executor.layers.activation import ensure_kimi_k3_activation_ops
from vllm.models.kimi_k3.nvidia.kda import ensure_fused_kda_decode_op
from vllm.models.kimi_k3.nvidia.ops.fused_mla_key_concat_kv_cache import (
    ensure_kimi_k3_cache_ops,
)

required = ("Caps", "plan", "bind", "compile", "run")
missing = [name for name in required if not hasattr(dense_mla, name)]
if missing:
    raise RuntimeError(f"incomplete B12X dense MLA API: {missing}")
ensure_kimi_k3_cache_ops()
if not ensure_fused_kda_decode_op():
    raise RuntimeError("Kimi K3 fused KDA decode op is unavailable")
if not ensure_kimi_k3_activation_ops():
    raise RuntimeError("Kimi K3 fused SiTU activation op is unavailable")
print(f"B12X dense MLA preflight: {dense_mla.__file__}", flush=True)
print("Kimi K3 no-DSpark target preflight: OK", flush=True)
PY

if [[ "${KIMI_NO_DSPARK_PREFLIGHT_ONLY:-0}" == 1 ]]; then
  exit 0
fi

exec "${SCRIPT_DIR}/serve-kimi-k3-instanttensor.sh" \
  --language-model-only \
  --attention-backend B12X_MLA \
  --decode-context-parallel-size "${DCP_SIZE}" \
  --dcp-comm-backend a2a \
  --dcp-kv-cache-interleave-size 1 \
  --kda-prefill-backend "${KDA_PREFILL_BACKEND}" \
  --kv-cache-dtype fp8 \
  --kv-cache-memory-bytes "${KV_CACHE_MEMORY_BYTES}" \
  --no-enable-prefix-caching \
  --additional-config "${KIMI_ADDITIONAL_CONFIG}" \
  "$@"
