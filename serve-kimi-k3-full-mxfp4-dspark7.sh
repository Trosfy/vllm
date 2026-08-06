#!/usr/bin/env bash
# Full stock Kimi K3 MXFP4 plus the five-layer Inferact MLA-native DSpark
# draft (seven-token speculative block) on 16x RTX PRO 6000 Blackwell.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# Multiprocessing workers inherit sys.path from the launcher's working
# directory.  Always launch from this checkout so an image-baked vLLM tree
# cannot precede the selected source tree in forkserver children.
cd -- "${SCRIPT_DIR}"
DEFAULT_PYTHON_BIN="${SCRIPT_DIR}/.venv/bin/python"
if [[ ! -x "${DEFAULT_PYTHON_BIN}" && -x /opt/venv/bin/python ]]; then
  DEFAULT_PYTHON_BIN=/opt/venv/bin/python
fi
PYTHON_BIN="${PYTHON_BIN:-${DEFAULT_PYTHON_BIN}}"
SPARKINFER_DIR="${SPARKINFER_DIR:-/mnt/luke/sparkinfer-k3-hh-dense-mla-dcp8-latest}"
export PYTHON_BIN

if [[ -e "${SCRIPT_DIR}/vllm/_C_stable_libtorch.abi3.so" ]]; then
  export PYTHONPATH="${SCRIPT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
fi
if [[ ! -f "${SPARKINFER_DIR}/sparkinfer/attention/dense_mla/__init__.py" ]]; then
  echo "SparkInfer dense_mla source is missing: ${SPARKINFER_DIR}" >&2
  exit 1
fi
export PYTHONPATH="${SCRIPT_DIR}:${SPARKINFER_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

# The runtime image carries a conservative host-staging fallback. Kimi-K3
# M=8 verification executes roughly two NCCL reductions per target layer;
# forcing host staging makes that exact sequence about 2.4x slower on this
# validated TP16 topology. Keep a profile-specific escape hatch, and never
# pass an empty graph-file path to NCCL.
export NCCL_P2P_DISABLE="${KIMI_NCCL_P2P_DISABLE:-0}"
unset NCCL_GRAPH_FILE

TP_SIZE="${TP_SIZE:-16}"
DCP_SIZE="${DCP_SIZE:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-2048}"
KV_CACHE_MEMORY_BYTES="${KV_CACHE_MEMORY_BYTES:-500000000}"
DSPARK_DRAFT_KV_WINDOW="${DSPARK_DRAFT_KV_WINDOW:-0}"
DSPARK_DRAFT_WEIGHT_FORMAT="${DSPARK_DRAFT_WEIGHT_FORMAT:-bf16}"
DSPARK_DRAFT_MXFP8_BACKEND="${DSPARK_DRAFT_MXFP8_BACKEND:-marlin}"
DSPARK_SHARD_MARKOV_HEAD="${DSPARK_SHARD_MARKOV_HEAD:-0}"
DSPARK_REPLICATE_MARKOV_W1="${DSPARK_REPLICATE_MARKOV_W1:-0}"
DSPARK_B12X_ARGMAX="${DSPARK_B12X_ARGMAX:-1}"
DSPARK_CAPTURE_SHARDED_MARKOV="${DSPARK_CAPTURE_SHARDED_MARKOV:-0}"
# Model-free TP16 measurements show a small launch-level win for the composed
# B12X AR + vLLM RMSNorm path.  Keep it opt-in: the full v74 model run lost
# 0.59% target cycles/s because it removed useful NCCL/FlashInfer overlap.
DSPARK_PREFER_B12X_ALLREDUCE_RMS="${DSPARK_PREFER_B12X_ALLREDUCE_RMS:-0}"
KIMI_TARGET_MXFP8_PROFILE="${KIMI_TARGET_MXFP8_PROFILE:-none}"
NUM_SPECULATIVE_TOKENS="${NUM_SPECULATIVE_TOKENS:-7}"
DRAFT_ATTENTION_BACKEND="${DRAFT_ATTENTION_BACKEND:-B12X_MLA}"
DRAFT_SAMPLE_METHOD="${DRAFT_SAMPLE_METHOD:-greedy}"
REJECTION_SAMPLE_METHOD="${REJECTION_SAMPLE_METHOD:-block}"
DSPARK_SPS_CURVE="${DSPARK_SPS_CURVE:-}"
DSPARK_SPS_OVERHEAD_MS="${DSPARK_SPS_OVERHEAD_MS:-0.0}"
DSPARK_ADAPTIVE_SPECULATIVE_TOKENS_WINDOW="${DSPARK_ADAPTIVE_SPECULATIVE_TOKENS_WINDOW:-0}"
DSPARK_BATCH_SIZE_SPECULATIVE_SCHEDULE="${DSPARK_BATCH_SIZE_SPECULATIVE_SCHEDULE:-}"
DSPARK_CAPACITY_ACTIVATION_BATCH_SIZE="${DSPARK_CAPACITY_ACTIVATION_BATCH_SIZE:-0}"
DSPARK_PROFILE_SPS_ONLY="${DSPARK_PROFILE_SPS_ONLY:-0}"
if (( DCP_SIZE > 1 )); then
  DCP_COMM_BACKEND="${DCP_COMM_BACKEND:-a2a}"
else
  DCP_COMM_BACKEND="${DCP_COMM_BACKEND:-ag_rs}"
fi

if (( TP_SIZE != 16 )); then
  echo "This profile is validated only for TP_SIZE=16, got ${TP_SIZE}" >&2
  exit 2
fi
if (( DCP_SIZE != 1 && DCP_SIZE != 8 && DCP_SIZE != 16 )); then
  echo "This profile supports DCP_SIZE=1, 8, or 16, got ${DCP_SIZE}" >&2
  exit 2
fi
if (( TP_SIZE % DCP_SIZE != 0 )); then
  echo "DCP_SIZE=${DCP_SIZE} must divide TP_SIZE=${TP_SIZE}" >&2
  exit 2
fi
if (( DCP_SIZE > 1 )) && [[ "${DCP_COMM_BACKEND}" != "a2a" ]]; then
  echo "DSpark DCP requires DCP_COMM_BACKEND=a2a" >&2
  exit 2
fi
if (( NUM_SPECULATIVE_TOKENS != 7 )); then
  echo "Inferact/Kimi-K3-DSpark has a fixed seven-token block; got ${NUM_SPECULATIVE_TOKENS}" >&2
  exit 2
fi
if [[ "${DRAFT_ATTENTION_BACKEND}" != "B12X_MLA" ]]; then
  echo "This SM120 profile is validated only with DRAFT_ATTENTION_BACKEND=B12X_MLA" >&2
  exit 2
fi
case "${DRAFT_SAMPLE_METHOD}" in
  probabilistic | greedy) ;;
  *)
    echo "DRAFT_SAMPLE_METHOD must be probabilistic or greedy" >&2
    exit 2
    ;;
esac
case "${REJECTION_SAMPLE_METHOD}" in
  block | standard | synthetic) ;;
  *)
    echo "Unsupported REJECTION_SAMPLE_METHOD=${REJECTION_SAMPLE_METHOD}" >&2
    exit 2
    ;;
esac
if [[ ! "${DSPARK_ADAPTIVE_SPECULATIVE_TOKENS_WINDOW}" =~ ^[0-9]+$ ]]; then
  echo "DSPARK_ADAPTIVE_SPECULATIVE_TOKENS_WINDOW must be a non-negative integer" >&2
  exit 2
fi
if [[ ! "${DSPARK_CAPACITY_ACTIVATION_BATCH_SIZE}" =~ ^[0-9]+$ ]]; then
  echo "DSPARK_CAPACITY_ACTIVATION_BATCH_SIZE must be a non-negative integer" >&2
  exit 2
fi
case "${DSPARK_PROFILE_SPS_ONLY}" in
  0 | 1) ;;
  *)
    echo "DSPARK_PROFILE_SPS_ONLY must be 0 or 1" >&2
    exit 2
    ;;
esac
if [[ ! "${KV_CACHE_MEMORY_BYTES}" =~ ^[1-9][0-9]*$ ]]; then
  echo "KV_CACHE_MEMORY_BYTES must be a positive integer" >&2
  exit 2
fi
if [[ ! "${DSPARK_DRAFT_KV_WINDOW}" =~ ^[0-9]+$ ]]; then
  echo "DSPARK_DRAFT_KV_WINDOW must be a non-negative integer" >&2
  exit 2
fi
case "${DSPARK_DRAFT_WEIGHT_FORMAT}" in
  bf16 | mxfp8) ;;
  *)
    echo "DSPARK_DRAFT_WEIGHT_FORMAT must be bf16 or mxfp8" >&2
    exit 2
    ;;
esac
case "${DSPARK_DRAFT_MXFP8_BACKEND}" in
  auto | marlin) ;;
  *)
    echo "DSPARK_DRAFT_MXFP8_BACKEND must be auto or marlin" >&2
    exit 2
    ;;
esac
case "${DSPARK_SHARD_MARKOV_HEAD}" in
  0 | 1) ;;
  *)
    echo "DSPARK_SHARD_MARKOV_HEAD must be 0 or 1" >&2
    exit 2
    ;;
esac
case "${DSPARK_REPLICATE_MARKOV_W1}" in
  0 | 1) ;;
  *)
    echo "DSPARK_REPLICATE_MARKOV_W1 must be 0 or 1" >&2
    exit 2
    ;;
esac
case "${DSPARK_B12X_ARGMAX}" in
  0 | 1) ;;
  *)
    echo "DSPARK_B12X_ARGMAX must be 0 or 1" >&2
    exit 2
    ;;
esac
case "${DSPARK_CAPTURE_SHARDED_MARKOV}" in
  0 | 1) ;;
  *)
    echo "DSPARK_CAPTURE_SHARDED_MARKOV must be 0 or 1" >&2
    exit 2
    ;;
esac
case "${DSPARK_PREFER_B12X_ALLREDUCE_RMS}" in
  0 | 1) ;;
  *)
    echo "DSPARK_PREFER_B12X_ALLREDUCE_RMS must be 0 or 1" >&2
    exit 2
    ;;
esac
if [[ "${DSPARK_REPLICATE_MARKOV_W1}" == 1 && "${DSPARK_SHARD_MARKOV_HEAD}" != 1 ]]; then
  echo "DSPARK_REPLICATE_MARKOV_W1=1 requires DSPARK_SHARD_MARKOV_HEAD=1" >&2
  exit 2
fi
if [[ "${DSPARK_CAPTURE_SHARDED_MARKOV}" == 1 ]]; then
  if [[ "${DSPARK_SHARD_MARKOV_HEAD}" != 1 ]]; then
    echo "DSPARK_CAPTURE_SHARDED_MARKOV=1 requires DSPARK_SHARD_MARKOV_HEAD=1" >&2
    exit 2
  fi
  if [[ "${DSPARK_B12X_ARGMAX}" != 1 ]]; then
    echo "DSPARK_CAPTURE_SHARDED_MARKOV=1 requires DSPARK_B12X_ARGMAX=1" >&2
    exit 2
  fi
  if [[ "${KIMI_DSPARK_PCIE_ALLREDUCE_BACKEND:-b12x}" != b12x ]]; then
    echo "Captured sharded Markov W1 requires the b12x all-reduce backend" >&2
    exit 2
  fi
fi
case "${KIMI_TARGET_MXFP8_PROFILE}" in
  none | shared_experts | kda_in_proj | attention_o_proj | kda_in_and_o_proj) ;;
  *)
    echo "Unsupported KIMI_TARGET_MXFP8_PROFILE=${KIMI_TARGET_MXFP8_PROFILE}" >&2
    exit 2
    ;;
esac
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python interpreter not found or not executable: ${PYTHON_BIN}" >&2
  exit 1
fi

export MODEL="${MODEL:-/root/.cache/huggingface/hub/models--moonshotai--Kimi-K3/snapshots/2496450e92e425c886db095102a52a6682ca3970}"
export DRAFT_MODEL="${DRAFT_MODEL:-/root/.cache/huggingface/hub/models--Inferact--Kimi-K3-DSpark/snapshots/cf6b8244620e7ea4b0651d214f28e89eac75bed6}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Kimi-K3-MXFP4-HH-DSpark7}"
export TP_SIZE DCP_SIZE MAX_MODEL_LEN MAX_NUM_SEQS MAX_NUM_BATCHED_TOKENS
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.985}"

if [[ ! -f "${MODEL}/model.safetensors.index.json" ]]; then
  echo "Kimi K3 target checkpoint is incomplete: ${MODEL}" >&2
  exit 1
fi
if [[ ! -f "${DRAFT_MODEL}/model.safetensors" || ! -f "${DRAFT_MODEL}/config.json" ]]; then
  echo "Kimi K3 DSpark checkpoint is incomplete: ${DRAFT_MODEL}" >&2
  exit 1
fi

# These are the same lossless BF16 projection shards used by the validated
# full-MXFP4 target. KDA f_a is enabled through additional_config below.
export VLLM_KIMI_SHARD_QKV_A="${VLLM_KIMI_SHARD_QKV_A:-1}"
export VLLM_KIMI_SHARD_ROUTED_DOWN_PROJ="${VLLM_KIMI_SHARD_ROUTED_DOWN_PROJ:-1}"
export VLLM_KIMI_SHARD_ROUTED_UP_PROJ="${VLLM_KIMI_SHARD_ROUTED_UP_PROJ:-1}"
export VLLM_KIMI_SHARD_ROUTER="${VLLM_KIMI_SHARD_ROUTER:-1}"

export VLLM_ENABLE_PCIE_ALLREDUCE="${VLLM_ENABLE_PCIE_ALLREDUCE:-1}"
# The base Docker image exports backend=cpp. TP16 is unsupported by that
# legacy C++ implementation, so select the validated B12X hierarchical path
# through a profile-specific override instead of inheriting the image value.
export VLLM_PCIE_ALLREDUCE_BACKEND="${KIMI_DSPARK_PCIE_ALLREDUCE_BACKEND:-b12x}"
export VLLM_PCIE_ONESHOT_SINGLE_CHANNEL="${KIMI_DSPARK_PCIE_ONESHOT_SINGLE_CHANNEL:-1}"
if (( DCP_SIZE > 1 )); then
  export VLLM_USE_B12X_DCP_A2A=1
  # A single request produces at most eight MLA rows while verifying the
  # trained seven-token block. Larger prefill batches fall back to NCCL and
  # do not reserve oversized eager/graph PCIe staging slabs.
  export VLLM_DCP_A2A_MAX_TOKENS="${VLLM_DCP_A2A_MAX_TOKENS:-8}"
  # Keep SparkInfer's low-latency A2A for decode, but avoid the large hidden
  # ProcessGroupNCCL allocation when a prefill chunk exceeds the B12X cap.
  export VLLM_DCP_A2A_LARGE_BACKEND="${VLLM_DCP_A2A_LARGE_BACKEND:-ag_rs}"
else
  export VLLM_USE_B12X_DCP_A2A=0
fi
# Keep the external DSpark draft on its native DCP1 layout.  The target still
# runs with DCP8, but the five draft layers keep a complete KV sequence on each
# TP rank.  This is vLLM's intentional default for external drafts; forcing the
# draft itself through DCP8 destroys acceptance because its cached context no
# longer matches the target-derived hidden-state stream.
export VLLM_DCP_SHARD_DRAFT="${VLLM_DCP_SHARD_DRAFT:-0}"
export VLLM_DSPARK_DRAFT_KV_WINDOW="${VLLM_DSPARK_DRAFT_KV_WINDOW:-${DSPARK_DRAFT_KV_WINDOW}}"
export VLLM_DSPARK_SHARD_MARKOV_HEAD="${VLLM_DSPARK_SHARD_MARKOV_HEAD:-${DSPARK_SHARD_MARKOV_HEAD}}"
export VLLM_DSPARK_REPLICATE_MARKOV_W1="${VLLM_DSPARK_REPLICATE_MARKOV_W1:-${DSPARK_REPLICATE_MARKOV_W1}}"
export VLLM_KIMI_K3_B12X_DSPARK_ARGMAX="${VLLM_KIMI_K3_B12X_DSPARK_ARGMAX:-${DSPARK_B12X_ARGMAX}}"
export VLLM_DSPARK_CAPTURE_SHARDED_MARKOV="${VLLM_DSPARK_CAPTURE_SHARDED_MARKOV:-${DSPARK_CAPTURE_SHARDED_MARKOV}}"
export VLLM_DSPARK_PREFER_B12X_ALLREDUCE_RMS="${VLLM_DSPARK_PREFER_B12X_ALLREDUCE_RMS:-${DSPARK_PREFER_B12X_ALLREDUCE_RMS}}"
# One 384-thread CTA per verification row is the measured TP16 optimum for
# the exact fused paired projection gather plus K3 sigmoid top-k path.
export SPARKINFER_PCIE_KIMI_TOPK_THREADS="${SPARKINFER_PCIE_KIMI_TOPK_THREADS:-384}"
if [[ "${VLLM_DSPARK_PREFER_B12X_ALLREDUCE_RMS}" == 1 ]]; then
  # The fixed DSpark verification batch is [8, 7168] BF16 = 112 KiB. Reserve
  # that B12X capacity only for the explicit composed-collective experiment.
  export VLLM_PCIE_ONESHOT_FUSED_ADD_RMS_NORM_MAX_SIZE="${VLLM_PCIE_ONESHOT_FUSED_ADD_RMS_NORM_MAX_SIZE:-112KB}"
fi
export VLLM_DSPARK_CAPACITY_ACTIVATION_BATCH_SIZE="${VLLM_DSPARK_CAPACITY_ACTIVATION_BATCH_SIZE:-${DSPARK_CAPACITY_ACTIVATION_BATCH_SIZE}}"
export VLLM_DSPARK_PROFILE_SPS_ONLY="${VLLM_DSPARK_PROFILE_SPS_ONLY:-${DSPARK_PROFILE_SPS_ONLY}}"

# At DSpark's fixed eight-row forward, FlashInfer's dynamic W8A8 MXFP8 path
# measures 7-14x slower than BF16 because every small projection requantizes
# its activation. Marlin consumes the same block-32 MXFP8 weights as W8A16;
# our shape harness measures only ~2x per GEMM, or roughly 0.3 ms for the
# complete five-layer draft.  The optional target shared-expert overlay uses
# the same W8A16 backend; routed MXFP4 experts and every retained BF16 target
# projection keep their original kernel selection.
if [[ ( "${DSPARK_DRAFT_WEIGHT_FORMAT}" == mxfp8 || "${KIMI_TARGET_MXFP8_PROFILE}" != none ) && "${DSPARK_DRAFT_MXFP8_BACKEND}" == marlin ]]; then
  for kernel in \
    B12xMxfp8LinearKernel \
    FlashInferCutedslMxfp8LinearKernel \
    FlashInferCutlassMxfp8LinearKernel; do
    case ",${VLLM_DISABLED_KERNELS:-}," in
      *,"${kernel}",*) ;;
      *) VLLM_DISABLED_KERNELS="${VLLM_DISABLED_KERNELS:+${VLLM_DISABLED_KERNELS},}${kernel}" ;;
    esac
  done
  export VLLM_DISABLED_KERNELS
fi

# K3 has no sparse indexer. These GLM-specific sparse/DCP policies remain
# disabled; dense target and draft MLA use the B12X DCP all-to-all path.
export VLLM_DCP_INDEXER_SHARDS=0
export VLLM_DCP_QUERY_SPLIT=0
export VLLM_DCP_GLOBAL_TOPK=0
export VLLM_DCP_PROJECT_BEFORE_MERGE=0

export KDA_PREFILL_BACKEND="${KDA_PREFILL_BACKEND:-flashkda}"
export CUDA_MODULE_LOADING="${CUDA_MODULE_LOADING:-LAZY}"
export CUDA_MODULE_DATA_LOADING="${CUDA_MODULE_DATA_LOADING:-LAZY}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export VLLM_MLA_CHUNKED_PREFILL_WORKSPACE_SIZE="${VLLM_MLA_CHUNKED_PREFILL_WORKSPACE_SIZE:-8192}"

# Manual cache sizing and measured [1,8] graphs are more accurate than the
# conservative graph reservation for this extremely tight target+draft fit.
# At the validated 500 MB/rank setting this creates 8,894 physical KV tokens,
# enough for max_model_len=8,192 with one request.
export VLLM_MEMORY_PROFILE_INCLUDE_ATTN="${VLLM_MEMORY_PROFILE_INCLUDE_ATTN:-0}"
export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS="${VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS:-0}"
if [[ -z "${COMPILATION_CONFIG:-}" ]]; then
  export COMPILATION_CONFIG='{"mode":0,"cudagraph_mode":"PIECEWISE","cudagraph_capture_sizes":[1,8],"pass_config":{"fuse_allreduce_rms":true}}'
fi

# Fail before loading 1.4 TiB of target weights if either the draft contract or
# the native K3 runtime is missing.
export KDA_PREFILL_BACKEND DSPARK_DRAFT_WEIGHT_FORMAT DSPARK_DRAFT_MXFP8_BACKEND
export KIMI_TARGET_MXFP8_PROFILE VLLM_DSPARK_SHARD_MARKOV_HEAD
export VLLM_DSPARK_REPLICATE_MARKOV_W1
export VLLM_KIMI_K3_B12X_DSPARK_ARGMAX
"${PYTHON_BIN}" - <<'PY'
import json
import os
from pathlib import Path

from sparkinfer.attention import dense_mla
from vllm.config import LoadConfig, ModelConfig, replace
from vllm.config.quantization import QuantizationConfigArgs, resolve_quantization_config
from vllm.model_executor.layers.quantization.online.base import (
    OnlineQuantizationConfig,
)
from vllm.model_executor.model_loader.weight_utils import get_quant_config
from vllm.model_executor.models.registry import ModelRegistry
from vllm.model_executor.kernels.linear import init_mxfp8_linear_kernel
from vllm.model_executor.layers.activation import ensure_kimi_k3_activation_ops
from vllm.models.kimi_k3.nvidia.kda import ensure_fused_kda_decode_op
from vllm.models.kimi_k3.nvidia.ops.fused_mla_key_concat_kv_cache import (
    ensure_kimi_k3_cache_ops,
)
from vllm.transformers_utils.config import get_config
from vllm.v1.attention.backends.mla.b12x_mla import (
    B12xMLABackend,
    B12xMLAMetadataBuilder,
    _kernel_query_heads,
)
from vllm.platforms.interface import DeviceCapability

draft = Path(os.environ["DRAFT_MODEL"])
raw = json.loads((draft / "config.json").read_text())
assert raw["architectures"] == ["K3DSparkModel"]
assert raw["model_type"] == "k3_dspark"
assert raw["num_hidden_layers"] == 5
assert raw["target_layer_ids"] == [2, 23, 47, 71, 89]
assert raw["max_position_embeddings"] >= int(os.environ["MAX_MODEL_LEN"])
config = get_config(str(draft), trust_remote_code=False)
assert type(config).__name__ == "K3DSparkConfig"
assert ModelRegistry._try_load_model_cls("K3DSparkModel").__name__ == "K3DSparkForCausalLM"
print(f"K3 DSpark preflight: {draft} ({type(config).__name__})", flush=True)

required = ("Caps", "plan", "bind", "compile", "run")
missing = [name for name in required if not hasattr(dense_mla, name)]
if missing:
    raise RuntimeError(f"incomplete sparkinfer.attention.dense_mla: {missing}")
print(f"SparkInfer dense MLA preflight: {dense_mla.__file__}", flush=True)
assert B12xMLABackend.supports_compute_capability(DeviceCapability(12, 0))
assert B12xMLABackend.supports_block_size(944)
assert B12xMLABackend.supports_non_causal()
assert B12xMLAMetadataBuilder.supports_non_causal_multi_token_decode
assert _kernel_query_heads(6, 1) == 8  # full-K3 target at TP16
assert _kernel_query_heads(4, 1) == 8  # Inferact draft at TP16
print("SparkInfer target/draft TP16 MLA contract: OK", flush=True)
ensure_kimi_k3_cache_ops()
if not ensure_fused_kda_decode_op():
    raise RuntimeError("HH Kimi-K3 fused KDA decode op is unavailable")
if os.environ["KDA_PREFILL_BACKEND"] == "flashkda":
    import vllm._flashkda_C  # noqa: F401
if not ensure_kimi_k3_activation_ops():
    raise RuntimeError("HH Kimi-K3 fused SiTU activation ops are unavailable")
print("K3 target native-op preflight: OK", flush=True)
target_mxfp8_profile = os.environ["KIMI_TARGET_MXFP8_PROFILE"]
target_profiles = {
    "shared_experts": {
        "linear": "mxfp8",
        "shared_experts": "mxfp8",
        "ignore": ["re:^(?!.*shared_experts).*$"],
    },
    "kda_in_proj": {
        "linear": "mxfp8",
        "ignore": [
            "re:^(?!.*self_attn\\.(?:q_proj|k_proj|v_proj|b_proj|f_a_proj)$).*$",
        ],
    },
    "attention_o_proj": {
        "linear": "mxfp8",
        "ignore": ["re:^(?!.*self_attn\\.o_proj$).*$"],
    },
    "kda_in_and_o_proj": {
        "linear": "mxfp8",
        "ignore": [
            "re:^(?!.*self_attn\\.(?:q_proj|k_proj|v_proj|b_proj|f_a_proj|o_proj)$).*$",
        ],
    },
}
if target_mxfp8_profile != "none":
    target_overlay = resolve_quantization_config(
        "mxfp4",
        target_profiles[target_mxfp8_profile],
    )
    if not isinstance(target_overlay, QuantizationConfigArgs):
        raise RuntimeError("target selective MXFP8 overlay did not resolve")
    assert target_overlay.linear is not None
    print(
        f"K3 target online MXFP8 preflight: {target_mxfp8_profile}",
        flush=True,
    )
needs_online_mxfp8 = (
    os.environ["DSPARK_DRAFT_WEIGHT_FORMAT"] == "mxfp8"
    or target_mxfp8_profile != "none"
)
if needs_online_mxfp8:
    kernel = init_mxfp8_linear_kernel()
    selected = type(kernel).__name__
    requested = os.environ["DSPARK_DRAFT_MXFP8_BACKEND"]
    if requested == "marlin" and selected != "MarlinMxfp8LinearKernel":
        raise RuntimeError(
            f"requested online MXFP8/Marlin, selected {selected} instead"
        )

if os.environ["VLLM_DSPARK_SHARD_MARKOV_HEAD"] == "1":
    if os.environ["VLLM_DSPARK_REPLICATE_MARKOV_W1"] == "1":
        print(
            "K3 DSpark Markov preflight: replicated-W1/sharded-W2 BF16",
            flush=True,
        )
    else:
        print("K3 DSpark Markov preflight: TP-sharded BF16", flush=True)
else:
    print("K3 DSpark Markov preflight: replicated", flush=True)

if os.environ["DSPARK_DRAFT_WEIGHT_FORMAT"] == "mxfp8":
    # Resolve the same online-quantized draft ModelConfig used by the worker.
    # In particular, exercise callable hf_overrides here so a config-regression
    # fails before the multi-terabyte target checkpoint is loaded.
    draft_model_config = ModelConfig(
        model=str(draft),
        runner="draft",
        tokenizer_mode="skip",
        max_model_len=int(os.environ["MAX_MODEL_LEN"]),
        quantization="mxfp8",
        quantization_config=resolve_quantization_config(
            "mxfp8",
            {
                "linear": "mxfp8",
                "ignore": [
                    "re:.*fused_qkv_a_proj$",
                ],
            },
        ),
        hf_overrides=lambda hf_config: hf_config,
    )
    draft_quant_config = get_quant_config(draft_model_config, LoadConfig())
    if not isinstance(draft_quant_config, OnlineQuantizationConfig):
        raise RuntimeError(
            "draft MXFP8 resolved to "
            f"{type(draft_quant_config).__name__}, expected OnlineQuantizationConfig"
        )
    draft_window = int(os.environ["VLLM_DSPARK_DRAFT_KV_WINDOW"])
    if draft_window:
        bounded_draft_config = replace(
            draft_model_config,
            max_model_len=draft_window + 768 - 1,
        )
        assert bounded_draft_config.max_model_len == draft_window + 768 - 1
    print(f"K3 DSpark MXFP8 kernel preflight: {selected}", flush=True)
PY

if [[ "${KIMI_DSPARK_PREFLIGHT_ONLY:-0}" == 1 ]]; then
  exit 0
fi

if [[ "${DSPARK_DRAFT_WEIGHT_FORMAT}" == mxfp8 ]]; then
  # The five TP-sharded transformer layers and context projection are
  # quantized online and execute through the W8A16 backend selected above.
  # Keep qkv-a in BF16 so the cross-layer KV-only context fusion remains
  # active. The replicated Markov embedding and vocabulary projection use
  # online MXFP8 too; this saves 77.5 MiB/rank without changing target logits.
  DRAFT_QUANT_JSON=',"quantization":"mxfp8","quantization_config":{"linear":"mxfp8","ignore":["re:.*fused_qkv_a_proj$"]}'
else
  DRAFT_QUANT_JSON=''
fi
DSPARK_RUNTIME_JSON=""
if [[ -n "${DSPARK_SPS_CURVE}" ]]; then
  if [[ "${DSPARK_SPS_CURVE}" == auto ]]; then
    DSPARK_RUNTIME_JSON+=',"dspark_sps_curve":"auto"'
  else
    DSPARK_RUNTIME_JSON+=',"dspark_sps_curve":'"${DSPARK_SPS_CURVE}"
  fi
  DSPARK_RUNTIME_JSON+=',"dspark_sps_overhead_ms":'"${DSPARK_SPS_OVERHEAD_MS}"
fi
if (( DSPARK_ADAPTIVE_SPECULATIVE_TOKENS_WINDOW > 0 )); then
  DSPARK_RUNTIME_JSON+=',"adaptive_speculative_tokens_window":'"${DSPARK_ADAPTIVE_SPECULATIVE_TOKENS_WINDOW}"
fi
if [[ -n "${DSPARK_BATCH_SIZE_SPECULATIVE_SCHEDULE}" ]]; then
  DSPARK_RUNTIME_JSON+=',"num_speculative_tokens_per_batch_size":'"${DSPARK_BATCH_SIZE_SPECULATIVE_SCHEDULE}"
fi
printf -v SPECULATIVE_CONFIG \
  '{"method":"dspark","model":"%s","num_speculative_tokens":7,"attention_backend":"%s","kv_cache_dtype":"fp8","draft_sample_method":"%s","rejection_sample_method":"%s"%s%s}' \
  "${DRAFT_MODEL}" "${DRAFT_ATTENTION_BACKEND}" \
  "${DRAFT_SAMPLE_METHOD}" "${REJECTION_SAMPLE_METHOD}" \
  "${DRAFT_QUANT_JSON}" "${DSPARK_RUNTIME_JSON}"

TARGET_QUANT_ARGS=()
case "${KIMI_TARGET_MXFP8_PROFILE}" in
  none) ;;
  shared_experts)
    TARGET_QUANT_JSON='{"linear":"mxfp8","shared_experts":"mxfp8","ignore":["re:^(?!.*shared_experts).*$"]}'
    ;;
  kda_in_proj)
    TARGET_QUANT_JSON='{"linear":"mxfp8","ignore":["re:^(?!.*self_attn\\.(?:q_proj|k_proj|v_proj|b_proj|f_a_proj)$).*$"]}'
    ;;
  attention_o_proj)
    TARGET_QUANT_JSON='{"linear":"mxfp8","ignore":["re:^(?!.*self_attn\\.o_proj$).*$"]}'
    ;;
  kda_in_and_o_proj)
    TARGET_QUANT_JSON='{"linear":"mxfp8","ignore":["re:^(?!.*self_attn\\.(?:q_proj|k_proj|v_proj|b_proj|f_a_proj|o_proj)$).*$"]}'
    ;;
esac
if [[ -n "${TARGET_QUANT_JSON:-}" ]]; then
  TARGET_QUANT_ARGS+=(--quantization-config "${TARGET_QUANT_JSON}")
fi

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
  --additional-config '{"kda_shard_f_a":true}' \
  --speculative-config "${SPECULATIVE_CONFIG}" \
  "${TARGET_QUANT_ARGS[@]}" \
  "$@"
