#!/usr/bin/env bash
set -euo pipefail

: "${MODEL:?Set MODEL to the GLM-5.2 NF3 checkpoint path}"

MODE="${MODE:-stock}"
IMAGE="${IMAGE:-local/vllm:glm52-dcp-post35-poc}"
NAME="${NAME:-glm52-dcp-post35-${MODE}}"
PORT="${PORT:-5694}"
GPUS="${GPUS:-0,1,2,3}"
TP="${TP:-4}"
DCP="${DCP:-4}"
MTP="${MTP:-0}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
GRAPH="${GRAPH:-6}"
JIT_CACHE="${JIT_CACHE:-/root/.cache/vllm-glm52-release/v20-consolidated}"
TMP_ROOT="${TMP_ROOT:-/root/vllm/tmp}"

selected=0
remote=off
bulk=0
prefetch_depth=1
pool_records=0

case "${MODE}" in
  stock)
    ;;
  materialized)
    selected=1
    ;;
  materialized-bulk)
    selected=1
    bulk=1
    prefetch_depth=3
    pool_records=32768
    ;;
  remote-ce)
    selected=1
    remote=ce
    ;;
  remote-peer)
    selected=1
    remote=peer
    ;;
  remote-storage)
    selected=1
    remote=storage
    ;;
  *)
    printf 'unknown MODE=%s\n' "${MODE}" >&2
    exit 2
    ;;
esac

docker rm -f "${NAME}" >/dev/null 2>&1 || true
rm -rf "${TMP_ROOT:?}/${NAME}"
mkdir -p "${TMP_ROOT}/${NAME}"

docker run -d \
  --name "${NAME}" \
  --network host \
  --ipc host \
  --privileged \
  --init \
  --shm-size 32g \
  --gpus all \
  --ulimit memlock=-1:-1 \
  --ulimit stack=67108864:67108864 \
  --ulimit nofile=1048576:1048576 \
  --entrypoint /usr/local/bin/serve-gilded-gnosis.sh \
  -e MODEL_FAMILY=glm52-hybrid \
  -e GPUS="${GPUS}" \
  -e PORT="${PORT}" \
  -e MODEL="${MODEL}" \
  -e SERVED_MODEL_NAME=GLM-5.2-MXFP8-NVFP4-NF3-Hybrid \
  -e TP="${TP}" \
  -e DCP="${DCP}" \
  -e MTP="${MTP}" \
  -e MAX_NUM_SEQS="${MAX_NUM_SEQS}" \
  -e GRAPH="${GRAPH}" \
  -e MAX_MODEL_LEN=300000 \
  -e MAX_BATCHED_TOKENS=2048 \
  -e GPU_MEMORY_UTILIZATION=0.96 \
  -e MOE_MODE=a16 \
  -e MOE_BACKEND=b12x \
  -e LINEAR_BACKEND=auto \
  -e QUANTIZATION=nvfp4_nf3_hybrid \
  -e ONLINE_QUANT=nf3-mxfp8 \
  -e QUANTIZATION_CONFIG_JSON= \
  -e NF3_GRID188=1 \
  -e VLLM_B12X_ABSORB_BMM=0 \
  -e KV_CACHE_DTYPE=nvfp4_ds_mla \
  -e KV_FP8_ROPE=1 \
  -e LOAD_FORMAT=instanttensor \
  -e INSTANTTENSOR_BACKEND=BUFFERED \
  -e F8_DMA=0 \
  -e DCP_BACKEND=a2a \
  -e DCP_A2A_MAX_TOKENS=64 \
  -e DCP_A2A_LARGE_BACKEND=ag_rs \
  -e DCP_QUERY_SPLIT=1 \
  -e DCP_CKV_GATHER=1 \
  -e DCP_PREFILL_WORKSPACE=auto \
  -e VLLM_DCP_TOPK_OWNER_MERGE=1 \
  -e VLLM_DCP_TOPK_SPARKINFER=0 \
  -e VLLM_DCP_INDEXER_SHARDS=0 \
  -e VLLM_B12X_MLA_CKV_PREFETCH_DEPTH="${prefetch_depth}" \
  -e VLLM_B12X_MLA_CKV_PREFETCH_WORKSPACE_MIB=1024 \
  -e VLLM_B12X_MLA_SPARSE_DECODE_CKV_GATHER="${selected}" \
  -e VLLM_B12X_MLA_SPARSE_DECODE_TRANSPORT=ce \
  -e VLLM_B12X_MLA_SPARSE_DECODE_REMOTE_RECORDS="${remote}" \
  -e VLLM_B12X_MLA_SPARSE_DECODE_BULK_PREFETCH="${bulk}" \
  -e VLLM_B12X_MLA_SPARSE_DECODE_MAX_SEQS="${MAX_NUM_SEQS}" \
  -e VLLM_B12X_MLA_SPARSE_DECODE_POOL_RECORDS="${pool_records}" \
  -e HF_HUB_OFFLINE=1 \
  -e TRANSFORMERS_OFFLINE=1 \
  -v /root/.cache/huggingface:/root/.cache/huggingface \
  -v /root/models:/root/models:ro \
  -v "${JIT_CACHE}:/cache" \
  -v "${TMP_ROOT}/${NAME}:/container-tmp" \
  "${IMAGE}"

printf 'started %s on http://127.0.0.1:%s (MODE=%s)\n' \
  "${NAME}" "${PORT}" "${MODE}"
