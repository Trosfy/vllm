#!/usr/bin/env bash
# Fast-loading stock Kimi K3 profile. Requires the consumer-stream-safe
# InstantTensor wheel from voipmonitor/InstantTensor:dev/gg-k3-consumer-event.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# The stock checkpoint contains two 2.19-GiB BF16 tensors. InstantTensor
# streams all smaller weights through this ring and sends only those two
# through the CPU safetensors fallback, avoiding a loader-time GPU OOM.
export INSTANTTENSOR_BUFFER_SIZE="${INSTANTTENSOR_BUFFER_SIZE:-536870912}"

exec "${SCRIPT_DIR}/serve-kimi-k3.sh" --load-format instanttensor "$@"
