# SPDX-License-Identifier: Apache-2.0
"""Memory-bounded TP reductions for full-width Kimi-K3 projections."""

import torch

from vllm.distributed import (
    tensor_model_parallel_all_reduce,
    tensor_model_parallel_all_reduce_in_place,
)

_KIMI_OUTPUT_BUFFER_REUSE_MIN_TOKENS = 1024


def should_reuse_kimi_full_width_output(output_buffer: torch.Tensor) -> bool:
    """Whether a full-width Kimi buffer is large enough to donate safely.

    Decode-sized projections stay on their original backend/cudagraph path.
    During chunked prefill the normalized layer input is dead by the time the
    final row-parallel projection runs, and reusing it avoids a 28 MiB
    allocation for Kimi-K3's [2048, 7168] output.
    """
    return output_buffer.ndim >= 2 and output_buffer.shape[0] >= (
        _KIMI_OUTPUT_BUFFER_REUSE_MIN_TOKENS
    )


def reduce_kimi_full_width_output(
    output: torch.Tensor,
    tp_size: int,
) -> torch.Tensor:
    """Reduce a dead rank-local projection output without a full-size copy."""
    if tp_size <= 1:
        return output
    if should_reuse_kimi_full_width_output(output):
        # Large messages fall through TP16's graph-oriented custom AR to NCCL.
        # The rank-local result is dead, so let NCCL overwrite it instead of
        # allocating another 28 MiB [2048, 7168] output tensor.
        return tensor_model_parallel_all_reduce_in_place(output)
    # Preserve the existing custom/cudagraph path for decode-sized tensors.
    return tensor_model_parallel_all_reduce(output)
