# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Storage helpers for CUDA library operations with bounded read-ahead."""

from __future__ import annotations

import math

import torch

CUBLAS_BMM_TAIL_PADDING_BYTES = 64 * 1024


def tail_padded_empty(
    shape: tuple[int, ...],
    *,
    device: torch.device,
    dtype: torch.dtype,
    tail_padding_bytes: int = CUBLAS_BMM_TAIL_PADDING_BYTES,
) -> torch.Tensor:
    """Allocate a contiguous tensor with live storage after its last item."""
    if tail_padding_bytes < 0:
        raise ValueError("tail_padding_bytes must be non-negative")
    numel = math.prod(shape)
    pad_numel = math.ceil(tail_padding_bytes / dtype.itemsize)
    storage = torch.empty(numel + pad_numel, device=device, dtype=dtype)
    return storage[:numel].view(shape)


def storage_tail_bytes(tensor: torch.Tensor) -> int:
    """Return storage bytes following the tensor's highest addressed item."""
    if tensor.numel() == 0:
        end = tensor.storage_offset() * tensor.element_size()
    else:
        if any(stride < 0 for stride in tensor.stride()):
            return 0
        last_item = tensor.storage_offset() + sum(
            (size - 1) * stride
            for size, stride in zip(tensor.shape, tensor.stride(), strict=True)
        )
        end = (last_item + 1) * tensor.element_size()
    return max(0, tensor.untyped_storage().nbytes() - end)


def ensure_cublas_tail_padding(
    tensor: torch.Tensor,
    tail_padding_bytes: int = CUBLAS_BMM_TAIL_PADDING_BYTES,
) -> torch.Tensor:
    """Ensure a downstream cuBLAS BMM can perform bounded mapped read-ahead."""
    if storage_tail_bytes(tensor) >= tail_padding_bytes:
        return tensor
    output = tail_padded_empty(
        tuple(tensor.shape),
        device=tensor.device,
        dtype=tensor.dtype,
        tail_padding_bytes=tail_padding_bytes,
    )
    output.copy_(tensor)
    return output
