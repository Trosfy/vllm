# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

import vllm.envs as envs

_b12x_compile_only_warmup: ContextVar[bool] = ContextVar(
    "b12x_compile_only_warmup",
    default=False,
)


def is_b12x_compile_only_warmup() -> bool:
    """Whether the current forward may compile but must not launch B12X compute."""
    return _b12x_compile_only_warmup.get()


@contextmanager
def b12x_compile_only_warmup() -> Iterator[None]:
    """Traverse a model to resolve exact B12X bindings without running them."""
    token = _b12x_compile_only_warmup.set(True)
    try:
        yield
    finally:
        _b12x_compile_only_warmup.reset(token)


def b12x_cuda_graph_prewarm_enabled() -> bool:
    return (
        envs.VLLM_USE_B12X_SPARSE_INDEXER
        or envs.VLLM_USE_B12X_MHC
        or envs.VLLM_USE_B12X_FP8_GEMM
        or envs.VLLM_USE_B12X_WO_PROJECTION
        or envs.VLLM_USE_B12X_MOE
        or envs.VLLM_USE_B12X_MINIMAX_M3_MSA
    )


def b12x_cuda_graph_wrapper_prewarm_enabled(is_piecewise: bool) -> bool:
    if not b12x_cuda_graph_prewarm_enabled():
        return False
    if is_piecewise:
        return envs.VLLM_B12X_CUDAGRAPH_PIECEWISE_PREWARM
    return True


@contextmanager
def guard_b12x_kernel_resolution(reason: str) -> Iterator[None]:
    if not b12x_cuda_graph_prewarm_enabled():
        yield
        return

    try:
        from sparkinfer import (
            freeze_kernel_resolution,
            kernel_resolution_frozen,
            unfreeze_kernel_resolution,
        )
    except ImportError:
        yield
        return

    if kernel_resolution_frozen():
        yield
        return

    freeze_kernel_resolution(reason)
    try:
        yield
    finally:
        unfreeze_kernel_resolution()
