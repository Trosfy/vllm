# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline tests for the opt-in fp8 w8a16 lm_head (lm_head_quant override).

No GPU, no checkpoint: a small ParallelLMHead is built directly (mirroring
test_ple_mmap.py's construction), a synthetic bf16 weight is streamed through
the online-processing weight loader, and the CPU dequant fallback of
Qwen4ExpLMHeadFp8Method.apply is checked against an fp32 reference.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import vllm.model_executor.layers.vocab_parallel_embedding as embedding_module
import vllm.model_executor.parameter as parameter_module
from vllm.config import CompilationConfig, set_current_vllm_config
from vllm.model_executor.kernels.linear.scaled_mm.humming import (
    HummingFP8ScaledMMLinearKernel,
)
from vllm.model_executor.kernels.linear.scaled_mm.marlin import (
    MarlinFP8ScaledMMLinearKernel,
)
from vllm.model_executor.kernels.linear.scaled_mm.ScaledMMLinearKernel import (
    FP8ScaledMMLinearLayerConfig,
)
from vllm.model_executor.layers.quantization.utils.humming_utils import (
    convert_linear_layer_to_humming_standard,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    get_fp8_min_max,
    kFp8DynamicTensorSym,
    kFp8StaticTensorSym,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    UnquantizedEmbeddingMethod,
)
from vllm.models.qwen4_exp.nvidia.lm_head_fp8 import (
    Qwen4ExpLMHeadFp8Method,
    get_lm_head_quant_method,
)
from vllm.utils.import_utils import has_humming

VOCAB = 128
HIDDEN = 32


@pytest.fixture(autouse=True)
def _allow_single_rank_tensor_parallel(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stand in a rank-0/size-1 TP world (mirrors test_ple_mmap.py)."""
    monkeypatch.setattr(embedding_module, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        embedding_module, "get_tensor_model_parallel_world_size", lambda: 1
    )
    monkeypatch.setattr(parameter_module, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        parameter_module, "get_tensor_model_parallel_world_size", lambda: 1
    )


def _vllm_config(
    lm_head_quant: str | None = None,
    tie_word_embeddings: bool = False,
    head_dtype: torch.dtype = torch.bfloat16,
) -> SimpleNamespace:
    hf_attrs = {} if lm_head_quant is None else {"lm_head_quant": lm_head_quant}
    return SimpleNamespace(
        model_config=SimpleNamespace(
            dtype=torch.bfloat16,
            head_dtype=head_dtype,
            hf_config=SimpleNamespace(**hf_attrs),
            hf_text_config=SimpleNamespace(
                tie_word_embeddings=tie_word_embeddings, **hf_attrs
            ),
        ),
        # Read by the wfp8-a16 kernel selection during create_weights.
        kernel_config=SimpleNamespace(linear_backend="auto"),
        compilation_config=CompilationConfig(custom_ops=["none"]),
    )


def test_helper_returns_none_without_override() -> None:
    assert get_lm_head_quant_method(_vllm_config()) is None


def test_helper_rejects_unknown_value() -> None:
    with pytest.raises(ValueError, match="lm_head_quant"):
        get_lm_head_quant_method(_vllm_config(lm_head_quant="int4"))


def test_helper_rejects_tied_embeddings() -> None:
    with pytest.raises(NotImplementedError, match="tie_word_embeddings"):
        get_lm_head_quant_method(
            _vllm_config(lm_head_quant="fp8", tie_word_embeddings=True)
        )


def test_helper_rejects_head_dtype_override() -> None:
    with pytest.raises(ValueError, match="head_dtype"):
        get_lm_head_quant_method(
            _vllm_config(lm_head_quant="fp8", head_dtype=torch.float32)
        )


def test_helper_selects_fp8_method() -> None:
    vllm_config = _vllm_config(lm_head_quant="fp8")
    with set_current_vllm_config(vllm_config):
        method = get_lm_head_quant_method(vllm_config)
    assert isinstance(method, Qwen4ExpLMHeadFp8Method)


def _build_lm_head(quant_method) -> ParallelLMHead:
    return ParallelLMHead(
        VOCAB,
        HIDDEN,
        params_dtype=torch.bfloat16,
        prefix="lm_head",
        quant_method=quant_method,
    )


def test_default_off_is_unquantized() -> None:
    head = _build_lm_head(quant_method=None)
    assert isinstance(head.quant_method, UnquantizedEmbeddingMethod)
    assert head.weight.dtype == torch.bfloat16


def test_online_quant_matches_dequant_reference() -> None:
    vllm_config = _vllm_config(lm_head_quant="fp8")
    with set_current_vllm_config(vllm_config):
        method = get_lm_head_quant_method(vllm_config)
        head = _build_lm_head(quant_method=method)
    assert head.quant_method is method
    assert head.weight.device.type == "meta"

    torch.manual_seed(7)
    checkpoint_weight = torch.randn(VOCAB, HIDDEN, dtype=torch.bfloat16)
    head.weight.weight_loader(head.weight, checkpoint_weight)
    method.process_weights_after_loading(head)

    fp8_max = get_fp8_min_max()[1]
    expected_scale = (
        checkpoint_weight.abs().amax().to(torch.float32).reshape(1) / fp8_max
    )
    assert head.weight.dtype == torch.float8_e4m3fn
    assert head.weight.shape == (HIDDEN, VOCAB)  # canonical (K, N)
    assert torch.allclose(head.weight_scale, expected_scale)

    # Idempotent on a second call (weight-reload guard).
    method.process_weights_after_loading(head)
    assert head.weight.dtype == torch.float8_e4m3fn

    x = torch.randn(3, HIDDEN, dtype=torch.bfloat16)
    logits = method.apply(head, x)
    reference = torch.nn.functional.linear(
        x.to(torch.float32),
        checkpoint_weight.to(torch.float32),
    )
    assert logits.shape == (3, VOCAB)
    # fp8 per-tensor round-trip tolerance: the quant error dominates.
    torch.testing.assert_close(
        logits.to(torch.float32), reference, atol=0.35, rtol=0.06
    )


def test_processed_weight_survives_humming_layout_prep() -> None:
    """Regression test for a live-serve crash.

    process_weights_after_loading() canonicalizes the weight to (K, N) via
    ``.t()``, which is a non-contiguous view (stride(-1) == HIDDEN, not 1).
    Humming's own prep, convert_linear_layer_to_humming_standard(), reads the
    weight's ``input_dim``/``output_dim`` tags to decide whether to
    transpose-and-contiguous it back before a dtype-reinterpret
    ``view(int32)`` cast, which requires a contiguous last dim; Marlin's prep
    tolerates the view directly and never caught a missing/wrong tag. Without
    the tags this raises ``RuntimeError: self.stride(-1) must be 1 to view
    Float8_e4m3fn as Int``. Exercises the real prep function directly (pure
    tensor ops, no CUDA/Humming runtime needed) so this is CPU-safe.
    """
    vllm_config = _vllm_config(lm_head_quant="fp8")
    with set_current_vllm_config(vllm_config):
        method = get_lm_head_quant_method(vllm_config)
        head = _build_lm_head(quant_method=method)

    torch.manual_seed(11)
    checkpoint_weight = torch.randn(VOCAB, HIDDEN, dtype=torch.bfloat16)
    head.weight.weight_loader(head.weight, checkpoint_weight)
    method.process_weights_after_loading(head)

    assert head.weight.shape == (HIDDEN, VOCAB)  # canonical (K, N)
    assert not head.weight.is_contiguous()  # a transposed view, by design
    assert head.weight.input_dim == 0
    assert head.weight.output_dim == 1

    convert_linear_layer_to_humming_standard(
        layer=head, name_map={"weight": "weight", "weight_scale": "weight_scale"}
    )
    assert head.weight.is_contiguous()
    assert head.weight.stride(-1) == 1


def _build_and_process(method) -> ParallelLMHead:
    head = _build_lm_head(quant_method=method)
    torch.manual_seed(29)
    checkpoint_weight = torch.randn(VOCAB, HIDDEN, dtype=torch.bfloat16)
    head.weight.weight_loader(head.weight, checkpoint_weight)
    method.process_weights_after_loading(head)
    return head


def _assert_only_cpu_op_gap(exc: Exception) -> None:
    """The only failure a CPU test runner may legitimately hit past the
    Python-level attribute contract these kernel preps read off ``layer``:
    the real CUDA-only repack op has no CPU kernel registered. Anything else
    -- notably an AttributeError for a layer attribute the prep reads --
    is a real regression, not an environment limitation.
    """
    assert isinstance(exc, NotImplementedError), (
        f"expected only the CPU-backend op gap, got {type(exc).__name__}: {exc}"
    )
    assert "CPU' backend" in str(exc), f"unexpected NotImplementedError: {exc}"


@pytest.mark.skipif(not has_humming(), reason="humming is not installed")
def test_humming_prep_does_not_hit_a_missing_layer_attribute() -> None:
    """Regression test for a live-serve crash one step past the layout fix.

    ParallelLMHead is not a LinearBase, so create_weights() has to duck-type
    every attribute the real w8a16 kernels' prep functions read off `layer`.
    A missing ``output_partition_sizes`` crashed
    process_weights_after_loading()'s Humming path with ``AttributeError:
    'ParallelLMHead' object has no attribute 'output_partition_sizes'``.

    Calls the REAL (unbound) ``HummingFP8ScaledMMLinearKernel.
    process_weights_after_loading`` -- not a mock -- against a layer built
    exactly as create_weights()/process_weights_after_loading() leave it,
    bypassing only ``__init__``'s CUDA-capability assertion (the kernel
    instance is a bare namespace carrying just the one attribute the method
    reads off itself, ``self.config``). Every layer attribute the method
    touches -- through convert_linear_layer_to_humming_standard() and
    prepare_humming_linear_layer_config() -- is read in plain Python before
    the first CUDA-only op (humming::repack_weight), so on a CPU test runner
    that op's NotImplementedError is the only exception this may
    legitimately raise; any AttributeError fails the test.
    """
    vllm_config = _vllm_config(lm_head_quant="fp8")
    with set_current_vllm_config(vllm_config):
        method = get_lm_head_quant_method(vllm_config)
        head = _build_and_process(method)

    fake_kernel = SimpleNamespace(
        config=FP8ScaledMMLinearLayerConfig(
            weight_quant_key=kFp8StaticTensorSym,
            activation_quant_key=kFp8DynamicTensorSym,
            weight_shape=head.weight.shape,
            input_dtype=torch.bfloat16,
            out_dtype=torch.bfloat16,
        )
    )
    try:
        HummingFP8ScaledMMLinearKernel.process_weights_after_loading(fake_kernel, head)
    except Exception as exc:  # noqa: BLE001 - re-asserted below
        _assert_only_cpu_op_gap(exc)


def test_marlin_prep_does_not_hit_a_missing_layer_attribute() -> None:
    """Same attribute-contract check as the Humming test above, for Marlin.

    Marlin's prep already read nothing beyond what create_weights() set
    before this fix (output_size_per_partition, input_size_per_partition,
    weight_block_size, orig_dtype, logical_widths) -- verified by reverting
    the fix locally and confirming Marlin still only hits the CUDA-only-op
    wall, never an AttributeError. This test exists so both kernels'
    attribute contracts are covered by one convention rather than relying on
    that having stayed true.
    """
    vllm_config = _vllm_config(lm_head_quant="fp8")
    with set_current_vllm_config(vllm_config):
        method = get_lm_head_quant_method(vllm_config)
        head = _build_and_process(method)

    fake_kernel = SimpleNamespace(
        marlin_input_dtype=None, block_quant=False, size_k_first=True
    )
    try:
        MarlinFP8ScaledMMLinearKernel.process_weights_after_loading(fake_kernel, head)
    except Exception as exc:  # noqa: BLE001 - re-asserted below
        _assert_only_cpu_op_gap(exc)
