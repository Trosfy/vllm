# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

# MXFP8 constants
MXFP8_VALUE_DTYPE = torch.float8_e4m3fn
MXFP8_SCALE_DTYPE = torch.uint8
MXFP8_BLOCK_SIZE = 32


@triton.jit
def _mxfp8_embedding_kernel(
    weight_ptr,
    scale_ptr,
    ids_ptr,
    output_ptr,
    embedding_dim: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    col_block = tl.program_id(1)
    cols = col_block * BLOCK + tl.arange(0, BLOCK)
    mask = cols < embedding_dim
    token_id = tl.load(ids_ptr + row).to(tl.int64)
    values = tl.load(
        weight_ptr + token_id * embedding_dim + cols,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    scales_per_row: tl.constexpr = embedding_dim // 32
    scale_exp = tl.load(
        scale_ptr
        + token_id * scales_per_row
        + cols // 32,
        mask=mask,
        other=127,
    ).to(tl.float32)
    values *= tl.exp2(scale_exp - 127.0)
    tl.store(output_ptr + row * embedding_dim + cols, values, mask=mask)


def _mxfp8_embedding_impl(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    token_ids: torch.Tensor,
) -> torch.Tensor:
    if weight.dtype != MXFP8_VALUE_DTYPE or weight.ndim != 2:
        raise TypeError(
            "MXFP8 embedding expects a 2-D float8_e4m3fn weight, got "
            f"shape={tuple(weight.shape)}, dtype={weight.dtype}."
        )
    if weight.shape[1] % MXFP8_BLOCK_SIZE:
        raise ValueError(
            f"MXFP8 embedding dimension {weight.shape[1]} must be divisible "
            f"by {MXFP8_BLOCK_SIZE}."
        )
    expected_scale_shape = (
        weight.shape[0],
        weight.shape[1] // MXFP8_BLOCK_SIZE,
    )
    if weight_scale.dtype != MXFP8_SCALE_DTYPE or tuple(weight_scale.shape) != (
        expected_scale_shape
    ):
        raise TypeError(
            "MXFP8 embedding scale must be row-major uint8 with shape "
            f"{expected_scale_shape}, got shape={tuple(weight_scale.shape)}, "
            f"dtype={weight_scale.dtype}."
        )
    if token_ids.device != weight.device:
        raise ValueError("MXFP8 embedding token ids and weight must share a device.")

    output = torch.empty(
        (*token_ids.shape, weight.shape[1]),
        dtype=torch.bfloat16,
        device=weight.device,
    )
    flat_ids = token_ids.reshape(-1)
    block = min(256, triton.next_power_of_2(weight.shape[1]))
    _mxfp8_embedding_kernel[(flat_ids.numel(), triton.cdiv(weight.shape[1], block))](
        weight,
        weight_scale,
        flat_ids,
        output,
        embedding_dim=weight.shape[1],
        BLOCK=block,
    )
    return output


def _mxfp8_embedding_fake(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    token_ids: torch.Tensor,
) -> torch.Tensor:
    del weight_scale
    return torch.empty(
        (*token_ids.shape, weight.shape[1]),
        dtype=torch.bfloat16,
        device=weight.device,
    )


direct_register_custom_op(
    op_name="mxfp8_embedding",
    op_func=_mxfp8_embedding_impl,
    fake_impl=_mxfp8_embedding_fake,
)


def mxfp8_embedding(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    token_ids: torch.Tensor,
) -> torch.Tensor:
    """Lookup and dequantize only the requested rows of an MXFP8 table."""
    return torch.ops.vllm.mxfp8_embedding(weight, weight_scale, token_ids)


def swizzle_mxfp8_scale(sf: torch.Tensor, M: int, K: int) -> torch.Tensor:
    """Swizzle MXFP8 scales from row-major 2D to F8_128x4 layout."""
    scaling_vector_size = MXFP8_BLOCK_SIZE  # 32 for MXFP8
    factor = scaling_vector_size * 4  # 128

    num_m_tiles = (M + 127) // 128
    num_k_tiles = (K + factor - 1) // factor

    m_padded = num_m_tiles * 128
    k_scale_padded = num_k_tiles * 4

    scale_cols = K // scaling_vector_size
    sf_padded = torch.zeros(
        (m_padded, k_scale_padded), dtype=sf.dtype, device=sf.device
    )
    sf_padded[:M, :scale_cols] = sf

    sf_reshaped = sf_padded.view(num_m_tiles, 4, 32, num_k_tiles, 4)

    sf_swizzled = sf_reshaped.transpose(1, 3)

    return sf_swizzled.contiguous().view(-1)


def _mxfp8_e4m3_quantize_torch(
    x: torch.Tensor,
    is_sf_swizzled_layout: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Naive MXFP8 quantization.
    For each block of 32 elements along the last dimension, compute a
    shared e8m0 scale that fits the block-wise amax into the finite
    float8_e4m3fn range, and quantize each element to float8_e4m3fn.

    Returns (quantized_values [same shape, fp8], scales uint8).
    Scale shape depends on is_sf_swizzled_layout:
      False -> [..., K//32]  (row-major 2D)
      True  -> [flat swizzled 1D]
    """
    assert x.shape[-1] % MXFP8_BLOCK_SIZE == 0
    orig_shape = x.shape
    num_blocks = x.shape[-1] // MXFP8_BLOCK_SIZE

    x_fp32 = x.to(torch.float32)
    x_blocked = x_fp32.view(*orig_shape[:-1], num_blocks, MXFP8_BLOCK_SIZE)

    amax = x_blocked.abs().amax(dim=-1)
    amax = amax.clamp(min=torch.finfo(torch.float32).tiny)
    fp8_max = torch.finfo(MXFP8_VALUE_DTYPE).max
    scale_biased = torch.ceil(torch.log2(amax / fp8_max)) + 127.0
    scale_biased = scale_biased.clamp(0, 254)
    scales_uint8 = scale_biased.to(torch.uint8)

    descale = torch.exp2(scale_biased - 127.0)
    x_scaled = x_blocked / descale.unsqueeze(-1)

    x_fp8 = x_scaled.view(orig_shape).to(MXFP8_VALUE_DTYPE)

    if x.ndim == 2:
        M, K = x.shape
        scales_uint8 = scales_uint8.view(M, -1)
        if is_sf_swizzled_layout:
            scales_uint8 = swizzle_mxfp8_scale(scales_uint8, M=M, K=K)
    elif x.ndim == 3:
        B, M, K = x.shape
        scales_uint8 = scales_uint8.view(B, M, -1)
        if is_sf_swizzled_layout:
            swizzled = []
            for i in range(B):
                swizzled.append(swizzle_mxfp8_scale(scales_uint8[i], M=M, K=K))
            scales_uint8 = torch.cat(swizzled)

    return x_fp8, scales_uint8


def _mxfp8_quant_triton_kernel():
    """Lazily-built Triton kernel: per-32-block E8M0 scale + FP8-E4M3 quant.

    Fuses what ``_mxfp8_e4m3_quantize_torch`` does in several elementwise passes
    into one launch. Each program handles ``[BLOCK_M, 32]`` (one MX block).
    """
    from vllm.triton_utils import tl, triton

    @triton.jit
    def _kernel(
        x_ptr,
        xq_ptr,
        s_ptr,
        M,
        K,
        sxm,
        sxk,
        sqm,
        sqk,
        ssm,
        ssk,
        BLOCK_M: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_b = tl.program_id(1)  # which 32-element block along K
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_k = pid_b * 32 + tl.arange(0, 32)
        m_mask = offs_m < M
        x = tl.load(
            x_ptr + offs_m[:, None] * sxm + offs_k[None, :] * sxk,
            mask=m_mask[:, None],
            other=0.0,
        ).to(tl.float32)
        amax = tl.maximum(tl.max(tl.abs(x), axis=1), 1e-30)  # [BLOCK_M]
        sb = tl.floor(tl.log2(amax)) + 127.0
        sb = tl.minimum(tl.maximum(sb, 0.0), 254.0)
        descale = tl.exp2(sb - 127.0)
        xq = (x / descale[:, None]).to(xq_ptr.dtype.element_ty)
        tl.store(
            xq_ptr + offs_m[:, None] * sqm + offs_k[None, :] * sqk,
            xq,
            mask=m_mask[:, None],
        )
        tl.store(s_ptr + offs_m * ssm + pid_b * ssk, sb.to(tl.uint8), mask=m_mask)

    return _kernel


_MXFP8_QUANT_KERNEL = None


def _mxfp8_e4m3_quantize_triton(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused 2D MXFP8 quant (non-swizzled, row-major [M, K//32] scales)."""
    from vllm.triton_utils import triton

    global _MXFP8_QUANT_KERNEL
    if _MXFP8_QUANT_KERNEL is None:
        _MXFP8_QUANT_KERNEL = _mxfp8_quant_triton_kernel()

    M, K = x.shape
    x = x.contiguous()
    xq = torch.empty((M, K), dtype=MXFP8_VALUE_DTYPE, device=x.device)
    scales = torch.empty(
        (M, K // MXFP8_BLOCK_SIZE), dtype=MXFP8_SCALE_DTYPE, device=x.device
    )
    BLOCK_M = 64
    grid = (triton.cdiv(M, BLOCK_M), K // MXFP8_BLOCK_SIZE)
    _MXFP8_QUANT_KERNEL[grid](
        x,
        xq,
        scales,
        M,
        K,
        x.stride(0),
        x.stride(1),
        xq.stride(0),
        xq.stride(1),
        scales.stride(0),
        scales.stride(1),
        BLOCK_M=BLOCK_M,
    )
    return xq, scales


def _mxfp8_e4m3_quantize_impl(
    x: torch.Tensor,
    is_sf_swizzled_layout: bool = False,
    alignment: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    from vllm.platforms import current_platform

    if current_platform.has_device_capability(100):
        from flashinfer import mxfp8_quantize as flashinfer_mxfp8_quantize

        x_q, x_scales = flashinfer_mxfp8_quantize(
            x,
            is_sf_swizzled_layout=is_sf_swizzled_layout,
            alignment=alignment if alignment > 0 else 32,
            backend="cute-dsl",
        )
        if x_scales.ndim == 1 and x.ndim == 2 and not is_sf_swizzled_layout:
            x_scales = x_scales.view(x.size(0), -1)
        return x_q, x_scales

    # ROCm: a single fused Triton kernel beats the multi-pass torch path for the
    # common 2D, non-swizzled activation-quant case (used by the native MX
    # linear/MoE). Falls back to torch otherwise (3D weights, swizzled layout).
    if (
        current_platform.is_rocm()
        and not is_sf_swizzled_layout
        and x.ndim == 2
        and x.shape[-1] % MXFP8_BLOCK_SIZE == 0
    ):
        return _mxfp8_e4m3_quantize_triton(x)

    return _mxfp8_e4m3_quantize_torch(x, is_sf_swizzled_layout)


def mxfp8_e4m3_quantize(
    x: torch.Tensor,
    is_sf_swizzled_layout: bool = False,
    alignment: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.ops.vllm.mxfp8_quantize(x, is_sf_swizzled_layout, alignment)


def dequant_mxfp8_to_bf16(x: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Dequantize MXFP8 tensor to BF16."""
    x_float = x.to(torch.float32)

    num_blocks = x.shape[-1] // MXFP8_BLOCK_SIZE
    x_blocked = x_float.view(*x.shape[:-1], num_blocks, MXFP8_BLOCK_SIZE)

    descale = torch.exp2(scales.to(torch.float32) - 127.0)

    dequantized = x_blocked * descale.unsqueeze(-1)

    dequantized = dequantized.view(*x.shape)

    return dequantized.to(torch.bfloat16)


def mxfp8_e4m3_quantize_fake(
    x: torch.Tensor,
    is_sf_swizzled_layout: bool = False,
    alignment: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fake implementation for torch.compile tracing."""
    fp_data = torch.empty_like(x, dtype=MXFP8_VALUE_DTYPE)

    block_size = MXFP8_BLOCK_SIZE

    if x.ndim == 2:
        M, N = x.shape
        K = (N + block_size - 1) // block_size
        if is_sf_swizzled_layout:
            M_padded = ((M + 127) // 128) * 128
            K_padded = ((K + 3) // 4) * 4
            scales = torch.empty(
                M_padded * K_padded, dtype=MXFP8_SCALE_DTYPE, device=x.device
            )
        else:
            scales = torch.empty((M, K), dtype=MXFP8_SCALE_DTYPE, device=x.device)
    elif x.ndim == 3:
        B, M, N = x.shape
        K = (N + block_size - 1) // block_size
        if is_sf_swizzled_layout:
            M_padded = ((M + 127) // 128) * 128
            K_padded = ((K + 3) // 4) * 4
            scales = torch.empty(
                B * M_padded * K_padded, dtype=MXFP8_SCALE_DTYPE, device=x.device
            )
        else:
            scales = torch.empty((B, M, K), dtype=MXFP8_SCALE_DTYPE, device=x.device)
    else:
        scale_shape = list(x.shape)
        scale_shape[-1] = (x.shape[-1] + block_size - 1) // block_size
        scales = torch.empty(scale_shape, dtype=MXFP8_SCALE_DTYPE, device=x.device)

    return fp_data, scales


direct_register_custom_op(
    op_name="mxfp8_quantize",
    op_func=_mxfp8_e4m3_quantize_impl,
    fake_impl=mxfp8_e4m3_quantize_fake,
)


def xpu_mxfp8_quantize(
    x: torch.Tensor, dtype: torch.dtype | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.ops.vllm.xpu_mxfp8_quantize(x, dtype)
