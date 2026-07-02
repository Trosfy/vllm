# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""rp-native tiny-decode (M=1) MoE for the b12x w4a8_mx mode.

Reads the N256/K128 in-place-repacked FP4 expert weights and e8m0 sfb grids
directly via their verified inverse bit mappings (see b12x
tests/test_w4a8_rp_inverse_mapping.py), with BF16 activations (no input
quantization) and fp32 accumulation: SiLU-gated FC1 -> FC2, router-weighted
fp32-atomic scatter. At DS4-Flash TP2 shapes this runs one decode token in
~29 us/layer vs ~35 us for the dynamic grouped kernel (and ~3 us more of
wrapper fills/copies it also bypasses), with better numerics than the
w4a8 path (cos vs fp32 oracle 0.999999 vs 0.9990).

Enabled with VLLM_B12X_W4A8_MX_TINY_DECODE=1; engages only for
quant_mode=w4a8_mx, silu activation, w31 layout, M==1, and shape multiples
the kernels require. Anything else falls through to the b12x dynamic path.
"""
import os

import torch
import triton
import triton.language as tl

from vllm.logger import init_logger

logger = init_logger(__name__)

_TINY_DECODE_ENABLED = os.getenv("VLLM_B12X_W4A8_MX_TINY_DECODE", "0") == "1"
_MAX_M = 4


@triton.jit
def _fp4_val(nib):
    # e2m1 nibble -> fp32 by direct bit assembly (no SFU):
    #   e>0: (1.m) * 2^(e-1)  -> fp32 exp field = e + 126
    #   e=0: m * 0.5
    s = (nib >> 3) & 1
    e = (nib >> 1) & 3
    m = nib & 1
    bits = tl.where(e > 0, ((e + 126) << 23) | (m << 22), m * (126 << 23))
    bits = bits | (s << 31)
    return bits.to(tl.float32, bitcast=True)


@triton.jit
def _tiny_fc1_kernel(
    x_ptr, ids_ptr, w13_ptr, sfb13_ptr, inter_ptr,
    TOPK: tl.constexpr,
    K: tl.constexpr, N2: tl.constexpr,
    W13_STRIDE: tl.constexpr, SFB13_STRIDE: tl.constexpr,
    ROT: tl.constexpr, KT_TILES: tl.constexpr, KT_PER_PROG: tl.constexpr,
):
    # grid: (M*topk, N2//256, KT_TILES//KT_PER_PROG)
    pid_rt = tl.program_id(0)
    nt = tl.program_id(1)
    pid_k = tl.program_id(2)

    eid = tl.load(ids_ptr + pid_rt).to(tl.int64)
    tok = pid_rt // TOPK
    w_base = eid * W13_STRIDE
    s_base = eid * SFB13_STRIDE

    k32 = tl.arange(0, 4)[:, None, None, None, None]
    n8c = tl.arange(0, 8)[None, :, None, None, None]
    r8 = tl.arange(0, 8)[None, None, :, None, None]
    cgrp = tl.arange(0, 4)[None, None, None, :, None]
    v = tl.arange(0, 4)[None, None, None, None, :]

    acc = tl.zeros((4, 8, 8, 4, 4), dtype=tl.float32)
    for kt_i in tl.range(0, KT_PER_PROG):
        kt = pid_k * KT_PER_PROG + kt_i
        tile_base = w_base + (nt * KT_TILES + kt).to(tl.int64) * 4096
        flat = tl.arange(0, 4096)
        words = tl.reshape(tl.load(w13_ptr + tile_base + flat), (4, 8, 8, 4, 4))
        # per-word 32-group scale: col = kt*4 + k32 -> kb=k32, col-tile=kt
        sfb_off = k32 | (r8 << 2) | ((n8c * 4 + v) << 5) | ((nt * KT_TILES + kt) << 10)
        wscale = (tl.load(sfb13_ptr + s_base + sfb_off).to(tl.int32) << 23).to(
            tl.float32, bitcast=True
        )
        a128 = tl.load(x_ptr + tok * K + kt * 128 + tl.arange(0, 128)).to(tl.float32)
        aw = tl.reshape(a128, (16, 8))  # (k32*4+cgrp, j)
        jj = tl.arange(0, 8)[None, :]
        part = tl.zeros((4, 8, 8, 4, 4), dtype=tl.float32)
        for j in tl.static_range(8):
            wv = _fp4_val((words >> (4 * j)) & 0xF)
            aj = tl.sum(tl.where(jj == j, aw, 0.0), axis=1)          # (16,)
            a = tl.reshape(aj, (4, 1, 1, 4, 1))                       # (k32,-,-,cgrp,-)
            part += wv * a
        acc += part * wscale
    row_part = tl.sum(tl.sum(acc, axis=3), axis=0)  # -> (n8c=8, r8=8, n8i=4)
    n8c_r = tl.arange(0, 8)[:, None, None]
    r8_r = tl.arange(0, 8)[None, :, None]
    v_r = tl.arange(0, 4)[None, None, :]
    p_full = nt * 256 + n8c_r * 32 + v_r * 8 + r8_r
    r_log = (p_full + ROT) % N2
    tl.atomic_add(inter_ptr + pid_rt.to(tl.int64) * N2 + r_log, row_part, sem='relaxed')


@triton.jit
def _tiny_fc2_kernel(
    inter_ptr, ids_ptr, tw_ptr, w2_ptr, sfb2_ptr, out_ptr,
    TOPK: tl.constexpr,
    N: tl.constexpr, K_OUT: tl.constexpr,
    W2_STRIDE: tl.constexpr, SFB2_STRIDE: tl.constexpr,
    KT_TILES: tl.constexpr, KT_PER_PROG: tl.constexpr,
):
    # grid: (M*topk, K_OUT//256, KT_TILES//KT_PER_PROG)
    pid_rt = tl.program_id(0)
    nt = tl.program_id(1)
    pid_k = tl.program_id(2)

    eid = tl.load(ids_ptr + pid_rt).to(tl.int64)
    tok = pid_rt // TOPK
    rw = tl.load(tw_ptr + pid_rt)
    w_base = eid * W2_STRIDE
    s_base = eid * SFB2_STRIDE

    k32 = tl.arange(0, 4)[:, None, None, None, None]
    n8c = tl.arange(0, 8)[None, :, None, None, None]
    r8 = tl.arange(0, 8)[None, None, :, None, None]
    cgrp = tl.arange(0, 4)[None, None, None, :, None]
    v = tl.arange(0, 4)[None, None, None, None, :]

    ibase = pid_rt.to(tl.int64) * (2 * N)
    acc = tl.zeros((4, 8, 8, 4, 4), dtype=tl.float32)
    for kt_i in tl.range(0, KT_PER_PROG):
        kt = pid_k * KT_PER_PROG + kt_i
        tile_base = w_base + (nt * KT_TILES + kt).to(tl.int64) * 4096
        flat = tl.arange(0, 4096)
        words = tl.reshape(tl.load(w2_ptr + tile_base + flat), (4, 8, 8, 4, 4))
        sfb_off = k32 | (r8 << 2) | ((n8c * 4 + v) << 5) | ((nt * KT_TILES + kt) << 10)
        wscale = (tl.load(sfb2_ptr + s_base + sfb_off).to(tl.int32) << 23).to(
            tl.float32, bitcast=True
        )
        g128 = tl.load(inter_ptr + ibase + kt * 128 + tl.arange(0, 128))
        u128 = tl.load(inter_ptr + ibase + N + kt * 128 + tl.arange(0, 128))
        act128 = (g128 / (1.0 + tl.exp(-g128))) * u128               # silu(gate)*up
        aw = tl.reshape(act128, (16, 8))
        jj = tl.arange(0, 8)[None, :]
        part = tl.zeros((4, 8, 8, 4, 4), dtype=tl.float32)
        for j in tl.static_range(8):
            wv = _fp4_val((words >> (4 * j)) & 0xF)
            aj = tl.sum(tl.where(jj == j, aw, 0.0), axis=1)
            a = tl.reshape(aj, (4, 1, 1, 4, 1))
            part += wv * a
        acc += part * wscale
    row_part = tl.sum(tl.sum(acc, axis=3), axis=0)  # (8, 8, 4)
    n8c_r = tl.arange(0, 8)[:, None, None]
    r8_r = tl.arange(0, 8)[None, :, None]
    v_r = tl.arange(0, 4)[None, None, :]
    p_full = nt * 256 + n8c_r * 32 + v_r * 8 + r8_r
    tl.atomic_add(out_ptr + tok.to(tl.int64) * K_OUT + p_full, row_part * rw, sem='relaxed')




_BUFFERS: dict = {}


def _get_buffers(device: torch.device, n2: int, k: int, topk: int):
    key = (device, n2, k)
    bufs = _BUFFERS.get(key)
    if bufs is None:
        if torch.cuda.is_current_stream_capturing():
            return None
        bufs = (
            torch.zeros(_MAX_M * topk, n2, dtype=torch.float32, device=device),
            torch.zeros(_MAX_M, k, dtype=torch.float32, device=device),
        )
        _BUFFERS[key] = bufs
    return bufs


def maybe_run_tiny_w4a8mx_moe(
    *,
    a: torch.Tensor,
    experts,
    output: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    plan,
) -> bool:
    """Run the tiny-decode path if applicable; returns True when handled."""
    if not _TINY_DECODE_ENABLED:
        return False
    m = int(a.shape[0])
    if m != 1:
        return False
    caps = getattr(plan, "caps", None)
    if caps is None or getattr(caps, "quant_mode", None) != "w4a8_mx":
        return False
    if getattr(caps, "apply_router_weight_on_input", False):
        return False
    if getattr(experts, "activation", None) != "silu":
        return False
    if getattr(experts, "w13_layout", None) != "w31":
        return False
    if getattr(experts, "source_format", None) != "fp4_e8m0_k32":
        return False
    k = int(a.shape[1])
    w13 = experts.w1_fp4
    w2 = experts.w2_fp4
    if w13.dtype != torch.int32 or w2.dtype != torch.int32:
        return False
    e = int(w13.shape[0])
    words13 = w13.numel() // e
    words2 = w2.numel() // e
    if k % 256 != 0 or words2 % k != 0:
        return False
    n = words2 * 8 // k
    if n % 256 != 0 or words13 != 2 * n * k // 8:
        return False
    topk = int(topk_ids.shape[1])
    bufs = _get_buffers(a.device, 2 * n, k, topk)
    if bufs is None:
        logger.warning_once(
            "b12x tiny-decode buffers not pre-allocated before graph capture; "
            "falling back to the dynamic path"
        )
        return False
    inter_buf, out_buf = bufs
    logger.info_once("b12x w4a8_mx tiny-decode path engaged (M=1, K=%d, N=%d)", k, n)

    rt = m * topk
    flat_ids = topk_ids.reshape(-1)
    flat_w = topk_weights.reshape(-1).float()
    inter = inter_buf[:rt]
    inter.zero_()
    out_buf[:m].zero_()

    w13_words = w13.reshape(e, -1)
    w2_words = w2.reshape(e, -1)
    sfb13_b = experts.w1_blockscale.view(torch.uint8).reshape(e, -1)
    sfb2_b = experts.w2_blockscale.view(torch.uint8).reshape(e, -1)

    kt13 = k // 128
    _tiny_fc1_kernel[(rt, (2 * n) // 256, kt13)](
        a, flat_ids, w13_words, sfb13_b, inter,
        TOPK=topk, K=k, N2=2 * n,
        W13_STRIDE=w13_words.shape[1], SFB13_STRIDE=sfb13_b.shape[1],
        ROT=n, KT_TILES=kt13, KT_PER_PROG=1,
        num_warps=8,
    )
    kt2 = n // 128
    _tiny_fc2_kernel[(rt, k // 256, kt2)](
        inter, flat_ids, flat_w, w2_words, sfb2_b, out_buf,
        TOPK=topk, N=n, K_OUT=k,
        W2_STRIDE=w2_words.shape[1], SFB2_STRIDE=sfb2_b.shape[1],
        KT_TILES=kt2, KT_PER_PROG=1,
        num_warps=8,
    )
    output[:m].copy_(out_buf[:m])
    return True
