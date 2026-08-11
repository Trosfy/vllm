#!/usr/bin/env python3
"""Bit-exactness + performance harness for the cx fused Kimi-K3 top-16 kernel.

Compares against vLLM's real moeSigmoid+moeTopK op (ops.topk_sigmoid) across
adversarial inputs (ties, saturation, NaN/Inf), then measures CUDA-graph
latency isolated AND under concurrent-GEMM contention (the in-context regime
this box's power capping creates).
"""

import sys

import torch

from vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router import (
    vllm_topk_sigmoid,
)
from vllm.models.kimi_k3.nvidia.ops.topk16_cx import kimi_topk16_sigmoid

EXPERTS = 896
TOPK = 16
DEV = "cuda:0"


def reference(logits, bias, renormalize=True, rsf=1.0):
    rows = logits.shape[0]
    weights = torch.empty((rows, TOPK), dtype=torch.float32, device=DEV)
    indices = torch.empty((rows, TOPK), dtype=torch.int32, device=DEV)
    token_expert = torch.empty((rows, TOPK), dtype=torch.int32, device=DEV)
    vllm_topk_sigmoid(
        weights, indices, token_expert, logits, renormalize, bias, rsf
    )
    return weights, indices


def gen_cases():
    g = torch.Generator(device=DEV).manual_seed(1234)
    bias_sets = {
        "zero": torch.zeros(EXPERTS, dtype=torch.float32, device=DEV),
        "randn": torch.randn(EXPERTS, generator=g, device=DEV),
        "neg": -torch.rand(EXPERTS, generator=g, device=DEV),
    }
    for rows in (1, 2, 4, 8):
        base = torch.randn(rows, EXPERTS, generator=g, device=DEV)
        yield f"randn_m{rows}", base, bias_sets["randn"]
        yield f"saturated_m{rows}", base * 100.0, bias_sets["randn"]
        # Heavy exact ties: quantize to very few distinct values.
        yield f"ties_m{rows}", (base * 2).round() * 0.5, bias_sets["zero"]
        yield f"allequal_m{rows}", torch.full(
            (rows, EXPERTS), 0.25, device=DEV
        ), bias_sets["zero"]
        bad = base.clone()
        bad[:, ::7] = float("nan")
        bad[:, 3::11] = float("inf")
        bad[:, 5::13] = float("-inf")
        yield f"naninf_m{rows}", bad, bias_sets["neg"]
        # Near-ties at float precision.
        nt = base.clone()
        nt[:, 1] = nt[:, 0]
        nt[:, 2] = torch.nextafter(nt[:, 0], torch.tensor(float("inf"), device=DEV))
        yield f"neartie_m{rows}", nt, bias_sets["zero"]


def check_exactness() -> bool:
    ok = True
    for name, logits, bias in gen_cases():
        rw, ri = reference(logits.contiguous(), bias)
        cw, ci = kimi_topk16_sigmoid(logits.contiguous(), bias)
        torch.cuda.synchronize()
        if not torch.equal(ri, ci):
            print(f"FAIL indices {name}: ref={ri[0][:8].tolist()} "
                  f"cx={ci[0][:8].tolist()}")
            ok = False
        elif not torch.equal(rw, cw):
            db = (rw - cw).abs().max().item()
            print(f"FAIL weights {name}: max|d|={db}")
            ok = False
    print("exactness:", "PASS" if ok else "FAIL")
    return ok


def bench(rows=8, iters=2000, contention=False):
    g = torch.Generator(device=DEV).manual_seed(7)
    logits = torch.randn(rows, EXPERTS, generator=g, device=DEV)
    bias = torch.randn(EXPERTS, generator=g, device=DEV)

    stop = torch.zeros(1, device=DEV)
    if contention:
        a = torch.randn(4096, 4096, device=DEV, dtype=torch.bfloat16)
        b = torch.randn(4096, 4096, device=DEV, dtype=torch.bfloat16)
        side = torch.cuda.Stream()

    def timed(fn):
        for _ in range(50):
            fn()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(10):
                fn()
        if contention:
            with torch.cuda.stream(side):
                for _ in range(6000):
                    torch.mm(a, b)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters // 10):
            graph.replay()
        end.record()
        end.synchronize()
        if contention:
            torch.cuda.synchronize()
        return start.elapsed_time(end) * 1000 / iters

    ref_us = timed(lambda: reference(logits, bias))
    cx_us = timed(lambda: kimi_topk16_sigmoid(logits, bias))
    tag = "contended" if contention else "isolated"
    print(f"{tag}: reference(sigmoid+topk)={ref_us:.2f}us  cx_fused={cx_us:.2f}us "
          f"(x{ref_us / cx_us:.2f})")


if __name__ == "__main__":
    torch.cuda.set_device(0)
    ok = check_exactness()
    bench(contention=False)
    bench(contention=True)
    sys.exit(0 if ok else 1)
