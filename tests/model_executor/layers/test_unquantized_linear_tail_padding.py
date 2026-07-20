# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.utils.cublas import storage_tail_bytes


def test_unquantized_linear_writes_directly_to_tail_padded_storage() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    layer = torch.nn.Module().to(device)
    layer.weight = torch.nn.Parameter(torch.randn(7, 5, device=device))
    layer.output_tail_padding_bytes = 64 * 1024
    source = torch.randn(3, 5, device=device)

    with torch.inference_mode():
        output = UnquantizedLinearMethod().apply(layer, source)

    torch.testing.assert_close(output, torch.nn.functional.linear(source, layer.weight))
    assert storage_tail_bytes(output) >= layer.output_tail_padding_bytes


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_unquantized_linear_tail_padding_survives_cuda_graph_replay() -> None:
    layer = torch.nn.Module().cuda()
    layer.weight = torch.nn.Parameter(
        torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16),
        requires_grad=False,
    )
    layer.output_tail_padding_bytes = 64 * 1024
    source = torch.randn(6, 2048, device="cuda", dtype=torch.bfloat16)

    with torch.inference_mode():
        expected = UnquantizedLinearMethod().apply(layer, source).clone()
        UnquantizedLinearMethod().apply(layer, source)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = UnquantizedLinearMethod().apply(layer, source)
        for _ in range(3):
            graph.replay()
        torch.cuda.synchronize()

    assert storage_tail_bytes(output) >= layer.output_tail_padding_bytes
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
