# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import glob
import inspect

import pytest
import torch
from safetensors.torch import save_file

from vllm.model_executor.model_loader.reload.layerwise import (
    _own_deferred_accelerator_tensors,
)
from vllm.model_executor.model_loader.weight_utils import (
    download_weights_from_hf,
    instanttensor_weights_iterator,
    safetensors_weights_iterator,
)
from vllm.platforms import current_platform


@pytest.mark.skipif(
    not current_platform.is_cuda(),
    reason="InstantTensor requires NVIDIA GPUs",
)
def test_instanttensor_model_loader():
    model_dir = download_weights_from_hf(
        "openai-community/gpt2", cache_dir=None, allow_patterns=["*.safetensors"]
    )
    safetensors = glob.glob(f"{model_dir}/*.safetensors")
    assert len(safetensors) > 0

    instanttensor_tensors = {}
    hf_safetensors_tensors = {}

    for name, tensor in instanttensor_weights_iterator(safetensors, True):
        instanttensor_tensors[name] = tensor.to("cpu")

    for name, tensor in safetensors_weights_iterator(safetensors, True):
        hf_safetensors_tensors[name] = tensor

    assert len(instanttensor_tensors) == len(hf_safetensors_tensors)

    for name, instanttensor_tensor in instanttensor_tensors.items():
        assert instanttensor_tensor.dtype == hf_safetensors_tensors[name].dtype
        assert instanttensor_tensor.shape == hf_safetensors_tensors[name].shape
        assert torch.all(instanttensor_tensor.eq(hf_safetensors_tensors[name]))


@pytest.mark.skipif(
    not current_platform.is_cuda(),
    reason="InstantTensor requires NVIDIA GPUs",
)
def test_instanttensor_honors_tensor_to_shard_index(tmp_path):
    base_shard = tmp_path / "model-00001-of-00002.safetensors"
    overlay_shard = tmp_path / "model-00002-of-00002.safetensors"
    save_file(
        {
            "model.dense.weight": torch.tensor([2.0]),
            "model.expert.weight": torch.tensor([1.0, 1.0]),
        },
        base_shard,
    )
    save_file(
        {"model.expert.weight": torch.tensor([3.0, 3.0, 3.0])},
        overlay_shard,
    )
    indexed_tensor_files = {
        "model.dense.weight": str(base_shard.resolve()),
        "model.expert.weight": str(overlay_shard.resolve()),
    }

    weights = {
        name: tensor.cpu()
        for name, tensor in instanttensor_weights_iterator(
            [str(base_shard), str(overlay_shard)],
            use_tqdm_on_load=False,
            indexed_tensor_files=indexed_tensor_files,
        )
    }

    assert set(weights) == {"model.dense.weight", "model.expert.weight"}
    assert weights["model.dense.weight"].tolist() == [2.0]
    assert weights["model.expert.weight"].tolist() == [3.0, 3.0, 3.0]


@pytest.mark.skipif(
    not current_platform.is_cuda(),
    reason="InstantTensor requires NVIDIA GPUs",
)
def test_instanttensor_falls_back_for_tensor_larger_than_ring(tmp_path, monkeypatch):
    shard = tmp_path / "model.safetensors"
    small = torch.arange(256 * 1024, dtype=torch.float32)
    large = torch.arange(9 * 256 * 1024, dtype=torch.float32)
    save_file({"model.small": small, "model.large": large}, shard)
    monkeypatch.setenv("INSTANTTENSOR_BUFFER_SIZE", str(8 * 1024 * 1024))

    weights = {
        name: tensor.cpu()
        for name, tensor in instanttensor_weights_iterator(
            [str(shard)], use_tqdm_on_load=False
        )
    }

    torch.testing.assert_close(weights["model.small"], small)
    torch.testing.assert_close(weights["model.large"], large)


@pytest.mark.skipif(
    not current_platform.is_cuda(),
    reason="InstantTensor requires NVIDIA GPUs",
)
def test_instanttensor_deferred_tensors_survive_ring_reuse(tmp_path, monkeypatch):
    shard = tmp_path / "model.safetensors"
    # Five 4 MiB tensors force an 8 MiB ring to wrap more than once.
    source = {
        f"model.weight_{index}": torch.full(
            (2 * 1024 * 1024,), index, dtype=torch.bfloat16
        )
        for index in range(5)
    }
    save_file(source, shard)
    monkeypatch.setenv("INSTANTTENSOR_BUFFER_SIZE", str(8 * 1024 * 1024))

    def deferred_loader(param, loaded_weight):
        raise AssertionError("The loader must not run while arguments are queued")

    signature = inspect.signature(deferred_loader)
    retained = {}
    for name, tensor in instanttensor_weights_iterator(
        [str(shard)],
        use_tqdm_on_load=False,
    ):
        bound_args = signature.bind(None, tensor)
        _own_deferred_accelerator_tensors(bound_args)
        retained[name] = bound_args.arguments["loaded_weight"]
    torch.cuda.synchronize()

    for name, expected in source.items():
        torch.testing.assert_close(retained[name].cpu(), expected)


if __name__ == "__main__":
    test_instanttensor_model_loader()
