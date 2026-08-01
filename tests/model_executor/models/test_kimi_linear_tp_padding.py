import torch

import vllm.model_executor.models.kimi_linear as kimi_module


def _set_tp(monkeypatch, rank: int, world_size: int = 12) -> None:
    monkeypatch.setattr(
        kimi_module, "get_tensor_model_parallel_world_size", lambda: world_size
    )
    monkeypatch.setattr(
        "vllm.model_executor.layers.linear.get_tensor_model_parallel_world_size",
        lambda: world_size,
    )
    monkeypatch.setattr(
        "vllm.model_executor.layers.linear.get_tensor_model_parallel_rank",
        lambda: rank,
    )
    monkeypatch.setattr(
        "vllm.model_executor.parameter.get_tensor_model_parallel_world_size",
        lambda: world_size,
    )
    monkeypatch.setattr(
        "vllm.model_executor.parameter.get_tensor_model_parallel_rank", lambda: rank
    )


def test_kimi_router_tp12_zero_pads_and_slices(monkeypatch):
    source = torch.arange(896 * 4, dtype=torch.float32).view(896, 4)
    x = torch.arange(8, dtype=torch.float32).view(2, 4)
    local_outputs = []
    gates = []
    for rank in range(12):
        _set_tp(monkeypatch, rank)
        gate = kimi_module.KimiColumnParallelGate(4, 896, "gate")
        gate.weight.weight_loader(gate.weight, source)
        local_outputs.append(torch.nn.functional.linear(x, gate.weight))
        gates.append(gate)

    gathered = torch.cat(local_outputs, dim=-1)
    assert gathered.shape == (2, 900)
    assert gates[-1].weight[-4:].count_nonzero().item() == 0
    monkeypatch.setattr(
        kimi_module, "tensor_model_parallel_all_gather", lambda _: gathered
    )
    output, _ = gates[0](x)
    assert output.shape == (2, 896)
    assert output.is_contiguous()
    assert torch.equal(output, torch.nn.functional.linear(x, source))


def test_kimi_latent_tp12_padding_preserves_projection(monkeypatch):
    down_source = torch.arange(3584 * 4, dtype=torch.float32).view(3584, 4)
    up_source = torch.arange(3 * 3584, dtype=torch.float32).view(3, 3584)
    x = torch.arange(8, dtype=torch.float32).view(2, 4)
    latent = torch.nn.functional.linear(x, down_source)
    down_outputs = []
    up_outputs = []
    for rank in range(12):
        _set_tp(monkeypatch, rank)
        down = kimi_module.KimiPaddedColumnParallelLinear(4, 3584, "down")
        down.weight.weight_loader(down.weight, down_source)
        down_outputs.append(torch.nn.functional.linear(x, down.weight))

        up = kimi_module.KimiPaddedRowParallelLinear(3584, 3, "up")
        up.weight.weight_loader(up.weight, up_source)
        padded_latent = torch.nn.functional.pad(latent, (0, 4))
        local_latent = padded_latent[..., rank * 299 : (rank + 1) * 299]
        up_outputs.append(torch.nn.functional.linear(local_latent, up.weight))

    gathered_latent = torch.cat(down_outputs, dim=-1)[..., :3584]
    assert torch.equal(gathered_latent, latent)
    padded_gather = torch.nn.functional.pad(latent, (0, 4))
    monkeypatch.setattr(
        kimi_module, "tensor_model_parallel_all_gather", lambda _: padded_gather
    )
    monkeypatch.setattr(
        "vllm.model_executor.layers.linear.tensor_model_parallel_all_gather",
        lambda _: padded_gather,
    )
    down_output, _ = down(x)
    assert down_output.shape == (2, 3584)
    assert down_output.is_contiguous()
    torch.testing.assert_close(
        sum(up_outputs),
        torch.nn.functional.linear(latent, up_source),
        rtol=1e-5,
        atol=1e-3,
    )
