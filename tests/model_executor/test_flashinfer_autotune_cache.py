# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.warmup import flashinfer_autotune_cache, kernel_warmup


def test_resolve_flashinfer_autotune_file_default_layout(
    monkeypatch, tmp_path: Path
) -> None:
    fake_jit = SimpleNamespace(
        env=SimpleNamespace(
            FLASHINFER_WORKSPACE_DIR=Path("/flashinfer-cache/0.6.11.post2/103a")
        )
    )
    fake_flashinfer = SimpleNamespace(jit=fake_jit)
    monkeypatch.setitem(sys.modules, "flashinfer", fake_flashinfer)
    monkeypatch.setitem(sys.modules, "flashinfer.jit", fake_jit)
    monkeypatch.setattr(
        flashinfer_autotune_cache,
        "aot_compile_hash_factors",
        lambda _: ["env-hash", "config-hash"],
    )
    monkeypatch.setattr(
        flashinfer_autotune_cache.envs, "VLLM_CACHE_ROOT", str(tmp_path)
    )
    monkeypatch.setattr(
        flashinfer_autotune_cache.envs,
        "VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR",
        None,
    )

    runner = SimpleNamespace(vllm_config=SimpleNamespace())
    cache_hash = sha256(str(["env-hash", "config-hash"]).encode()).hexdigest()

    path = flashinfer_autotune_cache.resolve_flashinfer_autotune_file(runner)

    assert path == (
        tmp_path
        / "flashinfer_autotune_cache"
        / "0.6.11.post2"
        / "103a"
        / cache_hash
        / "autotune_configs.json"
    )
    assert path.parent.is_dir()


def test_resolve_flashinfer_autotune_file_uses_override_dir(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        flashinfer_autotune_cache.envs,
        "VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR",
        str(tmp_path),
    )
    monkeypatch.setattr(
        flashinfer_autotune_cache,
        "aot_compile_hash_factors",
        lambda _: ["env-hash", "config-hash"],
    )

    runner = SimpleNamespace(vllm_config=SimpleNamespace())
    cache_hash = sha256(str(["env-hash", "config-hash"]).encode()).hexdigest()

    path = flashinfer_autotune_cache.resolve_flashinfer_autotune_file(runner)

    assert path == tmp_path / cache_hash / "autotune_configs.json"


def _flashinfer_autotune_worker(model, *, attn_groups=None):
    runner = SimpleNamespace(
        attn_groups=attn_groups or [],
        block_tables=object(),
        is_pooling_model=True,
    )
    return SimpleNamespace(
        get_model=lambda: model,
        use_v2_model_runner=True,
        model_runner=runner,
        scheduler_config=SimpleNamespace(
            max_num_batched_tokens=8,
            max_num_scheduled_tokens=None,
        ),
        vllm_config=SimpleNamespace(
            compilation_config=SimpleNamespace(
                cudagraph_capture_sizes=[],
                compile_sizes=[],
            ),
            kernel_config=SimpleNamespace(
                enable_flashinfer_autotune=True,
                enable_cutedsl_warmup=False,
                enable_jit_warmup=False,
                enable_bf16x3_router_gemm=False,
            ),
            model_config=SimpleNamespace(),
        ),
        model_config=SimpleNamespace(dtype=torch.bfloat16),
    )


def _patch_flashinfer_autotune_deps(monkeypatch):
    calls = []
    monkeypatch.setattr(kernel_warmup, "deepseek_v4_mhc_warmup", lambda *a, **k: None)
    monkeypatch.setattr(
        kernel_warmup,
        "flashinfer_sparse_mla_decode_autotune_warmup",
        lambda *a, **k: None,
    )
    monkeypatch.setattr(
        kernel_warmup,
        "deepseek_v4_sparse_mla_attention_warmup",
        lambda *a, **k: None,
    )
    monkeypatch.setattr(kernel_warmup, "minimax_m3_msa_warmup", lambda *a, **k: None)
    monkeypatch.setattr(kernel_warmup, "warmup_b12x_mxfp8_linear", lambda *a, **k: 0)
    monkeypatch.setattr(kernel_warmup, "warmup_b12x_moe_dynamic", lambda *a, **k: 0)
    monkeypatch.setattr(kernel_warmup, "has_flashinfer", lambda: True)
    monkeypatch.setattr(
        kernel_warmup.current_platform, "has_device_capability", lambda _: True
    )
    monkeypatch.setattr(
        kernel_warmup,
        "flashinfer_autotune",
        lambda runner: calls.append(runner),
    )
    return calls


@pytest.mark.parametrize(("draft_dcp", "expected_warmups"), [(1, 1), (2, 2)])
def test_b12x_dcp_warmup_uses_each_module_dcp_geometry(
    monkeypatch, draft_dcp: int, expected_warmups: int
) -> None:
    from vllm.distributed import parallel_state
    from vllm.model_executor.layers.attention.mla_attention import MLAAttention
    from vllm.v1.attention.ops import dcp_alltoall

    monkeypatch.setenv("VLLM_USE_B12X_DCP_A2A", "1")
    attention = MLAAttention.__new__(MLAAttention)
    torch.nn.Module.__init__(attention)
    attention.register_parameter(
        "device_probe",
        torch.nn.Parameter(torch.empty(1)),
    )
    attention.dcp_b12x = True
    attention.num_heads = 16
    attention.kv_lora_rank = 512
    attention.qk_rope_head_dim = 64
    attention.impl = SimpleNamespace(dcp_world_size=2)

    draft_attention = MLAAttention.__new__(MLAAttention)
    torch.nn.Module.__init__(draft_attention)
    draft_attention.register_parameter(
        "device_probe",
        torch.nn.Parameter(torch.empty(1)),
    )
    draft_attention.dcp_b12x = True
    draft_attention.num_heads = 8
    draft_attention.kv_lora_rank = 512
    draft_attention.qk_rope_head_dim = 64
    draft_attention.impl = SimpleNamespace(dcp_world_size=draft_dcp)

    model = torch.nn.Module()
    model.add_module("attention", attention)
    draft_model = torch.nn.Module()
    draft_model.add_module("attention", draft_attention)
    compilation_config = SimpleNamespace(
        static_forward_context={"model.layers.0.attn": attention}
    )
    worker = SimpleNamespace(
        get_model=lambda: model,
        use_v2_model_runner=True,
        model_runner=SimpleNamespace(
            is_pooling_model=True,
            speculator=SimpleNamespace(model=draft_model),
        ),
        model_config=SimpleNamespace(dtype=torch.bfloat16),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=4096),
        vllm_config=SimpleNamespace(
            parallel_config=SimpleNamespace(
                decode_context_parallel_size=2,
                # The explicit B12X path also accelerates query all-gather
                # when the output reduction uses the AG+RS fallback.
                dcp_comm_backend="ag_rs",
            ),
            compilation_config=compilation_config,
            model_config=SimpleNamespace(),
        ),
    )
    group = object()
    calls = []
    monkeypatch.setattr(parallel_state, "get_dcp_group", lambda: group)
    monkeypatch.setattr(
        dcp_alltoall,
        "warmup_b12x_dcp_a2a",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    assert kernel_warmup._warmup_b12x_dcp_a2a(worker) == expected_warmups
    expected_calls = [
        (
            (group,),
            {
                "device": torch.device("cpu"),
                "dtype": torch.bfloat16,
                "max_batch_size": 4096,
                "total_heads": 32,
                "head_dim": 512,
                "query_head_dim": 576,
            },
        )
    ]
    if draft_dcp > 1:
        expected_calls.append(
            (
                (group,),
                {
                    "device": torch.device("cpu"),
                    "dtype": torch.bfloat16,
                    "max_batch_size": 4096,
                    "total_heads": 16,
                    "head_dim": 512,
                    "query_head_dim": 576,
                },
            )
        )
    assert calls == expected_calls


@pytest.mark.parametrize("dcp_size", [8, 16])
def test_kimi_projection_warmup_uses_runtime_projection_group(
    monkeypatch: pytest.MonkeyPatch,
    dcp_size: int,
) -> None:
    from vllm.distributed import parallel_state
    from vllm.v1.attention.ops import dcp_alltoall

    monkeypatch.setenv("VLLM_USE_B12X_DCP_A2A", "1")
    monkeypatch.setenv("VLLM_KIMI_USE_B12X_PROJECTION_GATHER", "1")
    monkeypatch.setenv("VLLM_KIMI_USE_B12X_PAIRED_PROJECTION_GATHER", "1")
    model = torch.nn.Module()
    model.register_parameter("probe", torch.nn.Parameter(torch.empty(1)))
    dcp_group = SimpleNamespace(world_size=dcp_size)
    tp_group = SimpleNamespace(world_size=16)
    monkeypatch.setattr(parallel_state, "get_dcp_group", lambda: dcp_group)
    monkeypatch.setattr(parallel_state, "get_tp_group", lambda: tp_group)
    warmed_groups: list[object] = []
    monkeypatch.setattr(
        dcp_alltoall,
        "warmup_b12x_kimi_projection_gathers",
        lambda group, **kwargs: warmed_groups.append(group) or 1,
    )
    worker = SimpleNamespace(
        get_model=lambda: model,
        model_runner=SimpleNamespace(speculator=None),
        model_config=SimpleNamespace(dtype=torch.bfloat16),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=4096),
        vllm_config=SimpleNamespace(
            parallel_config=SimpleNamespace(
                decode_context_parallel_size=dcp_size,
            ),
            compilation_config=SimpleNamespace(static_forward_context={}),
        ),
    )

    assert kernel_warmup._warmup_b12x_dcp_a2a(worker) == 1
    assert warmed_groups == [dcp_group if dcp_size == 16 else tp_group]


def test_kernel_warmup_runs_b12x_mxfp8_linear_warmup(monkeypatch) -> None:
    calls = []
    model = torch.nn.Linear(2, 2)
    worker = _flashinfer_autotune_worker(model)
    worker.scheduler_config.max_num_batched_tokens = 2048
    worker.vllm_config.compilation_config.cudagraph_capture_sizes = [1, 2, 4, 8]
    worker.vllm_config.kernel_config.enable_flashinfer_autotune = False
    worker.model_config.dtype = torch.float16

    monkeypatch.setattr(kernel_warmup, "deepseek_v4_mhc_warmup", lambda *a, **k: None)
    monkeypatch.setattr(
        kernel_warmup,
        "flashinfer_sparse_mla_decode_autotune_warmup",
        lambda *a, **k: None,
    )
    monkeypatch.setattr(
        kernel_warmup,
        "deepseek_v4_sparse_mla_attention_warmup",
        lambda *a, **k: None,
    )
    monkeypatch.setattr(kernel_warmup, "minimax_m3_msa_warmup", lambda *a, **k: None)
    monkeypatch.setattr(kernel_warmup, "warmup_b12x_moe_dynamic", lambda *a, **k: 0)

    def fake_mxfp8_warmup(*args, **kwargs):
        calls.append((args, kwargs))
        return 3

    monkeypatch.setattr(
        kernel_warmup,
        "warmup_b12x_mxfp8_linear",
        fake_mxfp8_warmup,
    )

    kernel_warmup.kernel_warmup(worker)

    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args == (model,)
    assert kwargs == {
        "max_tokens": 2048,
        "cudagraph_capture_sizes": [1, 2, 4, 8],
        "output_dtype": torch.float16,
    }


def test_kernel_warmup_limits_fused_mla_query_to_graph_sizes(monkeypatch) -> None:
    calls = []
    model = torch.nn.Linear(2, 2)
    worker = _flashinfer_autotune_worker(model)
    worker.vllm_config.compilation_config.cudagraph_capture_sizes = [2, 4, 32, 64]
    worker.vllm_config.kernel_config.enable_flashinfer_autotune = False
    worker.vllm_config.kernel_config.enable_cutedsl_warmup = False

    monkeypatch.setattr(kernel_warmup, "deepseek_v4_mhc_warmup", lambda *a, **k: None)
    monkeypatch.setattr(
        kernel_warmup,
        "flashinfer_sparse_mla_decode_autotune_warmup",
        lambda *a, **k: None,
    )
    monkeypatch.setattr(
        kernel_warmup,
        "deepseek_v4_sparse_mla_attention_warmup",
        lambda *a, **k: None,
    )
    monkeypatch.setattr(kernel_warmup, "minimax_m3_msa_warmup", lambda *a, **k: None)
    monkeypatch.setattr(kernel_warmup, "warmup_b12x_mxfp8_linear", lambda *a, **k: 0)
    monkeypatch.setattr(kernel_warmup, "warmup_b12x_moe_dynamic", lambda *a, **k: 0)

    def fake_mla_query_warmup(*args, **kwargs):
        calls.append((args, kwargs))
        return 4

    monkeypatch.setattr(
        kernel_warmup,
        "warmup_fused_mla_query",
        fake_mla_query_warmup,
    )

    kernel_warmup.kernel_warmup(worker)

    assert calls == [((model,), {"m_values": [1, 2, 4, 32]})]


def test_flashinfer_attention_gate_accepts_pre_kv_runner() -> None:
    runner = SimpleNamespace()

    assert kernel_warmup._uses_flashinfer_attention(runner) is False


def test_kernel_warmup_runs_b12x_moe_warmup(monkeypatch) -> None:
    calls = []
    model = torch.nn.Linear(2, 2)
    worker = _flashinfer_autotune_worker(model)
    worker.scheduler_config.max_num_batched_tokens = 2048
    worker.scheduler_config.max_num_scheduled_tokens = 3072
    worker.vllm_config.compilation_config.cudagraph_capture_sizes = [1, 2, 4, 8]
    worker.vllm_config.compilation_config.compile_sizes = [17, 4096]
    worker.vllm_config.kernel_config.enable_flashinfer_autotune = False

    monkeypatch.setattr(kernel_warmup, "deepseek_v4_mhc_warmup", lambda *a, **k: None)
    monkeypatch.setattr(
        kernel_warmup,
        "flashinfer_sparse_mla_decode_autotune_warmup",
        lambda *a, **k: None,
    )
    monkeypatch.setattr(
        kernel_warmup,
        "deepseek_v4_sparse_mla_attention_warmup",
        lambda *a, **k: None,
    )
    monkeypatch.setattr(kernel_warmup, "minimax_m3_msa_warmup", lambda *a, **k: None)
    monkeypatch.setattr(kernel_warmup, "warmup_b12x_mxfp8_linear", lambda *a, **k: 0)

    def fake_moe_warmup(*args, **kwargs):
        calls.append((args, kwargs))
        return 4

    monkeypatch.setattr(
        kernel_warmup,
        "warmup_b12x_moe_dynamic",
        fake_moe_warmup,
    )

    kernel_warmup.kernel_warmup(worker)

    assert calls == [
        (
            (model,),
            {
                "max_tokens": 4096,
                "token_counts": [2048, 1, 2, 4, 8, 17, 4096, 3072],
                "direct_token_counts": range(1, 9),
            },
        )
    ]


def test_kernel_warmup_runs_b12x_sparse_indexer_warmup(monkeypatch) -> None:
    calls = []
    model = torch.nn.Linear(2, 2)
    worker = _flashinfer_autotune_worker(model)
    worker.vllm_config.kernel_config.enable_flashinfer_autotune = False

    monkeypatch.setattr(kernel_warmup, "deepseek_v4_mhc_warmup", lambda *a, **k: None)
    monkeypatch.setattr(
        kernel_warmup,
        "flashinfer_sparse_mla_decode_autotune_warmup",
        lambda *a, **k: None,
    )
    monkeypatch.setattr(
        kernel_warmup,
        "deepseek_v4_sparse_mla_attention_warmup",
        lambda *a, **k: None,
    )
    monkeypatch.setattr(kernel_warmup, "minimax_m3_msa_warmup", lambda *a, **k: None)
    monkeypatch.setattr(kernel_warmup, "warmup_b12x_mxfp8_linear", lambda *a, **k: 0)
    monkeypatch.setattr(kernel_warmup, "warmup_b12x_moe_dynamic", lambda *a, **k: 0)

    def fake_sparse_indexer_warmup(arg):
        calls.append(arg)
        return 16

    monkeypatch.setattr(
        kernel_warmup,
        "warmup_b12x_sparse_indexer",
        fake_sparse_indexer_warmup,
    )

    kernel_warmup.kernel_warmup(worker)

    assert calls == [worker]


def test_kernel_warmup_skips_flashinfer_autotune_without_flashinfer_kernels(
    monkeypatch,
) -> None:
    calls = _patch_flashinfer_autotune_deps(monkeypatch)
    worker = _flashinfer_autotune_worker(torch.nn.Linear(2, 2))

    kernel_warmup.kernel_warmup(worker)

    assert calls == []


def test_kernel_warmup_runs_flashinfer_autotune_for_attention(
    monkeypatch,
) -> None:
    calls = _patch_flashinfer_autotune_deps(monkeypatch)
    backend = SimpleNamespace(get_name=lambda: "FLASHINFER")
    group = SimpleNamespace(backend=backend)
    worker = _flashinfer_autotune_worker(
        torch.nn.Linear(2, 2),
        attn_groups=[[group]],
    )

    kernel_warmup.kernel_warmup(worker)

    assert calls == [worker.model_runner]


def test_kernel_warmup_runs_flashinfer_autotune_for_model_kernel(
    monkeypatch,
) -> None:
    calls = _patch_flashinfer_autotune_deps(monkeypatch)
    flashinfer_kernel_cls = type(
        "FlashInferKernel",
        (),
        {"__module__": "vllm.model_executor.kernels.linear.scaled_mm.flashinfer"},
    )

    class ModuleWithKernel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.quant_method = SimpleNamespace(
                scheme=SimpleNamespace(fp8_linear=flashinfer_kernel_cls())
            )

    worker = _flashinfer_autotune_worker(ModuleWithKernel())

    kernel_warmup.kernel_warmup(worker)

    assert calls == [worker.model_runner]


def test_kernel_warmup_defers_flashinfer_autotune_until_kv_cache(
    monkeypatch,
) -> None:
    calls = _patch_flashinfer_autotune_deps(monkeypatch)
    flashinfer_kernel_cls = type(
        "FlashInferKernel",
        (),
        {"__module__": "vllm.model_executor.kernels.linear.scaled_mm.flashinfer"},
    )

    class ModuleWithKernel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.quant_method = SimpleNamespace(
                scheme=SimpleNamespace(fp8_linear=flashinfer_kernel_cls())
            )

    worker = _flashinfer_autotune_worker(ModuleWithKernel())
    del worker.model_runner.block_tables

    assert kernel_warmup.kernel_warmup(worker) is False
    assert calls == []
