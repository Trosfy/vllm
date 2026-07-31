# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import subprocess
from pathlib import Path


_REPO_ROOT = Path(__file__).parents[2]
_LAUNCHER = _REPO_ROOT / "serve-ds4-flash.sh"


def _dry_run(tmp_path: Path, **overrides: str) -> str:
    env = {
        "PATH": os.environ["PATH"],
        "HOME": str(tmp_path),
        "XDG_CACHE_HOME": str(tmp_path / "cache"),
        "DRY_RUN": "1",
        **overrides,
    }
    result = subprocess.run(
        ["bash", str(_LAUNCHER)],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stderr


def test_ds4_launcher_defaults_to_0731_fixed_k7(tmp_path: Path) -> None:
    output = _dry_run(tmp_path)

    assert "mode=dspark depth=fixed" in output
    assert "model=deepseek-ai/DeepSeek-V4-Flash-0731" in output
    assert "9e165c30e2704aec5d9d593cce3eebd58bbef1cb" in output
    assert "num_speculative_tokens\\\":7" in output
    assert "max_seqs=16 graph=128" in output
    assert "--max-model-len 131072" in output
    assert "--gpu-memory-utilization 0.97" in output


def test_ds4_launcher_dynamic_depth_enables_capacity_mode(tmp_path: Path) -> None:
    output = _dry_run(tmp_path, DSPARK_DEPTH_MODE="dynamic")

    assert "mode=dspark depth=dynamic" in output
    assert "dspark_capacity_verification_mode\\\":\\\"varlen" in output
    assert "dspark_sps_curve\\\":\\\"auto" in output


def test_ds4_launcher_accepts_cluster_style_aliases(tmp_path: Path) -> None:
    output = _dry_run(
        tmp_path,
        DCP="1",
        NUM_SPECULATIVE_TOKENS="5",
    )

    assert "dcp=1" in output
    assert "num_speculative_tokens\\\":5" in output


def test_ds4_launcher_standard_mtp_uses_standard_checkpoint(tmp_path: Path) -> None:
    output = _dry_run(tmp_path, MODE="mtp2")

    assert "mode=mtp2 depth=disabled" in output
    assert "model=deepseek-ai/DeepSeek-V4-Flash" in output
    assert "DeepSeek-V4-Flash-0731" not in output
    assert "method\\\":\\\"mtp" in output
    assert "num_speculative_tokens\\\":2" in output


def test_ds4_launcher_can_disable_dspark_on_0731(tmp_path: Path) -> None:
    output = _dry_run(tmp_path, MODE="dspark-mtp0")

    assert "mode=dspark-mtp0 depth=disabled" in output
    assert "model=deepseek-ai/DeepSeek-V4-Flash-0731" in output
    assert "--speculative-config" not in output
