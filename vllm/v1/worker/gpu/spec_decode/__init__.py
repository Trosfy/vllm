# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from vllm.config import VllmConfig


def init_speculator(vllm_config: VllmConfig, device: torch.device):
    speculative_config = vllm_config.speculative_config
    assert speculative_config is not None
    if speculative_config.method == "dflash":
        from vllm.v1.worker.gpu.spec_decode.dflash.speculator import (
            DFlashSpeculator,
        )

        return DFlashSpeculator(vllm_config, device)
    elif speculative_config.method == "dspark":
        draft_arch = getattr(
            speculative_config.draft_model_config.hf_config,
            "architectures",
            [None],
        )[0]
        if draft_arch == "Qwen3DSparkModel":
            from vllm.v1.worker.gpu.spec_decode.dspark.qwen3_speculator import (
                Qwen3DSparkSpeculator,
            )

            return Qwen3DSparkSpeculator(vllm_config, device)
        else:
            from vllm.v1.worker.gpu.spec_decode.dspark.speculator import (
                DSparkSpeculator,
            )

            return DSparkSpeculator(vllm_config, device)
    elif speculative_config.use_gemma4_mtp():
        from vllm.v1.worker.gpu.spec_decode.gemma4.speculator import (
            Gemma4Speculator,
        )

        return Gemma4Speculator(vllm_config, device)
    elif speculative_config.method == "mtp":
        from vllm.v1.worker.gpu.spec_decode.mtp.speculator import MTPSpeculator

        return MTPSpeculator(vllm_config, device)
    elif speculative_config.method == "dflash":
        from vllm.v1.worker.gpu.spec_decode.dflash.speculator import (
            DFlashSpeculator,
        )

        return DFlashSpeculator(vllm_config, device)
    elif speculative_config.use_eagle():
        from vllm.v1.worker.gpu.spec_decode.eagle.speculator import (
            EagleSpeculator,
        )

        return EagleSpeculator(vllm_config, device)
    else:
        raise NotImplementedError(f"{speculative_config.method} is not supported yet.")
