# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch.nn as nn

from vllm.config import VllmConfig, replace
from vllm.distributed.parallel_state import get_pp_group
from vllm.model_executor.model_loader import get_model


def load_qwen3_dspark_model(
    target_model: nn.Module, vllm_config: VllmConfig
) -> nn.Module:
    speculative_config = vllm_config.speculative_config
    assert speculative_config is not None
    draft_model_config = speculative_config.draft_model_config

    from vllm.compilation.backends import set_model_tag

    # DSpark uses non-causal attention.
    causal = False
    draft_cache_config = vllm_config.cache_config
    # The external Qwen3 DSpark drafter is not an MLA model. It must not inherit
    # the target model's DeepSeek/GLM MLA KV layout (fp8_ds_mla), which no
    # non-causal Qwen3 attention backend can consume.
    draft_cache_dtype = getattr(speculative_config, "draft_kv_cache_dtype", None)
    if draft_cache_dtype is None and draft_cache_config.cache_dtype == "fp8_ds_mla":
        draft_cache_dtype = "bfloat16"
    if draft_cache_dtype is not None:
        draft_cache_config = replace(
            draft_cache_config,
            cache_dtype=draft_cache_dtype,
        )
    draft_vllm_config = replace(
        vllm_config,
        cache_config=draft_cache_config,
        attention_config=replace(
            vllm_config.attention_config,
            use_non_causal=not causal,
            backend=speculative_config.attention_backend,
        ),
    )

    with set_model_tag("dspark_head"):
        draft_model = get_model(
            vllm_config=draft_vllm_config, model_config=draft_model_config
        )

    if get_pp_group().world_size != 1:
        raise NotImplementedError("DSpark does not support pipeline parallelism.")

    # Self-contained dense DSpark drafts ship their own embed_tokens and lm_head,
    # so aliasing the target's would clobber the loaded weights.
    if not getattr(draft_model, "dspark_shares_target_embeddings", False):
        return draft_model

    target_language_model = (
        target_model.get_language_model()
        if hasattr(target_model, "get_language_model")
        else target_model
    )
    target_inner = target_language_model.model
    draft_inner = draft_model.model

    # Share the vocab embedding (target.model.embed_tokens -> draft.model).
    target_embed = getattr(target_inner, "embed_tokens", None)
    if target_embed is not None:
        if getattr(draft_inner, "embed_tokens", None) is not None:
            del draft_inner.embed_tokens
        draft_inner.embed_tokens = target_embed

    # Share the LM head (target.lm_head -> draft.lm_head).
    target_lm_head = getattr(target_model, "lm_head", None)
    if target_lm_head is not None:
        if getattr(draft_model, "lm_head", None) is not None:
            del draft_model.lm_head
        draft_model.lm_head = target_lm_head

    return draft_model
