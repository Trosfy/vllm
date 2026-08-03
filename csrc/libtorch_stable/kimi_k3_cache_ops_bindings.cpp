// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

// Companion registration fragment for running HH Python against an older
// _C_stable_libtorch image. A current full vLLM build already carries these
// schemas in torch_bindings.cpp; Python loads this fragment only when they are
// absent from the main extension.

#include "ops.h"
#include "core/registration.h"

#include <torch/csrc/stable/library.h>

STABLE_TORCH_LIBRARY_FRAGMENT(_C, ops) {
  ops.def(
      "fused_kimi_k3_mla_key_concat_kv_cache_insert("
      "Tensor! q, Tensor k_nope, Tensor k_pe, Tensor kv_c_normed, "
      "Tensor! k_out, Tensor! k_cache, Tensor slot_mapping, "
      "int cache_block_size, Tensor? position_ids=None, "
      "Tensor? cos_sin_cache=None) -> ()");
  ops.def(
      "fused_kimi_k3_mla_key_concat_ds_mla_insert("
      "Tensor! q, Tensor k_nope, Tensor k_pe, Tensor kv_c_normed, "
      "Tensor! k_out, Tensor! k_cache, Tensor slot_mapping, "
      "int cache_block_size, Tensor? position_ids=None, "
      "Tensor? cos_sin_cache=None) -> ()");
  ops.def(
      "fused_kimi_k3_mla_qkv_quant_kv_cache_fp8_insert("
      "Tensor q, Tensor k_nope, Tensor k_pe, Tensor kv_c_normed, Tensor v, "
      "Tensor! q_fp8, Tensor! k_fp8, Tensor! v_fp8, Tensor! k_cache, "
      "Tensor slot_mapping, Tensor q_scale_inv, Tensor k_scale_inv, "
      "Tensor v_scale_inv, Tensor cache_scale_inv, int cache_block_size, "
      "Tensor? position_ids=None, Tensor? cos_sin_cache=None) -> ()");
  ops.def(
      "fused_kimi_k3_mla_decode_q_concat_kv_cache_insert("
      "Tensor ql_nope, Tensor q_pe, Tensor kv_c_normed, Tensor k_pe, "
      "Tensor! mqa_q, Tensor! k_cache, Tensor slot_mapping, "
      "int cache_block_size, Tensor? position_ids=None, "
      "Tensor? cos_sin_cache=None) -> ()");
  ops.def(
      "fused_kimi_k3_mla_decode_q_concat_kv_cache_fp8_insert("
      "Tensor ql_nope, Tensor q_pe, Tensor kv_c_normed, Tensor k_pe, "
      "Tensor! mqa_q, Tensor! k_cache, Tensor slot_mapping, "
      "Tensor q_scale_inv, Tensor cache_scale_inv, int cache_block_size, "
      "Tensor? position_ids=None, Tensor? cos_sin_cache=None) -> ()");
  ops.def(
      "fused_kimi_k3_mla_decode_q_concat_ds_mla_insert("
      "Tensor ql_nope, Tensor q_pe, Tensor kv_c_normed, Tensor k_pe, "
      "Tensor! mqa_q, Tensor! k_cache, Tensor slot_mapping, "
      "int cache_block_size, Tensor? position_ids=None, "
      "Tensor? cos_sin_cache=None) -> ()");
}

STABLE_TORCH_LIBRARY_IMPL(_C, CUDA, ops) {
  ops.impl("fused_kimi_k3_mla_key_concat_kv_cache_insert",
           TORCH_BOX(&fused_kimi_k3_mla_key_concat_kv_cache_insert));
  ops.impl("fused_kimi_k3_mla_key_concat_ds_mla_insert",
           TORCH_BOX(&fused_kimi_k3_mla_key_concat_ds_mla_insert));
  ops.impl("fused_kimi_k3_mla_qkv_quant_kv_cache_fp8_insert",
           TORCH_BOX(&fused_kimi_k3_mla_qkv_quant_kv_cache_fp8_insert));
  ops.impl("fused_kimi_k3_mla_decode_q_concat_kv_cache_insert",
           TORCH_BOX(&fused_kimi_k3_mla_decode_q_concat_kv_cache_insert));
  ops.impl("fused_kimi_k3_mla_decode_q_concat_kv_cache_fp8_insert",
           TORCH_BOX(&fused_kimi_k3_mla_decode_q_concat_kv_cache_fp8_insert));
  ops.impl("fused_kimi_k3_mla_decode_q_concat_ds_mla_insert",
           TORCH_BOX(&fused_kimi_k3_mla_decode_q_concat_ds_mla_insert));
}

REGISTER_EXTENSION(_kimi_k3_cache_ops)
