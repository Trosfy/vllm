// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

// Companion registration fragment for HH's fused KDA decode kernel.  It lets
// HH Python sources run against a preserved image whose full stable-libtorch
// extension was built before this operator was added.

#include "ops.h"
#include "core/registration.h"

#include <torch/csrc/stable/library.h>

STABLE_TORCH_LIBRARY_FRAGMENT(_C, ops) {
  ops.def(
      "fused_kda_decode("
      "Tensor x, Tensor weight, Tensor? bias, Tensor! conv_state, "
      "Tensor raw_g, Tensor raw_beta, Tensor A_log, Tensor dt_bias, "
      "Tensor state_indices, Tensor! state, Tensor! out, "
      "float? lower_bound=None, Tensor? output_gate=None, "
      "Tensor? norm_weight=None, float norm_eps=1e-5) -> ()");
}

STABLE_TORCH_LIBRARY_IMPL(_C, CUDA, ops) {
  ops.impl("fused_kda_decode", TORCH_BOX(&fused_kda_decode));
}

REGISTER_EXTENSION(_kimi_k3_kda_ops)
