// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

// Companion registration fragment for the SiTU activations introduced with
// Kimi K3.  The implementation is compiled from activation_kernels.cu while
// the preserved GG image continues to provide every older activation op.

#include "ops.h"
#include "core/registration.h"

#include <torch/csrc/stable/library.h>

STABLE_TORCH_LIBRARY_FRAGMENT(_C, ops) {
  ops.def(
      "situ_and_mul(Tensor! out, Tensor input, float beta=1.0, float "
      "linear_beta=-1.0) -> ()");
  ops.def(
      "masked_situ_and_mul(Tensor! out, Tensor input, Tensor "
      "expert_num_tokens, float beta=1.0, float linear_beta=-1.0) -> ()");
}

STABLE_TORCH_LIBRARY_IMPL(_C, CUDA, ops) {
  ops.impl("situ_and_mul", TORCH_BOX(&situ_and_mul));
  ops.impl("masked_situ_and_mul", TORCH_BOX(&masked_situ_and_mul));
}

REGISTER_EXTENSION(_kimi_k3_activation_ops)
