//===----------------------------------------------------------------------===//
// Copyright (C) 2026 Sophgo Technologies Inc.  All rights reserved.
// TPU-MLIR is licensed under the 2-Clause BSD License.
//===----------------------------------------------------------------------===//

#include "ppl_dyn_fw.h"
#ifndef RECURRENT_GATED_DELTA_RULE_DEVICE_INCLUDED
#include "recurrent_gated_delta_rule.c"
#endif

#define SET_SEQ_PARAM(TYPE, SUFFIX)                                            \
  do {                                                                         \
    tpu_kernel_api_fenced_recurrent_gated_delta_rule_##SUFFIX##_t *p =         \
        (tpu_kernel_api_fenced_recurrent_gated_delta_rule_##SUFFIX##_t *)      \
            param;                                                             \
    const int seq_len = input_spec[0].shape[1];                                \
    const int conv_dim = input_spec[6].shape[1];                               \
    const int conv_width = input_spec[6].shape[2];                             \
    output_spec[0].dtype = input_spec[0].dtype;                               \
    output_spec[0].dims = 4;                                                   \
    output_spec[0].shape[0] = input_spec[0].shape[0];                          \
    output_spec[0].shape[1] = seq_len;                                         \
    output_spec[0].shape[2] = p->num_v_heads;                                 \
    output_spec[0].shape[3] = p->d;                                           \
    output_spec[0].elem_num = seq_len * p->num_v_heads * p->d;                 \
    output_spec[1].dtype = input_spec[0].dtype;                               \
    output_spec[1].dims = 2;                                                   \
    output_spec[1].shape[0] = seq_len;                                         \
    output_spec[1].shape[1] = p->num_v_heads * p->d * p->d;                   \
    output_spec[1].elem_num = seq_len * output_spec[1].shape[1];               \
    output_spec[2].dtype = input_spec[0].dtype;                               \
    output_spec[2].dims = 3;                                                   \
    output_spec[2].shape[0] = seq_len;                                         \
    output_spec[2].shape[1] = conv_dim;                                        \
    output_spec[2].shape[2] = conv_width;                                      \
    output_spec[2].elem_num = seq_len * conv_dim * conv_width;                 \
    if (p->core_num > tpu_core_num())                                          \
      p->core_num = tpu_core_num();                                            \
    /*                                                                        \
     * v8 deliberately uses separate device submissions. The v7 monolithic  \
     * entry let PPL pipeline state leak from the snapshot copy into the       \
     * recurrent tile loop; BM1684X then corrupted 2047 elements in tile 2.   \
     * Each recurrent entry below is now byte-for-byte the standalone cache    \
     * command stream, while copy/conv remain device-side and host-free.        \
     */                                                                        \
    const unsigned long long elem_bytes = 2;                                  \
    const unsigned long long q_stride =                                       \
        (unsigned long long)p->num_k_heads * p->d * elem_bytes;                \
    const unsigned long long v_stride =                                       \
        (unsigned long long)p->num_v_heads * p->d * elem_bytes;                \
    const unsigned long long state_stride =                                   \
        (unsigned long long)p->num_v_heads * p->d * p->d * elem_bytes;         \
    const unsigned long long scalar_stride =                                  \
        (unsigned long long)p->num_v_heads * elem_bytes;                       \
    for (int s = 0; s < seq_len; ++s) {                                        \
      const unsigned long long state_out =                                    \
          output_spec[1].addr + (unsigned long long)s * state_stride;          \
      const unsigned long long state_in =                                     \
          s == 0 ? input_spec[5].addr : state_out - state_stride;              \
      tpu_kernel_api_copy_recurrent_state_##SUFFIX##_t copy;                  \
      memset(&copy, 0, sizeof(copy));                                          \
      copy.ptr_state_out = state_out;                                          \
      copy.ptr_state_in = state_in;                                            \
      copy.B = 1;                                                              \
      copy.core_num = p->core_num;                                             \
      copy.num_k_heads = p->num_k_heads;                                       \
      copy.num_v_heads = p->num_v_heads;                                       \
      copy.d = p->d;                                                           \
      copy.block_h = p->block_h;                                               \
      copy_recurrent_state_##SUFFIX##_entry(&copy);                            \
                                                                                \
      tpu_kernel_api_fenced_recurrent_gated_delta_rule_##SUFFIX##_t step;      \
      memset(&step, 0, sizeof(step));                                          \
      step.ptr_core_attn_out =                                                 \
          output_spec[0].addr + (unsigned long long)s * v_stride;              \
      step.ptr_last_recurrent_state = state_out;                               \
      step.ptr_Q = input_spec[0].addr + (unsigned long long)s * q_stride;       \
      step.ptr_K = input_spec[1].addr + (unsigned long long)s * q_stride;       \
      step.ptr_V = input_spec[2].addr + (unsigned long long)s * v_stride;       \
      step.ptr_g = input_spec[3].addr + (unsigned long long)s * scalar_stride;  \
      step.ptr_beta =                                                          \
          input_spec[4].addr + (unsigned long long)s * scalar_stride;          \
      step.B = 1;                                                              \
      step.scale = p->scale;                                                   \
      step.core_num = p->core_num;                                             \
      step.num_k_heads = p->num_k_heads;                                       \
      step.num_v_heads = p->num_v_heads;                                       \
      step.d = p->d;                                                           \
      step.use_qk_l2norm = p->use_qk_l2norm;                                   \
      step.block_h = p->block_h;                                               \
      memcpy(step.addrs, p->addrs, sizeof(step.addrs));                        \
      fenced_recurrent_gated_delta_rule_##SUFFIX##_entry(&step);               \
    }                                                                          \
    tpu_kernel_api_sequential_conv_state_##SUFFIX##_t conv;                   \
    memset(&conv, 0, sizeof(conv));                                            \
    conv.ptr_conv_steps = output_spec[2].addr;                                 \
    conv.ptr_conv_state = input_spec[6].addr;                                  \
    conv.ptr_conv_updates = input_spec[7].addr;                                \
    conv.S = seq_len;                                                          \
    conv.conv_dim = conv_dim;                                                  \
    conv.conv_width = conv_width;                                              \
    conv.core_num = p->core_num;                                               \
    sequential_conv_state_##SUFFIX##_entry(&conv);                            \
  } while (0)

void dynamic_glb_sequential_recurrent_gated_delta_rule_ctrl(
    void *ctx, void *param, global_tensor_spec_t *input_spec,
    global_tensor_spec_t *output_spec) {
  const int32_t dtype = input_spec[0].dtype;
  const int B = input_spec[0].shape[0];
  const int S = input_spec[0].shape[1];
  output_spec[0].dtype = dtype;
  output_spec[0].dims = 4;
  output_spec[0].shape[0] = B;
  output_spec[0].shape[1] = S;
  output_spec[1].dtype = dtype;
  output_spec[1].dims = 2;
  output_spec[1].shape[0] = S;
  if (dtype == FW_DTYPE_FP16) {
    SET_SEQ_PARAM(fp16, f16);
  } else if (dtype == FW_DTYPE_BFP16) {
    SET_SEQ_PARAM(bf16, bf16);
  }
}

REGISTER_PPL_DYN_OP(PPL_FW_SEQUENTIAL_RECURRENT_GATED_DELTA_RULE,
                    dynamic_glb_sequential_recurrent_gated_delta_rule_ctrl, 0);
