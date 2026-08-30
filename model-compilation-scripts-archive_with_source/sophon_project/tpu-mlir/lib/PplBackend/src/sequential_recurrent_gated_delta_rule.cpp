//===----------------------------------------------------------------------===//
// Copyright (C) 2026 Sophgo Technologies Inc.  All rights reserved.
// TPU-MLIR is licensed under the 2-Clause BSD License.
//===----------------------------------------------------------------------===//

#include "recurrent_gated_delta_rule.h"
#include "ppl_static_host.h"
#include <cstdio>
#include <cstdlib>

namespace {
int run_sequential(tensor_spec_t *in, tensor_spec_t *out,
                   const recurrent_gated_delta_rule_spec_t *p, int block_h) {
  const int B = in[0].shape[0];
  const int S = in[0].shape[1];
  const int cores = get_core_num();
  if (in[0].dtype == DTYPE_BFP16) {
    return sequential_recurrent_gated_delta_rule_bf16(
        out[0].addr, out[1].addr, out[2].addr, in[5].addr, in[6].addr,
        in[7].addr, in[0].addr, in[1].addr, in[2].addr, in[3].addr,
        in[4].addr, B, S, in[6].shape[1], in[6].shape[2], p->scale, cores,
        p->num_k_heads, p->num_v_heads, p->d,
        p->use_qk_l2norm ? 1 : 0, block_h);
  }
  return sequential_recurrent_gated_delta_rule_f16(
      out[0].addr, out[1].addr, out[2].addr, in[5].addr, in[6].addr,
      in[7].addr, in[0].addr, in[1].addr, in[2].addr, in[3].addr,
      in[4].addr, B, S, in[6].shape[1], in[6].shape[2], p->scale, cores,
      p->num_k_heads, p->num_v_heads, p->d,
      p->use_qk_l2norm ? 1 : 0, block_h);
}

int choose_block_h(tensor_spec_t *in, tensor_spec_t *out,
                   const recurrent_gated_delta_rule_spec_t *p) {
  int block_h = p->num_k_heads / get_core_num() / 2;
  if (block_h < 1)
    block_h = 1;
  int ret = 0;
  while (block_h > 0) {
    ret = run_sequential(in, out, p, block_h);
    if (ret == PplL2AddrAssignErr || ret == PplLocalAddrAssignErr)
      block_h /= 2;
    else
      break;
  }
  if (block_h == 0 || ret != 0) {
    std::fprintf(stderr, "sequential recurrent GDN kernel failed: %d\n", ret);
    std::exit(-1);
  }
  return block_h;
}

int run_fenced(tensor_spec_t *in, tensor_spec_t *out,
               const recurrent_gated_delta_rule_spec_t *p, int block_h) {
  const int cores = get_core_num();
  if (in[0].dtype == DTYPE_BFP16) {
    return fenced_recurrent_gated_delta_rule_bf16(
        out[0].addr, out[1].addr, in[0].addr, in[1].addr, in[2].addr,
        in[3].addr, in[4].addr, 1, p->scale, cores, p->num_k_heads,
        p->num_v_heads, p->d, p->use_qk_l2norm ? 1 : 0, block_h);
  }
  return fenced_recurrent_gated_delta_rule_f16(
      out[0].addr, out[1].addr, in[0].addr, in[1].addr, in[2].addr,
      in[3].addr, in[4].addr, 1, p->scale, cores, p->num_k_heads,
      p->num_v_heads, p->d, p->use_qk_l2norm ? 1 : 0, block_h);
}

int choose_fenced_block_h(tensor_spec_t *in, tensor_spec_t *out,
                          const recurrent_gated_delta_rule_spec_t *p) {
  int block_h = p->num_k_heads / get_core_num() / 2;
  if (block_h < 1)
    block_h = 1;
  int ret = 0;
  while (block_h > 0) {
    ret = run_fenced(in, out, p, block_h);
    if (ret == PplL2AddrAssignErr || ret == PplLocalAddrAssignErr)
      block_h /= 2;
    else
      break;
  }
  if (block_h == 0 || ret != 0) {
    std::fprintf(stderr, "fenced recurrent GDN kernel failed: %d\n", ret);
    std::exit(-1);
  }
  return block_h;
}
} // namespace

extern "C" {

void api_sequential_recurrent_gated_delta_rule_global(void *param,
                                                       size_t,
                                                       void *input,
                                                       void *output) {
  auto *p = static_cast<recurrent_gated_delta_rule_spec_t *>(param);
  auto *in = static_cast<tensor_spec_t *>(input);
  auto *out = static_cast<tensor_spec_t *>(output);
  choose_block_h(in, out, p);
}

int api_dyn_sequential_recurrent_gated_delta_rule_global(
    void *param, void *input, void *output, void *buffer) {
  auto *p = static_cast<recurrent_gated_delta_rule_spec_t *>(param);
  auto *in = static_cast<tensor_spec_t *>(input);
  auto *out = static_cast<tensor_spec_t *>(output);
  int block_h = p->num_k_heads;
  if (buffer != nullptr)
    block_h = choose_fenced_block_h(in, out, p);
  if (in[0].dtype == DTYPE_BFP16) {
    return fill_fenced_recurrent_gated_delta_rule_bf16_struct(
        out[0].addr, in[5].addr, in[0].addr, in[1].addr, in[2].addr,
        in[3].addr, in[4].addr, 1, p->scale, get_core_num(),
        p->num_k_heads, p->num_v_heads, p->d,
        p->use_qk_l2norm ? 1 : 0, block_h, buffer);
  }
  return fill_fenced_recurrent_gated_delta_rule_f16_struct(
      out[0].addr, in[5].addr, in[0].addr, in[1].addr, in[2].addr,
      in[3].addr, in[4].addr, 1, p->scale, get_core_num(), p->num_k_heads,
      p->num_v_heads, p->d, p->use_qk_l2norm ? 1 : 0, block_h, buffer);
}

} // extern "C"
