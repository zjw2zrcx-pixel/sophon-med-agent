//===----------------------------------------------------------------------===//
// Copyright (C) 2026 Sophgo Technologies Inc.  All rights reserved.
// TPU-MLIR is licensed under the 2-Clause BSD License.
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/Transforms/Codegen/Dynamic/DynamicLayer.hpp"
#include "tpu_mlir/Support/MathUtils.h"
using namespace tpu_mlir::backend;

static recurrent_gated_delta_rule_spec_t
make_param(tpu::SequentialRecurrentGatedDeltaRuleOp op) {
  recurrent_gated_delta_rule_spec_t p = {0};
  p.num_k_heads = op.getNumKHeads();
  p.num_v_heads = op.getNumVHeads();
  p.d = op.getD();
  p.use_qk_l2norm = op.getUseQkL2norm();
  p.scale = static_cast<float>(op.getScale().convertToDouble());
  return p;
}

void tpu::SequentialRecurrentGatedDeltaRuleOp::codegen_global_bm1684x() {
  auto in = BM168x::get_input_spec(getOperation());
  auto out = BM168x::get_output_spec(getOperation());
  auto p = make_param(*this);
  BM168x::call_ppl_global_func(
      "api_sequential_recurrent_gated_delta_rule_global", &p, sizeof(p),
      in->data(), out->data());
}

int64_t tpu::SequentialRecurrentGatedDeltaRuleOp::get_fw_type_bm1684x() {
  return PPL_FW_SEQUENTIAL_RECURRENT_GATED_DELTA_RULE;
}

int64_t tpu::SequentialRecurrentGatedDeltaRuleOp::dyn_codegen_global_bm1684x(
    void *buffer) {
  auto in = BM168x::get_input_spec(getOperation());
  auto out = BM168x::get_output_spec(getOperation());
  auto p = make_param(*this);
  return BM168x::call_ppl_dyn_func(
      "api_dyn_sequential_recurrent_gated_delta_rule_global", &p, in->data(),
      out->data(), buffer);
}
