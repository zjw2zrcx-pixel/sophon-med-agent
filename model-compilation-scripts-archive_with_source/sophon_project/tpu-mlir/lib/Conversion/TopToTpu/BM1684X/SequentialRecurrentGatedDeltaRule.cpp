//===----------------------------------------------------------------------===//
// Copyright (C) 2026 Sophgo Technologies Inc.  All rights reserved.
// TPU-MLIR is licensed under the 2-Clause BSD License.
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Conversion/TopToTpu/LoweringBM1684X.h"

namespace tpu_mlir {
namespace bm1684x {

void SequentialRecurrentGatedDeltaRuleLowering::LoweringF32(
    PatternRewriter &rewriter,
    top::SequentialRecurrentGatedDeltaRuleOp op) const {
  lowering_common_f32<tpu::SequentialRecurrentGatedDeltaRuleOp>(rewriter, op);
}

void SequentialRecurrentGatedDeltaRuleLowering::LoweringINT8(
    PatternRewriter &rewriter, top::SequentialRecurrentGatedDeltaRuleOp op,
    bool asymmetric) const {
  lowering_common_f16<tpu::SequentialRecurrentGatedDeltaRuleOp>(rewriter, op);
}

void SequentialRecurrentGatedDeltaRuleLowering::LoweringINT4(
    PatternRewriter &rewriter, top::SequentialRecurrentGatedDeltaRuleOp op,
    bool asymmetric) const {
  lowering_common_f16<tpu::SequentialRecurrentGatedDeltaRuleOp>(rewriter, op);
}

void SequentialRecurrentGatedDeltaRuleLowering::LoweringBF16(
    PatternRewriter &rewriter,
    top::SequentialRecurrentGatedDeltaRuleOp op) const {
  lowering_common_bf16<tpu::SequentialRecurrentGatedDeltaRuleOp>(rewriter, op);
}

void SequentialRecurrentGatedDeltaRuleLowering::LoweringF16(
    PatternRewriter &rewriter,
    top::SequentialRecurrentGatedDeltaRuleOp op) const {
  lowering_common_f16<tpu::SequentialRecurrentGatedDeltaRuleOp>(rewriter, op);
}

void SequentialRecurrentGatedDeltaRuleLowering::LoweringF8(
    PatternRewriter &, top::SequentialRecurrentGatedDeltaRuleOp op) const {
  UNREACHABLE_OP("Not Implemented", op);
}

void SequentialRecurrentGatedDeltaRuleLowering::LoweringQuantized(
    PatternRewriter &rewriter,
    top::SequentialRecurrentGatedDeltaRuleOp op) const {
  std::vector<Type> types(op->getResultTypes().begin(),
                          op->getResultTypes().end());
  lowering_common<tpu::SequentialRecurrentGatedDeltaRuleOp>(
      rewriter, op.getOperation(), types);
}

} // namespace bm1684x
} // namespace tpu_mlir
