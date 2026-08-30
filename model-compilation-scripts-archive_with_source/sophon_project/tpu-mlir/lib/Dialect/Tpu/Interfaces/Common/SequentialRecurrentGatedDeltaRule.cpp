//===----------------------------------------------------------------------===//
// Copyright (C) 2026 Sophgo Technologies Inc.  All rights reserved.
// TPU-MLIR is licensed under the 2-Clause BSD License.
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/Float16.h"
#include "tpu_mlir/Support/MathUtils.h"

LogicalResult
tpu::SequentialRecurrentGatedDeltaRuleOp::init(InferenceParameter &) {
  return success();
}
void tpu::SequentialRecurrentGatedDeltaRuleOp::deinit(InferenceParameter &) {}
LogicalResult
tpu::SequentialRecurrentGatedDeltaRuleOp::inference(InferenceParameter &) {
  return success();
}
bool tpu::SequentialRecurrentGatedDeltaRuleOp::support_multi_core() {
  return true;
}
