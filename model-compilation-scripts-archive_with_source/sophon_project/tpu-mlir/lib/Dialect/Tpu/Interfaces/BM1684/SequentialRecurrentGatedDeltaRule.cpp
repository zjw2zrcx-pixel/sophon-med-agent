//===----------------------------------------------------------------------===//
// Copyright (C) 2026 Sophgo Technologies Inc.  All rights reserved.
// TPU-MLIR is licensed under the 2-Clause BSD License.
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Backend/BM168x/BM1684X.h"
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"

using namespace tpu_mlir::backend;

void tpu::SequentialRecurrentGatedDeltaRuleOp::codegen_global_bm1684() {
  UNREACHABLE_THIS("BM1684 is not supported");
}

uint32_t tpu::SequentialRecurrentGatedDeltaRuleOp::dyn_codegen_global_bm1684(
    void *) {
  UNREACHABLE_THIS("BM1684 is not supported");
  return 0;
}

int64_t tpu::SequentialRecurrentGatedDeltaRuleOp::get_fw_type_bm1684() {
  return -1;
}

