//===----------------------------------------------------------------------===//
// Copyright (C) 2026 Sophgo Technologies Inc.  All rights reserved.
// TPU-MLIR is licensed under the 2-Clause BSD License.
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/MathUtils.h"
#include "tpu_mlir/Support/Module.h"
#include <cmath>
#include <cstring>
#include <vector>

int64_t top::SequentialRecurrentGatedDeltaRuleOp::getFLOPs() {
  auto v = module::getShape(getValue());
  return v[0] * v[1] * getNumVHeads() * (7 * getD() * getD() + 2 * getD());
}

LogicalResult
top::SequentialRecurrentGatedDeltaRuleOp::init(InferenceParameter &) {
  return success();
}
void top::SequentialRecurrentGatedDeltaRuleOp::deinit(InferenceParameter &) {}

LogicalResult
top::SequentialRecurrentGatedDeltaRuleOp::inference(InferenceParameter &p) {
  auto q_shape = module::getShape(getQuery());
  auto conv_shape = module::getShape(getConvState());
  const int64_t B = q_shape[0], S = q_shape[1];
  const int64_t nkh = getNumKHeads(), nvh = getNumVHeads(), d = getD();
  if (B != 1 || S < 1 || S > 16)
    return failure();
  const int64_t groups = nvh / nkh;
  const int64_t state_elems = nvh * d * d;
  std::vector<float> state(p.inputs[5], p.inputs[5] + state_elems);

  for (int64_t s = 0; s < S; ++s) {
    for (int64_t h = 0; h < nvh; ++h) {
      const int64_t kh = h / groups;
      const float *q = p.inputs[0] + (s * nkh + kh) * d;
      const float *k = p.inputs[1] + (s * nkh + kh) * d;
      const float *v = p.inputs[2] + (s * nvh + h) * d;
      float *st = state.data() + h * d * d;
      std::vector<float> qn(q, q + d), kn(k, k + d);
      if (getUseQkL2norm()) {
        float qs = 1e-6f, ks = 1e-6f;
        for (int64_t x = 0; x < d; ++x) {
          qs += qn[x] * qn[x];
          ks += kn[x] * kn[x];
        }
        qs = 1.0f / std::sqrt(qs);
        ks = 1.0f / std::sqrt(ks);
        for (int64_t x = 0; x < d; ++x) {
          qn[x] *= qs;
          kn[x] *= ks;
        }
      }
      for (auto &x : qn)
        x *= static_cast<float>(getScale().convertToDouble());
      const float decay = std::exp(p.inputs[3][s * nvh + h]);
      const float beta = p.inputs[4][s * nvh + h];
      for (int64_t x = 0; x < d * d; ++x)
        st[x] *= decay;
      std::vector<float> delta(d);
      for (int64_t j = 0; j < d; ++j) {
        float mem = 0;
        for (int64_t x = 0; x < d; ++x)
          mem += kn[x] * st[x * d + j];
        delta[j] = (v[j] - mem) * beta;
      }
      for (int64_t x = 0; x < d; ++x)
        for (int64_t j = 0; j < d; ++j)
          st[x * d + j] += kn[x] * delta[j];
      float *out = p.outputs[0] + (s * nvh + h) * d;
      for (int64_t j = 0; j < d; ++j) {
        out[j] = 0;
        for (int64_t x = 0; x < d; ++x)
          out[j] += qn[x] * st[x * d + j];
      }
    }
    std::memcpy(p.outputs[1] + s * state_elems, state.data(),
                state_elems * sizeof(float));
  }
  const int64_t conv_channels = conv_shape[1];
  const int64_t conv_width = conv_shape[2];
  for (int64_t s = 0; s < S; ++s) {
    float *snapshot = p.outputs[2] + s * conv_channels * conv_width;
    for (int64_t c = 0; c < conv_channels; ++c) {
      for (int64_t w = 0; w < conv_width; ++w) {
        const int64_t source = s + w + 1;
        snapshot[c * conv_width + w] =
            source < conv_width
                ? p.inputs[6][c * conv_width + source]
                : p.inputs[7][c * S + source - conv_width];
      }
    }
  }
  return success();
}

void top::SequentialRecurrentGatedDeltaRuleOp::shape_inference() {
  auto q = module::getShape(getQuery());
  auto conv = module::getShape(getConvState());
  const int64_t B = q[0], S = q[1], nvh = getNumVHeads(), d = getD();
  module::setShapeOrVerify(getAttnOut(), {B, S, nvh, d});
  module::setShapeOrVerify(getRecurrentSteps(), {S, nvh * d * d});
  module::setShapeOrVerify(getConvSteps(), {S, conv[1], conv[2]});
}
