// Backend-neutral subset of sherpa-onnx's keyword transducer decoder.
//
// This file deliberately owns only ContextGraph-like keyword state and
// modified-beam bookkeeping.  The caller supplies RNN-T joiner logits, so
// encoder/decoder/joiner execution can remain on Sophgo TPU through SAIL.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <stdexcept>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace {
constexpr int kBlank = 0;
constexpr int kUnk = 2;  // <unk> in the bundled tokens.txt
constexpr int kInitialContextToken = -1;

double LogAdd(double x, double y) {
  if (x < y) std::swap(x, y);
  if (y == -std::numeric_limits<double>::infinity()) return x;
  return x + std::log1p(std::exp(y - x));
}

struct Hypothesis {
  std::vector<int> ys;
  std::vector<float> ys_probs;
  double log_prob = 0;
  int trailing_blanks = 0;
  int context_state = 0;  // length of the phrase prefix currently matched
};

struct Candidate {
  double acoustic_score;
  int hyp_index;
  int token;
  float token_log_prob;
};
}  // namespace

class KeywordSearch {
 public:
  KeywordSearch(std::vector<int> keyword, float score = 2.0f,
                float threshold = 0.18f, int max_active_paths = 4,
                int num_trailing_blanks = 1)
      : keyword_(std::move(keyword)),
        score_(score),
        threshold_(threshold),
        max_active_paths_(max_active_paths),
        num_trailing_blanks_(num_trailing_blanks) {
    if (keyword_.empty()) throw std::invalid_argument("keyword must not be empty");
    if (max_active_paths_ <= 0) throw std::invalid_argument("max_active_paths must be positive");
    BuildFailureTable();
    Reset();
  }

  void Reset() {
    hyps_.clear();
    // TransducerKeywordDecoder::GetEmptyResult() uses context-size blanks
    // {-1, 0}; -1 is the padding/sos history token, not the RNN-T blank.
    hyps_.push_back({{kInitialContextToken, kBlank}, {}, 0.0, 0, 0});
  }

  // Histories are deliberately exposed separately.  Python evaluates one TPU
  // decoder/joiner pair for each active path and passes the logits to Step().
  std::vector<std::vector<int>> Histories() const {
    std::vector<std::vector<int>> ans;
    ans.reserve(hyps_.size());
    for (const auto &h : hyps_) {
      ans.push_back({h.ys[h.ys.size() - 2], h.ys[h.ys.size() - 1]});
    }
    return ans;
  }

  bool Step(py::array_t<float, py::array::c_style | py::array::forcecast> logits) {
    const auto info = logits.request();
    if (info.ndim != 2 || static_cast<size_t>(info.shape[0]) != hyps_.size()) {
      throw std::invalid_argument("logits must be float32 [num_active_paths, vocab]");
    }
    const int vocab = static_cast<int>(info.shape[1]);
    if (vocab <= kBlank) throw std::invalid_argument("vocabulary must contain blank");
    const auto *p = static_cast<const float *>(info.ptr);

    // This is the same flattened paths x vocabulary global TopK used by
    // OnlineTransducerModifiedBeamSearchDecoder::GetTopK().
    std::vector<Candidate> candidates;
    candidates.reserve(hyps_.size() * static_cast<size_t>(vocab));
    for (size_t i = 0; i != hyps_.size(); ++i) {
      const float *row = p + i * vocab;
      float maximum = row[0];
      for (int j = 1; j != vocab; ++j) maximum = std::max(maximum, row[j]);
      double sum = 0;
      for (int j = 0; j != vocab; ++j) sum += std::exp(static_cast<double>(row[j] - maximum));
      const float log_z = maximum + static_cast<float>(std::log(sum));
      for (int j = 0; j != vocab; ++j) {
        const float logp = row[j] - log_z;
        candidates.push_back({hyps_[i].log_prob + logp, static_cast<int>(i), j, logp});
      }
    }
    const size_t keep = std::min(candidates.size(), static_cast<size_t>(max_active_paths_));
    std::partial_sort(candidates.begin(), candidates.begin() + keep, candidates.end(),
                      [](const Candidate &a, const Candidate &b) {
                        return a.acoustic_score > b.acoustic_score;
                      });

    // Sherpa Hypotheses::Add merges identical complete token histories by
    // log-add-exp.  The highest individual path supplies auxiliary state.
    std::map<std::vector<int>, Hypothesis> merged;
    for (size_t n = 0; n != keep; ++n) {
      const auto &c = candidates[n];
      const auto &old = hyps_[c.hyp_index];
      Hypothesis next = old;
      double total = c.acoustic_score;
      // Sherpa's keyword decoder deliberately treats <unk> as a blank too.
      // It must retain the ContextGraph state rather than reset the phrase.
      if (c.token == kBlank || c.token == kUnk) {
        ++next.trailing_blanks;
      } else {
        next.ys.push_back(c.token);
        next.trailing_blanks = 0;
        const auto transition = ForwardOneStep(old.context_state, c.token);
        next.context_state = transition.first;
        total += transition.second;
        if (next.context_state == 0) {
          // Matches sherpa's root-state cleanup: irrelevant token histories
          // do not keep influencing the prediction network indefinitely.
          next.ys = {kInitialContextToken, kBlank};
          next.ys_probs.clear();
        } else {
          // This is intentionally unconditional, like sherpa's
          // TransducerKeywordDecoder.  Only a transition all the way back to
          // root clears ys_probs; a failure-link transition retains it.
          next.ys_probs.push_back(std::exp(c.token_log_prob));
        }
      }
      next.log_prob = total;
      auto it = merged.find(next.ys);
      if (it == merged.end()) {
        merged.emplace(next.ys, std::move(next));
      } else {
        const double combined = LogAdd(it->second.log_prob, next.log_prob);
        if (next.log_prob > it->second.log_prob) it->second = std::move(next);
        it->second.log_prob = combined;
      }
    }
    hyps_.clear();
    for (auto &item : merged) hyps_.push_back(std::move(item.second));

    const auto best = std::max_element(hyps_.begin(), hyps_.end(),
                                       [](const Hypothesis &a, const Hypothesis &b) {
                                         return a.log_prob < b.log_prob;
                                       });
    if (best->context_state != static_cast<int>(keyword_.size()) ||
        best->trailing_blanks <= num_trailing_blanks_ ||
        best->ys_probs.size() < keyword_.size()) {
      return false;
    }
    double average = 0;
    for (size_t i = 0; i != keyword_.size(); ++i)
      average += best->ys_probs[i];
    average /= keyword_.size();
    if (average < threshold_) return false;
    Reset();  // KeywordSpotterTransducerImpl resets after each accepted phrase.
    return true;
  }

 private:
  void BuildFailureTable() {
    failure_.assign(keyword_.size(), 0);
    for (size_t i = 1; i != keyword_.size(); ++i) {
      int j = failure_[i - 1];
      while (j > 0 && keyword_[i] != keyword_[j]) j = failure_[j - 1];
      if (keyword_[i] == keyword_[j]) ++j;
      failure_[i] = j;
    }
  }

  // Equivalent to following ContextGraph failure links for a single keyword.
  // Node scores are the configured per-token keyword bonus, so a failure
  // transition removes the abandoned prefix bonus and adds the retained one.
  std::pair<int, double> ForwardOneStep(int state, int token) const {
    int next = state;
    while (next > 0 && (next == static_cast<int>(keyword_.size()) || keyword_[next] != token))
      next = failure_[next - 1];
    if (next < static_cast<int>(keyword_.size()) && keyword_[next] == token) ++next;
    double context_delta = (next - state) * static_cast<double>(score_);
    // ContextGraph::ForwardOneStep(..., strict=true) returns both the arc
    // score and output_score.  At a terminal node output_score is its full
    // accumulated phrase score; omitting it made the final phone far too weak.
    if (next == static_cast<int>(keyword_.size())) {
      context_delta += next * static_cast<double>(score_);
    }
    return {next, context_delta};
  }

  std::vector<int> keyword_;
  std::vector<int> failure_;
  float score_;
  float threshold_;
  int max_active_paths_;
  int num_trailing_blanks_;
  std::vector<Hypothesis> hyps_;
};

PYBIND11_MODULE(sail_keyword_search, m) {
  m.doc() = "Sherpa-style modified beam keyword search supplied with TPU logits";
  py::class_<KeywordSearch>(m, "KeywordSearch")
      .def(py::init<std::vector<int>, float, float, int, int>(), py::arg("keyword"),
           py::arg("score") = 2.0f, py::arg("threshold") = 0.18f,
           py::arg("max_active_paths") = 4, py::arg("num_trailing_blanks") = 1)
      .def("reset", &KeywordSearch::Reset)
      .def("histories", &KeywordSearch::Histories)
      .def("step", &KeywordSearch::Step);
}
