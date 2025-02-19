// Copyright (c) 2017 Personal (Binbin Zhang)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <algorithm>
#include <utility>

#include "frontend/feature_pipeline.h"

namespace wenet {

FeaturePipeline::FeaturePipeline(const FeaturePipelineConfig& config):
    config_(config),
    feature_dim_(config.num_bins),
    fbank_(config.num_bins, config.sample_rate,
           config.frame_length, config.frame_shift),
    num_frames_(0),
    input_finished_(false) {
}

void FeaturePipeline::AcceptWaveform(const std::vector<float>& wav) {
  std::vector<std::vector<float> > feats;
  std::vector<float> waves;
  waves.insert(waves.end(), remained_wav_.begin(), remained_wav_.end());
  waves.insert(waves.end(), wav.begin(), wav.end());
  int num_frames = fbank_.Compute(waves, &feats);
  for (size_t i = 0; i < feats.size(); i++) {
    feature_queue_.Push(std::move(feats[i]));
  }
  num_frames_ += num_frames;

  int left_samples = waves.size() - config_.frame_shift * num_frames;
  remained_wav_.resize(left_samples);
  std::copy(waves.begin() + config_.frame_shift * num_frames,
            waves.end(), remained_wav_.begin());
}

bool FeaturePipeline::ReadOne(std::vector<float> *feat) {
  if (input_finished_ && feature_queue_.Empty()) {
    return false;
  } else {
    *feat = std::move(feature_queue_.Pop());
    return true;
  }
}

bool FeaturePipeline::Read(int num_frames,
                           std::vector<std::vector<float> > *feats) {
  feats->clear();
  std::vector<float> feat;
  while (feats->size() < num_frames) {
    if (ReadOne(&feat)) {
      feats->push_back(std::move(feat));
    } else {
      return false;
    }
  }
  return true;
}

void FeaturePipeline::Reset() {
  input_finished_ = false;
  num_frames_ = 0;
  remained_wav_.clear();
  feature_queue_.Clear();
}

}  // namespace wenet

