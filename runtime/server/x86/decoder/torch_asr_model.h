// Copyright 2020 Mobvoi Inc. All Rights Reserved.
// Author: binbinzhang@mobvoi.com (Binbin Zhang)

#ifndef DECODER_TORCH_ASR_MODEL_H_
#define DECODER_TORCH_ASR_MODEL_H_

#include <torch/torch.h>
#include <torch/script.h>

#include <memory>
#include <string>

#include "utils/utils.h"

namespace wenet {

using TorchModule = torch::jit::script::Module;
// A wrapper for pytorch asr model
class TorchAsrModel {
 public:
  TorchAsrModel() = default;
  void Read(const std::string& model_path);
  int right_context() const { return right_context_; }
  int subsampling_rate() const { return subsampling_rate_; }
  int sos() const { return sos_; }
  int eos() const { return eos_; }
  std::shared_ptr<TorchModule> torch_model() const { return module_; }

 private:
  std::shared_ptr<TorchModule> module_ = nullptr;
  int right_context_ = 1;
  int subsampling_rate_ = 1;
  int sos_ = 0;
  int eos_ = 0;

  DISALLOW_COPY_AND_ASSIGN(TorchAsrModel);
};

}  // namespace wenet

#endif  // DECODER_TORCH_ASR_MODEL_H_
