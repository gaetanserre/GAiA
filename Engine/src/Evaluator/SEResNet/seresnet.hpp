//
// Created by Gaëtan Serré on 09/05/2021.
//

#ifndef GAiA_RESNET_H
#define GAiA_RESNET_H

#include <string>
#include <onnx/onnxruntime_cxx_api.h>

#if defined(USE_CUDA) || defined(USE_TENSORRT)
#include <onnx/tensorrt_provider_factory.h>
#endif

#define NB_CHANNELS 15

class SEResNet {
  public:
    SEResNet() = default;

    void init (const std::string& modelpath);
    double predict(int channel, std::array<float, NB_CHANNELS*8*8>& board);
  
  private:
    const char* input_names[7] = {"input_2"};
    const char* output_names[8] = {"dense_13"};

    std::array<float, NB_CHANNELS*8*8> input_data;
    std::array<float, 1> output_data;

    std::array<int64_t, 4> input_shape{1, 8, 8, NB_CHANNELS};
    std::array<int64_t, 2> output_shape{1, 1};

    std::unique_ptr<Ort::Env> env;
    Ort::SessionOptions session_options;
    std::unique_ptr<Ort::Session> session;

    std::unique_ptr<Ort::Value> input_tensor;
    std::unique_ptr<Ort::Value> output_tensor;

};

#endif //GAiA_RESNET_H
