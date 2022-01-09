//
// Created by Gaëtan Serré on 09/05/2021.
//
#include <onnx/onnxruntime_cxx_api.h>
#include <string>

#ifndef GAiA_RESNET_H
#define GAiA_RESNET_H

#define NB_CHANNELS 15

class ResNet {
  public:
    ResNet() = default;

    void init (const std::string& modelpath);
    double predict(int channel, std::array<float, NB_CHANNELS*8*8>& board);
  
  private:
    const char* input_names[7] = {"input_2"};
    const char* output_names[8] = {"dense_13"};

    std::array<float, NB_CHANNELS*8*8> input_data;
    std::array<float, 1> output_data;

    std::array<int64_t, 4> input_shape{1, 8, 8, NB_CHANNELS};
    std::array<int64_t, 2> output_shape{1, 1};

    std::unique_ptr<Ort::Session> p_session = NULL;
    std::unique_ptr<Ort::Value> input_tensor;
    std::unique_ptr<Ort::Value> output_tensor;

};

#endif //GAiA_RESNET_H
