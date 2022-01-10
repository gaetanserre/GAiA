#include "seresnet.hpp"


void SEResNet::init(const std::string& modelpath) {
  this->env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_FATAL, "Default");

  #ifdef USE_CUDA
  Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CUDA(this->session_options, 0));
  #endif

  #ifdef USE_TENSORRT
  Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_Tensorrt(this->session_options, 0));
  #endif

  const char * modelpath_c = modelpath.c_str();

  this->session = std::make_unique<Ort::Session>(*env, modelpath_c, session_options);

  Ort::MemoryInfo info("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault);
  this->input_tensor = std::make_unique<Ort::Value>(
                        Ort::Value::CreateTensor<float>(
                          info,
                          input_data.data(),
                          input_data.size(),
                          input_shape.data(),
                          input_shape.size()
                        )
                      );
  this->output_tensor = std::make_unique<Ort::Value>(
                          Ort::Value::CreateTensor<float>(
                            info,
                            output_data.data(),
                            output_data.size(),
                            output_shape.data(),
                            output_shape.size()
                          )
                        );
}

double SEResNet::predict(int channel, std::array<float, NB_CHANNELS*8*8>& board) {
  this->input_data = std::move(board);
  this->session->Run(Ort::RunOptions{nullptr}, input_names, input_tensor.get(), 1, output_names, output_tensor.get(), 1);
  return this->output_data[0];
}