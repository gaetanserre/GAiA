#include "resnet.hpp"

void ResNet::init(const std::string& modelpath) {
  Ort::Env env = Ort::Env(ORT_LOGGING_LEVEL_FATAL, "Default");
	Ort::SessionOptions session_options;
  const char * modelpath_c = modelpath.c_str();

  auto tmp_session = std::make_unique<Ort::Session>(env, modelpath_c, session_options);
  this->p_session = std::move(tmp_session);

  Ort::MemoryInfo info("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault);
  auto tmp_tensor = std::make_unique<Ort::Value>(Ort::Value::CreateTensor<float>(info, input_data.data(), input_data.size(), input_shape.data(), input_shape.size()));
  this->input_tensor = std::move(tmp_tensor);
  tmp_tensor = std::make_unique<Ort::Value>(Ort::Value::CreateTensor<float>(info, output_data.data(), output_data.size(), output_shape.data(), output_shape.size()));
  this->output_tensor = std::move(tmp_tensor);
}

double ResNet::predict(int channel, std::array<float, NB_CHANNELS*8*8>& board) {
  this->input_data = board;
  p_session->Run(Ort::RunOptions{nullptr}, input_names, input_tensor.get(), 1, output_names, output_tensor.get(), 1);
  return this->output_data[0];
}