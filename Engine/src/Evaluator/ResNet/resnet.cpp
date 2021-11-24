#include "resnet.hpp"

void ResNet::init(const std::string& modelpath) {
  this->model = fdeep::load_model(modelpath, false, 0);
}

double ResNet::predict(int channel, fdeep::tensor& board) {
  auto result = this->model.predict({board});
  std::vector<float> r = result[0].to_vector();
  return r[0];
}