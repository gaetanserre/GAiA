//
// Created by Gaëtan Serré on 09/05/2021.
//
#include <fdeep/fdeep.hpp>
#include <string>

#ifndef GAiA_RESNET_H
#define GAiA_RESNET_H

class ResNet {
  public:
    ResNet() = default;

    void init (const std::string& modelpath);
    double predict(int channel, fdeep::tensor& board);
  
  private:
    fdeep::model model;
};

#endif //GAiA_RESNET_H
