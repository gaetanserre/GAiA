//
// Created by Gaëtan on 11/05/2021.
//

#ifndef ACTIVATION_FUNCTIONS_H
#define ACTIVATION_FUNCTIONS_H

#include <vector>
#include <math.h>
#include <iostream>
#include <map>

static double dot(const std::vector<double>& u, const std::vector<double>& v) {
  int u_size = u.size(); int v_size = v.size();
  if (u_size != v_size) {
    std::string error = "Dot product between vector of size " + std::to_string(u_size) + " and " + "vector of size " + std::to_string(v_size) + ".";
    throw std::runtime_error(error);
  } else {
    double res = 0.0;
    for (int i = 0; i<u_size; i++) {
      res += u[i] * v[i];
    }
    return res;
  }
}

static inline double linear (const double& x) { return x; }
static inline double relu (const double& x) { return fmax(0.0, x); }
static inline double softplus (const double& x) { return log(exp(x) + 1); }
static inline double softsign (const double& x) { return x / (abs(x) + 1); }
static inline double tanh_ (const double& x) { return tanh(x); }

typedef double (*activationFunction)(const double&);
static std::map <std::string, activationFunction> f_dic;

static void initDic() {
  f_dic["linear"] = &linear;
  f_dic["relu"] = &relu;
  f_dic["softplus"] = &softplus;
  f_dic["softsign"] = &softsign;
  f_dic["tanh"] = &tanh_;
}

static activationFunction findAFunction(const std::string& name) {
  auto it = f_dic.find(name);
  if(it != f_dic.end()) {
    // activation function found;
    return it->second;
  } else {
    std::string error = "Activation function with name " + name + " not found.";
    throw std::runtime_error(error);
  }
}


#endif //ACTIVATION_FUNCTIONS_H
