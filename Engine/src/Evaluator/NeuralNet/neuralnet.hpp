//
// Created by Gaëtan Serré on 10/05/2021.
//

#ifndef NEURALNET_H
#define NEURALNET_H

#include "activation_functions.hpp"
#include "dot.hpp"

#include <map>

using namespace std;

class Neuron {
public:
  Neuron(vector<double> weights, double bias, activationFunction afunction, bool parallelization);
  double getOutput(const vector<double>& x);
private:
  vector<double> weights;
  double bias;
  activationFunction afunction;
  int nb_thread = 1;
};

class DenseLayer {
public:
  DenseLayer(vector<vector<double>> weights, vector<double> bias, string afunction_name, bool parallelization);
  vector<double> getOutput(const vector<double>& x);
  int getShape() { return this->neurons.size(); }
  string getAFunctionName() { return this->afunction_name; }
private:
  vector<Neuron> neurons;
  string afunction_name;
};

class NeuralNetwork {
public:
  NeuralNetwork() = default;
  NeuralNetwork(const string& modelpath, bool parallelization);

  void init (const string& modelpath);
  vector<vector<double>> predict(vector<vector<double>>& data);
  vector<double> single_predict(vector<double>& data);
  void summary();

  bool parallelization = false;


private:
  static string getAFunctionName (const string& line);
  void isInitiated();

  vector<DenseLayer> layers;
  bool initiated = false;
};

#endif //NEURALNET_H

