//
// Created by Gaëtan Serré on 10/05/2021.
//


#include "neuralnet.hpp"

#include <fstream>
#include <sstream>

Neuron::Neuron(vector<double> weights, double bias, activationFunction afunction, bool parallelization) {
  this->weights = std::move(weights);
  this->bias = bias;
  this->afunction = afunction;
  if (parallelization) {
    if (this->weights.size() <= 4)
      this->nb_thread = 1;
    else
      this->nb_thread = 8;
  }
}

double Neuron::getOutput(const vector<double>& x) {
  return this->afunction(dot(this->weights, x, this->nb_thread) + this->bias);
}


DenseLayer::DenseLayer(vector<vector<double>> weights, vector<double> bias, string afunction_name, bool parallelization) {
  this->afunction_name = std::move(afunction_name);
  for (int nb_neurons = 0; nb_neurons<weights.size(); nb_neurons++) {
    this->neurons.emplace_back(weights[nb_neurons], bias[nb_neurons], findAFunction(this->afunction_name), parallelization);
  }
}

vector<double> DenseLayer::getOutput(const vector<double>& x) {
  vector<double> res(neurons.size());
  for (int i = 0; i<this->neurons.size(); i++) {
    res[i] = neurons[i].getOutput(x);
  }
  return res;
}


NeuralNetwork::NeuralNetwork(const string& modelpath, bool parallelization) {
  this->parallelization = parallelization;
  this->init(modelpath);
}

string NeuralNetwork::getAFunctionName(const string& line) {
  return line.substr(6);
}

void NeuralNetwork::init(const string& modelpath) {
  initDic();

  vector<vector<double>> weights; vector<double> bias; string afunction;
  int nb_layer = 0;
  ifstream model_file(modelpath);

  if (!model_file) {
    string error = "File " + modelpath + " not found.";
    throw runtime_error(error);
  }

  for(string line; getline(model_file, line);) {
    if (line.rfind("layer", 0) == 0) {
      if (nb_layer > 0) {
        this->layers.emplace_back(weights, bias, afunction, parallelization);
        weights.clear();
        bias.clear();
      }
      afunction = getAFunctionName(line);
      nb_layer++;
    } else {
      vector<double> w;
      istringstream iss(line);
      string s;
      while ( getline(iss, s, ' ') ) {
        w.push_back(stod(s));
      }
      bias.push_back(w.back());
      w.pop_back();
      weights.emplace_back(w);
    }
  }
  if (!weights.empty() && !bias.empty())
    this->layers.emplace_back(weights, bias, afunction, parallelization);

  this->initiated = true;
}

void NeuralNetwork::isInitiated() {
  if (!this->initiated) {
    throw runtime_error("The network must be initiated.");
  }
}

vector<double> NeuralNetwork::single_predict(vector<double>& data) {
  this->isInitiated();
  for (DenseLayer layer : this->layers) {
    data = layer.getOutput(data);
  }
  return data;
}

vector<vector<double>> NeuralNetwork::predict(vector<vector<double>>& data) {
  this->isInitiated();
  vector<vector<double>> res;
  res.reserve(data.size());
  for (vector<double>& d : data) {
    res.emplace_back(this->single_predict(d));
  }
  return res;
}

void NeuralNetwork::summary() {
  this->isInitiated();
  cout << endl << "Network summary:" << endl;
  cout << "--------------------------------------------------------" << endl;
  for (int i = 0; i<this->layers.size(); i++) {
    int shape = this->layers[i].getShape();
    cout << "Layer " << i << " → " << shape
         << (shape > 1 ? " neurons" : " neuron") << " with " << this->layers[i].getAFunctionName()
         << " activation function." << endl;
  }
  cout << "--------------------------------------------------------" << endl << endl;
}
