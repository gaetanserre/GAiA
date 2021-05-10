//
// Created by Gaëtan Serré on 10/05/2021.
//


#include "neuralnet.h"

#include <fstream>
#include <sstream>
#include <utility>


Neuron::Neuron(vector<double> weights, double bias) {
  this->weights = std::move(weights);
  this->bias = bias;
}

double Neuron::getOutput(vector<double> x) {
  return relu(dot(this->weights, std::move(x)) + this->bias);
}

DenseLayer::DenseLayer(vector<vector<double>> weights, vector<double> bias) {
  for (int nb_neurons = 0; nb_neurons<weights.size(); nb_neurons++) {
    this->neurons.emplace_back(weights[nb_neurons], bias[nb_neurons]);
  }
}


vector<double> DenseLayer::getOutput(const vector<double>& x) {
  vector<double> res;
  for (Neuron neuron : this->neurons) {
    res.emplace_back(neuron.getOutput(x));
  }
  return res;
}


DenseLayer NeuralNetwork::createLayer(vector<vector<double>> weights, vector<double> bias) {
  return DenseLayer(std::move(weights), std::move(bias));
}

void NeuralNetwork::init(const string& modelpath) {
  vector<vector<double>> weights; vector<double> bias;
  int nb_layer = 0;
  ifstream model_file(modelpath);

  if (!model_file) {
    string error = "File " + modelpath + " not found.";
    throw runtime_error(error);
  }

  for(string line; getline(model_file, line);) {
    if (line == "layer") {
      if (nb_layer > 0) {
        this->layers.emplace_back(createLayer(weights, bias));
        weights.clear();
        bias.clear();
      }
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
    this->layers.emplace_back(createLayer(weights, bias));
}

double NeuralNetwork::single_predict(vector<double> data) {
  for (DenseLayer layer : this->layers) {
    data = layer.getOutput(data);
  }
  return data[0];
}

vector<double> NeuralNetwork::predict(const vector<vector<double>>& data) {
  vector<double> res;
  res.reserve(data.size());
for (const vector<double>& d : data) {
    res.emplace_back(this->single_predict(d));
  }
  return res;
}

void NeuralNetwork::printShape() {
  for (DenseLayer layer : this->layers)
    cout << layer.getShape() << endl;
}
