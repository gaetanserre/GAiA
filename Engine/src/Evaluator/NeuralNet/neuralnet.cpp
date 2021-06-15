//
// Created by Gaëtan Serré on 10/05/2021.
//


#include "neuralnet.h"

#include <fstream>
#include <sstream>

Neuron::Neuron(vector<double> weights, double bias, activationFunction afunction) {
  this->weights = std::move(weights);
  this->bias = bias;
  this->afunction = afunction;
}

double Neuron::getOutput(const vector<double>& x) {
  return this->afunction(dot(this->weights, x) + this->bias);
}


DenseLayer::DenseLayer(vector<vector<double>> weights, vector<double> bias, string afunction_name) {
  this->afunction_name = std::move(afunction_name);
  for (int nb_neurons = 0; nb_neurons<weights.size(); nb_neurons++) {
    this->neurons.emplace_back(weights[nb_neurons], bias[nb_neurons], findAFunction(this->afunction_name));
  }
}

vector<double> DenseLayer::getOutput(const vector<double>& x) {
  vector<double> res;
  for (Neuron neuron : this->neurons) {
    res.emplace_back(neuron.getOutput(x));
  }
  return res;
}


NeuralNetwork::NeuralNetwork(const string& modelpath) {
  this->init(modelpath);
}

DenseLayer NeuralNetwork::createLayer(vector<vector<double>> weights, vector<double> bias, string afunction_name) {
  return DenseLayer(std::move(weights), std::move(bias), std::move(afunction_name));
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
        this->layers.emplace_back(createLayer(weights, bias, afunction));
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
    this->layers.emplace_back(createLayer(weights, bias, afunction));

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
