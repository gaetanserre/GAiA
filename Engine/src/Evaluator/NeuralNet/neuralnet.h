//
// Created by Gaëtan Serré on 10/05/2021.
//

#ifndef NEURALNET_H
#define NEURALNET_H

#include <iostream>
#include <vector>

using namespace std;

static double relu(double x) { return max(0.0, x); }
static double dot(vector<double> u, vector<double> v) {
  int u_size = u.size(); int v_size = v.size();
  if (u_size != v_size) {
    string error = "Dot product between vector of size " + to_string(u_size) + " and " + "vector of size " + to_string(v_size) + ".";
    throw runtime_error(error);
  } else {
    double res = 0.0;
    for (int i = 0; i<u_size; i++) {
      res += u[i] * v[i];
    }
    return res;
  }
}

class Neuron {
  public:
    Neuron() = default;
    Neuron(vector<double> weights, double bias);
    double getOutput(vector<double> x);
  private:
    vector<double> weights;
    double bias;
};

class DenseLayer {
  public:
    DenseLayer() = default;
    DenseLayer(vector<vector<double>> weights, vector<double> bias);
    vector<double> getOutput(const vector<double>& x);
    int getShape() { return this->neurons.size(); }
  private:
    vector<Neuron> neurons;
};

class NeuralNetwork {
  public:
    void init (const string& modelpath);
    vector<double> predict(const vector<vector<double>>& data);
    double single_predict(vector<double> data);
    void printShape();

    
  private:
    static DenseLayer createLayer(vector<vector<double>> weights, vector<double> bias);
    vector<DenseLayer> layers;

};

#endif //NEURALNET_H

