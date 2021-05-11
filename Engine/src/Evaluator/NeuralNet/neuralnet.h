//
// Created by Gaëtan Serré on 10/05/2021.
//

#ifndef NEURALNET_H
#define NEURALNET_H

#include "activation_functions.h"

#include <map>

using namespace std;

class Neuron {
public:
    Neuron() = default;
    Neuron(vector<double> weights, double bias, activationFunction afunction);
    double getOutput(const vector<double>& x);
private:
    vector<double> weights;
    double bias;
    activationFunction afunction;
};

class DenseLayer {
public:
    DenseLayer() = default;
    DenseLayer(vector<vector<double>> weights, vector<double> bias, string afunction_name);
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
    NeuralNetwork(const string& modelpath);

    void init (const string& modelpath);
    vector<double> predict(const vector<vector<double>>& data);
    double single_predict(vector<double> data);
    void summary();


private:
    static DenseLayer createLayer(vector<vector<double>> weights, vector<double> bias, string afunction_name);
    static string getAFunctionName (const string& line);
    void isInitiated();

    vector<DenseLayer> layers;
    bool initiated = false;


};

#endif //NEURALNET_H

