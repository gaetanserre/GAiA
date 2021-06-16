//
// Created by Gaëtan Serré on 09/05/2021.
//

#ifndef DEEP_VICTORIA_EVALUATOR_H
#define DEEP_VICTORIA_EVALUATOR_H

#include "../Stockfish/position.h"
#include "../Stockfish/uci.h"
#include "NeuralNet/neuralnet.h"

using namespace Stockfish;

class Evaluator {
  public:
    void setModel (const std::string& modelpath);
    Value evalPosition(const Position& pos);

  private:
    NeuralNetwork network;

    static float getCastlingRights (const Position& pos);
    static std::vector<double> encodeBoard (const Position& pos);
    static double getPieceID(const Piece& p);
    static Value from_cp (const double& cp);
};


#endif //DEEP_VICTORIA_EVALUATOR_H
