//
// Created by Gaëtan Serré on 09/05/2021.
//

#ifndef DEEP_VICTORIA_EVALUATOR_H
#define DEEP_VICTORIA_EVALUATOR_H

#include "../Stockfish/position.h"
#include "../Stockfish/uci.h"
#include "model.h"

using namespace Stockfish;

class Evaluator {
  public:
    void setModel (const std::string& modelpath);
    Value evalPosition(const Position& pos);

  private:
    cppflow::model model;

    static float getCastlingRights (const Position& pos);
    static std::vector<float> encodeBoard (const Position& pos);
    static float getPieceID(Piece p);
    static int convertIdx (int idx);
    static Value from_cp (double cp);
};


#endif //DEEP_VICTORIA_EVALUATOR_H
