//
// Created by Gaëtan Serré on 09/05/2021.
//

#ifndef DEEP_VICTORIA_EVALUATOR_H
#define DEEP_VICTORIA_EVALUATOR_H
#include <vector>
#include "model.h"
#include "../Stockfish/position.h"

#define ModelFolderDefault "/Users/gaetanserre/Documents/Projets/Chess/Engines/Deep-ViCTORIA/Models/SF_model_batch_55M"

using namespace Stockfish;

class Evaluator {
  public:
    Value evalPosition(const Position& pos);

  private:
    cppflow::model model = cppflow::model(ModelFolderDefault);

    static float getCastlingRights (const Position& pos);
    static std::vector<float> encodeBoard (const Position& pos);
    static float getPieceID(Piece p);
    static int convertIdx (int idx);
    static Value from_cp (double cp);
};


#endif //DEEP_VICTORIA_EVALUATOR_H
