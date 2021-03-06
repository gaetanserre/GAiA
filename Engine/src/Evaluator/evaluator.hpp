//
// Created by Gaëtan Serré on 09/05/2021.
//

#ifndef GAiA_EVALUATOR_H
#define GAiA_EVALUATOR_H

#include "position.h"
#include "uci.h"
#include "seresnet.hpp"

using namespace Stockfish;

class Evaluator {
  public:
    Evaluator() = default;
    void set_model (const std::string& modelpath);
    Value eval_position(const Position& pos);

  private:
    SEResNet network;

    static float get_castling_rights (const Position& pos);
    static std::array<float, NB_CHANNELS*8*8> encode_position (const Position& pos);
    static int get_piece_idx(const Piece& p);
    static Value from_cp (const float& cp);
};


#endif //GAiA_EVALUATOR_H