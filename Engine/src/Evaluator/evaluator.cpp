//
// Created by Gaëtan Serré on 09/05/2021.
//

#include "evaluator.h"


void Evaluator::setModel(const std::string& modelpath) {
  this->network.init(modelpath);
}

float Evaluator::getCastlingRights(const Position& pos) {
  bool temp[] = {
          pos.can_castle(WHITE_OO),
          pos.can_castle(WHITE_OOO),
          pos.can_castle(BLACK_OO),
          pos.can_castle(BLACK_OOO)
  };

  float res = 0;
  int coeff = 1;
  for (int i = 0; i<4; i++) {
    if(temp[1])
      res += coeff;
    coeff *= 2;
  }
  return res;
}

double Evaluator::getPieceID(Piece p) {
  switch (p)
  {
    case W_PAWN: case B_PAWN: return 1.0;
    case W_ROOK: case B_ROOK: return 4.0;
    case W_KNIGHT: case B_KNIGHT: return 2.0;
    case W_BISHOP: case B_BISHOP: return 3.0;
    case W_QUEEN: case B_QUEEN: return 5.0;
    default: return 6.0;
  }
}

std::vector<double> Evaluator::encodeBoard(const Position& pos) {
  std::vector<double> res;

  for (int i = 0; i<64; i++) {

    Square s = Square(i);
    Piece p = pos.piece_on(s);

    if (p != NO_PIECE) {
      if (p < 9)
        res.push_back(1.0);
      else
        res.push_back(-1.0);

      res.push_back(getPieceID(p));
    } else {
      res.push_back(0.0);
      res.push_back(0.0);
    }
  }

  res.push_back(pos.side_to_move() == WHITE ? 1.0 : -1.0);
  res.push_back(getCastlingRights(pos));

  if (pos.ep_square() < 64)
    res.push_back(pos.ep_square());
  else
    res.push_back(-1.0);
  return res;
}

Value Evaluator::from_cp(double cp) {
  return Value(cp * double(PawnValueEg));
}

Value Evaluator::evalPosition(const Position &pos) {
  std::vector<double> encoded = encodeBoard(pos);
  double pred = this->network.single_predict(encoded)[0] / 100.0;
  Value v = from_cp(pred);
  return pos.side_to_move() == WHITE ? v : -v;
}

