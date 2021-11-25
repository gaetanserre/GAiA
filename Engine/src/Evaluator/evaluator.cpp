//
// Created by Gaëtan Serré on 09/05/2021.
//

#include "evaluator.hpp"


void Evaluator::set_model(const std::string& modelpath) {
  this->network.init(modelpath);
}

float Evaluator::get_castling_rights(const Position& pos) {
  bool temp[] = {
    pos.can_castle(WHITE_OO),
    pos.can_castle(WHITE_OOO),
    pos.can_castle(BLACK_OO),
    pos.can_castle(BLACK_OOO)
  };

  float res = 0;
  int coeff = 1;
  for (int i = 0; i<4; i++) {
    if(temp[i])
      res += coeff;
    coeff *= 2;
  }
  return res;
}

int Evaluator::get_piece_idx(const Piece& p) {
  switch (p)
  {
    case W_ROOK: return 3;
    case W_BISHOP: return 2;
    case W_KNIGHT: return 1;
    case W_PAWN: return 0;
    case W_KING: return 5;
    case W_QUEEN: return 4;

    case B_ROOK: return 9;
    case B_BISHOP: return 8;
    case B_KNIGHT: return 7;
    case B_PAWN: return 6;
    case B_KING: return 11;
    case B_QUEEN: return 10;

    default: return -1;
  }
}

int NB_CHANNELS = 16;
fdeep::tensor Evaluator::encode_position(const Position& pos) {
  fdeep::tensor board(fdeep::tensor_shape(8, 8, NB_CHANNELS), 0);

  float castlings_rights = get_castling_rights(pos);
  float ep_square = static_cast<float>(pos.ep_square());
  float is_ep_square = ep_square == 64 ? -1.0 : ep_square;

  for (int rank = 0; rank < 8; rank++) {
    for (int file = 0; file < 8; file++) {
      board.set(fdeep::tensor_pos(rank, file, 12), pos.side_to_move() == WHITE);
      board.set(fdeep::tensor_pos(rank, file, 13), pos.side_to_move() == BLACK);
      board.set(fdeep::tensor_pos(rank, file, 14), castlings_rights);
      board.set(fdeep::tensor_pos(rank, file, 15), is_ep_square);

      int square = (rank * 8) + file;
      Square s = Square(square);
      int p_idx = get_piece_idx(pos.piece_on(s));
      if (p_idx != -1)
        board.set(fdeep::tensor_pos(rank, file, p_idx), 1);
    }
  }
  return board;
}

Value Evaluator::from_cp(const float& cp) {
  return Value(cp * float(PawnValueEg));
}

Value Evaluator::eval_position(const Position &pos) {
  fdeep::tensor encoded = encode_position(pos);
  float pred = this->network.predict(NB_CHANNELS, encoded);
  pred /= 100.0;
  Value v = from_cp(pred);
  return pos.side_to_move() == WHITE ? v : -v;
}