#include "evaluator.h"

Evaluator::Evaluator(const cppflow::model& model) : model(model) {
    this->model = model;
}

float Evaluator::getCastlingRights(Board *board) {
    bool temp[] = {
            board->castling_short_w,
            board->castling_long_w,
            board->castling_short_b,
            board->castling_long_b,
    };

    float res = 0;
    int coeff = 1;
    for (int i = 0; i<4; i++) {
        if (temp[i])
            res += coeff;
        coeff *= 2;
    }
    return res;
}

float getPieceID(char c) {
    switch (c) {
        case 'p': case 'P': return 1.f;
        case 'r': case 'R': return 4.f;
        case 'n': case 'N': return 2.f;
        case 'b': case 'B': return 3.f;
        case 'q': case 'Q': return 5.f;
        default: return 6;
    }
}

vector<float> Evaluator::encodeBoard(Board *board) {
    vector<float> res;

    for (int i = 0; i<64; i++) {
        if(checkIfPiece(board->squares[i])) {
            if (board->squares[i]->isWhite())
                res.push_back(1.f);
            else
                res.push_back(-1.f);

            res.push_back(getPieceID(board->squares[i]->getName()));

        } else {
            res.push_back(0.f);
            res.push_back(0.f);
        }
    }

    res.push_back(board->isWhite() ? 1.f : -1.f);
    res.push_back(getCastlingRights(board));

    if (board->getEnPassant())
        res.push_back((float) squareToIdx(board->getEnPassantSquare()));
    else
        res.push_back(-1);
    return res;
}

Score Evaluator::evalPosition(Board *board) {
    vector<Ply> legal_moves = board->getLegalMoves();
    int size = legal_moves.size();
    bool white = board->isWhite();

    if (size == 0 || board->nb_plies_50_rule == 100) {
        if (board->isCheck())
            return Score( (white ? -mate_value : mate_value) );
        else return Score();
    } else {
        vector<float> encoded = encodeBoard(board);
        auto input = cppflow::tensor(encoded, {1, 131});
        auto output = this->model(input);
        int score = (int) output.get_data<float>()[0];
        return Score(score);
    }
}


