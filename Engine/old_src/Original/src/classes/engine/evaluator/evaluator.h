#ifndef VICTORIA_EVALUATOR_H
#define VICTORIA_EVALUATOR_H

#include <vector>
#include "../score/score.h"
#include "cppflow/model.h"

using namespace std;

class Evaluator {

    public:
        Evaluator(const cppflow::model& model);
        Score evalPosition(Board* board);

    private:

        cppflow::model model;

        float getCastlingRights (Board* board);
        vector<float> encodeBoard (Board* board);

};


#endif //VICTORIA_EVALUATOR_H
