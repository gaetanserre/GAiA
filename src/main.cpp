#include "classes/engine/engine.h"

int main (int argc, char** argv) {

    setenv("TF_CPP_MIN_LOG_LEVEL","3",1);

    Evaluator evaluator(cppflow::model("/Users/gaetanserre/Documents/Projets/Chess/Engines/Deep ViCTORIA/model"));
    Engine engine(argv[0], evaluator);

    string input;
    while (input != "quit") {
        getline(cin, input);
        engine.parseExpr(input);
    }
    return 0;
}