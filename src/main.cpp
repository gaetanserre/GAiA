#include "classes/engine/engine.h"

string transform_path(string path) {
    int count = 0;
    for (int i = path.size() - 1; i >= 0; i--) {
        if (path[i] == '/' || path[i] == '\\')
            count++;
        path.pop_back();
        if (count == 2) break;
    }
    return path;
}

int main (int argc, char** argv) {

    setenv("TF_CPP_MIN_LOG_LEVEL","3",1);

    Evaluator evaluator(cppflow::model( "/Users/gaetanserre/Documents/Projets/"
                                        "Chess/Engines/Deep-ViCTORIA/Models/SF_model_batch_32M"));
    Engine engine(transform_path(argv[0]), evaluator);

    string input;
    while (input != "quit") {
        getline(cin, input);
        engine.parseExpr(input);
    }
    return 0;
}
