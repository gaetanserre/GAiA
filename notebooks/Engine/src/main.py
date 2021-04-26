import sys
import chess
from evaluator import Evaluator
from engine import Engine

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Model path is necessary.")
        sys.exit(-1)
    else:
        evaluator = Evaluator(sys.argv[1])
        engine = Engine(evaluator)
        engine.run()
