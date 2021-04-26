import chess
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow import keras
from dataset_utils import encodeBoard
from LiteModel import LiteModel
import numpy as np

mate_value = 9999


class Evaluator:
    def __init__ (self, model_path):
        self.model = LiteModel.from_keras_model(keras.models.load_model(model_path))
    

    def eval (self, board):
        if board.is_game_over():
            if board.is_checkmate():
                return mate_value if board.turn == chess.BLACK else -mate_value
            else:
                return 0
        else:
            encoded_board = np.array(encodeBoard(board)).reshape(1, -1)
            return int(self.model.predict(encoded_board))
