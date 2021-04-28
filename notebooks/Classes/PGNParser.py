import chess.pgn
import numpy as np

def parseFromPGN(pgn_path, encoder, score_getter, nb_games=-1):
    
    data = []
    
    pgn = open(pgn_path)
    game = chess.pgn.read_game(pgn)
    int count = 0
    
    while game:
        count += 1

        if count > nb_games and nb_games != -1:
            break

        board = game.board()
        fen = board.fen()
        data.append(np.append(encoder(fen), score_getter(fen)))
        for move in game.mainline_moves():
            board.push(move)     
            fen = board.fen()
            try:
                data.append(np.append(encoder(fen), score_getter(fen)))
            except:
                continue
            
        game = chess.pgn.read_game(pgn)
        
    return data