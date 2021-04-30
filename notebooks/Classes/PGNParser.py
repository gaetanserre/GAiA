import chess.pgn
import numpy as np
from tqdm import tqdm
from multiprocessing import current_process

def parseFromPGN(pgn_path, nb_fens):
    
    data = []
    
    pgn = open(pgn_path)
    game = chess.pgn.read_game(pgn)
    count = 0
    
    current = current_process()
    pos = current._identity[0]-1 if len(current._identity) > 0 else 0
    pbar = tqdm(total=nb_fens, desc='Creating fens from pgn', position=pos)
    
    while game:

        if nb_fens != -1 and count > nb_fens:
            break

        board = game.board()
        fen = board.fen()
        data.append(fen)
        for move in game.mainline_moves():
            board.push(move)     
            fen = board.fen()
            
            count += 1
            pbar.update(1)
            
            try:
                data.append(fen)
            except:
                continue
            
        game = chess.pgn.read_game(pgn)
        
    pbar.close()
        
    return data