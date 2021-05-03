import chess
import numpy as np

def getNbPieceMaj(fen_splitted):
    nb_piece = 0
    for c in fen_splitted[0]:
        try:
            int(c)
            continue
        except:
            if c != '/' and c != 'p' and c != 'P' and c != 'k' and c != 'K':
                nb_piece += 1
    return nb_piece

def getNbMoves(fen_splitted):
    return int(fen_splitted[-1])

def checkIfEarlyMidEnd(fen):
    fen_splitted = fen.split(' ')
    nb_piece_maj = getNbPieceMaj(fen_splitted)

    if nb_piece_maj <= 4:
        return 'end_game'
    
    elif getNbMoves(fen_splitted) <= 10 and nb_piece_maj >= 14:
        return 'early_game'
    
    else:
        return 'mid_game'
    
    
def convertIdx(idx):
    rank = idx % 8
    file = int(idx/8) + 1
    
    return (8 - file) * 8 + rank

'''
We compute castling rights as if it were a 4-bit number.
'''
def getCastlingRights(board):
    rights = [
        board.has_kingside_castling_rights(chess.WHITE),
        board.has_queenside_castling_rights(chess.WHITE),
        board.has_kingside_castling_rights(chess.BLACK),
        board.has_queenside_castling_rights(chess.BLACK)
    ]
    
    res = 0
    coeff = 1
    for r in rights:
        if r:
            res += coeff
        coeff *= 2
    
    return res


'''
We encode a position like an image (shape: 64 * 2 + 3)
'''
def encodeBoard(fen):
    res = np.zeros((64, 2))
    board = chess.Board(fen)
    
    for i in range(64):
        square = convertIdx(i)
        piece_type = board.piece_type_at(square)
        if piece_type == None:
            continue
        else:
            piece_color = board.color_at(square)
            res[i] = [1 if piece_color else -1, piece_type]
    ep_square = -1 if board.ep_square == None else convertIdx(board.ep_square)
    
    res = np.append(res.flatten(), 1 if board.turn else 0)
    res = np.append(res, getCastlingRights(board))
    res = np.append(res, ep_square)
    
    return res.flatten()

def getColumns():
    columns = []
    for i in range (64 * 2):
        columns.append(str(i))
    columns.append('whites to play')
    columns.append('castling rights')
    columns.append('en passant square')
    return columns
