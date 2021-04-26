import chess
import time
from Score import Score

class Engine:
    def __init__(self, evaluator):
        self.evaluator = evaluator
        self.board = chess.Board()
        self.best_move = None
        self.searchPly = 0
        print("Deep ViCTORIA")

    def run (self):
        line = ""
        while (line != "quit"):
            line = input("")
            self.parseInput(line)
    
    def parseInput (self, line):
        line = line.split()
        if line[0] == "position":
            idx = 2
            if line[1] == "fen":
                fen = ""
                for i in range (2, 8):
                    fen += line[i] + " "
                self.board = chess.Board(fen)
                idx = 8
            else:
                self.board = chess.Board()
            
            if len(line) > idx and line[idx] == "moves":
                for i in range (idx+1, len(line)):
                    self.board.push_uci(line[i])
        
        elif line[0] == "go":
            if line[1] == "depth":
                depth = int(line[2])
                self.iterativeDeepening(depth)

        
        elif line[0] == "uci":
            print("id name Deep ViCTORIA")
            print("id author Gaëtan Serré")
            print("uciok")


        elif line[0] == "board":
            print(self.board)


        elif line[0] == "eval":
            before = time.time_ns() // 1_000_000
            score = self.evaluator.eval(self.board) / 100.
            after = time.time_ns() // 1_000_000 - before
            print(f"Took {after} ms")
            print(f"Final evaluation: {score} (white side)")

        elif line[0] == "fen":
            print(self.board.fen())
        
        elif line[0] == "board":
            print(self.board)
        

    def eval(self):
        score = self.evaluator.eval(self.board)
        if self.board.turn == chess.BLACK:
            score *= -1
        return Score(score=score, moves=[])

    def sortMove (self):
        captures = []
        others = []

        if self.searchPly == 0 and self.best_move != None:
            captures.append(self.best_move.moves[0])

        for move in self.board.legal_moves:
            if self.board.is_capture(move):
                captures.append(move)
            else:
                others.append(move)
        
        return captures + others



    def NegAlphaBeta (self, alpha, beta, depth, board):

        if board.is_repetition():
            return Score(0)
    
        elif board.is_game_over() or depth == 0:
            return self.eval()


        
        else:
            legal_moves = self.sortMove()
            for move in legal_moves:
                board.push(move)

                self.searchPly += 1
                temp = self.NegAlphaBeta(Score(-beta.score), Score(-alpha.score), depth-1, board)
                score = Score(-temp.score, temp.moves.copy())
                self.searchPly -= 1

                score.moves.append(move)
                board.pop()

                if score.score >= beta.score:
                    is_checkmate = abs(score.score) == 9999
                    if is_checkmate:
                        return Score(score.score, score.moves.copy())
                    else:
                        return beta
                
                if score > alpha:
                    alpha.score = score.score
                    alpha.moves = score.moves.copy()

            return alpha
    

    def iterativeDeepening (self, depth):
        self.best_move = None
        for i in range (1, depth+1):
            before = time.time_ns() // 1_000_000
            self.best_move = self.NegAlphaBeta(Score(-9999), Score(9999), i, self.board)
            after = time.time_ns() // 1_000_000

            self.best_move.moves.reverse()
            self.best_move.printInfo(self.board.turn==chess.WHITE, after-before)

            if abs(self.best_move.score) == 9999:
                break
        self.best_move.print()
            
