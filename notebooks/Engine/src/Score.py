class Score:
    def __init__(self, score, moves=[]):
        self.score = score
        self.moves = moves
    
    def printInfo(self, whites_to_play, elapsed_time):
        moves_str = ""
        for move in self.moves:
            moves_str += str(move) + " "
        moves_str = moves_str[:-1]

        score_str = "cp " + str(self.score)
        
        if abs(self.score) == 9999:

            nb = len(self.moves)
            nb //= 2

            if len(self.moves) % 2 == 1:
                nb += 1
                
            score_str = "mate " + str(nb if self.score > 0 else -nb)

        print("info depth {} score {} time {} pv {}".format(len(self.moves),
                score_str, elapsed_time, moves_str))
    
    def print(self):
        move = str(self.moves[0])
        ponder = ""
        if len(self.moves) > 1:
            ponder = str(self.moves[1])
        
        print("bestmove {} ponder {}".format(move, ponder))

    def __gt__(self, s):
        if self.score == -9999 and s.score == -9999:
            return len(self.moves) > len(s.moves)
        elif self.score == 9999 and s.score == 9999:
            return len(self.moves) < len(s.moves)
        else:
            return self.score > s.score

