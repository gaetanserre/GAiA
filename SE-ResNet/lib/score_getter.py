import subprocess
from enum import IntEnum

class Engine(IntEnum):
  STOCKFISH = 0
  LEELA     = 1

class ScoreGetter():
  def __init__(self, engine_path, engine=Engine.STOCKFISH):
    self.engine_type = engine
    self.eval = "eval\n".encode()
    self.depth1 = "go depth 1\n".encode()

    self.engine = subprocess.Popen(engine_path,
                  stdout=subprocess.PIPE,
                  stderr=subprocess.STDOUT,
                  stdin=subprocess.PIPE,
                  bufsize=0)
  
  def write_position(self, fen):
    set_pos = 'position fen ' + fen + '\n'
    self.engine.stdin.write(set_pos.encode())
  
  def get_score_sf(self, fen):
    self.write_position(fen)
    self.engine.stdin.write(self.depth1)
    return self.get_score_eval(fen)

  def get_score_eval(self, fen):
    self.engine.stdin.write(self.eval)

    out = self.engine.stdout.readline()
    score = None
    while out:
      line = out.decode("utf-8", "ignore")[:-1]

      if line.startswith('Final evaluation'):
        line_splitted = [s for s in line.split(' ') if s]

        if line_splitted[2] == 'none':
          return self.get_score_d1(fen)

        score = float(line_splitted[2])*100
        break

      out = self.engine.stdout.readline()
    return score

  def get_score_d1(self, fen):
    self.engine.stdin.write(self.depth1)

    out = self.engine.stdout.readline()
    score = None
    while out:
      line = out.decode("utf-8", "ignore")[:-1]

      if line.startswith('info depth'):
        line_splitted = line.split(' ')
        coeff = 1 if fen.split(' ')[1] == 'w' else -1
        idx = -1
        for i in range(len(line_splitted)):
          if line_splitted[i] == 'cp':
              idx = i+1
              break
        if idx == -1:
          raise Exception('Mate or stalemate position.')
        score = coeff * float(line_splitted[idx])

      if line.startswith('bestmove'):
          break

      out = self.engine.stdout.readline()
        
    if score == None:
      raise Exception('Mate or stalemate position.')

    return score
  
  def get_score_leela(self, fen):
    self.write_position(fen)
    self.engine.stdin.write(self.depth1)

    out = self.engine.stdout.readline()
    while out:
      line = out.decode("utf-8", "ignore")[:-1]
      if line.startswith("bestmove"):
        if line.split(' ')[1] == "a1a1":
          raise Exception('Mate or stalemate position.')
        break
      out = self.engine.stdout.readline()
    
    return self.get_score_eval(fen)

  
  def get_score(self, fen):
    if self.engine_type == Engine.STOCKFISH:
      return self.get_score_sf(fen)
    elif self.engine_type == Engine.LEELA:
      return self.get_score_leela(fen)