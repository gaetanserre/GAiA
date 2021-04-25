import subprocess
import time

class ScoreGetter:
  def __init__(self, engine_path, go_command, go_command2=None):
    self.go_command = go_command + '\n'
    if go_command2 != None:
      self.go_command2 = go_command2 + '\n'

    self.engine = subprocess.Popen(engine_path,
                 stdout=subprocess.PIPE,
                 stderr=subprocess.STDOUT,
                 stdin=subprocess.PIPE,
                 bufsize=0)

  
  def getScore(self, fen):
    set_pos = 'position fen ' + fen + '\n'
    self.engine.stdin.write(set_pos.encode())
    self.engine.stdin.write(self.go_command.encode())

    out = self.engine.stdout.readline()
    score = None
    while out:
      line = out.decode()[:-1]

      if line.startswith('Final evaluation'):
        line_splitted = line.split(' ')

        if line_splitted[2] == 'none':
          return self.getScore2(fen)

        score = int(float(line_splitted[6])*100)
        break

      out = self.engine.stdout.readline()

    return score

  def getScore2(self, fen):
    set_pos = 'position fen ' + fen + '\n'
    self.engine.stdin.write(set_pos.encode())
    self.engine.stdin.write(self.go_command2.encode())

    out = self.engine.stdout.readline()
    score = None
    while out:
      line = out.decode()[:-1]

      if line.startswith('info depth'):
        line_splitted = line.split(' ')
        coeff = 1 if fen.split(' ')[1] == 'w' else -1
        score = coeff * int(line_splitted[9])

      if line.startswith('bestmove'):
        break

      out = self.engine.stdout.readline()

    return score

  def quit(self):
    self.engine.stdin.write(b'quit\n')

  def __del__(self):
    self.quit()
  



if __name__ == '__main__':
  score_getter = ScoreGetter('/usr/local/bin/stockfish', 'eval', 'go depth 1')

  nb = 1
  start = time.time()
  for i in range(nb):
    score_getter.getScore('3r1k2/pp2bp1p/q7/1Qp1P1p1/P5P1/5PB1/1PP4P/2KR4 b - - 3 22')
  print(f'{(time.time() - start) / nb * 1000} ms per position.')



