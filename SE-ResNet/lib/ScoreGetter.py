import subprocess

class ScoreGetter:
  def __init__(self, engine_path, evalc="eval", depth1=None):
    self.engine_path = engine_path
    if evalc is not None:
      self.evalc = evalc + '\n'
      self.evalc = self.evalc.encode()
    else:
      self.evalc = None
    if depth1 is not None:
      self.depth1 = depth1 + '\n'
      self.depth1 = self.depth1.encode()
    else:
      self.depth1 = None

    self.engine = subprocess.Popen(engine_path,
                  stdout=subprocess.PIPE,
                  stderr=subprocess.STDOUT,
                  stdin=subprocess.PIPE,
                  bufsize=0)

  def get_score(self, fen):
    print("EFOEIFEJFOEJF")
    if self.evalc is None: return self.get_score2(fen)

    self.engine.stdin.write(self.evalc)

    out = self.engine.stdout.readline()
    score = None
    while out:
      line = out.decode("utf-8", "ignore")[:-1]
      print(line)

      if line.startswith('Final evaluation'):
        line_splitted = [s for s in line.split(' ') if s]

        if line_splitted[2] == 'none':
          return self.get_score2(fen)

        score = int(float(line_splitted[2])*100)
        break

      out = self.engine.stdout.readline()
    return score

  def go_depth1(self, fen):
    set_pos = 'position fen ' + fen + '\n'
    self.engine.stdin.write(set_pos.encode())
    self.engine.stdin.write("go depth 1\n".encode())
    out = self.engine.stdout.readline()
    while out:
      line = out.decode("utf-8", "ignore")[:-1]
      print(line)
      if line.startswith("bestmove"):
        break
      out = self.engine.stdout.readline()

  def get_score2(self, fen):
    set_pos = 'position fen ' + fen + '\n'
    self.engine.stdin.write(set_pos.encode())
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
        score = coeff * int(line_splitted[idx])

      if line.startswith('bestmove'):
          break

      out = self.engine.stdout.readline()
        
    if score == None:
      raise Exception('Mate or stalemate position.')

    return score

  def quit(self):
    self.engine.stdin.write(b'quit\n')

  def __del__(self):
    self.quit()
      
  def restart(self):
    self.engine = subprocess.Popen(self.engine_path,
                  stdout=subprocess.PIPE,
                  stderr=subprocess.STDOUT,
                  stdin=subprocess.PIPE,
                  bufsize=0)
