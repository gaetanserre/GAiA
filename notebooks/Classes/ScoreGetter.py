import subprocess
import time

class ScoreGetter:
    def __init__(self, engine_path, go_command, go_command2=None):
        self.engine_path = engine_path
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
        
  
