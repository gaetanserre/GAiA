import sys
sys.path.insert(1, 'Classes/')
import pandas as pd
import numpy as np
import random
from tqdm import tqdm
from multiprocessing import current_process

from ScoreGetter import ScoreGetter
from dataset_utils import checkIfEarlyMidEnd
from dataset_utils import encodeBoard
from dataset_utils import getColumns


df = pd.read_csv('Datasets/raw_dataset_13M.csv')
boards = df['board'].values


score_getter = ScoreGetter('/home/gaetan/Téléchargements/stockfish/stockfish', 'eval', 'go depth 1')


total_size = 5000000
batch_size = total_size / 3.0

earlies = []
mids = []
ends = []

current = current_process()
pos = current._identity[0]-1 if len(current._identity) > 0 else 0
pbar = tqdm(total=total_size, desc='Splitting and encoding', position=pos)

for i in range(len(boards)):
    board = boards[i]
    part = checkIfEarlyMidEnd(board)
    
    if len(earlies) < batch_size and part == "early_game":
        earlies.append(np.append(encodeBoard(board), score_getter.getScore(board)))
        pbar.update(1)
    
    elif len(mids) < batch_size and part == "mid_game":
        mids.append(np.append(encodeBoard(board), score_getter.getScore(board)))
        pbar.update(1)
        
    elif len(ends) < batch_size and part == "end_game":
        ends.append(np.append(encodeBoard(board), score_getter.getScore(board)))
        pbar.update(1)
    
    if len(earlies) >= batch_size and len(mids) >= batch_size and len(ends) >= batch_size:
        break
    
pbar.close()


data = earlies + mids + ends
random.shuffle(data)
len(data)


idx_dep = 0
idx_end = 1000000

for i in range(5):
    df = pd.DataFrame(data[idx_dep:idx_end], columns=np.append(getColumns(), 'cp (Stockfish 13)'))
    df.to_csv('Datasets/dataset'+ str(i) +'.csv', index=False)
    idx_dep += 1000000
    idx_end += 1000000
