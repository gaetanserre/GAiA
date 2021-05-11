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
from PGNParser import parseFromPGN

MILLION = 1e6


def parsePgn(pgn_path, nb_fens, output_path):
    data = parseFromPGN(pgn_path, nb_fens=nb_fens)
    df = pd.DataFrame(data, columns=['board'])
    df = df.drop_duplicates(subset=['board'])
    df.to_csv(output_path, index=False)
    print('Dataset shape:', df.shape)
    
def encodeBatch(dataset_path, batch_size, nb_sample, offset, engine, getScore):
    df = pd.read_csv('Datasets/raw_dataset.csv')
    boards = df['board'].values
    print('Dataset shape:', df.shape)
    
    current = current_process()
    pos = current._identity[0]-1 if len(current._identity) > 0 else 0
    pbar = tqdm(total=batch_size*nb_sample, desc='Splitting and encoding', position=pos)


    for i in range(nb_sample):
        data = []
        for j in range(i * batch_size, min(boards.shape[0], i * batch_size + batch_size)):
            board = boards[j]
            try:
                data.append(np.append(encodeBoard(board), getScore(board)))
                pbar.update(1)
            except Exception as e: 
                if str(e) == '[Errno 32] Broken pipe':
                    score_getter.restart()
                continue

        df = pd.DataFrame(data, columns=np.append(getColumns(), 'cp (' + engine + ')'))
        df.to_csv('Datasets/' + engine + '/dataset' + str(offset + i + 1) + '.csv', index=False)

    pbar.close()
    

def concatDatasets (datasets_path, output_path):
    dfs = []
    for path in datasets_path:
        dfs.append(pd.read_csv(path))

    pd.concat(dfs).to_csv(output_path, index=False)

    
#parsePgn('Datasets/lichess_db_standard_rated_2020-02.pgn', 40 * MILLION, 'Datasets/raw_dataset.csv')

score_getter = ScoreGetter('/home/gaetan/Documents/Chess/Engines/Lc0/lc0', 'eval', 'go depth 1')
encodeBatch('Datasets/raw_dataset.csv', MILLION, 58, 0, 'Leela Chess 0', score_getter.getScore2)

#concatDatasets (['Datasets/Stockfish 13/dataset56.csv', 'Datasets/Stockfish 13/dataset57.csv', 'Datasets/Stockfish 13/dataset58.csv'], 'Datasets/Stockfish 13/test_dataset.csv')

