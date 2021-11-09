import sys
sys.path.insert(1, "Classes/")
import pandas as pd
import numpy as np
from tqdm import tqdm
from multiprocessing import current_process

from ScoreGetter import ScoreGetter
from dataset_utils import encodeBoard, getColumns
from PGNParser import parseFromPGN

MILLION = 1000000

def parsePgn(pgn_path, nb_fens, output_path):
    data = parseFromPGN(pgn_path, nb_fens=nb_fens)
    df = pd.DataFrame(data, columns=["board"])
    df = df.drop_duplicates(subset=["board"])
    df.to_csv(output_path, index=False)
    print(f"Number of positions: {df.shape[0]}")
    
def encodeBatch(dataset_path, batch_size, nb_sample, offset, output_path, getScore, engine_name):
    df = pd.read_csv(dataset_path, nrows=100000)
    boards = df["board"].values
    print(f"Number of positions: {df.shape[0]}")

    columns = getColumns(engine_name)
    nb_columns = len(columns)
    
    current = current_process()
    pos = current._identity[0]-1 if len(current._identity) > 0 else 0
    pbar = tqdm(total=batch_size*nb_sample, desc="Splitting and encoding", position=pos)

    for i in range(nb_sample):
        data = np.zeros((batch_size, nb_columns))
        for j in range(i * batch_size, min(boards.shape[0], i * batch_size + batch_size)):
            try:
                data[j%batch_size, :-1] = encodeBoard(boards[j])
                data[j%batch_size, -1] = getScore(boards[j] + " 0 1")
                pbar.update(1)
            except Exception as e:
                if str(e) == "[Errno 32] Broken pipe":
                    score_getter.restart()
                continue

        df = pd.DataFrame(data, columns=columns)
        df.to_csv(output_path + "/dataset" + str(offset + i + 1) + ".csv", index=False)

    pbar.close()
    
def concatDatasets (datasets_path, output_path):
    dfs = []
    for path in datasets_path:
        dfs.append(pd.read_csv(path))

    pd.concat(dfs).to_csv(output_path, index=False)

pgn_dataset_path = "D:/IA/Deep_ViCTORIA/Datasets/lichess_db_standard_rated_2019-12.pgn"
engine_path = "C:/Users/Gaëtan/Downloads/stockfish_14.1_win_x64_avx2.exe"
fen_dataset_path = "D:/IA/Deep_ViCTORIA/Datasets/fen_dataset.csv"
dataset_dir = "D:/IA/Deep_ViCTORIA/Datasets/"
    
parsePgn(pgn_dataset_path, 100 * MILLION, fen_dataset_path)

#score_getter = ScoreGetter(engine_path, "eval", "go depth 1")
#encodeBatch(fen_dataset_path, 100, 2, 0, dataset_dir, score_getter.getScore, "Stockfish 14")

#concatDatasets ([dataset_dir + "/dataset57.csv", dataset_dir + "/dataset58.csv"], dataset_dir + "test_dataset.csv")

