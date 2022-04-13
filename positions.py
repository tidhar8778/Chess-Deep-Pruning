import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import multiprocessing
from Play import play_main
import csv

def utilize_test_positions(n):
    eval_threshold = 10
    posAndVals = pd.read_csv('.\dataset_posAndVals\chessData.csv', nrows=10000)
    posAndVals.Evaluation = pd.to_numeric(posAndVals.Evaluation, errors='coerce', downcast='integer')
    posAndVals = posAndVals.dropna()
    pos_vals_narrow = posAndVals[(posAndVals.Evaluation < eval_threshold) & (posAndVals.Evaluation > -eval_threshold)]
    pos_vals_shuffled = pos_vals_narrow.sample(frac=1, random_state=21)
    pos_vals = pos_vals_shuffled.head(n)

    return pos_vals


if __name__ == '__main__':
    pos_vals = utilize_test_positions(100)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    # pd.set_option('display.max_rows', None)
    # pd.set_option('display.max_columns', None)
    # pos_vals.to_csv('test_positions_sf_eval_prune.csv')
    outputs = []
    start_time = time.time()
    pool = multiprocessing.Pool(4)
    result = pool.map(func=play_main, iterable=pos_vals.FEN)
    pool.close()
    pool.join()
    end_time = time.time()
    print('Time collected data: ', str(end_time - start_time))
    for res in result:
        outputs.append(res)

    keys = outputs[0].keys()

    with open('dataset_from_search/test 3 - larger model (500)/test_positions_output_prune.csv', 'w', newline='') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(outputs)
