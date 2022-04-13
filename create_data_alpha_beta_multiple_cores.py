'''
Create dataset for neural network.
Each result row contains:
-State features
-Beta value at this state in the search
-Label: pruned or not (1,0)
'''

from extract_features import get_features
import pickle
from Play import play_main
import random
import time
import multiprocessing
import chess
import pandas as pd
from params import input_size

def create_set(folder):
    random.seed(10)
    eval_threshold = 500    # centi-pawns
    num_start_positions = 4000

    posAndVals = pd.read_csv('.\dataset_posAndVals\chessData.csv', nrows=num_start_positions*5)
    posAndVals.Evaluation = pd.to_numeric(posAndVals.Evaluation, errors='coerce', downcast='integer')
    posAndVals = posAndVals.dropna()
    pos_vals_narrow = posAndVals[(posAndVals.Evaluation < eval_threshold) & (posAndVals.Evaluation > -eval_threshold)]
    pos_vals_shuffled = pos_vals_narrow.sample(frac=1, random_state=21)
    pos_vals = pos_vals_shuffled.head(num_start_positions)

    data_collected = [] # each place contains dictionary: {fen: , beta: , pruned: [0,1], value (if pruned): }

    start_time = time.time()
    pool = multiprocessing.Pool(4)
    result = pool.map(func=play_main, iterable=pos_vals.FEN)
    pool.close()
    pool.join()
    end_time = time.time()

    print('Time collected data: ', str(end_time - start_time))

    for res in result:
        data_collected.extend(res)

    print('Collected {} rows'.format(len(data_collected)))

    # Shuffle data
    random.shuffle(data_collected)

    with open(folder+'/data_collected.pkl', 'wb') as f:
        pickle.dump(data_collected, f)

def get_features_help(data):
    # data is a dict: {fen, beta, prune}
    feat = get_features(chess.Board(data['fen']))
    data['features'] = feat
    return data


def create_X_and_labels(folder):

    with open(folder+'/data_collected.pkl', 'rb') as f:
        dataset = pickle.load(f)

    t1 = time.time()

    dataset_0 = [di for di in dataset if di['pruned'] == 0]
    dataset_1 = [di for di in dataset if di['pruned'] == 1]
    min_len = len(dataset_0) if len(dataset_0) <= len(dataset_1) else len(dataset_1)
    dataset_0 = dataset_0[:min_len]
    dataset_1 = dataset_1[:min_len]
    bal_dataset = []
    bal_dataset.extend(dataset_0)
    bal_dataset.extend(dataset_1)
    print('Start extracting features')
    pool = multiprocessing.Pool(4)
    result = pool.map(func=get_features_help, iterable=bal_dataset)
    pool.close()
    pool.join()
    t2 = time.time()
    print('\nTime extracting features from {} fens: {}'.format(len(result), t2 - t1))
    # Shuffle data
    random.shuffle(result)
    X = []
    labels = []
    for di in result:
        if len(di['features']+[di['beta']]) == input_size:
            X.append(di['features']+[di['beta']])
            labels.append(di['pruned'])

    t3 = time.time()
    print('\nTime creating X and labels: {}'.format(t3 - t2))

    with open(folder+'/X.pkl', 'wb') as f:
        pickle.dump(X, f)

    with open(folder+'/labels.pkl', 'wb') as f2:
        pickle.dump(labels, f2)


if __name__ == '__main__':
    folder = 'dataset_from_search'
    create_set(folder)
    create_X_and_labels(folder)
