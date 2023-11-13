'''
Create dataset for neural network.
Each result row contains:
-State features
-Beta value at this state in the search
-Label: pruned or not (1,0)
'''


import pickle
import random
import time
import multiprocessing
import chess
from params import folder_model, option_nn
from extract_features import get_features as special_features
from simple_input_features import get_features as halfka_features


def get_features_help(data):
    # data is a dict: {fen, beta, label}
    if option_nn in (1, 2):
        get_features = special_features
    elif option_nn == 3:
        get_features = halfka_features
    feat = get_features(chess.Board(data['fen']))
    data['features'] = feat
    return data


def create_X_and_labels():

    dataset = []
    with open('fen_labels.txt', 'r') as f:
        next(f)
        for line in f:
            (fen, beta, label, time_labeld) = line.split('\t')
            dataset.append({'fen': fen, 'beta': beta, 'label': label})

    t1 = time.time()

    dataset_0 = [di for di in dataset if di['label'] == '0']
    dataset_1 = [di for di in dataset if di['label'] == '1']
    min_len = len(dataset_0) if len(dataset_0) <= len(dataset_1) else len(dataset_1)
    dataset_0 = dataset_0[:min_len]
    dataset_1 = dataset_1[:min_len]
    bal_dataset = []
    bal_dataset.extend(dataset_0)
    bal_dataset.extend(dataset_1)
    print('Start extracting features')
    pool = multiprocessing.Pool(3)
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
        if option_nn == 3:
            X.append(di['features']+[di['beta']])
            labels.append(di['label'])
        elif option_nn in (1, 2):
            if len(di['features']+[di['beta']]) == 390:
                X.append(di['features'] + [di['beta']])
                labels.append(di['label'])

    t3 = time.time()
    print('\nTime creating X and labels: {}'.format(t3 - t2))

    with open(folder_model +'/X.pkl', 'wb') as f:
        pickle.dump(X, f)

    with open(folder_model + '/labels.pkl', 'wb') as f2:
        pickle.dump(labels, f2)


if __name__ == '__main__':
    create_X_and_labels()
