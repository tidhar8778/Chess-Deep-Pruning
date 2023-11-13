import torch
import chess
import numpy as np
from extract_features import get_features as special_features
from simple_input_features import get_features as halfka_features
from params import folder_model, input_size, num_classes, dropout, option_nn
from NeuralNetwork import NeuralNetwork
import pickle

def evaluate(fen, beta):
    '''

    :param board: fen - FEN presentation of board position
    :param beta: Value (lower than beta_max_value)
    :return: Prune or not
    '''
    if option_nn in (1, 2):
        get_features = special_features
    elif option_nn == 3:
        get_features = halfka_features

    path_for_model = folder_model + '/model.ckpt'
    model = NeuralNetwork()
    model.load_state_dict(torch.load(path_for_model))
    model.eval()
    if option_nn in (1, 2):
        par_train_mean = np.load(folder_model + '/par_train_mean.npy')
        par_train_std = np.load(folder_model + '/par_train_std.npy')

    board = chess.Board(fen)
    features = get_features(board)
    features.append(beta)
    # Sanity check. Sometimes get_features returns different sized list, when there is more than 2 rooks or 2 bishops or 2 queens or 3 knights on board
    if len(features) != input_size:
        return 0
    x = np.array(features)
    x = x.astype(np.float32)
    if option_nn in (1, 2):
        x -= par_train_mean
        x = np.divide(x, par_train_std, out=np.zeros_like(x), where=par_train_std!=0)   # save division
    x = torch.from_numpy(x).float()
    output = model(x)
    _, predicted = torch.max(output.data, 0)

    int_predicted = predicted.item()
    return int_predicted
