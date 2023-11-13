import torch
import torchvision
from params import folder_model
from NeuralNetwork import NeuralNetwork
from simple_input_features import get_features
import chess
import numpy as np

def create_sample():
    fen = 'r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/2KR3R w kq - 0 13'
    feat = get_features(chess.Board(fen))
    x = feat + [1]
    x = np.array(x, dtype='object').astype(np.float32)
    x = torch.from_numpy(x)
    return x


path_for_model = folder_model + '/model.ckpt'
model = NeuralNetwork()
model.load_state_dict(torch.load(path_for_model))
model.eval()

x = create_sample()
traced_script_module = torch.jit.trace(model, x)
traced_script_module.save(folder_model + '/trace_model.pt')
