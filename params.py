import numpy as np

verbose = False
input_size = 391
hidden_sizes = [256, 128, 64, 32] # [128, 64, 32]
num_classes = 2
num_epochs = 500
batch_size = 64
learning_rate = 0.001
dropout = 0.3
inf_value = 300
iterative = True
use_quiesce = True
quiesce_max_depth = 10
search_max_depth = 5
layer_depth_pruning = 3
margin_from_beta = 1
white_use_pruning = True
black_use_pruning = True
collect_data = False
test_positions = True
test_positions_full = False     # run full game
max_beta_data_collect = 5
min_beta_data_collect = -5
folder_model = 'dataset_from_search/test 3 - larger model (500)'