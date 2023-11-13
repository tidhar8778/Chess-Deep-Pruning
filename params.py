import numpy as np

verbose = True

num_classes = 2
num_epochs = 100
batch_size = 64
learning_rate = 0.001
dropout = 0.3
option_nn = 2

if option_nn in (1, 2):
    input_size = 390
elif option_nn == 3:
    input_size = 769

base_folder = 'C:\\Users\\User\\Documents\\tidhar\\barilan\\DeepLearning\\thesis\\chess\\tucano\\Tucano\\data\\train_model\\'
folder_model = base_folder + 'special_features'
