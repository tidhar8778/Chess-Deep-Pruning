from nn import evaluate
import chess
from NeuralNetwork import NeuralNetwork
from params import input_size, hidden_sizes, num_classes, dropout, batch_size
import torch
import torch.nn as nn
import pickle
import random
import numpy as np

def check_validation_new_data(model, new_data_x, new_data_y):
    ep_loss = 0
    correct = 0
    total = 0
    # Set confusion matrix
    confusion_matrix = torch.zeros(num_classes, num_classes)
    criterion = nn.CrossEntropyLoss()

    x = np.array(new_data_x, dtype='object').astype(float)
    labels = np.array(new_data_y)
    x = np.delete(x, par_indi_del, axis=1)
    x -= par_train_mean
    x /= par_train_std
    data_loader = torch.utils.data.DataLoader(dataset=list(zip(x, labels)), batch_size=batch_size, shuffle=False)
    total_step = len(data_loader)
    torch.set_grad_enabled(False)
    model.eval()
    #if train:
    #    torch.set_grad_enabled(True)
    #    model.train(True)

    for batch_number, (batch_features, batch_labels) in enumerate(data_loader):
        batch_features = batch_features.to(device)
        batch_labels = batch_labels.to(device)

        # Forward and loss
        outputs = model(batch_features.float())
        loss = criterion(outputs, batch_labels.long())
        ep_loss += loss.item()
        # if train:
        #     # Backward and optimize
        #     optimizer.zero_grad()
        #     loss.backward()
        #     optimizer.step()

        _, predicted = torch.max(outputs.data, 1)
        total += batch_labels.size(0)
        correct += (predicted == batch_labels).sum().item()

        for t, p in zip(batch_labels.view(-1), predicted.view(-1)):
            confusion_matrix[t.long(), p.long()] += 1

        if (batch_number + 1) % 100 == 0:
            print('\rStage [{}], Step [{}/{}], Loss: {:.4f}'
                  .format('extra validation', batch_number + 1, total_step,
                          loss.item()), end='')

    ep_loss = ep_loss / total_step
    ep_accuracy = correct / total

    print('\nStage [{}], Loss: {:.4f}, Accuracy: {:.4f}%'.format('extra validation',
                                                                                ep_loss, 100 * ep_accuracy))
    print('Confusion matrix (rows - true value. columns - predicted value):')
    print(confusion_matrix)
    print('Precision: ', str(confusion_matrix.diag() / confusion_matrix.sum(0)))
    print('Recall: ', str(confusion_matrix.diag() / confusion_matrix.sum(1)))

    return ep_loss, ep_accuracy

folder_model = 'dataset_from_search/test 3 - larger model (500)'
folder_data = 'dataset_beta_from_games_2m'
par_indi_del = np.load(folder_model+'/par_indi_del.npy')
par_train_mean = np.load(folder_model+'/par_train_mean.npy')
par_train_std = np.load(folder_model+'/par_train_std.npy')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNetwork(input_size-len(par_indi_del), hidden_sizes, num_classes, dropout).to(device)
model.load_state_dict(torch.load(folder_model+'/model.ckpt'))
model.eval()

with open(folder_data+'/balX.pkl', 'rb') as f:
    data_x = pickle.load(f)
    size = len(data_x)
    randomlist = random.sample(range(0, size), 10000)
    data_x = [data_x[i] for i in randomlist]
with open(folder_data+'/ballabels.pkl', 'rb') as f:
    data_y = pickle.load(f)
    data_y = [data_y[i] for i in randomlist]


loss, acc = check_validation_new_data(model, data_x, data_y)
print(loss)
print(acc)
