import torch
import torch.nn as nn
import time
import numpy as np
from sklearn.model_selection import train_test_split
import random
import os
from NeuralNetwork import NeuralNetwork
from params import learning_rate, num_epochs, num_classes, batch_size, dropout, folder_model, option_nn
import pickle
# from extract_features import get_features

# Random seed set
torch.backends.cudnn.deterministic = True
random.seed(10)
torch.manual_seed(10)
torch.cuda.manual_seed(10)
np.random.seed(10)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
verbose = True

def split_data(file_x, file_labels, test_ratio=0.2):
    t1 = time.time()
    with open(file_x, 'rb') as f1:
        if verbose:
            print('Start read X')
        X = pickle.load(f1)
        if verbose:
            print('Done read X')
    with open(file_labels, 'rb') as f2:
        labels = pickle.load(f2)
        if verbose:
            print('Done read labels')
            print(time.time()-t1)

    if option_nn in (1, 2):
        X = np.array(X, dtype='object').astype(np.float32)
    elif option_nn == 3:
        X = np.array(X, dtype='object').astype(np.float32)
    labels = np.array(labels)
    print(1, str(time.time() - t1))
    # low_indi = np.where(X[:, -1] <= min_beta_data_collect)
    print(2, str(time.time() - t1))
    # X = np.delete(X, low_indi, axis=0)
    # labels = np.delete(labels, low_indi, axis=0)
    print(3, str(time.time() - t1))
    train_x, val_x, train_y, val_y = train_test_split(X, labels, test_size=test_ratio, random_state=10)
    print(4, str(time.time() - t1))
    t2 = time.time()
    print('Time split data: {}'.format(round(t2-t1)))
    return train_x, val_x, train_y, val_y


def preprocessing(train_x, train_y, val_x, val_y):
    # preprocessing
    t3 = time.time()
    train_mean = np.mean(train_x, axis=0)
    train_std = np.std(train_x, axis=0)
    # delete columns with std=0, means they have the same value (per column) for all rows
    # indices_std_zero = np.where(train_std == 0)[0]
    # train_x = np.delete(train_x, indices_std_zero, axis=1)
    # val_x = np.delete(val_x, indices_std_zero, axis=1)
    # train_mean = np.delete(train_mean, indices_std_zero)
    # train_std = np.delete(train_std, indices_std_zero)
    # Normalize to mean 0 and std 1
    if option_nn in (1, 2):
        train_x -= train_mean
        train_x = np.divide(train_x, train_std, out=np.zeros_like(train_x), where=train_std != 0) # safe division
        val_x -= train_mean
        val_x = np.divide(val_x, train_std, out=np.zeros_like(val_x), where=train_std != 0)   # safe division
        # np.save(folder_model + '/par_indi_del.npy', indices_std_zero)
        np.save(folder_model + '/par_train_mean.npy', train_mean)
        np.save(folder_model + '/par_train_std.npy', train_std)
    with open(folder_model + '/train_x_normalize.pkl', 'wb') as f:
        pickle.dump(train_x, f)
    with open(folder_model + '/train_y.pkl', 'wb') as f:
        pickle.dump(train_y, f)
    with open(folder_model + '/val_x_normalize.pkl', 'wb') as f:
        pickle.dump(val_x, f)
    with open(folder_model + '/val_y.pkl', 'wb') as f:
        pickle.dump(val_y, f)
    train_loader = torch.utils.data.DataLoader(dataset=list(zip(train_x, train_y)), batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset=list(zip(val_x, val_y)), batch_size=batch_size, shuffle=False)
    t4 = time.time()
    print('Time preprocessing: {}'.format(round(t4-t3)))
    return train_loader, val_loader


def run_epoch(epoch, total_step, train):
    start_time = time.time()
    ep_loss = 0
    correct = 0
    total = 0
    # Set confusion matrix
    confusion_matrix = torch.zeros(num_classes, num_classes)

    if not train:
        model.eval()
    if train:
        model.train()

    for batch_number, (batch_features, batch_labels) in enumerate(train_loader if train else val_loader):
        batch_features = batch_features.to(device)
        batch_labels = torch.tensor(tuple(map(int, batch_labels)))
        batch_labels = batch_labels.long()

        if option_nn == 1:
            batch_labels = batch_labels.float().unsqueeze(1)
        # Forward and loss
        outputs = model(batch_features.float())

        loss = criterion(outputs, batch_labels)
        ep_loss += loss.item()
        if train:
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if option_nn == 1:
            sigmoid = nn.Sigmoid()
            predicted = torch.round(sigmoid(outputs))
        else:
            _, predicted = torch.max(outputs.data, 1)
        total += batch_labels.size(0)
        correct += (predicted == batch_labels).sum().item()

        if not train:
            for t, p in zip(batch_labels.view(-1), predicted.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1

        if (batch_number + 1) % 100 == 0:
            print('\rEpoch [{}/{}], Stage [{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, num_epochs, 'train' if train else 'validation', batch_number + 1, total_step, loss.item()), end='')

    ep_loss = ep_loss / total_step
    ep_accuracy = correct / total

    print('\nEpoch [{}/{}], Stage [{}], Loss: {:.4f}, Accuracy: {:.4f}%, Time epoch: {:.0f} seconds'.format(epoch+1, num_epochs, 'train' if train else 'validation', ep_loss, 100*ep_accuracy, time.time()-start_time))
    if train and (epoch+1) % 10 == 0:
        torch.save(model.state_dict(), folder_model + '/model.ckpt')

    if not train and ((epoch + 1) % 10 ==0 or epoch == 0):
        print('Confusion matrix (rows - true value. columns - predicted value):')
        print(confusion_matrix)
        print('Precision: ', str(confusion_matrix.diag() / confusion_matrix.sum(0)))
        print('Recall: ', str(confusion_matrix.diag() / confusion_matrix.sum(1)))
    return ep_loss, ep_accuracy


def train_and_val():
    total_step_train = len(train_loader)
    total_step_validation = len(val_loader)
    train_loss = []
    train_accuracy = []
    val_loss = []
    val_accuracy = []
    for epoch in range(num_epochs):
        ep_loss, ep_accuracy = run_epoch(epoch, total_step_train, train=True)
        train_loss.append(ep_loss)
        train_accuracy.append(ep_accuracy)

        # validation
        ep_val_loss, ep_val_accuracy = run_epoch(epoch, total_step_validation, train=False)
        val_loss.append(ep_val_loss)
        val_accuracy.append(ep_val_accuracy)

    # Save the model checkpoint
    torch.save(model.state_dict(), folder_model+'/model.ckpt')
    results = {}
    results['train_loss'] = train_loss
    results['train_accuracy'] = train_accuracy
    results['val_loss'] = val_loss
    results['val_accuracy'] = val_accuracy
    file_results = folder_model+'/loss_accuracy.pkl'
    with open(file_results, 'wb') as f:
        pickle.dump(results, f)
    return train_loss, train_accuracy, val_loss, val_accuracy


if __name__ == '__main__':
    # train_x, val_x, train_y, val_y = split_data(folder_model+'/X.pkl', folder_model+'/labels.pkl', 0.2)
    # train_loader, val_loader = preprocessing(train_x, train_y, val_x, val_y)
    with open(folder_model+'/train_x_normalize.pkl', 'rb') as f:
        train_x = pickle.load(f)
    with open(folder_model+'/train_y.pkl', 'rb') as f:
        train_y = pickle.load(f)
    with open(folder_model+'/val_x_normalize.pkl', 'rb') as f:
        val_x = pickle.load(f)
    with open(folder_model+'/val_y.pkl', 'rb') as f:
        val_y = pickle.load(f)

    train_loader = torch.utils.data.DataLoader(dataset=list(zip(train_x, train_y)), batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset=list(zip(val_x, val_y)), batch_size=batch_size, shuffle=False)

    model = NeuralNetwork().to(device)

    # Loss and optimizer
    if option_nn == 1:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_loss, train_accuracy, val_loss, val_accuracy = train_and_val()

    # Workaround to exit the process on my Windows 7 laptop
    os.kill(os.getpid(), 0)
