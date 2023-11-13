import time
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn import metrics
verbose = True


def logistic_regression(folder):
    t1 = time.time()
    with open(folder + '/train_x_normalize.pkl', 'rb') as f1:
        if verbose:
            print('Start read train X')
        train_x = pickle.load(f1)
        if verbose:
            print('Done read train X')
    with open(folder + '/val_x_normalize.pkl', 'rb') as f1:
        if verbose:
            print('Start read val X')
        val_x = pickle.load(f1)
        if verbose:
            print('Done read val X')

    with open(folder + '/train_y.pkl', 'rb') as f2:
        train_labels = pickle.load(f2)
        if verbose:
            print('Done read train labels')
            print(time.time() - t1)

    with open(folder + '/val_y.pkl', 'rb') as f2:
        val_labels = pickle.load(f2)
        if verbose:
            print('Done read val labels')
            print(time.time() - t1)

    #train_x, val_x = np.array(train_x, dtype='object').astype(float), np.array(val_x, dtype='object').astype(float)
    #train_labels, val_labels = np.array(train_labels), np.array(val_labels)
    print('Time after read and split data ', str(time.time() - t1))

    logreg = LogisticRegression()
    logreg.fit(train_x, train_labels)
    acc = logreg.score(val_x, val_labels)
    print('Time after fit model and predict validation: ', str(time.time() - t1))
    print('Logistic Regression accuracy: {:.2f}'.format(acc))

def svm(folder):
    t1 = time.time()
    with open(folder + '/train_x_normalize.pkl', 'rb') as f1:
        if verbose:
            print('Start read train X')
        train_x = pickle.load(f1)
        if verbose:
            print('Done read train X')
    with open(folder + '/val_x_normalize.pkl', 'rb') as f1:
        if verbose:
            print('Start read val X')
        val_x = pickle.load(f1)
        if verbose:
            print('Done read val X')

    with open(folder + '/train_y.pkl', 'rb') as f2:
        train_labels = pickle.load(f2)
        if verbose:
            print('Done read train labels')
            print(time.time() - t1)

    with open(folder + '/val_y.pkl', 'rb') as f2:
        val_labels = pickle.load(f2)
        if verbose:
            print('Done read val labels')
            print(time.time() - t1)

    #train_x, val_x = np.array(train_x, dtype='object').astype(float), np.array(val_x, dtype='object').astype(float)
    #train_labels, val_labels = np.array(train_labels), np.array(val_labels)
    print('Time after read and split data ', str(time.time() - t1))

    svc = SVC(kernel='linear')
    svc.fit(train_x, train_labels)
    y_predict = svc.predict(val_x)
    acc = metrics.accuracy_score(val_labels, y_predict)
    precision = metrics.precision_score(val_labels, y_predict)
    recall = metrics.recall_score(val_labels, y_predict)
    print('Time after fit model and predict validation: ', str(time.time() - t1))
    print('SVM accuracy: {:.2f}\nPrecision: {:.2f}\nRecall: {:.2f}'.format(acc, precision, recall))


if __name__ == '__main__':
    #logistic_regression('dataset_from_search')
    svm('dataset_from_search')