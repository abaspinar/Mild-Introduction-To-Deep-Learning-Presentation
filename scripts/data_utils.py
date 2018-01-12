import glob
import numpy as np
import os
import pickle

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='latin1')
    return dict

def load_CIFAR10(data_path='datasets/cifar-10-batches-py', train_percent=1, test_percent=1):
    x_train = []
    y_train = []

    #load all training data and their labels
    for file in glob.glob(os.path.join(data_path, 'data_batch_*')):
        dict = unpickle(file)
        x = dict['data']
        x_train.append(x.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float"))
        y = np.array(dict['labels'])
        y_train.append(y)
    x_train = np.concatenate(x_train)
    y_train = np.concatenate(y_train)

    #load all test data and their labels
    dict = unpickle(os.path.join(data_path, 'test_batch'))
    x = dict['data']
    x_test = x.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
    y_test = np.array(dict['labels'])

    #load label names
    dict = unpickle(os.path.join(data_path, 'batches.meta'))
    label_names = np.array(dict['label_names'])

    #return the percent of the total data
    train_length = int(len(x_train) * train_percent)
    test_length = int(len(x_test) * test_percent)

    return x_train[:train_length], y_train[:train_length], x_test[:test_length], y_test[:test_length], label_names









