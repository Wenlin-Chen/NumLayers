import gzip
import os
import pickle
import sys
import numpy as np


def load_mnist(train_transform, val_test_transfrom, root='./data/mnist'):
    f = gzip.open(os.path.join(root, 'mnist.pkl.gz'), 'rb')
    if sys.version_info[0] == 2:
        train, val, test = pickle.load(f)
    else:
        train, val, test = pickle.load(f, encoding='latin1')
    train_data, val_data, test_data = train[0].reshape(-1, 1, 28, 28), val[0].reshape(-1, 1, 28, 28), \
                                      test[0].reshape(-1, 1, 28, 28)
    train_data, val_data, test_data = train_transform.transform(train_data), val_test_transfrom.transform(val_data), \
                                      val_test_transfrom.transform(test_data)
    train_labels, val_labels, test_labels = train[1], val[1], test[1]

    return (train_data, train_labels), (val_data, val_labels), (test_data, test_labels)

def load_cifar10(train_transform, val_test_transfrom, root='./data/cifar10'):
    # from torchvision.datasets.CIFAR10
    base_folder = 'cifar-10-batches-py'

    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
    ]
    val_list = [
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb']
    ]
    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]

    # training set
    train_data = []
    train_labels = []
    for fentry in train_list:
        f = fentry[0]
        file = os.path.join(root, base_folder, f)
        fo = open(file, 'rb')
        if sys.version_info[0] == 2:
            entry = pickle.load(fo)
        else:
            entry = pickle.load(fo, encoding='latin1')
        train_data.append(entry['data'])
        if 'labels' in entry:
            train_labels += entry['labels']
        else:
            train_labels += entry['fine_labels']
        fo.close()
    train_data = np.concatenate(train_data)
    train_labels = np.array(train_labels)
    train_data = train_data.reshape(40000, 3, 32, 32)
    train_data = train_transform.transform(train_data)
    train = train_data, train_labels

    # test_set
    f = test_list[0][0]
    file = os.path.join(root, base_folder, f)
    fo = open(file, 'rb')
    if sys.version_info[0] == 2:
        entry = pickle.load(fo)
    else:
        entry = pickle.load(fo, encoding='latin1')
    test_data = entry['data']
    if 'labels' in entry:
        test_labels = entry['labels']
    else:
        test_labels = entry['fine_labels']
    fo.close()
    test_labels = np.array(test_labels)
    test_data = test_data.reshape(10000, 3, 32, 32)
    test_data = val_test_transfrom.transform(test_data)
    test = test_data, test_labels

    # validation set
    f = val_list[0][0]
    file = os.path.join(root, base_folder, f)
    fo = open(file, 'rb')
    if sys.version_info[0] == 2:
        entry = pickle.load(fo)
    else:
        entry = pickle.load(fo, encoding='latin1')
    val_data = entry['data']
    if 'labels' in entry:
        val_labels = entry['labels']
    else:
        val_labels = entry['fine_labels']
    fo.close()
    val_labels = np.array(val_labels)
    val_data = val_data.reshape(10000, 3, 32, 32)
    val_data = val_test_transfrom.transform(val_data)
    val = val_data, val_labels

    return train, val, test
