import numpy as np


def split_minibatch(x, y, batch_size, shuffle):
    size = x.shape[0]

    # randomly shuffle the data set
    if shuffle:
        permutation = np.random.permutation(size)
        x, y = x[permutation], y[permutation]

    # split the training set into mini batches
    split = size - size % batch_size
    idx = list(range(batch_size, split + 1, batch_size))
    if size % batch_size == 0:
        idx.pop()

    return np.split(x, idx), np.split(y, idx)
