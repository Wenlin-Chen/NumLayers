from blocks import layers, activations, losses
from utils import load_data, network, optimizers
from utils.transforms import *


# hyper-parameters
learning_rate = 0.1
num_epoches = 25
batch_size = 128
lr_decay = [0.2, 15]
l2_reg = 0.001
eval_freq = 100

# network
net = network.Network(l2_reg=l2_reg)

# blocks
net.add_block(layers.Linear(n_in=28 * 28 * 1, n_out=1024))
net.add_block(layers.BatchNorm1d(n_in=1024))
net.add_block(activations.ReLU(inplace=True))
net.add_block(layers.Dropout(keep_prob=0.5))
net.add_block(layers.Linear(n_in=1024, n_out=10))
net.add_block(losses.CrossEntropyLoss())

# optimizer
optimizer = optimizers.Momentum(learning_rate=learning_rate, momentum=0.9, Nesterov=True, lr_decay=lr_decay)

# data loading and augmentation
train_transform = Transforms([ToTensor()])
val_test_transform = Transforms([ToTensor()])
train, val, test = load_data.load_mnist(train_transform, val_test_transform)
net.load_data(train, val, test)

# training
net.train(optimizer, num_epoches, batch_size, eval_freq)
