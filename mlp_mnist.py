from blocks import layers, activations, losses
from utils import load_data, network, optimizers

# hyper-parameters
learning_rate = 0.2
num_iter = 10000
batch_size = 128
lr_decay = [0.2, 6000]
l2_reg = 0.001

# network
net = network.Network(num_iter=num_iter, batch_size=batch_size, l2_reg=l2_reg)

# blocks
net.add_block(layers.Linear(n_in=28 * 28 * 1, n_out=1024))
net.add_block(layers.BatchNorm1d(n_in=1024))
net.add_block(activations.ReLU(inplace=True))
net.add_block(layers.Dropout(keep_prob=0.5))
net.add_block(layers.Linear(n_in=1024, n_out=10))
net.add_block(losses.CrossEntropyLoss())

#optimizer
optimizer = optimizers.SGD(learning_rate=learning_rate, lr_decay=lr_decay)

# data
train, val, test = load_data.load_mnist()
net.load_data(train, val, test)

# training
net.train(optimizer)
