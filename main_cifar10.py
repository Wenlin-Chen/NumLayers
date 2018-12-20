from models import layers, activations, losses
from utils import load_data, network, optimizers

# hyper-parameters
learning_rate = 0.03
num_iter = 10000
batch_size = 128
lr_decay = None
l2_reg = 0.001

# network
net = network.Network(num_iter=num_iter, batch_size=batch_size, l2_reg=l2_reg)

# optimizer
optimizer = optimizers.SGD(learning_rate=learning_rate, lr_decay=lr_decay)

# layers
net.add_layer(layers.Linear(n_in=32 * 32 * 3, n_out=1024))
net.add_layer(layers.BatchNorm1d(n_in=1024))
net.add_layer(activations.ReLU())
net.add_layer(layers.Dropout(keep_prob=0.5))
net.add_layer(layers.Linear(n_in=1024, n_out=1024))
net.add_layer(layers.BatchNorm1d(n_in=1024))
net.add_layer(activations.ReLU())
net.add_layer(layers.Dropout(keep_prob=0.5))
net.add_layer(layers.Linear(n_in=1024, n_out=10))
net.add_layer(losses.CrossEntropyLoss())

# data
train, val, test = load_data.load_cifar10()
net.load_data(train, val, test)

# training
net.train(optimizer)
