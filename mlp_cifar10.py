from blocks import layers, activations, losses
from utils import load_data, network, optimizers
from utils.transforms import *
import time


# hyper-parameters
learning_rate = 0.001
num_iter = 10000
batch_size = 128
l2_reg = 0.001
print_freq = 100

# network
net = network.Network(l2_reg=l2_reg)

# blocks
net.add_block(layers.Linear(n_in=32 * 32 * 3, n_out=1024))
net.add_block(layers.BatchNorm1d(n_in=1024))
net.add_block(activations.LeakeyReLU(negative_slope=0.01))
net.add_block(layers.Dropout(keep_prob=0.5))
net.add_block(layers.Linear(n_in=1024, n_out=1024))
net.add_block(layers.BatchNorm1d(n_in=1024))
net.add_block(activations.LeakeyReLU(negative_slope=0.01))
net.add_block(layers.Dropout(keep_prob=0.5))
net.add_block(layers.Linear(n_in=1024, n_out=10))
net.add_block(losses.CrossEntropyLoss())

# optimizer
optimizer = optimizers.Adam(learning_rate=learning_rate, betas=(0.9, 0.999))
optimizer.load(net.params, net.grads)

# data and augmentation
train_transform = Transforms([ToTensor()])
val_test_transform = Transforms([ToTensor()])
train, val, test = load_data.load_cifar10(train_transform, val_test_transform)
net.load_data(train, val, test)

# training
t = time.time()
for step in range(num_iter):
    net.train_step(optimizer, step, batch_size)
    if step % print_freq == 0 or step == num_iter - 1:
        net.eval(step, num_iter)
net.plot(time.time() - t)
