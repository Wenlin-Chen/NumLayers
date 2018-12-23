from blocks import layers, activations, losses
from utils import load_data, network, optimizers
import time


# hyper-parameters
learning_rate = 0.001
num_iter = 5000
batch_size = 128
l2_reg = 1e-5
print_freq = 100

# network
net = network.Network(num_iter=num_iter, batch_size=batch_size, l2_reg=l2_reg)

# blocks
net.add_block(layers.Conv2d(in_channels=3, out_channels=8, kernel_size=(5, 5), padding=2, stride=1))
net.add_block(layers.BatchNorm2d(in_channels=8))
net.add_block(activations.ReLU(inplace=True))
net.add_block(layers.MaxPool2d(stride=2))
net.add_block(layers.Conv2d(in_channels=8, out_channels=16, kernel_size=(5, 5), padding=2, stride=1))
net.add_block(layers.BatchNorm2d(in_channels=16))
net.add_block(activations.ReLU(inplace=True))
net.add_block(layers.MaxPool2d(stride=2))
net.add_block(layers.Linear(n_in=8 * 8 * 16, n_out=512))
net.add_block(layers.BatchNorm1d(n_in=512))
net.add_block(activations.ReLU(inplace=True))
net.add_block(layers.Dropout(keep_prob=0.5))
net.add_block(layers.Linear(n_in=512, n_out=10))
net.add_block(losses.CrossEntropyLoss())

#optimizer
optimizer = optimizers.Adam(learning_rate=learning_rate, betas=(0.9, 0.999))
optimizer.load(net.params, net.grads)

# data
train, val, test = load_data.load_cifar10()
net.load_data(train, val, test)

# training
t = time.time()
for step in range(num_iter):
    net.train(optimizer, step)
    if step % print_freq == 0 or step == num_iter - 1:
        net.eval(step, split=100)
net.plot(time.time() - t)
