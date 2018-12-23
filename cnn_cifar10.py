from blocks import layers, activations, losses
from utils import load_data, network, optimizers
from utils.transforms import *
import time


# hyper-parameters
learning_rate = 0.003
num_iter = 10000
batch_size = 128
l2_reg = 1e-5
lr_decay = [0.1, 3000, 6000]
print_freq = 100

# network
net = network.Network(l2_reg=l2_reg)

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
optimizer = optimizers.Adam(learning_rate=learning_rate, betas=(0.9, 0.999), lr_decay=lr_decay)
optimizer.load(net.params, net.grads)

# data and augmentation
mean = [x / 255 for x in [125.3, 123.0, 113.9]]
std = [x / 255 for x in [63.0, 62.1, 66.7]]
train_transform = Transforms([ToTensor(), Pad(4), RandomCrop(32), RandomHorizontalFlip(), Normalize(mean, std)])
val_test_transform = Transforms([ToTensor(), Normalize(mean, std)])
train, val, test = load_data.load_cifar10(train_transform, val_test_transform)
net.load_data(train, val, test)

# training
t = time.time()
for step in range(num_iter):
    net.train_step(optimizer, step, batch_size)
    if step % print_freq == 0 or step == num_iter - 1:
        net.eval(step, num_iter, split=100)
net.plot(time.time() - t)
