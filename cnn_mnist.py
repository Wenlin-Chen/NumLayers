from blocks import layers, activations, losses
from utils import load_data, network, optimizers


# hyper-parameters
learning_rate = 0.001
num_iter = 4000
batch_size = 128
l2_reg = 1e-5

# network
net = network.Network(num_iter=num_iter, batch_size=batch_size, l2_reg=l2_reg)

# blocks
net.add_block(layers.Conv2d(in_channels=1, out_channels=8, kernel_size=(5, 5), padding=2, stride=1))
net.add_block(layers.BatchNorm2d(in_channels=8))
net.add_block(activations.ReLU(inplace=True))
net.add_block(layers.MaxPool2d(stride=2))
net.add_block(layers.Conv2d(in_channels=8, out_channels=16, kernel_size=(5, 5), padding=2, stride=1))
net.add_block(layers.BatchNorm2d(in_channels=16))
net.add_block(activations.ReLU(inplace=True))
net.add_block(layers.MaxPool2d(stride=2))
net.add_block(layers.Linear(n_in=7 * 7 * 16, n_out=512))
net.add_block(layers.BatchNorm1d(n_in=512))
net.add_block(activations.ReLU(inplace=True))
net.add_block(layers.Dropout(keep_prob=0.5))
net.add_block(layers.Linear(n_in=512, n_out=10))
net.add_block(losses.CrossEntropyLoss())

#optimizer
optimizer = optimizers.Adam(learning_rate=learning_rate, betas=(0.9, 0.999))

# data
train, val, test = load_data.load_mnist()
net.load_data(train, val, test)

# training
net.train(optimizer)
