from model import Layers, load_data
from model.network import Network

# hyper-parameters
learning_rate = 0.01
num_iter = 10000
batch_size = 128

# network
net = Network(learning_rate=learning_rate, num_iter=num_iter, batch_size=batch_size, lr_decay=None)
hidden1 = Layers.HiddenLayer(name='hidden1', n_in=32 * 32 * 3, n_out=1024, activation='relu', batch_norm=True)
drop1 = Layers.Dropout(name='dropout1', keep_prob=0.5)
hidden2 = Layers.HiddenLayer(name='hideen2', n_in=32 * 32 * 3, n_out=1024, activation='relu', batch_norm=True)
drop2 = Layers.Dropout(name='dropout2', keep_prob=0.5)
soft = Layers.Softmax(name='softmax', n_in=1024, n_out=10)
net.add_layer(hidden1)
net.add_layer(drop1)
net.add_layer(soft)

# data
train, val, test = load_data.load_cifar10()
net.load_data(train, val, test)

# training
net.train()
