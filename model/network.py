import time
import matplotlib.pyplot as plt
import numpy as np
from model import Layers


class Network(object):

    def __init__(self, learning_rate, num_iter, batch_size, lr_decay=None):
        self.layers = []
        self.lr = learning_rate
        self.num_iter = num_iter
        self.batch_size = batch_size
        self.x_tr, self.y_tr, self.x_val, self.y_val, self.x_te, self.y_te = None, None, None, None, None, None
        self.lr_decay_rate = None
        self.lr_decay_iter = None
        if lr_decay:
            self.lr_decay_iter = lr_decay[1:]
            self.lr_decay_rate = lr_decay[0]


    def add_layer(self, layer):
        self.layers.append(layer)

    def forward(self, x, label):
        temp = x
        loss = None
        for layer in self.layers:
            if type(layer) == Layers.Softmax:
                loss = layer.forward(input=temp, label=label)
            else:
                temp = layer.forward(input=temp)
        return loss

    def backward(self):
        temp = None
        for layer in reversed(self.layers):
            if type(layer) == Layers.Softmax:
                temp = layer.backward(learning_rate=self.lr)
            else:
                temp = layer.backward(forward_delta=temp, learning_rate=self.lr)

    def score(self, x, y):
        temp = x
        acc = None
        for layer in self.layers:
            if type(layer) == Layers.Softmax:
                acc = layer.score(input=temp, y=y)
            else:
                temp = layer.score(input=temp)
        return acc

    def train(self):
        # recorder
        val_iteration = []
        test_iteration = []
        loss_his = []
        val_acc = []
        test_acc = []
        best_val = 0
        best_te = 0

        # training
        t = time.time()
        sum_loss = 0
        sum_iter = 0
        for i in range(self.num_iter):
            if self.lr_decay_rate:
                if i in self.lr_decay_iter:
                    self.lr *= self.lr_decay_rate

            batches = np.random.choice(np.arange(self.x_tr.shape[0]), self.batch_size, True)
            x_batch, y_batch = self.x_tr[batches, :], self.y_tr[batches]
            loss = self.forward(x_batch, y_batch)
            sum_loss += loss
            sum_iter += 1
            self.backward()

            if (i != 0) and (i % 100 == 0 or i == self.num_iter - 1):
                val_iteration.append(i)
                loss_his.append(sum_loss / sum_iter)
                sum_loss = 0
                sum_iter = 0
                print('iteration:', i)
                print('   training loss', loss)

                acc = self.score(self.x_val, self.y_val)
                val_acc.append(acc)
                print('   validation accuracy:', acc)

                if acc > best_val or i == self.num_iter - 1:
                    test_iteration.append(i)
                    if acc > best_val:
                        best_val = acc
                    acc_te = self.score(self.x_te, self.y_te)
                    test_acc.append(acc_te)
                    print('   test accuracy:', acc_te)
                    if acc_te > best_te:
                        best_te = acc_te

                print()

        print('---------------------------------optimization complete---------------------------------')
        print('The optimization ran %fs, best validation accuracy %f with test accuracy %f'
              % ((time.time() - t), best_val, best_te))

        # plotting figure
        plt.subplot(1, 2, 1)
        plt.title('Loss')
        plt.xlabel('Iteration')
        plt.ylabel('Training Loss')
        plt.plot(val_iteration[1:], loss_his[1:])
        plt.subplot(1, 2, 2)
        plt.title('Accuracy')
        plt.xlabel('Iteration')
        plt.ylabel('Validation/Test accuracy')
        val_plot, = plt.plot(val_iteration[1:], val_acc[1:], label='Validation accuracy')
        test_plot, = plt.plot(test_iteration[1:], test_acc[1:], label='Test accuracy')
        plt.legend(handles=[val_plot, test_plot], loc='upper left')
        plt.show()

    def load_data(self, train, val, test):
        self.x_tr, self.y_tr = train
        self.x_val, self.y_val = val
        self.x_te, self. y_te = test
