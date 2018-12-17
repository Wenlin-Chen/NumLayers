import time
import matplotlib.pyplot as plt
import numpy as np


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

    def forward(self, x, labels):
        temp = x
        loss = None
        for layer in self.layers:
            if layer.layer == 'loss':
                loss = layer.forward(input=temp, labels=labels)
            else:
                temp = layer.forward(input=temp)
        return loss

    def backward(self):
        temp = None
        for layer in reversed(self.layers):
            if layer.layer == 'loss':
                temp = layer.backward()
            elif layer.layer == 'linear' or layer.layer == 'batch_norm':
                temp = layer.backward(grad=temp, learning_rate=self.lr)
            else:
                temp = layer.backward(grad=temp)

    def score(self, x, y):
        temp = x
        acc = None
        for layer in self.layers:
            if layer.layer == 'loss':
                acc = layer.score(input=temp, y=y)
            else:
                temp = layer.score(input=temp)
        return acc

    def train(self, task='classification'):
        if task != 'classification' and task != 'regression':
            raise Exception('Unknown task (neither classification nor regression)')
        # recorder
        val_iteration = []
        test_iteration = []
        loss_his = []
        val_acc = []
        test_acc = []
        if task == 'classification':
            best_val = 0
            best_te = 0
        else:
            best_val = np.inf
            best_te = np.inf

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

                if task == 'classification':
                    print('   validation accuracy:', acc)
                else:
                    print('   validation loss:', acc)


                if task == 'classification':
                    if acc > best_val or i == self.num_iter - 1:
                        test_iteration.append(i)
                        if acc > best_val:
                            best_val = acc
                        acc_te = self.score(self.x_te, self.y_te)
                        test_acc.append(acc_te)
                        print('   test accuracy:', acc_te)
                        if acc_te > best_te:
                            best_te = acc_te

                else:
                    if acc < best_val or i == self.num_iter - 1:
                        test_iteration.append(i)
                        if acc < best_val:
                            best_val = acc
                        acc_te = self.score(self.x_te, self.y_te)
                        test_acc.append(acc_te)
                        print('   test loss:', acc_te)
                        if acc_te < best_te:
                            best_te = acc_te

                print()

        print('---------------------------------optimization complete---------------------------------')
        if task == 'classification':
            print('The optimization ran %fs, best validation accuracy %f with test accuracy %f'
                  % ((time.time() - t), best_val, best_te))
        else:
            print('The optimization ran %fs, best validation loss %f with test loss %f'
                  % ((time.time() - t), best_val, best_te))

        # plotting figure
        plt.subplot(1, 2, 1)
        plt.title('Loss')
        plt.xlabel('Iteration')
        plt.ylabel('Training Loss')
        plt.plot(val_iteration[1:], loss_his[1:])
        plt.subplot(1, 2, 2)
        plt.xlabel('Iteration')
        if task == 'classification':
            plt.title('Accuracy')
            plt.ylabel('Validation/Test accuracy')
            val_plot, = plt.plot(val_iteration[1:], val_acc[1:], label='Validation accuracy')
            test_plot, = plt.plot(test_iteration[1:], test_acc[1:], label='Test accuracy')
        else:
            plt.title('Loss')
            plt.ylabel('Validation/Test loss')
            val_plot, = plt.plot(val_iteration[1:], val_acc[1:], label='Validation loss')
            test_plot, = plt.plot(test_iteration[1:], test_acc[1:], label='Test loss')
        plt.legend(handles=[val_plot, test_plot], loc='upper left')
        plt.show()

    def load_data(self, train, val, test):
        self.x_tr, self.y_tr = train
        self.x_val, self.y_val = val
        self.x_te, self. y_te = test
