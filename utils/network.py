import time
import matplotlib.pyplot as plt
import numpy as np


class Network(object):

    def __init__(self, num_iter, batch_size, l2_reg=None):
        self.layers = []
        self.num_layers = 0
        self.num_iter = num_iter
        self.batch_size = batch_size
        self.x_tr, self.y_tr, self.x_val, self.y_val, self.x_te, self.y_te = None, None, None, None, None, None
        self.l2_reg = l2_reg
        self.params = {}
        self.grads = {}


    def add_layer(self, layer):
        layer.num = self.num_layers
        self.num_layers += 1
        if layer.layer == 'linear':
            if self.l2_reg:
                layer.l2_reg = self.l2_reg
            self.params['W'+str(layer.num)] = layer.W
            self.params['b'+str(layer.num)] = layer.b
            self.grads['W'+str(layer.num)] = layer.W_grad
            self.grads['b' + str(layer.num)] = layer.b_grad
        if layer.layer == 'batch_norm':
            self.params['gamma' + str(layer.num)] = layer.gamma
            self.params['beta' + str(layer.num)] = layer.beta
            self.grads['gamma' + str(layer.num)] = layer.gamma_grad
            self.grads['beta' + str(layer.num)] = layer.beta_grad
        if layer.layer == 'loss' and self.l2_reg:
            layer.l2_reg = self.l2_reg
            layer.params = self.params
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

    def train(self, optimizer, task='classification'):
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
            # set all the gradients to be zero
            self.zero_grad()

            # forward
            batches = np.random.choice(np.arange(self.x_tr.shape[0]), self.batch_size, True)
            x_batch, y_batch = self.x_tr[batches, :], self.y_tr[batches]
            loss = self.forward(x_batch, y_batch)
            sum_loss += loss
            sum_iter += 1

            # backward
            self.backward()

            # parameters update
            optimizer.step(i)

            # print
            if (i != 0) and (i % 100 == 0 or i == self.num_iter - 1):
                val_iteration.append(i)
                ave_loss = sum_loss / sum_iter
                loss_his.append(ave_loss)
                sum_loss = 0
                sum_iter = 0
                print('iteration:', i)
                print('   training loss', ave_loss)

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

    def zero_grad(self):
        for _, grad in self.grads.items():
            grad[0] = 0
