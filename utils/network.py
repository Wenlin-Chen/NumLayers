import matplotlib.pyplot as plt
import numpy as np


class Network(object):

    def __init__(self, l2_reg=None, task='classification'):
        self.task = task
        if self.task != 'classification' and self.task != 'regression':
            raise Exception('Unknown task (neither classification nor regression)')
        self.blocks = []
        self.num_blocks = 0
        self.x_tr, self.y_tr, self.x_val, self.y_val, self.x_te, self.y_te = None, None, None, None, None, None
        self.l2_reg = l2_reg
        self.params = {}
        self.grads = {}
        # recorder
        self.val_iteration = []
        self.test_iteration = []
        self.loss_his = []
        self.val_acc = []
        self.test_acc = []
        if task == 'classification':
            self.best_val = 0
            self.best_te = 0
        else:
            self.best_val = np.inf
            self.best_te = np.inf

        self.sum_loss = 0
        self.sum_iter = 0


    def add_block(self, block):
        block.num = self.num_blocks
        self.num_blocks += 1
        if block.block == 'linear' or block.block == 'conv2d':
            if self.l2_reg:
                if self.l2_reg != 0:
                    block.l2_reg = self.l2_reg
            self.params['W'+str(block.num)] = block.W
            self.params['b'+str(block.num)] = block.b
            self.grads['W'+str(block.num)] = block.W_grad
            self.grads['b' + str(block.num)] = block.b_grad
        if block.block == 'batch_norm':
            self.params['gamma' + str(block.num)] = block.gamma
            self.params['beta' + str(block.num)] = block.beta
            self.grads['gamma' + str(block.num)] = block.gamma_grad
            self.grads['beta' + str(block.num)] = block.beta_grad
        if block.block == 'loss' and self.l2_reg:
            block.l2_reg = self.l2_reg
            block.params = self.params
        self.blocks.append(block)

    def forward(self, x, labels):
        temp = x
        loss = None
        for block in self.blocks:
            if block.block == 'loss':
                loss = block.forward(input=temp, labels=labels)
            else:
                temp = block.forward(input=temp)
        return loss

    def backward(self):
        temp = None
        for block in reversed(self.blocks):
            if block.block == 'loss':
                temp = block.backward()
            else:
                temp = block.backward(grad=temp)

    def score(self, x, y):
        temp = x
        acc = None
        for block in self.blocks:
            if block.block == 'loss':
                acc = block.score(input=temp, y=y)
            else:
                temp = block.score(input=temp)
        return acc

    def train_step(self, optimizer, step, batch_size):

        # set all the gradients to be zero
        self.zero_grad()

        # forward
        batches = np.random.choice(np.arange(self.x_tr.shape[0]), batch_size, replace=False)
        x_batch, y_batch = self.x_tr[batches, :], self.y_tr[batches]
        loss = self.forward(x_batch, y_batch)
        self.sum_loss += loss
        self.sum_iter += 1

        # backward
        self.backward()

        # parameters update
        optimizer.step(step)

    def eval(self, step, num_iter, split=1):
        # split validation and test data into subsets
        x_val, y_val = np.split(self.x_val, split), np.split(self.y_val, split)
        x_te, y_te = np.split(self.x_te, split), np.split(self.y_te, split)

        # print and record
        self.val_iteration.append(step)
        ave_loss = self.sum_loss / self.sum_iter
        self.loss_his.append(ave_loss)
        self.sum_loss = 0
        self.sum_iter = 0
        print('iteration:', step)
        print('   training loss', ave_loss)

        sum_acc = 0
        for index in range(split):
            sum_acc += self.score(x_val[index], y_val[index]) * x_val[index].shape[0]
        acc = sum_acc / self.x_val.shape[0]
        self.val_acc.append(acc)

        if self.task == 'classification':
            print('   validation accuracy:', acc)
        else:
            print('   validation loss:', acc)

        if self.task == 'classification':
            if acc > self.best_val or step == num_iter - 1:
                self.test_iteration.append(step)
                if acc > self.best_val:
                    self.best_val = acc
                sum_acc_te = 0
                for index in range(split):
                    sum_acc_te += self.score(x_te[index], y_te[index]) * x_te[index].shape[0]
                acc_te = sum_acc_te / self.x_te.shape[0]
                self.test_acc.append(acc_te)
                print('   test accuracy:', acc_te)
                if acc_te > self.best_te:
                    self.best_te = acc_te
        else:
            if acc < self.best_val or step == num_iter - 1:
                self.test_iteration.append(step)
                if acc < self.best_val:
                    self.best_val = acc
                sum_acc_te = 0
                for index in range(split):
                    sum_acc_te += self.score(x_te[index], y_te[index])
                acc_te = sum_acc_te / split
                self.test_acc.append(acc_te)
                print('   test loss:', acc_te)
                if acc_te < self.best_te:
                    self.best_te = acc_te

        print()

    def plot(self, t):
        # print result
        print('---------------------------------optimization complete---------------------------------')
        if self.task == 'classification':
            print('The optimization ran %fs, best validation accuracy %f with test accuracy %f'
                  % (t, self.best_val, self.best_te))
        else:
            print('The optimization ran %fs, best validation loss %f with test loss %f'
                  % (t, self.best_val, self.best_te))

        # plotting figure
        plt.subplot(1, 2, 1)
        plt.title('Loss')
        plt.xlabel('Iteration')
        plt.ylabel('Training Loss')
        plt.plot(self.val_iteration[1:], self.loss_his[1:])
        plt.subplot(1, 2, 2)
        plt.xlabel('Iteration')
        if self.task == 'classification':
            plt.title('Accuracy')
            plt.ylabel('Validation/Test accuracy')
            val_plot, = plt.plot(self.val_iteration[1:], self.val_acc[1:], label='Validation accuracy')
            test_plot, = plt.plot(self.test_iteration[1:], self.test_acc[1:], label='Test accuracy')
        else:
            plt.title('Loss')
            plt.ylabel('Validation/Test loss')
            val_plot, = plt.plot(self.val_iteration[1:], self.val_acc[1:], label='Validation loss')
            test_plot, = plt.plot(self.test_iteration[1:], self.test_acc[1:], label='Test loss')
        plt.legend(handles=[val_plot, test_plot], loc='upper left')
        plt.show()

    def load_data(self, train, val, test):
        self.x_tr, self.y_tr = train
        self.x_val, self.y_val = val
        self.x_te, self.y_te = test

    def zero_grad(self):
        for _, grad in self.grads.items():
            grad[0] = 0
