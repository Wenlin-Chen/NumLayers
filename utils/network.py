from utils.split import split_minibatch
import matplotlib.pyplot as plt
import numpy as np
import time


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

        # recorders
        self.epochs = []
        self.loss_his = []
        self.train_score = []
        self.val_score = []
        self.test_score = []

        self.best_val_epoch = 0
        if task == 'classification':
            self.best_val = 0
        else:
            self.best_val = np.inf


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
        for block in self.blocks:
            if block.block == 'loss':
                temp = block.forward(input=temp, labels=labels)
            else:
                temp = block.forward(input=temp)
        return temp

    def backward(self, grad=None):
        temp = grad
        for block in reversed(self.blocks):
            if block.block == 'loss':
                temp = block.backward()
            else:
                temp = block.backward(grad=temp)
        return temp

    def score(self, x, y):
        temp = x
        acc = None
        for block in self.blocks:
            if block.block == 'loss':
                acc = block.score(input=temp, y=y)
            else:
                temp = block.score(input=temp)
        return acc

    def train(self, optimizer, num_epoches, batch_size, eval_freq):
        print('\nSocre = Accuracy for classification task / Raw Loss for regression task\n')
        optimizer.load(self.params, self.grads)

        t = time.time()
        for epoch in range(num_epoches):
            self.epochs.append(epoch)

            x_tr_batches, y_tr_batches = split_minibatch(self.x_tr, self.y_tr, batch_size, shuffle=True)
            num_batches = len(x_tr_batches)
            sum_loss = 0
            for i in range(num_batches):
                # set all the gradients to be zero
                self.zero_grad()
                # forward
                loss = self.forward(x_tr_batches[i], y_tr_batches[i])
                sum_loss += loss
                # backward
                self.backward()
                # parameters update
                optimizer.step(epoch)

                # track
                if (i + 1) == num_batches:
                    ave_loss = sum_loss / (i + 1)
                    self.track(i, num_batches, epoch, num_epoches, batch_size, ave_loss, record=True)
                elif (i + 1) % eval_freq == 0:
                    ave_loss = sum_loss / (i + 1)
                    self.track(i, num_batches, epoch, num_epoches, batch_size, ave_loss, record=False)

        self.plot(time.time() - t)

    def eval(self, x, y, batch_size):
        x_batches, y_batches = split_minibatch(x, y, batch_size, shuffle=False)
        sum_score = 0
        for i in range(len(x_batches)):
            sum_score += self.score(x_batches[i], y_batches[i]) * x_batches[i].shape[0]
        return sum_score / x.shape[0]

    def track(self, i, num_batches, epoch, num_epoches, batch_size, loss, record):
        print('Epoch[{0}/{1}], Iteration[{2}/{3}]'.format(epoch, num_epoches-1, i+1, num_batches))
        print('    Training loss: {0}'.format(loss))

        # evaluate validation set
        val_score = self.eval(self.x_val, self.y_val, batch_size)
        print('    Validation score: {0:.4f}'.format(val_score))

        # evaluate test set
        te_score = self.eval(self.x_te, self.y_te, batch_size)
        print('    Test score: {0:.4f}'.format(te_score))

        print()

        if record:
            self.loss_his.append(loss)

            # evaluate training set
            tr_score = self.eval(self.x_tr, self.y_tr, batch_size)
            print('    Training score: {0:.4f}'.format(tr_score))
            print()

            self.train_score.append(tr_score)
            self.val_score.append(val_score)
            self.test_score.append(te_score)

            if self.task == 'classification':
                if val_score > self.best_val:
                    self.best_val = val_score
                    self.best_val_epoch = epoch
            else:
                if val_score < self.best_val:
                    self.best_val = val_score
                    self.best_val_epoch = epoch

    def plot(self, t):
        # print results
        print('---------------------------------optimization completed---------------------------------')
        print('The optimization ran {0}s, best validation score {1:.4f} with test score {2:.4f} at epoch {3}'
              .format(t, self.best_val, self.test_score[self.best_val_epoch], self.best_val_epoch))

        # plot figures
        plt.subplot(1, 3, 1)
        plt.title('Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Training Loss')
        loss_plot, = plt.plot(self.epochs, self.loss_his)
        plt.legend(handles=[loss_plot], loc='upper right')
        plt.grid()

        plt.subplot(1, 3, 2)
        plt.xlabel('Epoch')
        plt.title('Accuracy for classification / Raw Loss for regression')
        plt.ylabel('Training score')
        train_plot, = plt.plot(self.epochs, self.train_score, label='Training score')
        plt.legend(handles=[train_plot], loc='upper left')
        plt.grid()

        plt.subplot(1, 3, 3)
        plt.xlabel('Epoch')
        plt.title('Accuracy for classification / Raw Loss for regression')
        plt.ylabel('Validation/Test score')
        val_plot, = plt.plot(self.epochs, self.val_score, label='Validation score')
        test_plot, = plt.plot(self.epochs, self.test_score, label='Test score')
        plt.legend(handles=[val_plot, test_plot], loc='upper left')
        plt.grid()
        plt.show()

    def load_data(self, train, val, test):
        self.x_tr, self.y_tr = train
        self.x_val, self.y_val = val
        self.x_te, self.y_te = test

    def zero_grad(self):
        for _, grad in self.grads.items():
            grad[0] = 0
