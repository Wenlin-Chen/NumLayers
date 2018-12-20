import numpy as np


class CrossEntropyLoss(object):

    def __init__(self):
        self.layer = 'loss'
        self.num = None
        self.batch_size = None
        self.y = None
        self.p = None
        self.l2_reg =None
        self.params = None

    def forward(self, input, labels):
        self.y = labels
        self.batch_size = input.shape[0]

        exp_score = np.exp(input - np.max(input, axis=1).reshape(-1, 1))
        normalization = np.sum(exp_score, axis=1).reshape(-1, 1)
        self.p = exp_score / normalization
        loss = - np.sum(np.log(self.p[range(self.batch_size), self.y] + 1e-8)) / self.batch_size
        if self.l2_reg:
            for key in self.params:
                if key[0] == 'W':
                    loss += 0.5 * self.l2_reg * np.linalg.norm(self.params[key][0])
        return loss

    def score(self, input, y):
        exp_score = np.exp(input - np.max(input, axis=1).reshape(-1, 1))
        normalization = np.sum(exp_score, axis=1).reshape(-1, 1)
        p = exp_score / normalization
        y_pred = np.argmax(p, axis=1)
        return np.mean(y == y_pred)

    def backward(self):
        p = self.p
        p[range(self.batch_size), self.y] -= 1
        return (p + 1e-8) / self.batch_size


class HingeLoss(object):

    def __init__(self):
        self.layer = 'loss'
        self.num = None
        self.batch_size = None
        self.y = None
        self.margin = None
        self.l2_reg = None
        self.params = None

    def forward(self, input, labels):
        self.y = labels
        self.batch_size = input.shape[0]

        correct_scores = input[range(self.batch_size), list(self.y)].reshape(-1, 1)
        self.margin = input - correct_scores + 1
        self.margin[self.margin < 0] = 0
        self.margin[range(self.batch_size), list(self.y)] = 0
        loss = np.sum(self.margin) / self.batch_size
        if self.l2_reg:
            for key in self.params:
                if key[0] == 'W':
                    loss += 0.5 * self.l2_reg * np.linalg.norm(self.params[key][0])
        return loss

    def score(self, input, y):
        y_pred = np.argmax(input, axis=1)
        return np.mean(y == y_pred)

    def backward(self):
        grad = np.zeros(self.margin.shape)
        grad[self.margin > 0] = 1
        grad[range(self.batch_size), list(self.y)] = - np.sum(grad, axis=1)

        return grad


class MSELoss(object):

    def __init__(self):
        self.layer = 'loss'
        self.num = None
        self.input = None
        self.batch_size = None
        self.y = None
        self.l2_reg = None
        self.params = None

    def forward(self, input, labels):
        self.input = input
        self.batch_size = self.input.shape[0]
        self.y = labels.reshape(self.batch_size, -1)
        loss = 0.5 * np.linalg.norm(input - self.y) / self.batch_size
        if self.l2_reg:
            for key in self.params:
                if key[0] == 'W':
                    loss += 0.5 * self.l2_reg * np.linalg.norm(self.params[key][0])
        return loss

    def score(self, input, y):
        batch_size = input.shape[0]
        y = y.reshape(batch_size, -1)
        return 0.5 * np.linalg.norm(input - y) / batch_size

    def backward(self):
        return (self.input - self.y) / self.batch_size
