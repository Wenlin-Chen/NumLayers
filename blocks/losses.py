import numpy as np


class CrossEntropyLoss(object):

    def __init__(self):
        self.block = 'loss'
        self.num = None
        self.batch_size = None
        self.y = None
        self.p = None
        self.l2_reg =None
        self.params = None

    def softmax(self, z):
        exp_score = np.exp(z - np.max(z, axis=1).reshape(-1, 1))
        normalization = np.sum(exp_score, axis=1).reshape(-1, 1)
        return exp_score / normalization

    def forward(self, input, labels):
        self.y = labels.reshape(-1)
        self.batch_size = input.shape[0]

        self.p =self.softmax(input)
        loss = - np.sum(np.log(self.p[range(self.batch_size), self.y] + 1e-8)) / self.batch_size
        if self.l2_reg:
            for key in self.params:
                if key[0] == 'W':
                    loss += 0.5 * self.l2_reg * np.linalg.norm(self.params[key][0])
        return loss

    def score(self, input, y):
        p = self.softmax(input)
        y_pred = np.argmax(p, axis=1)
        return np.mean(y.reshape(-1) == y_pred)

    def backward(self):
        p = self.p
        p[range(self.batch_size), self.y] -= 1
        return (p + 1e-8) / self.batch_size


class BCELoss(object):

    def __init__(self):
        self.block = 'loss'
        self.num = None
        self.batch_size = None
        self.y = None
        self.p = None
        self.l2_reg =None
        self.params = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def forward(self, input, labels):
        self.y = labels.reshape(-1, 1)
        self.batch_size = input.shape[0]

        self.p =self.sigmoid(input)
        loss = - np.sum(self.y * np.log(self.p + 1e-8) + (1 - self.y) * np.log(1 - self.p + 1e-8)) / self.batch_size
        if self.l2_reg:
            for key in self.params:
                if key[0] == 'W':
                    loss += 0.5 * self.l2_reg * np.linalg.norm(self.params[key][0])
        return loss

    def score(self, input, y):
        p = self.sigmoid(input)
        y_pred = p > 0.5
        return np.mean(y.reshape(-1, 1) == y_pred)

    def backward(self):
        return (self.p - self.y + 1e-8) / self.batch_size


class HingeLoss(object):

    def __init__(self):
        self.block = 'loss'
        self.num = None
        self.batch_size = None
        self.y = None
        self.margin = None
        self.l2_reg = None
        self.params = None

    def forward(self, input, labels):
        self.y = labels.reshape(-1)
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
        return np.mean(y.reshape(-1) == y_pred)

    def backward(self):
        grad = np.zeros(self.margin.shape)
        grad[self.margin > 0] = 1
        grad[range(self.batch_size), list(self.y)] = - np.sum(grad, axis=1)

        return grad


class MSELoss(object):

    def __init__(self):
        self.block = 'loss'
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
