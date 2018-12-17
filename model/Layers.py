import numpy as np


class CrossEntropyLoss(object):

    def __init__(self):
        self.layer = 'loss'
        self.batch_size = None
        self.y = None
        self.p = None

    def forward(self, input, labels):
        self.y = labels
        self.batch_size = input.shape[0]

        exp_score = np.exp(input - np.max(input, axis=1).reshape(-1, 1))
        normalization = np.sum(exp_score, axis=1).reshape(-1, 1)
        self.p = exp_score / normalization
        loss = - np.sum(np.log(self.p[range(self.batch_size), self.y] + 1e-8)) / self.batch_size
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
        self.batch_size = None
        self.y = None
        self.margin = None

    def forward(self, input, labels):
        self.y = labels
        self.batch_size = input.shape[0]

        correct_scores = input[range(self.batch_size), list(self.y)].reshape(-1, 1)
        self.margin = input - correct_scores + 1
        self.margin[self.margin < 0] = 0
        self.margin[range(self.batch_size), list(self.y)] = 0
        loss = np.sum(self.margin) / self.batch_size
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
        self.input = None
        self.batch_size = None
        self.y = None

    def forward(self, input, labels):
        self.input = input
        self.batch_size = self.input.shape[0]
        self.y = labels.reshape(self.batch_size, -1)
        loss = 0.5 * np.linalg.norm(input - self.y) / self.batch_size
        return loss

    def score(self, input, y):
        batch_size = input.shape[0]
        y = y.reshape(batch_size, -1)
        return 0.5 * np.linalg.norm(input - y) / batch_size

    def backward(self):
        return (self.input - self.y) / self.batch_size


class Linear(object):

    def __init__(self, n_in, n_out):
        self.layer = 'linear'
        self.n_in = n_in
        self.n_out = n_out
        self.input = None
        self.batch_size = None
        self.W = np.random.randn(self.n_in, self.n_out) / np.sqrt(self.n_in)
        self.b = np.zeros(shape=[self.n_out])

    def forward(self, input):
        self.input = input
        self.batch_size = self.input.shape[0]
        self.input = self.input.reshape(self.batch_size, -1)

        return np.dot(self.input, self.W) + self.b

    def score(self, input):
        batch_size = input.shape[0]
        input = input.reshape(batch_size, -1)

        return np.dot(input, self.W) + self.b

    def backward(self, grad, learning_rate):
        W_grad = np.dot(self.input.T, grad)
        b_grad = np.sum(grad, axis=0)
        self.W -= learning_rate * W_grad
        self.b -= learning_rate * b_grad

        return np.dot(grad, self.W.T)


class Sigmoid(object):

    def __init__(self):
        self.layer = 'activation'
        self.activation = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def forward(self, input):
        self.activation = self.sigmoid(input)
        return self.activation

    def score(self, input):
        return self.sigmoid(input)

    def backward(self, grad):
        activation_derivative = self.activation * (1 - self.activation)
        return grad * activation_derivative


class Tanh(object):

    def __init__(self):
        self.layer = 'activation'
        self.activation = None

    def forward(self, input):
        self.activation = np.tanh(input)
        return self.activation

    def score(self, input):
        return np.tanh(input)

    def backward(self, grad):
        activation_derivative = 1 - np.square(self.activation)
        return grad * activation_derivative


class ReLU(object):

    def __init__(self):
        self.layer = 'activation'
        self.input = None

    def relu(self, z):
        return np.maximum(0, z)

    def forward(self, input):
        self.input = input
        return self.relu(self.input)

    def score(self, input):
        return self.relu(input)

    def backward(self, grad):
        tmp = np.where(self.input > 0, self.input, 0.0)
        activation_derivative = np.where(tmp <= 0, tmp, 1.0)
        return grad * activation_derivative


class Dropout(object):

    def __init__(self, keep_prob):
        self.layer = 'dropout'
        self.keep_prob = keep_prob
        self.keep = None

    def forward(self, input):
        self.keep = np.random.binomial(n=1, p=self.keep_prob, size=input.shape)

        return self.keep * input / self.keep_prob

    def score(self, input):
        return input

    def backward(self, grad):
        return self.keep * grad / self.keep_prob


class BatchNorm1d(object):

    def __init__(self, n_in):
        self.layer = 'batch_norm'
        self.input = None
        self.batch_size = None
        self.mu = None
        self.var = None
        self.normalized = None
        self.gamma = np.ones(shape=[1, n_in])
        self.beta = np.zeros(shape=[1, n_in])

    def forward(self, input):
        self.input = input
        self.batch_size = self.input.shape[0]
        self.mu = np.mean(self.input, axis=0)
        self.var = np.var(self.input, axis=0)
        self.normalized = (self.input - self.mu) / np.sqrt(self.var + 1e-8)

        return self.gamma * self.normalized + self.beta

    def score(self, input):
        mu = np.mean(input, axis=0)
        var = np.var(input, axis=0)
        normalized = (input - mu) / np.sqrt(var + 1e-8)

        return self.gamma * normalized + self.beta

    def backward(self, grad, learning_rate):
        beta_grad = np.sum(grad, axis=0)
        gamma_grad = np.sum(grad * self.normalized, axis=0)
        self.beta -= learning_rate * beta_grad
        self.gamma -= learning_rate * gamma_grad

        input_mu = self.input - self.mu
        std_inv = 1 / np.sqrt(self.var + 1e-8)
        dnormalized = grad * self.gamma
        dvar = np.sum(dnormalized * input_mu, axis=0) * (-0.5) * std_inv ** 3
        dmu = np.sum(dnormalized * -std_inv, axis=0) + dvar * np.mean(-2 * input_mu, axis=0)

        return dnormalized * std_inv + dvar * 2 * input_mu / self.batch_size + dmu / self.batch_size

