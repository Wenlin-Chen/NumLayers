import numpy as np


class Linear(object):

    def __init__(self, n_in, n_out):
        self.layer = 'linear'
        self.n_in = n_in
        self.n_out = n_out
        self.input = None
        self.batch_size = None
        self.W = np.random.randn(self.n_in, self.n_out) / np.sqrt(self.n_in)
        self.b = np.zeros(shape=[self.n_out])
        self.l2_reg = None

    def forward(self, input, Ws):
        self.input = input
        self.batch_size = self.input.shape[0]
        self.input = self.input.reshape(self.batch_size, -1)
        if self.l2_reg:
            Ws.append(self.W)

        return np.dot(self.input, self.W) + self.b, Ws

    def score(self, input):
        batch_size = input.shape[0]
        input = input.reshape(batch_size, -1)

        return np.dot(input, self.W) + self.b

    def backward(self, grad, learning_rate):
        W_grad = np.dot(self.input.T, grad)
        b_grad = np.sum(grad, axis=0)
        if self.l2_reg:
            self.W -= learning_rate * (W_grad + self.l2_reg * self.W)
        else:
            self.W -= learning_rate * W_grad
        self.b -= learning_rate * b_grad

        return np.dot(grad, self.W.T)


class Dropout(object):

    def __init__(self, keep_prob):
        self.layer = 'dropout'
        self.keep_prob = keep_prob
        self.keep = None

    def forward(self, input, Ws):
        self.keep = np.random.binomial(n=1, p=self.keep_prob, size=input.shape)

        return self.keep * input / self.keep_prob, Ws

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

    def forward(self, input, Ws):
        self.input = input
        self.batch_size = self.input.shape[0]
        self.mu = np.mean(self.input, axis=0)
        self.var = np.var(self.input, axis=0)
        self.normalized = (self.input - self.mu) / np.sqrt(self.var + 1e-8)

        return self.gamma * self.normalized + self.beta, Ws

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

