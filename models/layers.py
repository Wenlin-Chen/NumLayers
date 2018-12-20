import numpy as np


class Linear(object):

    def __init__(self, n_in, n_out):
        self.layer = 'linear'
        self.num = None
        self.n_in = n_in
        self.n_out = n_out
        self.input = None
        self.batch_size = None
        self.W = [np.random.randn(self.n_in, self.n_out) / np.sqrt(self.n_in)]
        self.b = [np.zeros(shape=[self.n_out])]
        self.W_grad = [np.zeros(self.W[0].shape)]
        self.b_grad = [np.zeros(self.b[0].shape)]
        self.l2_reg = None

    def forward(self, input):
        self.input = input
        self.batch_size = self.input.shape[0]
        self.input = self.input.reshape(self.batch_size, -1)

        return np.dot(self.input, self.W[0]) + self.b[0]

    def score(self, input):
        batch_size = input.shape[0]
        input = input.reshape(batch_size, -1)

        return np.dot(input, self.W[0]) + self.b[0]

    def backward(self, grad):
        if self.l2_reg:
            self.W_grad[0] = np.dot(self.input.T, grad) + self.l2_reg * self.W[0]
        else:
            self.W_grad[0] = np.dot(self.input.T, grad)
        self.b_grad[0] = np.sum(grad, axis=0)

        return np.dot(grad, self.W[0].T)


class Dropout(object):

    def __init__(self, keep_prob):
        self.layer = 'dropout'
        self.num = None
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
        self.num = None
        self.input = None
        self.batch_size = None
        self.mu = None
        self.var = None
        self.normalized = None
        self.input_mu = None
        self.std_inv = None
        self.gamma = [np.ones(shape=[1, n_in])]
        self.beta = [np.zeros(shape=[1, n_in])]
        self.gamma_grad = [np.zeros(self.gamma[0].shape)]
        self.beta_grad = [np.zeros(self.beta[0].shape)]

    def forward(self, input):
        self.input = input
        self.batch_size = self.input.shape[0]
        self.mu = np.mean(self.input, axis=0)
        self.var = np.var(self.input, axis=0)
        self.input_mu = self.input - self.mu
        self.std_inv = 1 / np.sqrt(self.var + 1e-8)
        self.normalized = self.input_mu * self.std_inv

        return self.gamma[0] * self.normalized + self.beta[0]

    def score(self, input):
        mu = np.mean(input, axis=0)
        var = np.var(input, axis=0)
        normalized = (input - mu) / np.sqrt(var + 1e-8)

        return self.gamma[0] * normalized + self.beta[0]

    def backward(self, grad):
        self.beta_grad[0] = np.sum(grad, axis=0)
        self.gamma_grad[0] = np.sum(grad * self.normalized, axis=0)

        dnormalized = grad * self.gamma[0]
        dvar = np.sum(dnormalized * self.input_mu, axis=0) * (-0.5) * self.std_inv ** 3
        dmu = np.sum(dnormalized * (- self.std_inv), axis=0) + dvar * np.mean(-2 * self.input_mu, axis=0)

        return dnormalized * self.std_inv + dvar * 2 * self.input_mu / self.batch_size + dmu / self.batch_size

