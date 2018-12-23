import numpy as np


class SGD(object):

    def __init__(self, learning_rate, lr_decay=None):
        self.params = None
        self.grads = None
        self.lr = learning_rate
        self.lr_decay_rate = None
        self.lr_decay_iter = None
        if lr_decay:
            self.lr_decay_rate = lr_decay[0]
            self.lr_decay_iter = lr_decay[1:]

    def step(self, iter):
        if self.lr_decay_rate and iter in self.lr_decay_iter:
            self.lr *= self.lr_decay_rate
        for key, param in self.params.items():
            param[0] -= self.lr * self.grads[key][0]

    def load(self, params, grads):
        self.params = params
        self.grads = grads


class Momentum(object):

    def __init__(self, learning_rate, momentum=0.9, lr_decay=None):
        self.params = None
        self.grads = None
        self.lr = learning_rate
        self.momentum = momentum
        self.velocity = {}
        self.lr_decay_rate = None
        self.lr_decay_iter = None
        if lr_decay:
            self.lr_decay_rate = lr_decay[0]
            self.lr_decay_iter = lr_decay[1:]

    def step(self, iter):
        if self.lr_decay_rate and iter in self.lr_decay_iter:
            self.lr *= self.lr_decay_rate
        for key, param in self.params.items():
            self.velocity[key] = self.velocity[key] * self.momentum + self.lr * self.grads[key][0]
            param[0] -= self.velocity[key]


    def load(self, params, grads):
        self.params = params
        self.grads = grads
        for key in self.params:
            self.velocity[key] = 0


class Adam(object):

    def __init__(self, learning_rate, betas=(0.9, 0.999), lr_decay=None):
        self.params = None
        self.grads = None
        self.m = {}
        self.v = {}
        self.lr = learning_rate
        self.beta1, self.beta2 = betas
        self.lr_decay_rate = None
        self.lr_decay_iter = None
        if lr_decay:
            self.lr_decay_rate = lr_decay[0]
            self.lr_decay_iter = lr_decay[1:]

    def step(self, iter):
        if self.lr_decay_rate and iter in self.lr_decay_iter:
            self.lr *= self.lr_decay_rate
        t = iter + 1
        for key, param in self.params.items():
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * self.grads[key][0]
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * np.square(self.grads[key][0])
            lr = self.lr * np.sqrt(1 - self.beta2**t) / (1 - self.beta1**t)
            param[0] -= lr * (self.m[key] / (np.sqrt(self.v[key] + 1e-8)))

    def load(self, params, grads):
        self.params = params
        self.grads = grads
        for key in self.params:
            self.v[key] = 0
            self.m[key] = 0
