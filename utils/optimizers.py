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

    def load_momentum(self):
        for key in self.params:
            self.velocity[key] = 0

    def load(self, params, grads):
        self.params = params
        self.grads = grads
        self.load_momentum()
