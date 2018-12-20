class SGD(object):

    def __init__(self, learning_rate, lr_decay, params, grads):
        self.params = params
        self.grads = grads
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
