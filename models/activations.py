import numpy as np


class Sigmoid(object):

    def __init__(self):
        self.layer = 'activation'
        self.num = None
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
        self.num = None
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

    def __init__(self, inplace=True):
        self.layer = 'activation'
        self.num = None
        self.input = None
        self.inplace = inplace

    def relu(self, z):
        if self.inplace:
            z[z < 0] = 0
            return z
        return np.maximum(0, z)

    def forward(self, input):
        self.input = input
        return self.relu(self.input)

    def score(self, input):
        return self.relu(input)

    def backward(self, grad):
        if self.inplace:
            self.input[self.input > 0] = 1
            return grad * self.input
        tmp = np.where(self.input > 0, self.input, 0.0)
        activation_derivative = np.where(tmp <= 0, tmp, 1.0)
        return grad * activation_derivative
