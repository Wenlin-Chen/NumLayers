import numpy as np


class Sigmoid(object):

    def __init__(self):
        self.block = 'activation'
        self.num = None
        self.activation = None
        self.input_shape = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def forward(self, input):
        self.input_shape = input.shape
        self.activation = self.sigmoid(input)
        return self.activation

    def score(self, input):
        return self.sigmoid(input)

    def backward(self, grad):
        grad = grad.reshape(self.input_shape)
        activation_derivative = self.activation * (1 - self.activation)
        return grad * activation_derivative


class Tanh(object):

    def __init__(self):
        self.block = 'activation'
        self.num = None
        self.activation = None
        self.input_shape = None

    def forward(self, input):
        self.input_shape = input.shape
        self.activation = np.tanh(input)
        return self.activation

    def score(self, input):
        return np.tanh(input)

    def backward(self, grad):
        grad = grad.reshape(self.input_shape)
        activation_derivative = 1 - np.square(self.activation)
        return grad * activation_derivative


class ReLU(object):

    def __init__(self, inplace=True):
        self.block = 'activation'
        self.num = None
        self.input = None
        self.input_shape = None
        self.inplace = inplace

    def relu(self, z):
        if self.inplace:
            z[z < 0] = 0
            return z
        return np.maximum(0, z)

    def forward(self, input):
        self.input = input
        self.input_shape = self.input.shape
        return self.relu(self.input)

    def score(self, input):
        return self.relu(input)

    def backward(self, grad):
        grad = grad.reshape(self.input_shape)
        if self.inplace:
            self.input[self.input > 0] = 1
            return grad * self.input
        tmp = np.where(self.input > 0, self.input, 0.0)
        activation_derivative = np.where(self.input <= 0, tmp, 1.0)
        return grad * activation_derivative


class LeakeyReLU(object):

    def __init__(self, negative_slope=0.01):
        self.block = 'activation'
        self.num = None
        self.negative_slope = negative_slope
        self.input = None
        self.input_shape = None

    def leaky_relu(self, z):
        return np.maximum(self.negative_slope * z, z)

    def forward(self, input):
        self.input = input
        self.input_shape = self.input.shape
        return self.leaky_relu(self.input)

    def score(self, input):
        return self.leaky_relu(input)

    def backward(self, grad):
        grad = grad.reshape(self.input_shape)
        tmp = np.where(self.input <= 0, self.input, 1.0)
        activation_derivative = np.where(self.input > 0, tmp, self.negative_slope)
        return grad * activation_derivative
