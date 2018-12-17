import numpy as np


class Softmax(object):

    def __init__(self, name, n_in, n_out):
        self.name = name
        self.n_in = n_in
        self.n_out = n_out
        self.input = None
        self.batch_size = None
        self.y = None
        self.W = np.random.randn(self.n_in, self.n_out) / np.sqrt(self.n_in)
        self.b = np.zeros(shape=[self.n_out])
        self.p = None

    def forward(self, input, label):
        self.input = input
        self.y = label
        self.batch_size = self.input.shape[0]

        score = np.dot(self.input, self.W) + self.b
        exp_score = np.exp(score - np.max(score, axis=1).reshape(-1, 1))
        normalization = np.sum(exp_score, axis=1).reshape(-1, 1)
        self.p = exp_score / normalization
        loss = - 1 / self.batch_size * np.sum(np.log(self.p[range(self.batch_size), self.y] + 1e-8))
        return loss

    def score(self, input, y):
        score = np.dot(input, self.W) + self.b
        exp_score = np.exp(score - np.max(score, axis=1).reshape(-1, 1))
        normalization = np.sum(exp_score, axis=1).reshape(-1, 1)
        p = exp_score / normalization
        y_pred = np.argmax(p, axis=1)
        return np.mean(y == y_pred)

    def backward(self, learning_rate):
        p = self.p
        p[range(self.batch_size), self.y] -= 1
        delta = 1 / self.batch_size * (p + (1e-8))
        W_grad = np.dot(self.input.T, delta)
        b_grad = np.sum(delta, axis=0)
        self.W -= learning_rate * W_grad
        self.b -= learning_rate * b_grad
        return np.dot(delta, self.W.T)


class HiddenLayer(object):

    def __init__(self, name, n_in, n_out, activation, batch_norm=True):
        self.name = name
        self.n_in = n_in
        self.n_out = n_out
        self.activation = activation
        self.input = None
        self.batch_size = None
        self.W = np.random.randn(self.n_in, self.n_out) / np.sqrt(self.n_in)
        self.b = np.zeros(shape=[self.n_out])
        self.z = None
        self.batch_norm =batch_norm
        self.gamma = None
        self.beta = None
        self.mu = None
        self.var = None
        self.z_norm = None
        if self.batch_norm:
            self.gamma = np.ones(shape=[1, n_out])
            self.beta = np.zeros(shape=[1, n_out])

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def forward(self, input):
        self.input = input
        self.batch_size = self.input.shape[0]
        self.input = self.input.reshape(self.batch_size, -1)
        self.z = np.dot(self.input, self.W) + self.b
        temp = self.z
        if self.batch_norm:
            self.mu = np.mean(self.z, axis=0)
            self.var = np.var(self.z, axis=0)
            self.z_norm = (self.z - self.mu) / np.sqrt(self.var + 1e-8)
            temp = self.gamma * self.z_norm + self.beta

        if self.activation == 'relu':
            output = np.maximum(0, temp)
        elif self.activation == 'tanh':
            output = np.tanh(temp)
        elif self.activation == 'sigmoid':
            output = self.sigmoid(temp)
        elif self.activation == 'identity':
            output = temp
        else:
            raise Exception('The activation function of layer ' + self.name + ' is illegal.')

        return output

    def score(self, input):
        batch_size = input.shape[0]
        input = input.reshape(batch_size, -1)
        z = np.dot(input, self.W) + self.b
        temp = z
        if self.batch_norm:
            mu = np.mean(z, axis=0)
            var = np.var(z, axis=0)
            z_norm = (z - mu) / np.sqrt(var + 1e-8)
            temp = self.gamma * z_norm + self.beta

        if self.activation == 'relu':
            output = np.maximum(0, temp)
        elif self.activation == 'tanh':
            output = np.tanh(temp)
        elif self.activation == 'sigmoid':
            output = self.sigmoid(temp)
        elif self.activation == 'identity':
            output = temp
        else:
            raise Exception('The activation function of layer ' + self.name + ' is illegal.')

        return output

    def backward(self, forward_delta, learning_rate):
        temp  = self.z
        if self.batch_norm:
            temp = self.z_norm

        if self.activation == 'relu':
            tmp = np.where(temp > 0, temp, 0.0)
            activation_derivative = np.where(tmp <= 0, temp, 1.0)
        elif self.activation == 'tanh':
            activation_derivative = 1 - np.square(np.tanh(temp))
        elif self.activation == 'sigmoid':
            activation_derivative = self.sigmoid(temp) * (1 - self.sigmoid(temp))
        elif self.activation == 'identity':
            activation_derivative = temp
        else:
            raise Exception('The activation function of layer ' + self.name + ' is illegal.')

        if self.batch_norm:
            dout = forward_delta * activation_derivative
            beta_grad = np.sum(dout, axis=0)
            gamma_grad = np.sum(dout * self.z_norm, axis=0)

            z_mu = self.z - self.mu
            std_inv = 1 / np.sqrt(self.var + 1e-8)
            dz_norm = dout * self.gamma
            dvar = np.sum(dz_norm * z_mu, axis=0) * (-0.5) * std_inv**3
            dmu = np.sum(dz_norm * -std_inv, axis=0) + dvar * np.mean(-2 * z_mu, axis=0)
            delta = dz_norm * std_inv + dvar * 2 * z_mu / self.batch_size + dmu / self.batch_size

            self.beta -= learning_rate * beta_grad
            self.gamma -= learning_rate * gamma_grad


        else:
            # A[i-1] == input
            # Z[i] = dot(A[i-1], W[i]) + b[i]
            # A[i] = activation(Z_norm[i])
            # Z[i+1] = dot(A[i], W[i+1]) + b[i+1]

            # activation_derivative = actiavtion'(Z[i])
            # dl/dA[i] = dl/dZ[i+1] * dZ[i+1]/dA[i] = dot(delta[i+1], W[i+1].T) == forward_delta
            # dl/dZ[i] = dl/dA[i] * dA[i]/dZ[i] = forward_delta * activation_derivative == delta
            # dl/dW[i] = dl/dZ[i] * dZ[i]/dW[i] = dot(input.T, dZ[i]) == W_grad
            # db[i] = dl/dZ[i] * dZ[i]/db[i] = sum(dZ[i], axis=0) == b_grad

            delta = forward_delta * activation_derivative

        W_grad = np.dot(self.input.T, delta)
        b_grad = np.sum(delta, axis=0)
        self.W -= learning_rate * W_grad
        self.b -= learning_rate * b_grad

        return np.dot(delta, self.W.T)


class Dropout(object):

    def __init__(self, name, keep_prob):
        self.name = name
        self.keep_prob = keep_prob
        self.keep = None

    def forward(self, input):
        self.keep = np.random.binomial(n=1, p=self.keep_prob, size=input.shape)
        return self.keep * input / self.keep_prob

    def score(self, input):
        return input

    def backward(self, forward_delta, learning_rate):
        return self.keep * forward_delta / self.keep_prob

