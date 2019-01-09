import numpy as np
try:
    from utils.im2col.im2col_cython import im2col_cython, col2im_cython
except ImportError:
    print('WARNING: Falied to enable Cython acceleration for Conv2d')
    print('Please run the following command from the ./utils/im2col/ directory and try again:')
    print('python setup.py build_ext --inplace')
    from utils.im2col.im2col import im2col, col2im


class Linear(object):

    def __init__(self, n_in, n_out):
        self.block = 'linear'
        self.num = None
        self.n_in = n_in
        self.n_out = n_out
        self.input = None
        self.batch_size = None
        self.W = [np.random.randn(self.n_in, self.n_out) * np.sqrt(2 / self.n_in)]
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
            self.W_grad[0] += np.dot(self.input.T, grad) + self.l2_reg * self.W[0]
        else:
            self.W_grad[0] += np.dot(self.input.T, grad)
        self.b_grad[0] += np.sum(grad, axis=0)

        return np.dot(grad, self.W[0].T)


class Conv2d(object):

    def __init__(self, in_channels, out_channels, kernel_size, padding, stride):
        self.block = 'conv2d'
        self.num = None
        self.in_C = in_channels
        self.out_C = out_channels
        self.in_H = None
        self.in_W = None
        self.batch_size = None
        self.input = None
        self.kernel_H, self.kernel_W = kernel_size
        self.stride = stride
        self.padding = padding
        self.out_H = None
        self.out_W = None
        self.W = [np.random.randn(self.out_C, self.in_C, self.kernel_H, self.kernel_W) *
                  np.sqrt(2 / (self.kernel_H * self.kernel_W * self.in_C))]
        self.b = [np.zeros(shape=[self.out_C, 1])]
        self.W_grad = [np.zeros(self.W[0].shape)]
        self.b_grad = [np.zeros(self.b[0].shape)]
        self.l2_reg = None
        self.cols = None

    def forward(self, input):
        self.input = input
        self.batch_size, _, self.in_H, self.in_W = self.input.shape
        assert (self.in_H + 2 * self.padding - self.kernel_H) % self.stride == 0
        assert (self.in_W + 2 * self.padding - self.kernel_W) % self.stride == 0

        self.out_H = (self.in_H + 2 * self.padding - self.kernel_H) // self.stride + 1
        self.out_W = (self.in_W + 2 * self.padding - self.kernel_W) // self.stride + 1

        try:
            self.cols = im2col_cython(self.input, self.kernel_H, self.kernel_W, self.padding, self.stride)
        except:
            self.cols = im2col(self.input, (self.kernel_H, self.kernel_W), self.padding, self.stride)

        output = np.dot(self.W[0].reshape(self.out_C, -1), self.cols) + self.b
        output = output.reshape(self.out_C, self.out_H, self.out_W, self.batch_size)
        output = np.transpose(output, (3, 0, 1, 2))
        return output

    def score(self, input):
        batch_size, _, in_H, in_W = input.shape
        assert (in_H + 2 * self.padding - self.kernel_H) % self.stride == 0
        assert (in_W + 2 * self.padding - self.kernel_W) % self.stride == 0

        out_H = (in_H + 2 * self.padding - self.kernel_H) // self.stride + 1
        out_W = (in_W + 2 * self.padding - self.kernel_W) // self.stride + 1

        try:
            cols = im2col_cython(input, self.kernel_H, self.kernel_W, self.padding, self.stride)
        except:
            cols = im2col(input, (self.kernel_H, self.kernel_W), self.padding, self.stride)

        output = np.dot(self.W[0].reshape(self.out_C, -1), cols) + self.b
        output = output.reshape(self.out_C, out_H, out_W, batch_size)
        output = np.transpose(output, (3, 0, 1, 2))
        return output

    def backward(self, grad):
        grad = grad.reshape(self.batch_size, self.out_C, self.out_H, self.out_W)
        self.b_grad[0] += np.sum(grad, axis=(0, 2, 3)).reshape(-1, 1)
        grad_reshaped = np.transpose(grad, (1, 2, 3, 0)).reshape(self.out_C, -1)
        if self.l2_reg:
            self.W_grad[0] += np.dot(grad_reshaped, self.cols.T).reshape(self.W[0].shape) + self.l2_reg * self.W[0]
        else:
            self.W_grad[0] += np.dot(grad_reshaped, self.cols.T).reshape(self.W[0].shape)
        dcols = np.dot(self.W[0].reshape(self.out_C, -1).T, grad_reshaped)

        try:
            output = col2im_cython(dcols, self.input.shape[0], self.input.shape[1], self.input.shape[2],
                                   self.input.shape[3], self.kernel_H, self.kernel_W, self.padding, self.stride)
        except:
            output = col2im(dcols, self.input.shape, (self.kernel_H, self.kernel_W), self.padding, self.stride)
        return output

class Dropout(object):

    def __init__(self, keep_prob):
        self.block = 'dropout'
        self.num = None
        self.keep_prob = keep_prob
        self.keep = None
        self.input_shape = None

    def forward(self, input):
        self.input_shape = input.shape
        self.keep = np.random.binomial(n=1, p=self.keep_prob, size=self.input_shape)
        return self.keep * input / self.keep_prob

    def score(self, input):
        return input

    def backward(self, grad):
        grad = grad.reshape(self.input_shape)
        return self.keep * grad / self.keep_prob


class BatchNorm1d(object):

    def __init__(self, n_in):
        self.block = 'batch_norm'
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
        self.beta_grad[0] += np.sum(grad, axis=0)
        self.gamma_grad[0] += np.sum(grad * self.normalized, axis=0)

        dnormalized = grad * self.gamma[0]
        dvar = np.sum(dnormalized * self.input_mu, axis=0) * (-0.5) * self.std_inv ** 3
        dmu = np.sum(dnormalized * (- self.std_inv), axis=0) + dvar * np.mean(-2 * self.input_mu, axis=0)

        return dnormalized * self.std_inv + dvar * 2 * self.input_mu / self.batch_size + dmu / self.batch_size


class BatchNorm2d(object):

    def __init__(self, in_channels):
        self.block = 'batch_norm'
        self.num = None
        self.input = None
        self.batch_size = None
        self.ori_shape = None
        self.mu = None
        self.var = None
        self.normalized = None
        self.input_mu = None
        self.std_inv = None
        self.gamma = [np.ones(shape=[1, in_channels])]
        self.beta = [np.zeros(shape=[1, in_channels])]
        self.gamma_grad = [np.zeros(self.gamma[0].shape)]
        self.beta_grad = [np.zeros(self.beta[0].shape)]

    def forward(self, input):
        N, C, H, W = input.shape
        self.ori_shape = N, C, H, W
        self.input = np.transpose(input, (0, 2, 3, 1)).reshape(-1, C)

        self.batch_size = self.input.shape[0]
        self.mu = np.mean(self.input, axis=0)
        self.var = np.var(self.input, axis=0)
        self.input_mu = self.input - self.mu
        self.std_inv = 1 / np.sqrt(self.var + 1e-8)
        self.normalized = self.input_mu * self.std_inv

        out = self.gamma[0] * self.normalized + self.beta[0]
        return np.transpose(out.reshape(N, H, W, C), (0, 3, 1, 2))

    def score(self, input):
        N, C, H, W = input.shape
        input = np.transpose(input, (0, 2, 3, 1)).reshape(-1, C)

        mu = np.mean(input, axis=0)
        var = np.var(input, axis=0)
        normalized = (input - mu) / np.sqrt(var + 1e-8)
        out = self.gamma[0] * normalized + self.beta[0]
        return np.transpose(out.reshape(N, H, W, C), (0, 3, 1, 2))

    def backward(self, grad):
        N, C, H, W = self.ori_shape
        grad = grad.reshape(self.ori_shape)
        grad = np.transpose(grad, (0, 2, 3 ,1)).reshape(-1, self.ori_shape[1])

        self.beta_grad[0] += np.sum(grad, axis=0)
        self.gamma_grad[0] += np.sum(grad * self.normalized, axis=0)

        dnormalized = grad * self.gamma[0]
        dvar = np.sum(dnormalized * self.input_mu, axis=0) * (-0.5) * self.std_inv ** 3
        dmu = np.sum(dnormalized * (- self.std_inv), axis=0) + dvar * np.mean(-2 * self.input_mu, axis=0)

        grad = dnormalized * self.std_inv + dvar * 2 * self.input_mu / self.batch_size + dmu / self.batch_size

        return np.transpose(grad.reshape(N, H, W, C), (0, 3, 1, 2))


class MaxPool2d(object):

    # Only support the case that kernel_H == kernel_W == stride for now
    def __init__(self, stride):
        self.block = 'maxpool2d'
        self.num = None
        self.stride = stride
        self.input_shape = None
        self.input_reshaped = None
        self.output = None

    def forward(self, input):
        self.input_shape = input.shape
        N, C, H, W = self.input_shape
        assert H % self.stride == 0
        assert W % self.stride == 0

        self.input_reshaped = input.reshape(N, C, H // self.stride, self.stride, W // self.stride, self.stride)
        self.output = self.input_reshaped.max(axis=3).max(axis=4)
        return self.output

    def score(self, input):
        N, C, H, W = input.shape
        assert H % self.stride == 0
        assert W % self.stride == 0

        input_reshaped = input.reshape(N, C, H // self.stride, self.stride, W // self.stride, self.stride)
        output = input_reshaped.max(axis=3).max(axis=4)
        return output

    def backward(self, grad):
        grad = grad.reshape(self.output.shape)

        out_grad_reshaped = np.zeros_like(self.input_reshaped)
        output_newaxis = self.output[:, :, :, np.newaxis, :, np.newaxis]
        mask = (self.input_reshaped == output_newaxis)
        grad_newaxis = grad[:, :, :, np.newaxis, :, np.newaxis]
        grad_broadcast, _ = np.broadcast_arrays(grad_newaxis, out_grad_reshaped)
        out_grad_reshaped[mask] = grad_broadcast[mask]
        out_grad_reshaped /= np.sum(mask, axis=(3, 5), keepdims=True)
        return out_grad_reshaped.reshape(self.input_shape)