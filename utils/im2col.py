import numpy as np


def get_indices(input_shape, kernel_size, padding, stride):
    N, C, H, W = input_shape
    kernel_H, kernel_W = kernel_size
    assert (H + 2 * padding - kernel_H) % stride == 0
    assert (W + 2 * padding - kernel_W) % stride == 0
    out_H = (H + 2 * padding - kernel_H) // stride + 1
    out_W = (W + 2 * padding - kernel_W) // stride + 1

    i0 = np.tile(np.repeat(np.arange(kernel_H), kernel_W), C)
    i1 = stride * np.repeat(np.arange(out_H), out_W)
    j0 = np.tile(np.arange(kernel_W), kernel_H * C)
    j1 = stride * np.tile(np.arange(out_W), out_H)

    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)
    k = np.repeat(np.arange(C), kernel_H * kernel_W).reshape(-1, 1)
    return k, i, j


def im2col(input, kernel_size, padding, stride):
    ori_input = input
    if padding > 0:
        input = np.pad(ori_input, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode='constant')
    k, i, j = get_indices(ori_input.shape, kernel_size, padding, stride)

    cols = input[:, k, i, j]
    C = input.shape[1]
    cols = np.transpose(cols, (1, 2, 0)).reshape(kernel_size[0] * kernel_size[1] * C, -1)
    return cols


def col2im(cols, input_shape, kernel_size, padding, stride):
    N, C, H, W = input_shape
    H_padded, W_padded = H + 2 * padding, W + 2 * padding
    output = np.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)
    k, i, j = get_indices(input_shape, kernel_size, padding, stride)

    cols_reshaped = np.transpose(cols.reshape(kernel_size[0] * kernel_size[1] * C, -1, N), (2, 0, 1))
    np.add.at(output, (slice(None), k, i, j), cols_reshaped)
    if padding == 0:
        return output
    return output[:, :, padding: -padding, padding: -padding]
