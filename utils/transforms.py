import numpy as np
import matplotlib.pyplot as plt


class Transforms(object):

    def __init__(self, transform_list):
        self.transform_list = transform_list

    def transform(self, x_data):
        transformed = x_data
        for transform in self.transform_list:
            transformed = transform.transform(transformed)
        return transformed


class Pad(object):

    def __init__(self, padding=4, mode='reflect'):
        self.padding = padding
        self.mode = mode

    def transform(self, x_data):
        return np.pad(x_data, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), self.mode)


class Normalize(object):

    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std

    def transform(self, x_data):
        if self.mean is not None and self.std is not None:
            self.mean = np.array(self.mean).reshape(1, -1, 1, 1)
            self.std = np.array(self.std).reshape(1, -1, 1, 1)
            return (x_data - self.mean) / self.std
        else:
            mean = np.mean(x_data, axis=(0, 2, 3)).reshape(1, -1, 1, 1)
            std = np.std(x_data, axis=(0, 2, 3)).reshape(1, -1, 1, 1)
            return (x_data - mean) / std


class ToTensor(object):

    def __init__(self):
        pass

    def transform(self, x_data):
        return x_data / 255


class RandomHorizontalFlip(object):

    def __init__(self, flip_prob=0.5):
        self.flip_prob = flip_prob

    def transform(self, x_data):
        x_shape = x_data.shape
        for i in range(x_shape[0]):
            if np.random.rand() < self.flip_prob:
                x_data[i] = x_data[i, :, :, ::-1]
        return x_data


class RandomVerticaltalFlip(object):

    def __init__(self, flip_prob=0.5):
        self.flip_prob = flip_prob

    def transform(self, x_data):
        x_shape = x_data.shape
        for i in range(x_shape[0]):
            if np.random.rand() < self.flip_prob:
                x_data[i] = x_data[i, :, ::-1, :]
        return x_data


class RandomCrop(object):

    def __init__(self, crop_size):
        if type(crop_size) is int:
            crop_size = (crop_size, crop_size)
        self.crop_size = crop_size

    def transform(self, x_data):
        x_shape = x_data.shape
        h = x_shape[2]
        w = x_shape[3]
        x_new = np.zeros(shape=[x_shape[0], x_shape[1], self.crop_size[0], self.crop_size[1]])
        for i in range(x_shape[0]):
            bottom = np.random.randint(0, h - self.crop_size[0])
            left = np.random.randint(0, w - self.crop_size[1])
            top = bottom + self.crop_size[0]
            right = left + self.crop_size[1]
            x_new[i] = x_data[i, :, bottom: top, left: right]
        return x_new
