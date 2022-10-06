import numpy as np

class standard:
    def __init__(self, data):

        self.mean = data.mean()
        self.std = data.std()
        print("std:", self.std, "mean:", self.mean)

    def transform(self, data):

        return (data - self.mean) / self.std

    def inverse_transform(self, data):

        return (data * self.std) + self.mean

    def rmse_transform(self, data):
        return data * self.std

    def mse_transform(self, data):
        return data * self.std * self.std


class minmax:
    def __init__(self, data):
        self.min = data.min()
        self.max = data.max()
        print("min:", self.min, "max:", self.max)

    def trasform(self, data):
        data = 1. * (data - self.min) / (self.max - self.min)
        return data * 2. - 1.

    def inverse_transform(self, data):
        data = (data + 1.) / 2.
        return 1. * data * (self.max - self.min)

    def rmse_transform(self, data):
        return data * (self.max - self.min)

    def mse_transform(self, data):
        return data * (self.max - self.min)*(self.max - self.min)


class node_standard:
    def __init__(self, data):
        print("node_standard...........")
        self.node = data.shape[-1]
        x = data.reshape((-1, self.node))
        self.mean = np.mean(x, axis=0)
        self.std = np.std(x, axis=0)
    def transform(self, data):
        shape = data.shape
        x = data.reshape((-1, self.node))
        x = (x-self.mean) / self.std
        return x.reshape(shape)

    def inverse_transform(self, data):
        shape = data.shape
        x = data.reshape((-1, self.node))
        x = x * self.std + self.mean
        return x.reshape(shape)

class node_minmax:
    def __init__(self, data):
        print("node_minmax...........")
        self.node = data.shape[-1]
        x = data.reshape((-1, self.node))
        self.max = np.max(x, axis=0)
        self.min = np.min(x, axis=0)

    def transform(self, data):
        shape = data.shape
        x = data.reshape((-1, self.node))
        x = (x-self.min) / (self.max - self.min)
        return x.reshape(shape)

    def inverse_transform(self, data):
        shape = data.shape
        x = data.reshape((-1, self.node))
        x = x * (self.max - self.min) + self.min
        return x.reshape(shape)


class node_standard_nyc:
    def __init__(self, data):
        print("node_standard nyc...........")
        self.node = data.shape[-2]
        x = data.swapaxes(-1,-2).reshape((-1, self.node))
        self.mean = np.mean(x, axis=0)
        self.std = np.std(x, axis=0)
    def transform(self, data):
        x = data.swapaxes(-1,-2)
        shape = x.shape
        x = x.reshape((-1,self.node))
        x = (x-self.mean) / self.std
        return x.reshape(shape).swapaxes(-1,-2)
    def inverse_transform(self, data):
        x = data.swapaxes(-1, -2)
        shape = x.shape
        x = x.reshape((-1, self.node))
        x = x * self.std + self.mean
        return x.reshape(shape).swapaxes(-1,-2)

class nonormal:
    def __init__(self, data=None):
        print("No normal................")

    def transform(self, data):
        return data

    def inverse_transform(self, data):
        return data

    def rmse_transform(self, data):
        return data

    def mse_transform(self, data):
        return data


