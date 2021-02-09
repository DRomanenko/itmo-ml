import numpy as np


# Kernels
def linear(x, y, _):
    return np.dot(x, y)


def polynomial(x, y, p):
    return (1 + np.dot(x, y)) ** p


def gaussian(x, y, beta):
    return np.exp((-beta) * (np.sqrt(np.sum((x - y) ** 2)) ** 2))


kernels = {
    "linear": linear,
    "polynomial": polynomial,
    "gaussian": gaussian
}
