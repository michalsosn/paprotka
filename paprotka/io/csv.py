import numpy as np


def save_matrix(path, matrix, *args, **kwargs):
    np.savetxt(path, matrix, delimiter=',', *args, **kwargs)


def load_matrix(path, *args, **kwargs):
    return np.loadtxt(path, delimiter=',', ndmin=2, *args, **kwargs)
