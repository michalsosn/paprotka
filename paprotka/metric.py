import numpy as np


METRICS = {
    'manhattan': lambda x, y: np.sum(np.abs(x - y)),
    'euclidean': lambda x, y: np.sum((x - y) ** 2),
    'minkowski3': lambda x, y: np.sum((x - y) ** 3),
    'chebyshev': lambda x, y: np.max(np.abs(x - y))
}


def calculate_distances(metric, matrix, point):
    row_num, _ = matrix.shape
    distances = np.zeros(row_num)
    for i in range(row_num):
        distances[i] = metric(matrix[i], point)
    return distances
