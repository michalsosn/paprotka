import math
import numpy as np


def make_rotation_matrix(angle):
    return np.array([[math.cos(angle), -math.sin(angle)],
                     [math.sin(angle), math.cos(angle)]])


def interpolate_bilinear(image, point):
    height, width = image.shape
    y, x = point
    y1, x1 = int(y), int(x)
    ys = np.array([y1, y1, y1 + 1, y1 + 1]).clip(0, height - 1)
    xs = np.array([x1, x1 + 1, x1, x1 + 1]).clip(0, width - 1)
    weights_before = np.array([y1 + 1 - y, y - y1])
    weights_after = np.array([[x1 + 1 - x], [x - x1]])
    values = image[ys, xs].reshape(2, 2)
    return weights_before.dot(values).dot(weights_after)


def rotate_image(image, angle):
    rev_matrix = make_rotation_matrix(-angle)
    height, width = image.shape
    center_translation = np.array([height / 2, width / 2])
    result = np.zeros(image.shape)
    height, width = image.shape
    for row in range(height):
        for col in range(width):
            source = np.dot(rev_matrix, [row, col] - center_translation) + center_translation
            result[row, col] = interpolate_bilinear(image, source)
    return result
