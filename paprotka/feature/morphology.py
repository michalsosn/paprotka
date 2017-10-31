import heapq as hq
import math
import numpy as np
from paprotka.struct.union import Union
from .moment import moment


def erode_image(image, by=1):
    new_image = image
    for _ in range(by):
        image = new_image
        new_image = image.copy()
        new_image[1:, :] = np.minimum(new_image[1:, :],  image[:-1, :])
        new_image[:-1, :] = np.minimum(new_image[:-1, :],  image[1:, :])
        new_image[:, 1:] = np.minimum(new_image[:, 1:],  image[:, :-1])
        new_image[:, :-1] = np.minimum(new_image[:, :-1],  image[:, 1:])
    return new_image


def dilate_image(image, by=1):
    new_image = image
    for _ in range(by):
        image = new_image
        new_image = image.copy()
        new_image[1:, :] = np.maximum(new_image[1:, :],  image[:-1, :])
        new_image[:-1, :] = np.maximum(new_image[:-1, :],  image[1:, :])
        new_image[:, 1:] = np.maximum(new_image[:, 1:],  image[:, :-1])
        new_image[:, :-1] = np.maximum(new_image[:, :-1],  image[:, 1:])
    return new_image


def open_image(image, by=1):
    eroded_image = erode_image(image, by)
    return dilate_image(eroded_image, by)


def close_image(image, by=1):
    dilated_image = dilate_image(image, by)
    return erode_image(dilated_image, by)


def center_image(image):
    height, width = image.shape
    mean = image.sum()

    center_col = moment(image, 1, 0) / mean
    center_row = moment(image, 0, 1) / mean

    vertical_roll = int(round(height / 2 - center_row))
    horizontal_roll = int(round(width / 2 - center_col))

    vertically_centered = np.roll(image, vertical_roll, axis=0)
    return np.roll(vertically_centered, horizontal_roll, axis=1)


def watershed(gradient, markers):
    labels = np.zeros_like(markers)

    queue = [(gradient[row, col], 0, markers[row, col], row, col)
             for row, col in np.transpose(markers.nonzero())]
    hq.heapify(queue)

    todo = markers == 0
    time = 1

    while queue:
        _, _, label, row, col = hq.heappop(queue)
        labels[row, col] = label
        neighbors = np.transpose(
            todo[row - 1:row + 2, col - 1:col + 2].nonzero()
        ) + [max(0, row - 1), max(0, col - 1)]
        for nrow, ncol in neighbors:
            item = (gradient[nrow, ncol], time, label, nrow, ncol)
            time += 1
            hq.heappush(queue, item)
            todo[nrow, ncol] = False

    return labels


def distance_to_background(background):
    height, width = background.shape
    todo = background.copy()
    distances = np.zeros(background.shape)

    queue = [(0, row, col) for row, col in np.transpose((background == 0).nonzero())]
    hq.heapify(queue)

    while queue:
        distance, row, col = hq.heappop(queue)
        distance = min(distance, row + 1, col + 1, height - row, width - col)
        distances[row, col] = distance
        todo[row, col] = False
        neighbors = np.transpose(
            todo[row - 1:row + 2, col - 1:col + 2].nonzero()
        ) + [max(0, row - 1), max(0, col - 1)]
        for nrow, ncol in neighbors:
            difference = math.hypot(nrow - row, ncol - col)
            item = (distance + difference, nrow, ncol)
            hq.heappush(queue, item)
            if difference <= 1:
                todo[nrow, ncol] = False

    return distances


def label_unique(image, start=2):
    height, width = image.shape
    classes = np.zeros(image.shape, dtype=np.int32) - 1
    unions = []

    next_unique = 1
    for row in range(height):
        for col in range(width):
            if image[row, col]:
                neighbors = []
                if row > 0:
                    if col > 0 and classes[row - 1, col - 1] >= 0:
                        neighbors.append((row - 1, col - 1))
                    if classes[row - 1, col] >= 0:
                        neighbors.append((row - 1, col))
                    if col < width - 1 and classes[row - 1, col + 1] >= 0:
                        neighbors.append((row - 1, col + 1))
                if col > 0 and classes[row, col - 1] >= 0:
                    neighbors.append((row, col - 1))

                if len(neighbors) == 0:
                    classes[row, col] = len(unions)
                    unions.append(Union(next_unique))
                    next_unique += 1
                else:
                    frow, fcol = neighbors[0]
                    first_class = classes[frow, fcol]
                    classes[row, col] = first_class
                    union = unions[first_class]
                    for nrow, ncol in neighbors[1:]:
                        unions[classes[nrow, ncol]].merge(union, min)

    label_mapping = {}
    labels = np.zeros(image.shape, dtype=np.uint16)
    for row in range(height):
        for col in range(width):
            if image[row, col]:
                union = unions[classes[row, col]]
                unique = union.getv()
                if unique not in label_mapping:
                    label_mapping[unique] = start
                    start += 1
                labels[row, col] = label_mapping[unique]

    return labels
