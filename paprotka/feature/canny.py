import functools as ft
import numpy as np
import scipy.signal as sg


convolve2d_cut = ft.partial(sg.convolve2d, mode='same', boundary='symm')

SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
SOBEL_Y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])


def apply_sobel(image):
    edges_x = convolve2d_cut(image, SOBEL_X)
    edges_y = convolve2d_cut(image, SOBEL_Y)
    edge_gradient = np.hypot(edges_x, edges_y)
    edge_direction = np.arctan2(edges_x, edges_y)
    return edge_gradient, edge_direction


def round_angles(direction):
    mod_direction = direction % np.pi
    cut_direction = np.floor(mod_direction * (8 / np.pi))
    return ((cut_direction + 1) // 2) % 4


def suppress_non_maximum(gradient, direction, inplace=False):
    gradient = gradient if inplace else gradient.copy()

    ns = direction == 0
    ns_1 = np.zeros(gradient.shape, dtype=bool)
    ns_1[:-1, :] = gradient[:-1, :] < gradient[1:, :]
    ns_2 = np.zeros(gradient.shape, dtype=bool)
    ns_2[1:, :] = gradient[1:, :] < gradient[:-1, :]
    ns = np.logical_and(ns, np.logical_or(ns_1, ns_2))
    gradient[ns] = 0

    nw_se = direction == 1
    nw_se_1 = np.zeros(gradient.shape, dtype=bool)
    nw_se_1[:-1, :-1] = gradient[:-1, :-1] < gradient[1:, 1:]
    nw_se_2 = np.zeros(gradient.shape, dtype=bool)
    nw_se_2[1:, 1:] = gradient[1:, 1:] < gradient[:-1, :-1]
    nw_se = np.logical_and(nw_se, np.logical_or(nw_se_1, nw_se_2))
    gradient[nw_se] = 0

    ew = direction == 2
    ew_1 = np.zeros(gradient.shape, dtype=bool)
    ew_1[:, :-1] = gradient[:, :-1] < gradient[:, 1:]
    ew_2 = np.zeros(gradient.shape, dtype=bool)
    ew_2[:, 1:] = gradient[:, 1:] < gradient[:, :-1]
    ew = np.logical_and(ew, np.logical_or(ew_1, ew_2))
    gradient[ew] = 0

    ne_sw = direction == 3
    ne_sw_1 = np.zeros(gradient.shape, dtype=bool)
    ne_sw_1[1:, :-1] = gradient[1:, :-1] < gradient[:-1, 1:]
    ne_sw_2 = np.zeros(gradient.shape, dtype=bool)
    ne_sw_2[:-1, 1:] = gradient[:-1, 1:] < gradient[1:, :-1]
    ne_sw = np.logical_and(ne_sw, np.logical_or(ne_sw_1, ne_sw_2))
    gradient[ne_sw] = 0

    return gradient


def apply_threshold(image, threshold, inplace=False):
    greater = image >= threshold
    if inplace:
        image[:, :] = 0
        image[greater] = 1
    else:
        image = np.zeros(image.shape, dtype=np.bool)
        image[greater] = 1
    return image


def detect_borders(image, threshold=50):
    edge_gradient, edge_direction = apply_sobel(image)
    rounded_direction = round_angles(edge_direction)
    suppressed_gradient = suppress_non_maximum(edge_gradient, rounded_direction, True)
    return apply_threshold(suppressed_gradient, threshold)
