import numpy as np


def moment(image, p, q):
    height, width = image.shape
    by_rows = np.arange(width) * np.ones((height, 1)) if p > 0 else 1
    if p > 1:
        by_rows **= p
    by_cols = np.ones(width) * np.arange(height).reshape(height, 1) if q > 0 else 1
    if q > 1:
        by_cols **= q
    return (by_rows * by_cols * image).sum()


def central_moment(image, p, q, center=None):
    """if center is None it is assumed that the image is centered around (h/2, w/2)"""
    height, width = image.shape
    if center is None:
        center = (height / 2, width / 2)
    by_rows = (np.arange(width) - center[1]) * np.ones((height, 1)) if p > 0 else 1
    if p > 1:
        by_rows **= p
    by_cols = np.ones(width) * (np.arange(height) - center[0]).reshape(height, 1) if q > 0 else 1
    if q > 1:
        by_cols **= q
    return (by_rows * by_cols * image).sum()


def scale_invariant(image, p, q, central00, center=None):
    return central_moment(image, p, q) / (central00 ** (1 + (p + q) / 2))


eta = scale_invariant


def first_hu_invariant(im):
    cm00 = central_moment(im, 0, 0)
    return eta(im, 2, 0, cm00) + eta(im, 0, 2, cm00)


def second_hu_invariant(im):
    cm00 = central_moment(im, 0, 0)
    return (eta(im, 2, 0, cm00) - eta(im, 0, 2, cm00)) ** 2 + 4 * eta(im, 1, 1, cm00) ** 2


def third_hu_invariant(im):
    cm00 = central_moment(im, 0, 0)
    eta30 = eta(im, 3, 0, cm00)
    eta12 = eta(im, 1, 2, cm00)
    eta21 = eta(im, 2, 1, cm00)
    eta03 = eta(im, 0, 3, cm00)
    return (eta30 - 3 * eta12) ** 2 + (3 * eta21 - eta03) ** 2


def fourth_hu_invariant(im):
    cm00 = central_moment(im, 0, 0)
    eta30 = eta(im, 3, 0, cm00)
    eta12 = eta(im, 1, 2, cm00)
    eta21 = eta(im, 2, 1, cm00)
    eta03 = eta(im, 0, 3, cm00)
    return (eta30 + eta12) ** 2 + (eta21 + eta03) ** 2


def fifth_hu_invariant(im):
    cm00 = central_moment(im, 0, 0)
    eta30 = eta(im, 3, 0, cm00)
    eta12 = eta(im, 1, 2, cm00)
    eta21 = eta(im, 2, 1, cm00)
    eta03 = eta(im, 0, 3, cm00)
    left = (eta30 - 3 * eta12) * (eta30 + eta12) * ((eta30 + eta12) ** 2 - 3 * (eta21 + eta03) ** 2)
    right = (3 * eta21 - eta03) * (eta21 + eta03) * (3 * (eta30 + eta12) ** 2 - (eta21 + eta03) ** 2)
    return left + right


def sixth_hu_invariant(im):
    cm00 = central_moment(im, 0, 0)
    eta20 = eta(im, 2, 0, cm00)
    eta11 = eta(im, 1, 1, cm00)
    eta02 = eta(im, 0, 2, cm00)
    eta30 = eta(im, 3, 0, cm00)
    eta21 = eta(im, 2, 1, cm00)
    eta12 = eta(im, 1, 2, cm00)
    eta03 = eta(im, 0, 3, cm00)
    left = (eta20 - eta02) * ((eta30 + eta12) ** 2 - (eta21 + eta03) ** 2)
    right = 4 * eta11 * (eta30 + eta12) * (eta21 + eta03)
    return left + right


def seventh_hu_invariant(im):
    cm00 = central_moment(im, 0, 0)
    eta30 = eta(im, 3, 0, cm00)
    eta12 = eta(im, 1, 2, cm00)
    eta21 = eta(im, 2, 1, cm00)
    eta03 = eta(im, 0, 3, cm00)
    left = (3 * eta21 - eta03) * (eta30 + eta12) * ((eta30 + eta12) ** 2 - 3 * (eta21 + eta03) ** 2)
    right = (eta30 - 3 * eta12) * (eta21 + eta03) * (3 * (eta30 + eta12) ** 2 - (eta21 + eta03) ** 2)
    return left - right


HU_INVARIANTS = [
    first_hu_invariant,
    second_hu_invariant,
    third_hu_invariant,
    fourth_hu_invariant,
    fifth_hu_invariant,
    sixth_hu_invariant,
    seventh_hu_invariant
]
