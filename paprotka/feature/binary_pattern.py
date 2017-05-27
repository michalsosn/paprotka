import math
import numpy as np


def interpolate_bilinear(image, y, x):
    height, width = image.shape

    def get_pixel(py, px):
        if py < 0 or py >= height:
            return 0
        if px < 0 or px >= width:
            return 0
        return image[py, px]

    floor_y = math.floor(y)
    floor_x = math.floor(x)
    ceil_y = math.ceil(y)
    ceil_x = math.ceil(x)

    frac_y = y - floor_y
    frac_x = x - floor_x

    top = (1 - frac_x) * get_pixel(floor_y, floor_x) + frac_x * get_pixel(floor_y, ceil_x)
    bot = (1 - frac_x) * get_pixel(ceil_y, floor_x) + frac_x * get_pixel(ceil_y, ceil_x)
    return (1 - frac_y) * top + frac_y * bot


def local_binary_pattern(image):
    HYPOT = 0.70711
    POSITIONS = [(-HYPOT, -HYPOT), (-1, 0), (-HYPOT, HYPOT), (0, 1),
                 (HYPOT, HYPOT), (1, 0), (HYPOT, -HYPOT), (0, -1)]
    FIRST_WEIGHT = 128
    patterns = np.zeros_like(image)

    height, width = image.shape
    for y in range(height):
        for x in range(width):
            pattern = 0
            current = image[y, x]
            for py, px in POSITIONS:
                if current <= interpolate_bilinear(image, y + py, x + px):
                    pattern = (pattern * 2) + 1
                else:
                    pattern *= 2

            min_pattern = pattern
            for i in range(7):
                first = pattern & 1
                pattern >>= 1
                if first:
                    pattern += FIRST_WEIGHT
                if pattern < min_pattern:
                    min_pattern = pattern
            patterns[y, x] = min_pattern

    return patterns
