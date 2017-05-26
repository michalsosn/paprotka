from scipy import ndimage


def load_color(path, *args, **kwargs):
    return ndimage.imread(path, mode='RGB', *args, **kwargs)


def load_gray(path, *args, **kwargs):
    return ndimage.imread(path, flatten=True, *args, **kwargs)