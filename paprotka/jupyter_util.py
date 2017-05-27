import matplotlib.pyplot as plt


def display_image(image, root=plt, title=None, cmap=plt.cm.gray, interpolation='nearest', show=True, *args, **kwargs):
    root.imshow(image, cmap=cmap, interpolation=interpolation, *args, **kwargs)
    if title is not None:
        plt.title(title)
    root.axis('off')
    if show:
        root.show()
