import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from sklearn import metrics


def wrap_display(display):
    def wrapped(data, title=None, root=None, show=True, *args, **kwargs):
        if root is None:
            fig, root = plt.subplots()
        if title is not None:
            plt.title(title)
        display(root, data, *args, **kwargs)
        if show:
            plt.show()
    return wrapped


@wrap_display
def display_image(root, image, cmap=plt.cm.gray, interpolation='nearest', *args, **kwargs):
    root.imshow(image, cmap=cmap, interpolation=interpolation, *args, **kwargs)
    root.axis('off')


@wrap_display
def display_sound(root, sound, xlabel='Time [sec]', ylabel='Amplitude', *args, **kwargs):
    domain = np.arange(0, sound.data.size / sound.rate, 1.0 / sound.rate)
    root.plot(domain, sound.data, *args, **kwargs)
    root.set_xlabel(xlabel)
    root.set_ylabel(ylabel)


@wrap_display
def display_spectrogram(root, sound, result_scaling=np.log1p, view_range=None,
                        xlabel='Time [sec]', ylabel='Frequency [Hz]', *args, **kwargs):
    frequencies, times, spectrogram = signal.spectrogram(sound.data, sound.rate, *args, **kwargs)
    if view_range is not None:
        freq_start = np.searchsorted(frequencies, view_range[0], 'left')
        freq_end = np.searchsorted(frequencies, view_range[1], 'right')
        frequencies = frequencies[freq_start:freq_end]
        spectrogram = spectrogram[freq_start:freq_end, :]
    root.pcolormesh(times, frequencies, result_scaling(spectrogram))
    root.set_xlabel(xlabel)
    root.set_ylabel(ylabel)


@wrap_display
def display_covariance(root, data, column_names=None, cmap=plt.cm.winter, *args, **kwargs):
    cov_data = np.corrcoef(data)
    root.matshow(cov_data, cmap=cmap, *args, **kwargs)
    if column_names is not None:
        root.set_xticks(range(len(column_names)))
        root.set_xticklabels(column_names, rotation='vertical')
        root.set_yticks(range(len(column_names)))
        root.set_yticklabels(column_names)


@wrap_display
def display_confusion(root, t_labels, t_predictions, xlabel='Predicted label', ylabel='True label', *args, **kwargs):
    unique_labels = np.unique(t_labels)
    matrix = metrics.confusion_matrix(t_labels, t_predictions, unique_labels)
    plt.imshow(matrix, interpolation='nearest', *args, **kwargs)
    plt.colorbar()
    plt.tight_layout()
    root.set_ylabel(ylabel)
    root.set_xlabel(xlabel)
    root.set_yticks(range(len(unique_labels)))
    root.set_yticklabels(unique_labels)
    root.set_xticks(range(len(unique_labels)))
    root.set_xticklabels(unique_labels, rotation='vertical')


def row_map(func, matrix, cols=(), dtype=np.float64):
    result = np.zeros((matrix.shape[0],) + cols, dtype=dtype)
    for i, row in enumerate(matrix):
        result[i] = func(row)
    return result
