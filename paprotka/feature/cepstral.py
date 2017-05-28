import math
import numpy as np
from scipy import signal, fftpack


def pre_emphasize(data, pre_emphasis=0.97):
    return np.append(data[0], data[1:] - pre_emphasis * data[:-1])


def hz_to_mel(hz):
    return 2595 * math.log10(1 + hz / 700)


def mel_to_hz(mel):
    return 700 * (10 ** (mel / 2595) - 1)


def make_mel_filters(half, rate, filter_num):
    min_mel = 0
    max_mel = hz_to_mel(rate / 2)
    mel_points = np.linspace(min_mel, max_mel, filter_num + 2)
    hz_points = mel_to_hz(mel_points)
    bin_points = np.floor((2 * half + 1) * hz_points / rate).astype(np.int32)

    filters = np.zeros((filter_num, half))
    for i in range(filter_num):
        start, mid, end = bin_points[i], bin_points[i + 1], bin_points[i + 2]
        filters[i, start:mid] = np.linspace(0, 1, mid - start, endpoint=False)
        filters[i, mid:end] = np.linspace(1, 0, end - mid, endpoint=True)
    return filters


def calculate_filter_bank(sound, filter_num=30, result_scaling=np.log1p, *args, **kwargs):
    frequencies, times, transform = signal.stft(sound.data, sound.rate, *args, **kwargs)
    power_spectrum = np.abs(transform) ** 2
    filters = make_mel_filters(frequencies.size, sound.rate, filter_num)
    coefficients = (filters @ power_spectrum).T
    return result_scaling(coefficients)


def calculate_mfcc(sound, num_ceps=12, *args, **kwargs):
    filter_banks = calculate_filter_bank(sound, *args, **kwargs)
    mfcc = fftpack.dct(filter_banks, norm='ortho')
    if num_ceps is None:
        return mfcc
    return mfcc[:, 1:(num_ceps + 1)]
