import math
import fastdtw
import numpy as np
from paprotka.metric import METRICS


def calculate_dtw(metric, pattern, sequence):
    pattern_size = len(pattern)
    sequence_size = len(sequence)

    if sequence_size < pattern_size:
        sequence_size, pattern_size = pattern_size, sequence_size
        sequence, pattern = pattern, sequence

    prev_row = np.full(pattern_size + 1, math.inf, dtype=np.float64)
    prev_row[0] = 0
    current_row = np.zeros(pattern_size + 1, dtype=np.float64)
    current_row[0] = math.inf

    for i, sequence_window in enumerate(sequence):
        for j, pattern_window in enumerate(pattern):
            distance = metric(pattern_window, sequence_window)
            prev_distance = min(prev_row[j], prev_row[j + 1], current_row[j])
            current_row[j + 1] = prev_distance + distance
        prev_row[:] = current_row
        current_row[0] = math.inf

    return prev_row[-1]


class DynamicTimeWarpingClassifier:
    def __init__(self, metric='euclidean'):
        self.metric = METRICS[metric]
        self.patterns = None
        self.labels = None

    def fit(self, features, labels):
        self.patterns = features
        self.labels = labels

    def predict(self, features):
        sequence_num = len(features)

        results = np.zeros(sequence_num, dtype=self.labels.dtype)
        for i, sequence in enumerate(features):
            min_distance = None
            min_label = None
            for j, pattern in enumerate(self.patterns):
                distance, _ = fastdtw.fastdtw(pattern, sequence, dist=self.metric)
                if min_distance is None or distance < min_distance:
                    min_distance = distance
                    min_label = self.labels[j]
            results[i] = min_label

        return results
