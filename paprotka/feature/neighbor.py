import collections as cl
import numpy as np
from paprotka.metric import METRICS, calculate_distances
from paprotka.struct.balltree import BallTree


class NearestCentroid:
    def __init__(self, metric='euclidean'):
        self.metric = METRICS[metric]
        self.patterns = None
        self.unique_labels = None

    def fit(self, features, labels):
        self.unique_labels = np.unique(labels)
        self.patterns = np.array([
            features[labels.ravel() == label].mean(axis=0)
            for label in self.unique_labels
        ])

    def predict(self, features):
        rows, _ = features.shape
        result = np.empty(rows, dtype=self.unique_labels.dtype)
        for i in range(rows):
            distances = calculate_distances(self.metric, self.patterns, features[i])
            label = self.unique_labels[np.argmin(distances)]
            result[i] = label
        return result


class KNeighborsClassifier:
    def __init__(self, n_neighbors=5, metric='euclidean'):
        self.n_neighbors = n_neighbors
        self.metric = METRICS[metric]
        self.tree = None
        self.labels_dtype = str

    def fit(self, features, labels):
        rows, dims = features.shape
        self.tree = BallTree(features, labels)
        self.labels_dtype = labels.dtype

    def predict(self, features):
        rows, dims = features.shape
        result = np.empty(rows, dtype=self.labels_dtype)
        for i in range(rows):
            closest_labels = self.tree.find_k_nearest(self.n_neighbors, features[i])
            closest_counter = cl.Counter(label[0] for label in closest_labels)
            most_common = closest_counter.most_common()
            max_count = most_common[0][1]
            result[i] = min(label for label, count in most_common if count == max_count)
        return result
