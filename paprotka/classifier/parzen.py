import numpy as np


class Parzen:
    def __init__(self):
        self.features = None
        self.labels = None

    def normalize(self, features):
        extended_features = np.c_[features, np.zeros(features.shape[0])]
        norms = np.linalg.norm(extended_features, axis=1)
        return (extended_features.T / norms).T

    def fit(self, features, labels):
        self.features = self.normalize(features)  # N x d
        self.labels = labels # N
        self.unique_labels = np.unique(labels) # k
        self.variance = 2 * labels.size

    def predict(self, features):
        points, dims = features.shape
        class_num = self.unique_labels.size

        features = self.normalize(features)

        sums = self.features @ features.T # N x d @ d x M = N x M
        outputs = np.exp((sums - 1) / self.variance) # N x M
        all_results = np.zeros((points, class_num), dtype=np.float) # M x k

        for i, label in enumerate(self.unique_labels):
            label_results = outputs.T @ (self.labels == label) # M x N @ N = M
            all_results[:, i] = label_results

        decisions = np.argmax(all_results, axis=1) # M

        return self.unique_labels[decisions]
