import numpy as np


class MyGaussianNB:
    classes = None
    priors = None
    means = None
    variances = None

    def __init__(self):
        pass

    def fit(self, features, labels, priors_equal=True):
        self.classes = np.unique(labels)

        self.priors = np.ones(self.classes.shape) / self.classes.size

        self.means = {}
        self.variances = {}
        for i, clazz in enumerate(self.classes):
            positions = (labels == clazz).ravel()
            if not priors_equal:
                self.priors[i] = positions.sum() / labels.size
            self.means[clazz] = features[positions, :].mean(axis=0)
            self.variances[clazz] = features[positions, :].var(axis=0)

    def predict(self, features):
        likelihoods = np.zeros((features.shape[0], self.classes.size))
        for i, clazz in enumerate(self.classes):
            gaussians = np.exp(-np.square(features - self.means[clazz]) / (2 * self.variances[clazz])) \
                        / np.sqrt(2 * np.pi * self.variances[clazz])
            likelihoods[:, i] = np.product(gaussians, axis=1)

        posteriors = likelihoods * self.priors
        decisions = np.argmax(posteriors, axis=1)
        return self.classes[decisions]