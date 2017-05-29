import numpy as np
from hmmlearn import hmm


def states_probability(a, states, initial=1.0):
    return initial * np.product(a[states[:-1], states[1:]])


def forward_algorithm(a, b, observations, initial_probs=None):
    state_n, _ = a.shape

    current_probs = initial_probs
    if initial_probs is None:
        current_probs = np.full(state_n, 1.0 / state_n, dtype=np.float64)

    for observation in observations:
        observation_probs = b[:, observation]
        current_probs = (a @ current_probs) * observation_probs
    return current_probs.sum()


class HMMGMMClassifier:
    def __init__(self):
        self.models = None
        self.unique_labels = None

    def fit(self, features, labels, *args, **kwargs):
        self.unique_labels = np.unique(labels)
        self.models = []

        for unique_label in self.unique_labels:
            relevant_ixs = (labels == unique_label).nonzero()[0]
            relevant_data = []

            for relevant_ix in relevant_ixs:
                row = features[relevant_ix]
                relevant_data.append(row)

            lengths = np.array([len(data) for data in relevant_data])
            concat = np.vstack(relevant_data)

            model = hmm.GMMHMM(*args, **kwargs)
            model.fit(concat, lengths)
            self.models.append(model)

    def predict(self, features):
        results = np.zeros_like(features, dtype=self.unique_labels.dtype)

        for i, sequence in enumerate(features):
            max_probability = None
            max_label = None
            for j, model in enumerate(self.models):
                probability = model.score(sequence)
                if max_probability is None or probability > max_probability:
                    max_probability = probability
                    max_label = self.unique_labels[j]
            results[i] = max_label

        return results
