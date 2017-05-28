import numpy as np


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
