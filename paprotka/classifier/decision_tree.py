import collections as cl
import numpy as np


class DecisionTreeNode:
    def __init__(self, column, division, lesser, greater, score):
        self.column = column
        self.division = division
        self.lesser = lesser
        self.greater = greater
        self.score = score

    def classify(self, point):
        if point[self.column] < self.division:
            return self.lesser.classify(point)
        else:
            return self.greater.classify(point)

    def print_tree(self, indent=0, indent_change=4):
        print(' ' * indent + 'if {} < {}:'.format(self.column, self.division))
        self.lesser.print_tree(indent + indent_change, indent_change)
        print(' ' * indent + 'else:')
        self.greater.print_tree(indent + indent_change, indent_change)

    def count_leaves(self):
        return self.lesser.count_leaves() + self.greater.count_leaves()

    def count_depth(self):
        return 1 + max(self.lesser.count_depth(), self.greater.count_depth())

    def _get_description(self):
        return '{} < {}\nscore={:4f}'.format(self.column, self.division, self.score)


class DecisionTreeLeaf:
    def __init__(self, labels, score):
        self.counts = cl.Counter(labels)
        self.label = self.counts.most_common(1)[0][0]
        self.score = score

    def classify(self, point):
        return self.label

    def print_tree(self, indent=0, indent_change=4):
        print(' ' * indent + str(self.label))

    def count_leaves(self):
        return 1

    def count_depth(self):
        return 1

    def _get_description(self):
        pretty_counts = ' '.join(str(v) for k, v in sorted(self.counts.items()))
        return 'label={}\ncounts={}\nscore={:4f}'.format(self.label, pretty_counts, self.score)


def count_uniques(values):
    unique_values = np.unique(values)
    return np.array([(values == unique).sum() for unique in unique_values])


def split_best(features, labels, score_function, current_score=None, min_samples_leaf=1):
    if current_score is None:
        current_score = score_function(labels)

    min_score = float('Inf')
    for column in features:
        series = features[column]
        for division in find_dividing_points(series):
            lesser_pos = series < division
            divided_features = (features[lesser_pos], features[~lesser_pos])
            divided_labels = (labels[lesser_pos], labels[~lesser_pos])

            divided_scores = [score_function(count_uniques(labels)) for labels in divided_labels]

            lesser_percentage = divided_labels[0].size / labels.size
            total_score = lesser_percentage * divided_scores[0] + (1 - lesser_percentage) * divided_scores[1]

            if any(labels.size < min_samples_leaf for labels in divided_labels):
                continue

            if total_score < min_score:
                min_score = total_score
                min_column = column
                min_division = division
                min_scores = divided_scores
                min_features = divided_features
                min_labels = divided_labels

    if min_score >= current_score:
        return DecisionTreeLeaf(labels, current_score)

    nodes = [split_best(features, labels, score_function, score, min_samples_leaf)
             for features, labels, score in zip(min_features, min_labels, min_scores)]
    return DecisionTreeNode(min_column, min_division, nodes[0], nodes[1], current_score)


def find_dividing_points(values):
    ordered = np.unique(np.sort(values))
    middles = (ordered[1:] + ordered[:1]) * 0.5
    yield from middles


def entropy(counts):
    percentages = counts / counts.sum()
    return -np.sum(percentages * np.log(percentages))


class DecisionTreeClassifier:
    CRITERIA = {'gini': lambda counts: np.sum((counts / counts.sum()) ** 2),
                'entropy': entropy}
    SPLITTERS = {'best': split_best}

    def __init__(self, criterion, splitter, **splitter_kwargs):
        self.criterion = self.CRITERIA[criterion]
        self.splitter = self.SPLITTERS[splitter]
        self.splitter_kwargs = splitter_kwargs

    def fit(self, features, labels):
        self.unique_labels = np.unique(labels)
        self.tree = self.splitter(features, labels, self.criterion, **self.splitter_kwargs)

    def predict(self, test_features):
        return test_features.apply(self.tree.classify, axis=1)
