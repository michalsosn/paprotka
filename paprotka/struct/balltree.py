import heapq as hq
import math
import random
import numpy as np
from paprotka.metric import METRICS, calculate_distances


class BallTree:
    points = None
    labels = None
    left = None
    right = None

    def __init__(self, points, labels, pivot=None, radius=None, leaf_size=20, metric=METRICS['euclidean']):
        self.pivot = pivot
        self.radius = radius
        self.metric = metric

        row_num, _ = points.shape
        if row_num < leaf_size:
            self.leaf = True
            self.points = points
            self.labels = labels
        else:
            self.leaf = False

            left_ix, right_ix = random.sample(range(row_num), 2)
            left_pivot, right_pivot = points[left_ix], points[right_ix]
            left_distances = calculate_distances(metric, points, left_pivot)
            right_distances = calculate_distances(metric, points, right_pivot)
            closer_to_left = left_distances < right_distances
            closer_to_right = np.logical_not(closer_to_left)

            if closer_to_left.any() and closer_to_right.any():
                left_radius = left_distances[closer_to_left].max()
                right_radius = right_distances[closer_to_right].max()
                self.left = BallTree(points[closer_to_left], labels[closer_to_left], left_pivot, left_radius, leaf_size,
                                     metric)
                self.right = BallTree(points[closer_to_right], labels[closer_to_right], right_pivot, right_radius,
                                      leaf_size, metric)
            else:  # all points the same
                self.leaf = True
                self.points = points
                self.labels = labels

    def depth(self):
        if self.leaf:
            return 1
        else:
            return 1 + max(self.left.depth(), self.right.depth())

    def node_count(self):
        if self.leaf:
            return 1
        else:
            return 1 + self.left.node_count() + self.right.node_count()

    def size(self):
        if self.leaf:
            row_num, _ = self.points.shape
            return row_num
        else:
            return self.left.size() + self.right.size()

    def find(self, point):
        if self.leaf:
            pos, = np.where((self.points == point).all(axis=1))
            return self.labels[pos] if pos.size > 0 else None
        else:
            left_distance = self.metric(self.left.pivot, point)
            right_distance = self.metric(self.right.pivot, point)
            downstream = self.left if left_distance < right_distance else self.right
            return downstream.find(point)

    def find_k_nearest(self, k, reference):
        k = min(k, self.size())
        best = [(-math.inf, None) for _ in range(k)]
        hq.heapify(best)
        self._find_k_nearest(k, reference, best)
        return [pair[1] for pair in best]

    def _find_k_nearest(self, k, reference, best):
        if self.leaf:
            distances = calculate_distances(self.metric, self.points, reference)
            k_closest = np.argpartition(distances, k - 1)[:k] if k <= distances.size else range(distances.size)
            for pos in k_closest:
                hq.heappushpop(best, (-distances[pos], self.labels[pos]))
        else:
            left_distance = self.metric(self.left.pivot, reference)
            right_distance = self.metric(self.right.pivot, reference)

            if left_distance < right_distance:
                first, second = self.left, self.right
                first_distance, second_distance = left_distance, right_distance
            else:
                first, second = self.right, self.left
                first_distance, second_distance = right_distance, left_distance

            if first_distance - first.radius <= -best[0][0]:
                first._find_k_nearest(k, reference, best)
            if second_distance - second.radius <= -best[0][0]:
                second._find_k_nearest(k, reference, best)
