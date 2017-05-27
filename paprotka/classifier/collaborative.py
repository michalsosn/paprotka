import numpy as np
from scipy import sparse
from sklearn import base as skbase


class CollaborativeRecommender(skbase.BaseEstimator):
    def __init__(self, feature_count=10, learning_rate=0.1, regularization_param=0, error_ratio_threshold=0.00001):
        self.feature_count = feature_count
        self.learning_rate = learning_rate
        self.regularization_param = regularization_param
        self.error_ratio_threshold = error_ratio_threshold

    def get_params(self, deep=True):
        return {
            'feature_count': self.feature_count,
            'learning_rate': self.learning_rate,
            'regularization_param': self.regularization_param,
            'error_ratio_threshold': self.error_ratio_threshold
        }

    def fit(self, train_df, label_df):
        person_ids = np.unique(train_df.person_id)
        self.person_pos = {person_id: index for index, person_id in enumerate(person_ids)}
        user_count = person_ids.size

        train_values = train_df.values
        label_values = label_df.values
        sparse_matrix = sparse.csr_matrix((label_values, (train_values[:, 0], train_values[:, 1] - 1)), dtype=np.uint8)
        item_count = sparse_matrix.shape[1]

        exist_values = np.ones_like(label_values)
        exist_sparse_matrix = sparse.csr_matrix((exist_values, (train_values[:, 0], train_values[:, 1] - 1)),
                                                dtype=np.bool)
        exist_count = exist_values.sum()

        train_matrix = np.zeros((user_count, item_count), dtype=np.uint8)
        exist_matrix = np.zeros((user_count, item_count), dtype=np.uint8)
        for person_id, index in self.person_pos.items():
            train_matrix[index] = sparse_matrix.getrow(person_id).toarray()
            exist_matrix[index] = exist_sparse_matrix.getrow(person_id).toarray()
        exist_per_user = exist_matrix.sum(axis=1)  # U
        exist_per_item = exist_matrix.sum(axis=0)  # I

        self.user_params = np.random.rand(user_count, self.feature_count)  # U x W
        self.item_features = np.random.rand(item_count, self.feature_count)  # I x W

        prev_total_error = None
        total_error = None
        while prev_total_error is None or abs(
                        (total_error - prev_total_error) / prev_total_error) >= self.error_ratio_threshold:
            prev_total_error = total_error

            prediction_corrections = (self.user_params @ self.item_features.T - train_matrix) * exist_matrix  # U x I
            user_corrections = (prediction_corrections @ self.item_features) / exist_per_user[:, None]  # U x W
            item_corrections = (prediction_corrections.T @ self.user_params) / exist_per_item[:, None]  # I x W
            total_error = np.square(prediction_corrections).sum() / exist_count

            if self.regularization_param != 0:
                user_corrections += self.regularization_param * self.user_params
                item_corrections += self.regularization_param * self.item_features
                total_error += self.regularization_param * (
                    np.square(self.user_params).sum() + np.square(self.item_features).sum()
                )

            self.user_params -= self.learning_rate * user_corrections
            self.item_features -= self.learning_rate * item_corrections

    def predict(self, task_df):
        def classify_collaborative(task_row):
            pos = self.person_pos[task_row.person_id]
            prediction = self.user_params[pos] @ self.item_features[task_row.movie_id - 1].T
            try:
                return int(np.clip(round(prediction), 0, 5))
            except ValueError:
                return -1

        return task_df.T.apply(classify_collaborative)