"""
    KNN is like use to predict label of new sample by mahority(for classification) and average(for regression) of k nearest neighbors.
    It is a lazy learning algorithm, meaning it does not learn a model during training, but rather stores the training data and makes predictions based on the stored data during inference.

    its use when you have a small dataset and want to make predictions based on the similarity of the data points.

    maths: dis (euclidean) = sqrt(sum((x1 - x2)^2))
    class (majority vote) = argmax(sum(y_i for i in knn))
    reg (avg)  = 1/k * sum(y_i for i in knn)

"""

import numpy as np
from collections import Counter
from utils.distances import euclidean_distance


class KNNClassifier:
    def __init__(self, n_neighbors: int = 5, metric: str = 'euclidean'):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.X_train = None
        self.y_train = None
        self.classes_ = None


    def _distance(self, x1: np.ndarray, x2: np.ndarray) -> float:
        if self.metric == "euclidean":
            return euclidean_distance(x1, x2)
        elif self.metric == "manhattan":
            return np.sum(np.abs(x1 - x2))
        else:
            raise ValueError(f"Unknown metric: {self.metric}")

    def fit(self, X:np.ndarray, y: np.ndarray) -> "KNNClassifier":
        self.X_train_ = np.array(X, dtype=np.float64)
        self.y_train_ = np.array(y, dtype=np.float64).ravel()
        self.classes_ = np.unique(self.y_train_)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.array(X, dtype=np.float64)
        predictions = np.zeros(X.shape[0], dtype=self.y_train_.dtype)

        for i, x in enumerate(X):
            distances = np.array([self._distance(x,x_train) for x_train in self.X_train_])
            k_idx = np.argsort(distances)[:self.n_neighbors]
            k_labels = self.y_train_[k_idx]
            predictions[i] = Counter(k_labels).most_common(1)[0][0]

        return predictions


    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.array(X, dtype=np.float64)
        probas = np.zeros((X.shape[0], len(self.classes_)))
        class_to_idx = {c: i for i, c in enumerate(self.classes_)}

        for i, x in enumerate(X):
            distances = np.array([self._distance(x, x_train) for x_train in self.X_train_])
            k_idx = np.argsort(distances)[:self.n_neighbors]
            k_labels = self.y_train_[k_idx]
            for label in k_labels:
                probas[i, class_to_idx[label]] += 1.0
            probas[i] /= self.n_neighbors

        return probas


class KNNRegressor:

    def __init__(self, n_neighbors: int = 5, metric: str = 'euclidean'):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.X_train = None
        self.y_train = None

    def _distance(self, x1: np.ndarray, x2: np.ndarray) -> float:
        if self.metric == "euclidean":
            return euclidean_distance(x1, x2)
        elif self.metric == "manhattan":
            return np.sum(np.abs(x1 - x2))
        else:
            raise ValueError(f"Unknown metric: {self.metric}")

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KNNRegressor":
        self.X_train_ = np.array(X, dtype=np.float64)
        self.y_train_ = np.array(y, dtype=np.float64).ravel()
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.array(X, dtype=np.float64)
        predictions = np.zeros(X.shape[0])

        for i, x in enumerate(X):
            distances = np.array([self._distance(x, x_train) for x_train in self.X_train_])
            k_idx = np.argsort(distances)[:self.n_neighbors]
            predictions[i] = np.mean(self.y_train_[k_idx])

        return predictions

