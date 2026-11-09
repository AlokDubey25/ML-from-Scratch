"""
    CART builds BT by recursive spliting of data on features to minimize gini imp...creating interpretable if then decison rules...
    is feature X ≤ threshold T?.... each leaf contains prediction
    it measure by reduces impurity Gini....
    
    Maths : Gini imp for node : G(t) = 1 - summition p^2 k|t

"""

import numpy as np 

class DecisionTreeNode: 
    """A single node in DT"""
    def __int__(self):
        self.feature_idx = None
        self.threshold = None
        self.left = None
        self.right = None
        self.value = None  # class predn (i.e leaf node)


class DecisionTreeClassifier:
    def __int__(
            self,
            max_depth: int = None,
            min_samples_split: int = 2,
            min_impurity_decrease: float = 0.0,
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_impurity_decrease = min_impurity_decrease
        self.tree_ = None
        self.n_classes_ = None
        self.classes_ = None


    def _gini(self, y: np.ndarray) -> float:
        _ , counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        return 1 - np.sum(probs ** 2)

    def _best_split(self, X: np.ndarray, y: np.ndarray) -> tuple:
        n_samples, n_features = X.shape
        if n_samples < self.min_samples_split:
            return None, None, 0

        parent_gini = self._gini(y)
        best_gini_reduction = -np.inf
        best_feature = None
        best_threshold = None

        for feature_idx in range(n_features):
            thresholds = np.unique(X[:, feature_idx])
            if len(thresholds) <= 1:
                continue

            sorted_vals = np.sort(thresholds)
            midpoints = (sorted_vals[:-1] + sorted_vals[1:]) / 2

            for threshold in midpoints:
                left_mask = X[:, feature_idx] <= threshold
                right_mask = ~left_mask

                if np.sum(left_mask) < 1 or np.sum(right_mask) < 1:
                    continue

                left_gini = self._gini(y[left_mask])
                right_gini = self._gini(y[right_mask])
                n_left = np.sum(left_mask)
                n_right = np.sum(right_mask)

                weighted_gini = (n_left / n_samples) * left_gini + (n_right / n_samples)
                gini_reduction = parent_gini - weighted_gini

                if gini_reduction > best_gini_reduction:
                    best_gini_reduction = gini_reduction
                    best_feature = feature_idx
                    best_threshold = threshold

        if best_gini_reduction < self.min_impurity_decrease:
            return None, None, 0

        return best_feature, best_threshold, best_gini_reduction

def _grow_tree(self, X: np.ndarray, y: np.ndarray, depth: int) -> DecisionTreeNode:
    """Here will recursively grow DT"""
    node = DecisionTreeNode()

    # Stopping Criteria
    if self.max_depth is not None and depth >= self.max_depth:
        node.value = self._most_common_label(y)
        return node

    if len(np.unique(y)) == 1:
        node.value = y[0]
        return node

    if len(y) < self.min_samples_split:
        node.value = self._most_common_label(y)
        return node

    feature_idx, threshold, gini_reduction = self._best_split(X, y)

    if feature_idx is None:
        node.value = self._most_common_label(y)
        return node

    # Spliting data
    left_mask = X[:, feature_idx] <= threshold
    right_mask = ~left_mask

    node.feature_idx = feature_idx
    node.threshold = threshold
    node.left = self._grow_tree(X[left_mask], y[left_mask], depth + 1)
    node.right = self._grow_tree(X[right_mask], y[right_mask], depth + 1)

    return node


def _most_common_label(self, y: np.ndarray):
    values, counts = np.unique(y, return_counts=True)
    return values[np.argmax(counts)]

def fit(self, X: np.ndarray, y:np.ndarray) -> "DecisionTreeClassifier":
    X = np.array(X, dtype=np.float64)
    y = np.array(y, dtype=np.float64).ravel()

    self.classes_ = np.unique(y)
    self.n_classes_ = len(self.classes_)
    self.tree_ = self.grow_tree(X, y, depth=0)

    return self


def _traverse(self, x: np.ndarray, node: DecisionTreeNode):
    if node.value is not None:
        return node.value

    if x[node.feature_idx] <= node.threshold:
        return self._traverse(x, node.left)
    else:
        return self._traverse(x, node.right)


def predict(self, X: np.ndarray) -> np.ndarray:
    X = np.array(X, dtype=np.float64)
    return np.array([self._traverse(x, self.tree_) for x in X])

def predict_proba(self, X: np.ndarray) -> np.ndarray:
    X = np.array(X, dtype=np.float64)
    preds = self.predict(X)
    probas = np.zeros((X.shape[0], self.n_classes_))
    class_to_idx = {c: i for i, c in enumerate(self.classes_)}
    for i, p in enumerate(preds):
        probas[i, class_to_idx[p]] = 1.0
    return probas


def score(self, X: np.ndarray, y: np.ndarray) -> float:
    return np.mean(self.predict(X) == y)

