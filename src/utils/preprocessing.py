import numpy as np

class StandardScaler:
    """Standardize features by removing the mean and scaling to unit variance..."""
    
    def __init__(self):
        self.mean_ = None
        self.std_     = None

        def fit(self, X: np. ndarray) -> "StandardScaler":
            self.mean_ = np.mean(X, axis=0)
            self.std_      = np.std(X, axis=0)
            self.std_[self.std_ == 0] = 1.0                                 # if we're getting zerp in deno
            return self

        def transform(self, X: np.ndarray) -> np.ndarray:
            return (X - self.mean_) / self.std_

        def fit_transform(self, X: np.ndarray) -> np.ndarray:
            return self.fit(X).transform(X)


class MinMaxScaler:
    """So that scaler features get range of [0,1]"""

    def __init__(self):
        self.min_ = None
        self.range_ = None

    def fit(self, X: np.ndarray) -> "MinMaxScaler":
        self.min_ = np.min(X, axis=0)
        self.range_ = np.max(X, axis=0) - self.min_
        self.range_[self.range_ == 0] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.min_) / self.range_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)



def train_test_split(
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.2,
        random_state: int = None,
) -> tuple:
    """It will return or split train test for x and y by those above datas...."""

    if random_state is not None:
        np.random.seed(random_state)

    n = len(X)
    indices = np.random.permutation(n)
    split = int(n* (1 - test_size))

    train_idx = indices[:split]
    test_idx  = indices[split:]

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def one_hot_encode(y: np.ndarray) -> np.ndarray:
    """It is array of integer class labels so it will return size of samples and classes"""

    n_classes = len(np.unique(y))
    one_hot = np.zeros((len(y), n_classes))
    one_hot[np.arange(len(y)), y.astype(int)] = 1
    return one_hot


def add_bias_columnn(X: np.ndarray) -> np.ndarray:
    """To prepend a column of ones to X for Bias or intercept terms..."""
    return np.hstack([np.ones((X.shape[0], 1)), X])

