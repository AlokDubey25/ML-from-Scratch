"""
    its bayes theorem with naive assumption that features are independent of each other
    Imagine we r tryin to identify a fruit as "apple" or "orange" based on its weight and color....
    We look at all the apples we've seen and compute the average weight and average color and how much they vary. 
    When a new fruit arrives, we will do: "How likely is this weight/color combo for an apple? How likely for an orange?" 
    Then we multiply by how common each fruit is overall. The class with the highest product wins.

The "naive" part: we pretend weight and color are independent, even though they might not be. 
This simplification makes the math tractable.

    maths: bayes theorem: P(A|B) = P(B|A) * P(A) / P(B)
                Gaussian: P(x|y) = 1/(sqrt(2*pi)*sigma) * exp(-(x-mu)^2/(2*sigma^2))
                log space(avoids underflow): log(P(y|x)) = log(P(x|y)) + log(P(y)) - log(P(x))
"""

import numpy as np

class GaussianNB:
    def __init__(self, var_smoothing: float = 1e-9):
        self.var_smoothing = self.var_smoothing
        self.classes_ = None
        self.class_prior_ = None
        self.theta_ = None
        self.var_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GaussianNB":
        X = np.array(X, dtype=np.float64)
        y = np.array(y, dtype=np.float64).ravel()

        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features = X.shape[1]

        self.theta_ = np.zeros((n_classes, n_features))
        self.var_ = np.zeros((n_classes, n_features))
        self.class_prior_ = np.zeros(n_classes)

        for i, cls in enumerate(self.classes_):
            X_c = X[y == cls]
            self.classes_prior_[i] = np.log(len(X_c) / len(X))
            self.theta_[i, :] = np.mean(X_c, axis=0)
            self.var_[i, :] = np.var(X_c, axis=0) + self.var_smoothing

        return self

    
    def _gaussian_pdf(self, X:np.ndarray, mean: np.ndarray, var:np.ndarray)  -> np.ndarray:
        # log(p(x)) = -0.5 * [log(2.pi.sigma^2) + (x - nu)^2/sigma^2]
        return -0.5 * (np.log(2 * np.pi * var) + (X - mean) ** 2 / var)

    def predict_log_proba(self, X:np.ndarray) -> np.ndarray:
        X = np.array(X, dtype=np.float64)
        n_samples = X.shape[0]
        n_classes = len(self.classes_)

        log_probs = np.zeros((n_samples, n_classes))
        for i in range(n_classes):
            class_log_likelihood = np.sum(
                self._gaussian_pdf(X, self.theta_[i] , self.var_[i]),
                axis=1,
            )
            log_probs[:, i] = self.class_prior_[i] + class_log_likelihood

        return log_probs

    def predict_proba(self, X: np.ndarray) -> np.ndarray:

        log_probs = self.predict_log_proba(X)
        return self.classes_[np.argmax(log_probs, axis=1)]

    def score(self, X:np.ndarray, y: np.ndarray) -> float:
        """For mean accuracy on test data and labels...."""
        return np.mean(self.predict(X) == y)
    