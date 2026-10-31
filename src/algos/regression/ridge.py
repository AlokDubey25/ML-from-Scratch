"""
    its L2 reg diff frm l1 which we did earlier like here we shrink wts towards zero, to prevent overfirtting and handling multicollinearity

    like if fitting line to nosiy data with many correlated features then for high wt  it gives high penenlty...

    maths : ƒ(theta) = 1/n | | X*theta - y | |^2 + K | | theta | |^2
                 theta = (XTX + KI)^-1XTy
"""

import numpy as np

class RidgeRegression:

    def __int__(self, alpha: float = 1.0, fit_intercept: bool = True):
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.weights_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RidgeRegression":

        X = np.ndarray(X, dtype=np.float64)
        y = np.ndarray(y, dtype=np.float64)

        if self.fit_intercept:
            X = np.hstack([np.ones((X.shape[0], 1)), X])

        n_features = X.shape[1]

        # for regularization matrix
        # not regularize bias term (first element)
        reg_matrix = self.alpha * np.eye(n_features)
        if self.fit_intercept:
            reg_matrix[0, 0] = 0        # no penalty for bias

        # Eq 
        self.weights_ = np.linalg.inv(X.T @ X + reg_matrix) @ X.T @ y

        return self


    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.array(X, dtype=np.float64)
        if self.fit_intercept:
            X = np.hstack([np.ones((X.shape[0], 1)), X])
        return X @ self.weights_

    @property
    def intercept_(self) -> float:
        if self.fit_intercept and self.weights_ is not None:
            return self.weights_
        return 0.0

    @property
    def coef_(self) -> np.ndarray:
        if self.weights_ is None:
            return None
        return self.weights_[1:] if self.fit_intercept else self.weights_
