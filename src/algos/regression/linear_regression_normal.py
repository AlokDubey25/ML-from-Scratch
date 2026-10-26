"""
    So first of all what is it - we're fiting st line to data by finding wt that minimize sq errors,
    and Normal eq solves for those wts in one shot using closed form matrix formula

    Like if we ve scatter plot of house sizes vs. prices. We wanna draw the "best fitting" straight line through the points. "Best" means the line that makes the total squared distance from every point to the line as small as possible.

    The Normal Equation is like having a magic formula cuz instead of repeatedly nudging the line to fit better (gradient descent), we plug our data into one equation and get the perfect answer instantly.

    Its like solving 2x = 10 directly (x = 5) vs. guessing and checking until you land on 5.
    
    eq : theta = (X^T X)^(-1) X^T y

"""

import numpy as np

class LinearRegressionNormalEquation:

    def __init__(self):
        self.fit_intercept = self.fit_intercept
        self.weights_ = None

    def _add_bias(self, X: np.ndarray) -> np.ndarray:
        return np.hstack([np.ones((X.shape[0], 1)), X])             # it prepend a column of ones for intercept term....

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegressionNormalEquation":
        X = np.ndarray(X, dtype=np.float64)             # Computing wt using normal eq
        y = np.ndarray(y, dtype=np.float64)

        if self.fit_intercept:
            X = self._add_bias(X)

        self.weights_ = np.linalg.pinv(X.T @ X) @ X.T @ y      # wt =(X^T X) ^(-1)  X^T y

        return self


    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.array(X, dtype=np.float64)                   # to predict target values for new data

        if self.fit_intercept:
            X = self._add_bias(X)

        return X @ self.weights_

    @ property
    def intercept_(self) -> float:
        if self.fit_intercept and self.weights_ is not None:            # to get bias term
            return self.weights_[0]
        return 0.0

    @property
    def coef_(self) -> np.ndarray:
        if self.weights_ is None:                                               # to get features coefficents excpet bias
            return None
        if self.fit_intercept:
            return self.weights_[1:]
        return self.weights_