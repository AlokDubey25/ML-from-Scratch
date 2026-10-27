"""
    Same lr model as the Normal Equation version, 
    but finds the optimal weights iteratively by repeatedly stepping in the direction that reduces the error fastest.

    The "gradient" tells you which direction is downhill, and the "learning rate" controls how big each step is 
    too big and you overshoot the valley, too small and it takes forever.

    F(x) = 1/n summition (yi^ - yi)^2

    when we have many features then we can use it over normal eq , or need to extend our data to online or streaming learning

    **Feature Scaling for fast conv...

"""

import numpy as np


class LinearRegressionGD:

    def __init__(self, learning_rate: float = 0.01, n_iterations: int  = 1000, fit_intercept: bool = True, verbose: bool = False):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.fit_intercept = fit_intercept
        self.verbose = verbose
        self.weights_ = None
        self.loss_history_ = []

    def _add_bias(self, X: np.ndarray) -> np.ndarray:
        return np.hstack([np.ones((X.shape[0], 1)), X])

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegressionGD":
        """
        We'll train model with gradient descent...
        
        for MSE wrt 0 :   J = (2/n) * X^T (Xtheta - y)
        update rule :  theta = theta - lr * J
        """

        X = np.array(X, dtype=np.float64)
        y = np.array(y, dtype=np.float64)

        if self.fit_intercept:
            X = self._add_bias(X)

        n_samples, n_features = X.shape

        np.random.seed(42)
        self.weights_  =  np.random.randn(n_features) * 0.01
        self.loss_history_ = []

        for i in range(self.n_iterations):
            y_pred = X @ self.weights_

            loss = np.mean((y_pred - y) ** 2)
            self.loss_history_.append(loss)

            gradient = (2/n_samples) * (X.T @ (y_pred - y))

            self.weights_ -= self.learning_rate * gradient

            if self.verbose and (i + 1) % 100 == 0:
                print(f"            Iteration {i+1:4d} | Loss: {loss:.6f}" )

        return self


    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.array(X, dtype=np.float64)
        if self.fit_intercept:
            X = self._add_bias(X)
        return X @ self.weights_


    @property
    def intercept_(self) -> float:
        if self.fit_intercept and self.weights_ is not None:
            return self.weights_[0]
        return 0.0

    @property
    def coef_(self) -> np.ndarray:
        if self.weights_ is None:
            return None
        return self.weights_[1:] if self.fit_intercept else self.weights_
    