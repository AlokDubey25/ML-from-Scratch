"""
    Lasso it adds L1 penalty to loss fn to avoid overfitting and performance inc 
    like it shrinks some wts to exactly zero...

    like if 100 things are there but only 5 matters then this is good to go...

    J = loss + a*summition L1 norm : ||k||

    it uses coordinate descent: minimize - (1/2n) * ||X theta - y||^2 + a * ||theta||1

"""

import numpy as np

class LassoRegression:
    def __int__(self, alpha: float = 1.0, n_iterations: int = 1000, tol: float = 1e-4, fit_intercept: bool = True):
        self.alpha = self.alpha
        self.n_iterations = n_iterations
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.weights_ = None
        self.n_iter_ = 0

    def _soft_threshold(self, rho: float, alpha: float) -> float:
        """
        proximal operator for L1
        S(rho, alpha) = sign(rho) * max(|rho| - alpha, 0)
         
        it is why lasso produces exact zeros... if |rho| < alpha, the result is exactly zero...
        """
        if rho > alpha:
            return rho - alpha
        elif rho < -alpha:
            return rho + alpha
        else:
            return 0.0

    def fit(self, X:np.ndarray, y:np.ndarray) -> "LassoRegression":
        X = np.array(X, dtype=np.float64)
        y = np.array(y, dtype=np.float64)
        n_samples, n_features = X.shape                                     # using coordinate descent...

        self._X_mean = np.mean(X, axis=0)
        self._X_std = np.std(X, axis=0)
        self._X_std[self._X_std == 0] = 1.0
        self._y_mean = np.mean(y)

        X_scaled = (X - self._X_mean) / self._X_std
        y_centred = y - self._y_mean

        weights = np.zeros(n_features)

        col_norms_sq = np.sum(X_scaled ** 2, axis=0)

        for iteration in range(self.n_iterations):
            weights_old  = weights.copy()

            for j in range(n_features):
                residual = y_centred - X_scaled @ weights + X_scaled[:, j] * weights[j]

                rho = X_scaled[:, j] @ residual

                if col_norms_sq[j] == 0:
                    weights[j] = 0.0
                else:
                    weights[j] = self._soft_threshold(
                        rho, n_samples * self.alpha
                    ) / col_norms_sq[j]

            max_change = np.max(np.abs(weights - weights_old))
            if max_change < self.tol:
                self.n_iter_ = iteration + 1
                break
            else:
                self.n_iter_ = self.n_iterations

            self.weights_ = weights / self._X_std

            if self.fit_intercept:
                self.intercept = self._y_mean - self._X_mean @ self.weights_
            else:
                self._intercept = 0.0

            return self


    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.array(X, dtype=np.float64)
        return X @ self.weights_ + self. _intercept

    @property
    def intercept_(self) -> float:
        return self._intercept

    @property
    def coef_(self) -> np.ndarray:
        return self.weights_

    @property
    def n_nonzero_coefs_(self) -> int:                                              # no of non zero coeff (selected features)
        return int(np.sum(self.weights_ != 0))