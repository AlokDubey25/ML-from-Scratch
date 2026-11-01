"""
    Sigmoid function vala...
    sample class using logistic function for binary cases or softmax function for multiclass by minimizing cross entropy loss via gradient descent

    decision boundary - 0.5

    for multiclass softmax generalizes this to multiple classes by producing prob dist over all classes that sums to 1...

    Binary : sigmoid + binary cross-entropy.
    Multiclass : softmax + categorical cross-entropy (one-vs-all training).

    maths : binary: ƒ = sigmoid(X.theta) where sigmoid(z) = 1/1+e^-z
                Loss (Binary cross entropy) : J(theta) = -1/n summition [ (yi*log(yi^) + (1-yi)*log(1-yi^)) ]
                Multiclass (softmax) : P(y=k | x) = e^(X thetak ) / summitione^(X thetaj)
                and
                Gradient (binary) : DJ = 1/n XT (y^ - y) + k theta
"""

import numpy as np
from utils.activations import sigmoid, softmax

class LogisticRegression:
    def __int__(
            self,
            learning_rate: float = 0.01,
            n_iter: int =  1000,
            reg_strength: float = 0.0,
            fit_intercept: bool = True,
            ):
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.reg_strength = reg_strength
        self.fit_intercept = fit_intercept
        self.weights_ = None
        self.classes_ = None
        self.n_classes_ = None


    def _add_bias(self, X: np.ndarray) -> np.ndarray:
        return np.hstack([np.one((X.shape[0], 1)), X])              # its mainly for prepending a column of ones for intercept term...

    def fit(self, X:np.ndarray, y: np.ndarray) -> "LogisticRegression":
        X = np.array(X, dtype=np.float64)
        y = np.array(y, dtrype=np.float64).ravel()          # its uses gradient descent and automatically detects binary or multiclass on unique labels...

        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)

        if self.fit_intercept:
            X = self._add_bias(X)

        n_samples, n_features= X.shape

        if self.n_classes_ == 2:
            y_bin = (y == self.classes_[1]).astype(float)          # binary case, convert labels to 0 and 1
            self.weights_ = np.zeros(n_features)

            for _ in range(self.n_iter):
                linear = X @ self.weights_
                preds = sigmoid(linear)
                error = preds - y_bin
                grad = (X.T @ error) / n_samples
                if self.reg_strength > 0:
                    grad += (self.reg_strength / n_samples) * self.weights_  # L2 regularization
                    grad[0] -= (self.reg_strength / n_samples) * self.weights_[0]  # Don't regularize bias term
                self.weights_ -= self.learning_rate * grad

        else:
            # Multiclass one vs all with softmax
            self.weights_ = np.zeros((n_features, self.n_classes_))
            y_one_hot = np.zeros((n_samples, self.n_classes_))
            for i, cls in enumerate(self.classes_):
                y_one_hot[:, i] = (y == cls).astype(float)

            for _ in range(self.n_iter):
                logits = X @ self.weights_
                probs = softmax(logits)
                error = probs - y_one_hot
                grad = (X.T @ error) / n_samples
                if self.reg_strength > 0:
                    grad[:-1, :] += (self.reg_strength / n_samples) * self.weights_[:-1, :]  # L2 regularization for weights only
                self.weights_ -= self.learning_rate * grad

        return self


    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.array(X, dtype=np.float64)
        if self.fit_intercept:
            X = self._add_bias(X)

        if self.n_classes_ == 2:
            linear = X @ self.weights_
            probs = sigmoid(linear)
            return np.vstack([1 - probs, probs]).T  # Return probabilities for both classes
        else:
            logits = X @ self.weights_
            return softmax(logits)  # Return probabilities for all classes

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        if self.n_classes_ == 2:
            return self.classes_[(probs[:, 1] >= 0.5).astype(int)]  # Binary case
        else:
            return self.classes_[np.argmax(probs, axis=1)]  # Multiclass case


    @property
    def intercept_(self) -> np.ndarray:
        if self.fit_intercept:
            return self.weights_[0]
        else:
            return np.array([])

    @property
    def coef_(self) -> np.ndarray:
        if self.fit_intercept:
            return self.weights_[1:]
        else:
            return self.weights_