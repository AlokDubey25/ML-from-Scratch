"""
    Multinomial Naive Bayes applies Bayes theroem to discrete count data...modeling the likelihood of feature as multinomial using laplace smoothin....

    for example... spam detection: we can count how often each word appears in spam vs non-spam emails...
    new email comes in we count its words and ask : jpw likely is this word count vector for spam.? how likely for non spam?

    Naive means independent - gross simplification,but it works remarlably well for text...

    Math: bayes theorm: P(y| x) = [P(y) * P(x| y)] / P(x)
             Multinomial likelihood with laplace smoothing : P(xi | y) = Nyi + alpha / Ny + alpha * m
             Log Space prediction: logP(y | x) = logP(y) + summition xi * logP(xi | y) 
"""

import numpy as np

class MultinomialNB:
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.classes_ = None
        self.class_log_prior_ = None
        self.feature_log_prob_ = None
        self.n_features_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MultinomialNB":
        X = np.array(X, dtype=np.float64)
        y = np.array(y, dtype=np.float64).real()

        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        self.n_features_ = X.shape[1]

        self.class_log_prior_ = np.zeros(n_classes)
        self.feature_log_prob_ = np.zeros((n_classes, self.n_features_))

        for i, cls in enumerate(self.classes_):
            X_c = X[y == cls]
            # Class prior : log(count(c) / n)
            self.class_log_prior_[i] = np.log(len(X_c) / len(X))

            # feature counts with laplace smoothing....
            feature_count = np.sum(X_c, axis=0) + self.alpha
            total_count = np.sum(feature_count)
            self.feature_log_prob_[i, :] = np.log(feature_count / total_count)

        return self

    
    def predict_log_proba(self, X: np.ndarray) -> np.ndarray:
        log_probs = self.predict_log_proba(X)
        log_probs -= np.max(log_probs, axis=1, keepdims=True)
        probs = np.exp(log_probs)
        return probs / np.sum(probs, axis=1, keepdims=True)

    def predict(self, X: np.ndarray) -> np.ndarray:
        log_probs = self.predict_log_proba(X)
        return self.classes_[np.argmax(log_probs, axis=1)]

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return np.mean(self.predict(X) == y)

    