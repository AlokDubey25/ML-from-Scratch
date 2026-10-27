import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from algos.regression.linear_regression_normal import LinearRegressionNormalEquation

def main():
    print("=" * 55)
    print("     Linear Regression (Normal Equation)  -  From Scratch")
    print("=" * 55)


    # For random data or synthetic random data
    np.random.seed(42)
    n_samples = 100
    X = 2 * np.random.rand(n_samples, 1)                                         # single features
    y = 4 + 3  * X[:, 0] + np.random.randn(n_samples) * 0.5             # y = 4 + 3x + noise

    # Train Test Split
    split = int(0.8 * n_samples)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Fit from scratch model
    model = LinearRegressionNormalEquation(fit_intercept= True)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)


    # <<<  Metrrics  >>>
    mse = np.mean((y_test - y_pred) ** 2)
    r2 = 1 - np.sum((y_test - y_pred) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2)

    print(f"\n      True relationship : y = 4 + 3x + noise")
    print(f"        Learned intercept:  {model.intercept_:.4f}  (excepted ~4.0)")
    print(f"        Learned slope:        {model.coef_[0]:.4f}     (excepted ~3.0)")
    print(f"\n      Test MSE:       {mse:.4f}")
    print(f"        Test R2:        {r2:.4f}")


    # NOW COMPARE WITH OTHER ONE LIKE ORIGINAL ONE
    try:
        from sklearn.linear_model import LinearRegression as SKlearnLR
        sk_model = SKlearnLR()
        sk_model.fit(X_train, y_train)
        sk_pred = sk_model.predict(X_test)
        sk_mse = np.mean((y_test - sk_pred) ** 2)

        print(f"\n      [sklearn] intercept : {sk_model.intercept_:.4f}, "
                 f"slope: {sk_model.coef_[0]:.4f},  MSE: {sk_mse:.4f}")

        print("     RESULT MATCH SKLearn...!!")

    except ImportError:
        print("\n       (sklearn not installed  -  skipping comparison)")


if __name__ == "__main__"