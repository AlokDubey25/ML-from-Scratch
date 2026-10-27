import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from algos.regression.linaer_regression_gradient import LinearRegressionGD

def main():
    print("=" * 55)
    print("     Linear Regression (Gradient) - from Scratch")
    print("="* 55)

    np.random.seed(42)
    X = 2 * np.random.rand(200, 1)
    y = 4 + 3 * X[:, 0] + np.random.randn(200) * 0.5

    X_train, X_test = X[:160], X[160:]
    y_train, y_test = y[:160], y[160:]

    model = LinearRegressionGD(learning_rate = 0.1, n_iterations = 500, verbose = True)
    model.fit(X_train,y_train)
    
    y_pred = model.predict(X_test)
    mse = np.mean((y_test - y_pred) ** 2)
    r2 = 1 - np.sum((y_test - y_pred) ** 2) / np.sum((y_test- np.mean(y_test)) ** 2)

    print(f"\n      Learned intercept: {model.intercept_:.4f}   (excepted ~4.0)")
    print(f"        Learned slope:      {model.coef_[0]:.4f}    (expected ~3.0)")
    print(f"        Test MSE:       {mse:.4f}")
    print(f"        Test R^2:       {r2:.4f}")

    print(f"\n      Loss at iteration 1:        {model.loss_history_[0]:.4f}")
    print(f"        Loss at iteration 500:     {model.loss_history_[-1]:.4f} ")
    print(f"        Converged:      {'Yes' if model.loss_history_[-1] < 0.5 else 'No'}")
    print()



if __name__ == "__main__":
    main()