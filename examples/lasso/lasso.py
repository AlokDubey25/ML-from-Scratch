import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from algos.regression.lasso import LassoRegression

def main():
    print("=" * 55)
    print("     Lasso Regression (L1) ")
    print("=" * 55)

    # irrelevent data --- 
    np.random.seed(42)
    n = 200
    X = np.random.randn(n, 8)   # 8 features

    y = 3.0 * X[:, 0] + 1.5 * X[:, 2] - 2.0 * X[:, 5] + np.random.randn(n) * 0.5

    X_train, X_test = X[:160], X[160:]
    y_train, y_test = y[:160], y[160:]

    print(f"\n      True weights: [3.0, 0, 1.5, 0, 0, -2.0, 0, 0]")
    print(f"        Only features 0, 2, 5 are relevant (5 are noise)")

    # comparing different values of alpha
    print(f"\n      {'Alpha': <8}   {'Non-zero': <10}   {'Coefficents': <50}    {'MSE'}")
    print(f"        {'-'*80}")

    for alpha in [0.001, 0.01,0.05, 0.1, 0.5, 1.0]:
        model = LassoRegression(alpha=alpha, n_iterations=5000)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = np.mean((y_test - y_pred) ** 2)

        coef_str = ", ".join(f"{c:6.3f}" for c in model.coef_)
        print(f"        {alpha: <8.3f}  {model.n_nonzero_coefs_:<10d}   [{coef_str}]    {mse:.4f}")


    print(f"\n  Key insight: ")
    print(f"        ☻ Small ƒ: all features kept (like OLS)")
    print(f"        ☻ Medium ƒ: noise features dropped to exactly 0 (feature selection!)")
    print(f"        ☻ Large ƒ: too agreesive - even useful features are zeroed out ")

    model = LassoRegression(alpha=0.5, n_iterations=5000)
    model.fit(X_train, y_train)
    print(f"\n      Converged in {model.n_iter_} iterations (ƒ=0.05)")
    print()

if __name__ == "__main__":
    main()