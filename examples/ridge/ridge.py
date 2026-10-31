import numpy as np
import sys, os 
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from src.algos.regression.ridge import RidgeRegression

def main():
    print("=" * 55)
    print("     Ridge Regression (L2)")
    print("=" * 55)

    # Collinear data pts : x2 = x1 + noise
    np.random.seed(42)
    n = 100
    x1 = np.random.randn(n)
    x2 = x1 + np.random.randn(n) * 0.01         # almost likely x1 but diff
    X = np.column_stack([x1, x2])
    y = 3 * x1 + 2 * x2 + np.random.randn(n) * 0.5

    X_train, X_test = X[:80], X[80:]
    y_train, y_test = y[:80], y[80:]

    print(f"\n      True weights: [3,2] | Intercept: 1")
    print(f"        Features are highly correalted (r app 0.99)")
    print(f"\n      {'Alpha': <10}  {'Coefs': <25}  {'Test MSE': <12}   {'Weight Norm'}")
    print(f"    {'-'*60}") 

    for alpha in [0, 0.01, 0.1, 1.0, 10.0, 100.0]:
        model = RidgeRegression(alpha=alpha)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = np.mean((y_test - y_pred) ** 2)
        weight_norm = np.sqrt(np.sum(model.coef_ ** 2))

        coef_str = f"[{model.coef_[0]: 7.2f}, {model.coef_[1]:7.2f}]"
        print(f"        {alpha: <10.2f} {coef_str: <25} {mse:<12.4f}    {weight_norm:.4f}")


    # INSIGHT ON WORK OF L2
    print(f"\n      Key insight: ")
    print(f"        ☻ ƒ=0: (no regularization): wt are unstable/huge due to collinearity")
    print(f"        ☻ ƒ=1: wt are shrunk and stable, MSE stays good")
    print(f"        ☻ ƒ=100: wt are over-shrunk, model underfits")
    print()


if __name__ == "__main__":
    main()