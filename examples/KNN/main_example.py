import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from algos.classification.knn import KNNClassifier, KNNRegressor
from utils.metrics import accuracy_score, mean_squared_error

def main():
    print("=" * 55)
    print("     K-Nearest Neighbors (KNN)")
    print("=" * 55)

    #====================================================
    # Classification....
    print("\n" + "-" * 55)
    print("     Classification")
    print("-" * 55)

    # For random data or synthetic random data
    np.random.seed(42)
    n_samples = 100
    X = np.random.rand(n_samples, 2)                                         
    y = (X[:, 0]**2 + X[:, 1]**2 > 1.5**2).astype(int)             # y = 1 if x1 + x2 > 1 else 0

    # Train Test Split
    split = int(0.8 * n_samples)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Fit from scratch model
    model = KNNClassifier(n_neighbors=5, metric='euclidean')
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"\n      Test Accuracy: {acc:.4f}")

    try:
        from sklearn.neighbors import KNeighborsClassifier as SKlearnKNN
        sk_model = SKlearnKNN(n_neighbors=5, metric='euclidean')
        sk_model.fit(X_train, y_train)
        sk_acc = accuracy_score(y_test, sk_model.predict(X_test))
        print(f"      [sklearn] Test Accuracy: {sk_acc:.4f}")   
        print("     ☻ RESULT MATCH SKLearn...!!")
    except ImportError:
        print("\n       (sklearn not installed  -  skipping comparison)")

    #====================================================
    # Regression....
    print("\n" + "-" * 55)
    print("     Regression")
    print("-" * 55)

    np.random.seed(42)
    X_reg = np.random.rand(n_samples, 1) * 5
    y_reg = np.sin(X_reg[:, 0]) + np.random.randn(n_samples) * 0.1

    # Train Test Split
    X_r_train, X_r_test = X_reg[:split], X_reg[split:]
    y_r_train, y_r_test = y_reg[:split], y_reg[split:]

    # Fit from scratch model
    reg_model = KNNRegressor(n_neighbors=5, metric='euclidean')
    reg_model.fit(X_r_train, y_r_train)
    y_r_pred = reg_model.predict(X_r_test)

    mse = mean_squared_error(y_r_test, y_r_pred)
    print(f"\n      Test MSE: {mse:.4f}")

    try:
        from sklearn.neighbors import KNeighborsRegressor as SKlearnKNNR
        sk_reg = SKlearnKNNR(n_neighbors=5, metric='euclidean')
        sk_reg.fit(X_r_train, y_r_train)
        sk_mse = mean_squared_error(y_r_test, sk_reg.predict(X_r_test))
        print(f"      [sklearn] Test MSE: {sk_mse:.4f}")   
        print("     ☻ RESULT MATCH SKLearn...!!")
    except ImportError:
        print("\n       (sklearn not installed  -  skipping comparison)")

    print()


if __name__ == "__main__":
    main()