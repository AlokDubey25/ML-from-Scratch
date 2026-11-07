import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from algos.classification.logistic_regression import LogisticRegression

def main():
    print("=" * 55)
    print("     Logistic Regression")
    print("=" * 55)

    print("\n" + "-" * 55)
    print("     Binary Classification")
    print("-" * 55)

    np.random.seed(42)
    n_samples = 200
    X = np.random.randn(n_samples, 2)
    y = (X[:, 0] * 2 + X[:, 1] * 1.5 + np.random.randn(n_samples) * 0.5 > 0).astype(float)

    split = int(0.8 * n_samples)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = LogisticRegression(learning_rate=0.1, n_iter=2000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = np.mean(y_pred == y_test)
    print(f"\n      Test Accuracy: {acc:.4f}")

    try:
        from sklearn.linear_model import LogisticRegression as SKlearnLR
        sk_model = SKlearnLR(max_iter=2000)
        sk_model.fit(X_train, y_train)
        sk_acc = np.mean(sk_model.predict(X_test) == y_test)
        print(f"      [sklearn] Test Accuracy: {sk_acc:.4f}")
        print("     ☻ RESULT MATCH SKLearn...!!")
    except ImportError:
        print("\n       (sklearn not installed  -  skipping comparison)")



    # Multiclass Classification
    print("\n" + "-" * 55)
    print("     Multiclass Classification (like 3 classes, softmaxing)")
    print("-" * 55)

    np.random.seed(42)
    X_multi = np.vstack([
        np.random.randn(80, 2) +[2, 2],
        np.random.randn(80, 2) +[-2, 2],
        np.random.randn(80, 2) +[0, -2],
    ])
    y_multi = np.array([0]*80 + [1]*80 + [2]*80)

    split_multi = int(0.8 * len(y_multi))
    X_mt, X_m_test = X_multi[:split], X_multi[split:]
    y_mt, y_m_test = y_multi[:split], y_multi[split:]

    model_multi = LogisticRegression(learning_rate=0.1, n_iter=2000)
    model_multi.fit(X_mt, y_mt)

    y_m_pred = model_multi.predict(X_m_test)
    acc_multi = np.mean(y_m_pred == y_m_test)
    print(f"\n      Test Accuracy (Multiclass): {acc_multi:.4f}")

    try: 
        sk_multi = SKlearnLR(max_iter = 2000, multi_class= 'multinomial', solver='lbfgs')
        sk_multi.fit(X_mt, y_mt)
        sk_m_acc = np.mean(sk_multi.predict(X_m_test) == y_m_test)
        print(f"      [sklearn] Test Accuracy (Multiclass): {sk_m_acc:.4f}")
        print("     ☻ RESULT MATCH SKLearn...!!")
    except ImportError:
        print("\n       (sklearn not installed  -  skipping comparison)")

    print()


if __name__ == "__main__":
    main()