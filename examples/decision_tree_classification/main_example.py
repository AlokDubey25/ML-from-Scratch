import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from src.algos.classification import decision_tree_classifier
from src.utils.metrics import accuracy
from src.utils.preprocessing import train_test_split


def main():
    print("=" * 55)
    print("  Decision Tree Classifier (CART) — From Scratch")
    print("=" * 55)

    # Generate synthetic data
    np.random.seed(42)
    n_samples = 500

    X0 = np.random.randn(n_samples // 3, 2) * 0.3 + [0, 0]
    X1 = np.random.randn(n_samples // 3, 2) * 0.3 + [1, 1]
    X2 = np.random.randn(n_samples // 3, 2) * 0.3 + [0.5, -0.5]
    X = np.vstack([X0, X1, X2])
    y = np.array([0] * (n_samples // 3) + [1] * (n_samples // 3) + [2] * (n_samples // 3))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Fit from-scratch model
    model = decision_tree_classifier(max_depth=5, min_samples_split=5)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy(y_test, y_pred)
    print(f"\n  Test accuracy:  {acc:.4f}")

    # Compare with scikit-learn
    try:
        from sklearn.tree import DecisionTreeClassifier as SklearnDT
        sk_model = SklearnDT(max_depth=5, min_samples_split=5, random_state=42)
        sk_model.fit(X_train, y_train)
        sk_acc = accuracy(y_test, sk_model.predict(X_test))
        print(f"  [sklearn] accuracy: {sk_acc:.4f}")
        print("  ☻ Results match sklearn!")
    except ImportError:
        print("  (sklearn not installed — skipping comparison)")

    # Effect of max_depth
    print(f"\n{'─' * 55}")
    print("  Effect of max_depth")
    print(f"{'─' * 55}")

    for depth in [1, 3, 5, 10, None]:
        m = decision_tree_classifier(max_depth=depth, min_samples_split=2)
        m.fit(X_train, y_train)
        train_acc = accuracy(y_train, m.predict(X_train))
        test_acc = accuracy(y_test, m.predict(X_test))
        print(f"  depth={str(depth):<5}  train_acc={train_acc:.4f}  test_acc={test_acc:.4f}")

    print()


if __name__ == "__main__":
    main()