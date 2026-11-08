import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from src.algos.classification.naive_bayes_Gaussian import GaussianNB
from src.utils.metrics import accuracy
from src.utils.preprocessing import train_test_split

def main():
    print("=" * 55) 
    print("     Gaussian Naive Bayes")
    print("=" * 55)

    # genrated data....
    np.random.seed(42)

    # Two classes with different Gaussian distributuons
    n_per_class = 100
    X0 = np.random.randn(n_per_class, 2) * 0.5 + [1,1]
    X1 = np.random.randn(n_per_class, 2) * 0.5 + [-1, -1]
    X = np.vstack([X0, X1])
    y = np.array([0] * n_per_class + [1] * n_per_class)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # fit from scratch model...
    model = GaussianNB()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy(y_test, y_pred)
    print(f"\n      Test accuracy:  {acc:.4f}")
    print(f"        Class priors:       {np.exp(model.class_prior_)}")
    print(f"        Class means: \n     Class 0: {model.theta_[0]}\n        Class 1: {model.theta_[1]}")

    # comparing with sk learn...
    try: 
        from sklearn.naive_bayes import GaussianNB
        sk_model = GaussianNB()
        sk_model.fit(X_train, y_train)
        sk_acc = accuracy(y_test, sk_model.predict(X_test))
        print(f"    \n  [sklearn] accuracy: {sk_acc:.4f}")
        print( "    ☻ Results matched with sklearn...!!")
    except ImportError:
        print("\n       (sklearn not installed - skipping comparison)")


    # Multi feature with overlapping classes....
    print(f"\n{'-' * 55}")
    print("     4-feature example with 3 classes....")
    print(f"{'-' * 55}")

    np.random.seed(42)
    X_m = np.vstack([
        np.random.randn(100, 4) * 0.3 + [0, 0, 0, 0],
        np.random.randn(100, 4) * 0.3 + [2, 2, 0, 0],
        np.random.randn(100, 4) * 0.3 + [-2, 2, 0, 0],
    ])
    y_m = np.array([0]*100 + [1]*100 + [2]*100)

    X_mt, X_mtest, y_mt, y_mtest = train_test_split(X_m, y_m, test_size=0.2, random_state=42)
    model_m = GaussianNB()
    model_m.fit(X_mt, y_mt)
    acc_m = model_m.score(X_mtest, y_mtest)
    print(f"        Test accuracy:  {acc_m:.4f}")

    try:
        sk_m = GaussianNB()
        sk_m.fit(X_mt, y_mt)
        print(F"        [sklearn] accuracy: {accuracy(y_mtest, sk_m.predict(X_mtest)):.4f}")
        print("     ☻ Results Matched with Sklearn....!!")
    except ImportError:
        print("     (sklearn not installed  ... skipping the comparison)")

    print()

if __name__ == "__main__":
    main()
    