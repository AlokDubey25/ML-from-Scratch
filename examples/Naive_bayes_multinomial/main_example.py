import numpy as np 
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from src.algos.classification.naive_bayes_multinomial import MultinomialNB
from src.utils.metrics import accuracy
from src.utils.preprocessing import train_test_split

def main():
    print("=" * 55)
    print("     Multinomial Naive Bayes")
    print("=" * 55)

    np.random.seed(42)

    # Class 0: word indices 0,1 are frequent
    # Class 1: word indices 2,3 are frequent
    # Class 2: word index 4 is frequent

    def generate_docs(n_docs: int, word_weights: np.ndarray, doc_length: int = 20) -> np.ndarray:
        word_weights = word_weights / word_weights.sum()
        return np.random.multinomial(doc_length, word_weights, size=n_docs)

    n_per_class = 50
    X0 = generate_docs(n_per_class, np.array([4, 3, 1, 1, 1]))
    X1 = generate_docs(n_per_class, np.array([1, 1, 4, 3, 1]))
    X2 = generate_docs(n_per_class, np.array([1, 1, 1, 1, 5]))

    X = np.vstack([X0, X1, X2])
    y = np.array([0]*n_per_class + [1]*n_per_class + [2]*n_per_class)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # fit model here
    model = MultinomialNB(alpha=1.0)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy(y_test, y_pred)
    print(f"\n      Test accuracy:      {acc:.4f}")
    print(f"        Feature log-prob (first class): \n      {np.exp(model.feature_log_prob_[0])}")

    # COmparing with sk learn

    try:
        from sklearn.naive_bayes import MultinomialNB
        sk_model = MultinomialNB(alpha=1.0)
        sk_model.fit(X_train, y_train)
        sk_acc = accuracy(y_test, sk_model.predict(X_test))
        print(f"\n      [sklearn] accuracy:  {sk_acc:.4f}")
        print("     ☻ Results match skllearn")
    except ImportError:
        print("\n       (sklearn not installed - skipping comparison)")


    # Binary case with fewer features
    print(f"\n{'-' * 55}")
    print("     Binary Classification of same")
    print(f"{'-' * 55}")

    Xb0 = generate_docs(30, np.array([5, 3, 1, 1]))
    Xb1 = generate_docs(30, np.array([1, 1, 5, 3]))
    Xb   = np.vstack([Xb0, Xb1])
    yb    = np.array([0]*30 + [1]*30)

    Xb_tr, Xb_te, yb_tr, yb_te = train_test_split(Xb, yb, test_size=0.2, random_state=42)
    model_b = MultinomialNB(alpha=1.0)
    model_b.fit(Xb_tr, yb_tr)
    acc_b = model_b.score(Xb_te, yb_te)
    print(f"        Test accuracy: {acc_b:.4f}")

    try:
        sk_b = MultinomialNB(alpha=1.0)
        sk_b.fit(Xb_tr, yb_tr)
        print(f"        [sklearn] accuracy: {accuracy(yb_te, sk_b.predict(Xb_te)):.4f}")
        print(F"        ☻ Results match sklearn...!!")
    except ImportError:
        print("     (sklearn not installed - skipping comparison)")

    print()

if __name__ == "__main__":
    main()