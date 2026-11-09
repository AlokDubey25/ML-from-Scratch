<div align="center">

<h1>🧠 ML From Scratch</h1>

<h3>⚡ 17 Classic Machine Learning Algorithms, Built From First Principles ⚡</h3>

<p><i>Pure Python + NumPy. No scikit-learn. No black boxes.</i></p>

![Algorithms](https://img.shields.io/badge/ALGORITHMS-17-0A66C2?style=flat-square)
![Python](https://img.shields.io/badge/PYTHON-3.9+-3776AB?style=flat-square&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NUMPY-ONLY-013243?style=flat-square&logo=numpy&logoColor=white)
![Examples](https://img.shields.io/badge/EXAMPLES-17-brightgreen?style=flat-square)

![scikit-learn](https://img.shields.io/badge/SCIKIT--LEARN-NONE-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
![Layout](https://img.shields.io/badge/LAYOUT-src%2F-6f42c1?style=flat-square)
![Install](https://img.shields.io/badge/INSTALL-pip%20install%20--e%20.-informational?style=flat-square)
![License](https://img.shields.io/badge/LICENSE-MIT-green?style=flat-square)

[Core Mechanism](#-core-mechanism-what-it-does) •
[Architecture](#-system-architecture) •
[Algorithms](#-algorithms) •
[Install](#-installation) •
[Examples](#-running-the-examples) •
[Structure](#-project-structure) •
[Future Work](#-future-work)

</div>

---

## 🔩 Core Mechanism: What It Does

Every model in this repo follows the same three-stage pipeline, and every stage is hand-written:

1. 🟩 **Prepare**: raw data is scaled, split and encoded by `utils/preprocessing.py`. No library does this for you.
2. 🟧 **Learn**: the algorithm's own `fit` runs the math: a gradient step, a tree split, an eigendecomposition, a Bayes update. It builds on shared `utils/` helpers (distances, activations, losses).
3. 🟦 **Evaluate**: predictions are scored by `utils/metrics.py` (accuracy, precision, recall, F1, MSE, R²).

> 📌 **The Rule:** no algorithm is allowed to call a machine learning library. If a step exists between your data and a prediction, you can open the file and read it.

---

## 🎯 Why This Exists

Calling `model.fit(X, y)` teaches you an API. Writing `fit` yourself teaches you the algorithm.

This repository re-implements the core of classical machine learning using only NumPy, so every step between raw data and a trained model is visible: the gradients, the splits, the probabilities, the update rules.

- 🔬 **Transparent**: every algorithm is readable top to bottom, with the math next to the code
- 🧱 **Modular**: metrics, distances, activations, losses and preprocessing live in `src/utils/` and are shared by every algorithm
- 📦 **Installable**: a real Python package (`pip install -e .`), not a folder of notebooks
- ▶️ **Runnable**: every algorithm ships with its own example in `examples/`

---

## 📚 Algorithms

| # | Family | Algorithm | Core idea |
|---|---|---|---|
| 1 | **Regression** | Linear Regression (Normal Equation) | Closed form: $\hat{w} = (X^\top X)^{-1} X^\top y$ |
| 2 | | Linear Regression (Gradient Descent) | Iterative minimization of MSE |
| 3 | | Ridge Regression | L2 penalty shrinks weights |
| 4 | | Lasso Regression | L1 penalty induces sparsity |
| 5 | | Decision Tree Regressor | Variance-reduction splits |
| 6 | **Classification** | Logistic Regression | Sigmoid + cross-entropy, gradient descent |
| 7 | | K-Nearest Neighbors | Majority vote over the *k* closest points |
| 8 | | Gaussian Naive Bayes | Bayes' rule with per-feature Gaussians |
| 9 | | Multinomial Naive Bayes | Bayes' rule over count features |
| 10 | | Decision Tree Classifier | Recursive Gini / entropy splits |
| 11 | **Ensembles** | Random Forest | Bagging + random feature subsets |
| 12 | | Gradient Boosting | Sequential fitting to residuals |
| 13 | **Clustering** | K-Means | Lloyd's algorithm |
| 14 | | K-Means++ | Spread-out centroid initialization |
| 15 | **Dimensionality Reduction** | PCA | Eigendecomposition of the covariance matrix |
| 16 | **Neural Networks** | Single-Layer Perceptron | Rosenblatt's perceptron rule |
| 17 | | Multi-Layer Perceptron | Backpropagation from scratch |

### Key equations

**Gradient descent update**

$$w \leftarrow w - \alpha \, \nabla_w J(w)$$

**Ridge vs. Lasso**

$$J_{\text{ridge}}(w) = \text{MSE}(w) + \lambda \lVert w \rVert_2^2 \qquad J_{\text{lasso}}(w) = \text{MSE}(w) + \lambda \lVert w \rVert_1$$

**Logistic regression loss**

$$J(w) = -\frac{1}{n}\sum_{i=1}^{n}\Big[y_i \log \hat{y}_i + (1-y_i)\log(1-\hat{y}_i)\Big], \quad \hat{y} = \sigma(w^\top x)$$

**Principal components**

$$\Sigma v_k = \lambda_k v_k, \qquad Z = X_{\text{centered}} \, V_k$$

---

## ⚙️ Installation

```bash
git clone https://github.com/AlokDubey25/ml-from-scratch.git
cd ml-from-scratch

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -e .
```

---

## ▶️ Running the Examples

```bash
# Regression
python examples/linear_regression_normal_equation/main_example.py
python examples/ridge/main_example.py

# Classification
python examples/logistic/main_example.py
python examples/KNN/main_example.py

# Ensembles, clustering, dimensionality reduction
python examples/random_forest/main_example.py
python examples/kmeans/main_example.py
python examples/pca/main_example.py

# Neural networks
python examples/mlp/main_example.py
```

---

## 🗂 Project Structure

```
ml-from-scratch/
│
├── pyproject.toml                       # pip install -e .
├── README.md
├── LICENSE
├── .gitignore
│
├── examples/
│   ├── linear_regression_normal_equation/main_example.py
│   ├── linear_regression_gradient_descent/main_example.py
│   ├── ridge/main_example.py
│   ├── lasso/main_example.py
│   ├── decision_tree_regressor/main_example.py
│   ├── logistic/main_example.py
│   ├── KNN/main_example.py
│   ├── Naive_bayes_gaussian/main_example.py
│   ├── Naive_bayes_multinomial/main_example.py
│   ├── decision_tree_classifier/main_example.py
│   ├── random_forest/main_example.py
│   ├── gradient_boosting/main_example.py
│   ├── kmeans/main_example.py
│   ├── kmeans_plus_plus/main_example.py
│   ├── pca/main_example.py
│   ├── perceptron/main_example.py
│   └── mlp/main_example.py
│
└── src/
    ├── utils/
    │   ├── activations.py
    │   ├── distances.py
    │   ├── losses.py
    │   ├── metrics.py
    │   └── preprocessing.py
    │
    └── algos/
        ├── regression/
        │   ├── linear_regression_normal_equation.py
        │   ├── linear_regression_gradient_descent.py
        │   ├── ridge.py
        │   ├── lasso.py
        │   └── decision_tree_regressor.py
        │
        ├── classification/
        │   ├── logistic_regression.py
        │   ├── knn.py
        │   ├── naive_bayes_gaussian.py
        │   ├── naive_bayes_multinomial.py
        │   └── decision_tree_classifier.py
        │
        ├── ensemble/
        │   ├── random_forest.py
        │   └── gradient_boosting.py
        │
        ├── clustering/
        │   ├── kmeans.py
        │   └── kmeans_plus_plus.py
        │
        ├── dimensionality_reduction/
        │   └── pca.py
        │
        └── neural_networks/
            ├── perceptron.py
            └── mlp.py
```

---

## 🧩 Design Principles

1. **NumPy only.** If it can be written with array operations, it is.
2. **One shared toolbox.** No algorithm re-implements a metric, distance or activation; they all import from `src/utils/`.
3. **Math first.** The code follows the derivation, not the other way around.
4. **Every algorithm is runnable.** Each one has a matching example in `examples/`.
5. **Layout mirrors the theory.** Algorithms are grouped by family, so the repo reads like a table of contents of classical ML.

---

## 🛣 Future Work

The implementations work. The next step is making them **comparable**, so the repo shows *why* algorithms behave differently, not just that they run.

### 🔬 Comparing models by modifying them

| Experiment | What gets modified | What it reveals |
|---|---|---|
| Normal Equation vs. Gradient Descent | Learning rate, iterations, feature scaling | Convergence behavior and when the closed form stops being practical |
| Ridge vs. Lasso | Regularization strength $\lambda$ | Weight shrinkage vs. sparsity; coefficient paths |
| KNN | Value of *k* and the distance metric (euclidean, manhattan, cosine) | Bias-variance trade-off; sensitivity to the metric |
| Decision Tree | Max depth, min samples per split, Gini vs. entropy | Where overfitting begins |
| Random Forest | Number of trees, feature subset size | Variance reduction from bagging |
| Gradient Boosting | Learning rate, number of estimators, tree depth | The shrinkage vs. rounds trade-off |
| K-Means vs. K-Means++ | Initialization strategy, number of restarts | Convergence speed and final inertia |
| MLP | Activation (sigmoid, tanh, ReLU), depth, width, learning rate | Vanishing gradients and capacity |
| Perceptron vs. Logistic Regression | Linearly separable vs. overlapping data | Why probabilistic outputs matter |

### 🧰 Making the framework better

- [ ] Benchmarks against scikit-learn on the same datasets
- [ ] A shared experiment runner that sweeps hyperparameters and logs metrics
- [ ] Plots for every experiment: decision boundaries, loss curves, coefficient paths, elbow plots
- [ ] Cross-validation and grid search built on top of `utils/`
- [ ] Better optimizers for the MLP and logistic regression: SGD with momentum, Adam
- [ ] Unit tests and CI with GitHub Actions

### 🧬 More algorithms

- [ ] Support Vector Machine
- [ ] Gaussian Mixture Models (EM)
- [ ] DBSCAN and hierarchical clustering
- [ ] Convolutional layers for the MLP

---

## 🤝 Contributing

Found a bug in the math or have a cleaner implementation? Open an issue or a pull request.

1. Fork the repo
2. Create a branch: `git checkout -b feature/your-feature`
3. Commit your changes and open a pull request

---

## 📄 License

Distributed under the MIT License. See [`LICENSE`](LICENSE) for details.

---

## 👤 Author

**Alok Dubey**
Integrated MSc Mathematics, SVNIT Surat
[GitHub](https://github.com/AlokDubey25)

<div align="center">

⭐ If this helped you understand how ML actually works, consider starring the repo.

</div>
