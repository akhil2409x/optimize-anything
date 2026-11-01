"""Quantum machine learning vs classical models on the same data.

Datasets (binary classification, features scaled to [0, pi] for angle encoding):
  moons  : sklearn make_moons, 2 features, noise 0.2 - non-linear boundary
  cancer : breast-cancer data reduced to 4 features with PCA - real data

Quantum models (4 qubits for cancer, 2 for moons, simulated with default.qubit):
  vqc    : variational quantum classifier. Data re-uploading: each layer encodes
           the features with RY angles, then a trainable StronglyEntanglingLayers
           block; output <Z_0> in [-1, 1] -> class. Trained with Adam on a square
           loss. Gradients on the simulator use backprop for speed; the reported
           "circuit evaluations" are what hardware would need with the
           parameter-shift rule: per sample per step, 1 + 2 * (#trainable gates).
  qkernel: fidelity quantum kernel k(x, x') = |<phi(x')|phi(x)>|^2 with an IQP-style
           (ZZ) feature map, plugged into a classical SVM. Needs one circuit per
           pair of points: n(n-1)/2 for training plus n_test * n_train for testing.

Classical models: logistic regression, RBF-kernel SVM, small MLP.

All models see identical train/test splits (stratified, seeded); scores are test
accuracy. Training time is wall-clock on this machine (simulation for the
quantum models, so it measures the cost of *simulating*, not of a real QPU).
"""
from __future__ import annotations

import time

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from sklearn.datasets import load_breast_cancer, make_moons
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC


# ------------------------------------------------------------------ data
def load(name: str, n_samples: int = 200, seed: int = 0, test_size: float = 0.3):
    if name == "moons":
        X, y = make_moons(n_samples=n_samples, noise=0.2, random_state=seed)
    elif name == "cancer":
        d = load_breast_cancer()
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(d.target), size=n_samples, replace=False)
        X, y = d.data[idx], d.target[idx]
    else:
        raise ValueError(name)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)
    if name == "cancer":                 # fit preprocessing on training data only
        sc = StandardScaler().fit(Xtr)
        pca = PCA(4, random_state=seed).fit(sc.transform(Xtr))
        Xtr, Xte = pca.transform(sc.transform(Xtr)), pca.transform(sc.transform(Xte))
    mm = MinMaxScaler((0, np.pi)).fit(Xtr)
    Xtr, Xte = mm.transform(Xtr), np.clip(mm.transform(Xte), 0, np.pi)
    return Xtr, Xte, ytr, yte


# ------------------------------------------------------------------ variational classifier
class VQC:
    def __init__(self, n_features: int, layers: int = 3, steps: int = 60, batch: int = 16,
                 lr: float = 0.05, seed: int = 0):
        self.n, self.layers, self.steps, self.batch, self.lr, self.seed = n_features, layers, steps, batch, lr, seed
        dev = qml.device("default.qubit", wires=self.n)

        @qml.qnode(dev, diff_method="backprop")
        def circuit(weights, x):
            for l in range(self.layers):
                for i in range(self.n):              # data re-uploading
                    qml.RY(x[i], wires=i)
                qml.StronglyEntanglingLayers(weights[l:l + 1], wires=range(self.n))
            return qml.expval(qml.Z(0))
        self.circuit = circuit

    @property
    def n_params(self):
        return self.layers * self.n * 3 + 1        # rotation angles + bias

    @property
    def trainable_gates(self):
        # each qml.Rot = RZ RY RZ -> 3 shiftable gates per qubit per layer
        return self.layers * self.n * 3

    def _forward(self, w, b, X):
        return pnp.stack([self.circuit(w, x) for x in X]) + b

    def fit(self, X, y):
        rng = np.random.default_rng(self.seed)
        w = pnp.array(rng.uniform(0, 2 * np.pi, (self.layers, self.n, 3)), requires_grad=True)
        b = pnp.array(0.0, requires_grad=True)
        t = pnp.array(2 * y - 1.0, requires_grad=False)     # labels in {-1, +1}
        opt = qml.AdamOptimizer(self.lr)
        self.circuit_evals = 0
        t0 = time.perf_counter()
        for _ in range(self.steps):
            idx = rng.choice(len(X), size=min(self.batch, len(X)), replace=False)
            Xb, tb = X[idx], t[idx]

            def loss(w, b):
                return pnp.mean((self._forward(w, b, Xb) - tb) ** 2)
            w, b = opt.step(loss, w, b)
            # hardware-equivalent cost of one parameter-shift step on this batch
            self.circuit_evals += len(idx) * (1 + 2 * self.trainable_gates)
        self.train_seconds = time.perf_counter() - t0
        self.w, self.b = w, b
        return self

    def predict(self, X):
        out = self._forward(self.w, self.b, X)
        self.circuit_evals_predict = len(X)
        return (np.array(out) > 0).astype(int)


# ------------------------------------------------------------------ quantum kernel
class QuantumKernelSVM:
    def __init__(self, n_features: int, reps: int = 2, C: float = 1.0):
        self.n, self.reps, self.C = n_features, reps, C
        dev = qml.device("default.qubit", wires=self.n)

        def feature_map(x):
            for _ in range(self.reps):
                for i in range(self.n):
                    qml.Hadamard(i)
                    qml.RZ(2 * x[i], wires=i)
                for i in range(self.n):
                    for j in range(i + 1, self.n):
                        qml.CNOT([i, j])
                        qml.RZ(2 * (np.pi - x[i]) * (np.pi - x[j]) / np.pi, wires=j)
                        qml.CNOT([i, j])

        @qml.qnode(dev)
        def overlap(x1, x2):
            feature_map(x1)
            qml.adjoint(feature_map)(x2)
            return qml.probs(wires=range(self.n))
        self.overlap = overlap

    def k(self, a, b):
        return float(self.overlap(a, b)[0])     # P(all zeros) = |<phi(b)|phi(a)>|^2

    def gram(self, A, B, symmetric=False):
        G = np.zeros((len(A), len(B)))
        evals = 0
        for i, a in enumerate(A):
            for j, b in enumerate(B):
                if symmetric and j < i:
                    G[i, j] = G[j, i]
                elif symmetric and j == i:
                    G[i, j] = 1.0                 # k(x, x) = 1, no circuit needed
                else:
                    G[i, j] = self.k(a, b)
                    evals += 1
        return G, evals

    def fit(self, X, y):
        t0 = time.perf_counter()
        G, self.circuit_evals = self.gram(X, X, symmetric=True)
        self.svm = SVC(kernel="precomputed", C=self.C).fit(G, y)
        self.train_seconds = time.perf_counter() - t0
        self.Xtr = X
        self.n_params = int(len(self.svm.support_) + 1)   # dual coefficients + intercept
        return self

    def predict(self, X):
        G, self.circuit_evals_predict = self.gram(X, self.Xtr)
        return self.svm.predict(G)


# ------------------------------------------------------------------ classical
def classical_models(seed: int = 0):
    return {
        "logistic regression": LogisticRegression(max_iter=2000),
        "SVM (RBF kernel)": SVC(kernel="rbf", C=1.0, gamma="scale"),
        "MLP (1 hidden layer, 8 units)": MLPClassifier((8,), max_iter=3000, random_state=seed),
    }


def classical_param_count(name, model):
    if name.startswith("logistic"):
        return int(model.coef_.size + model.intercept_.size)
    if name.startswith("SVM"):
        return int(len(model.support_) + 1)
    return int(sum(w.size for w in model.coefs_) + sum(b.size for b in model.intercepts_))
