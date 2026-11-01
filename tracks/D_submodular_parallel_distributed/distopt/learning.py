"""Distributed and multi-task optimization.

FedAvg (McMahan et al. 2017) vs centralized training
  Model: softmax (multinomial logistic) regression on scikit-learn's 8x8 digits
  (10 classes, 64 features). K clients each hold a shard of the training data.
  Each communication round: every client starts from the global model, runs E
  local epochs of mini-batch SGD on its own data, sends its model back; the
  server averages the models weighted by client data size.
  Shards are IID (random) or non-IID (each client holds only 2 digit classes).
  Communication = rounds x clients x 2 model transfers.
  Centralized baseline: the same SGD on all data together, same number of epochs.

Multi-task learning: one shared layer, two task heads
  h = tanh(X W);  task k prediction = h v_k;  loss_k = mean squared error.
  Task 2's targets are 30x larger, so with equal weights its gradient dominates
  the shared layer. Methods for combining the two gradients on the shared W:
    equal    : g = g1 + g2
    MGDA     : g = alpha g1 + (1-alpha) g2 with alpha minimising ||g|| (closed form
               for two tasks); a common descent direction (Desideri 2012, Sener & Koltun 2018)
    GradNorm : learn weights w_k so each task's weighted gradient norm on W tracks
               the average norm scaled by its relative training rate r_k^alpha
               (Chen et al. 2018); weights renormalised to sum to 2.
"""
from __future__ import annotations

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


# ------------------------------------------------------------------ softmax regression
def softmax(Z):
    Z = Z - Z.max(axis=1, keepdims=True)
    P = np.exp(Z)
    return P / P.sum(axis=1, keepdims=True)


def loss_grad(W, X, Y, lam=1e-4):
    P = softmax(X @ W)
    n = len(Y)
    loss = -np.mean(np.log(P[np.arange(n), Y] + 1e-12)) + 0.5 * lam * np.sum(W * W)
    P[np.arange(n), Y] -= 1
    return loss, X.T @ P / n + lam * W


def accuracy(W, X, Y):
    return float(np.mean((X @ W).argmax(axis=1) == Y))


def digits_data(seed=0):
    d = load_digits()
    X = d.data / 16.0
    X = np.hstack([X, np.ones((len(X), 1))])
    return train_test_split(X, d.target, test_size=0.3, random_state=seed, stratify=d.target)


def split_clients(Y, k, iid, seed=0):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(Y))
    if iid:
        rng.shuffle(idx)
        return np.array_split(idx, k)
    # non-IID: sort by label, cut into 2k shards, give each client 2 shards (~2 classes)
    order = idx[np.argsort(Y, kind="stable")]
    shards = np.array_split(order, 2 * k)
    perm = rng.permutation(2 * k)
    return [np.concatenate([shards[perm[2 * c]], shards[perm[2 * c + 1]]]) for c in range(k)]


def local_sgd(W, X, Y, epochs, lr, batch, rng):
    W = W.copy()
    n = len(Y)
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch):
            b = order[s:s + batch]
            _, g = loss_grad(W, X[b], Y[b])
            W -= lr * g
    return W


def fedavg(Xtr, Ytr, Xte, Yte, clients, rounds, local_epochs, lr=0.5, batch=16, seed=0):
    rng = np.random.default_rng(seed)
    W = np.zeros((Xtr.shape[1], 10))
    sizes = np.array([len(c) for c in clients], dtype=float)
    hist = []
    for r in range(rounds):
        models = [local_sgd(W, Xtr[c], Ytr[c], local_epochs, lr, batch, rng) for c in clients]
        W = sum(m * s for m, s in zip(models, sizes)) / sizes.sum()
        hist.append(dict(round=r + 1, test_acc=accuracy(W, Xte, Yte), train_loss=loss_grad(W, Xtr, Ytr)[0]))
    return W, hist


def centralized(Xtr, Ytr, Xte, Yte, epochs, lr=0.5, batch=16, seed=0):
    rng = np.random.default_rng(seed)
    W = np.zeros((Xtr.shape[1], 10))
    hist = []
    for e in range(epochs):
        W = local_sgd(W, Xtr, Ytr, 1, lr, batch, rng)
        hist.append(dict(epoch=e + 1, test_acc=accuracy(W, Xte, Yte), train_loss=loss_grad(W, Xtr, Ytr)[0]))
    return W, hist


# ------------------------------------------------------------------ multi-task
def two_task_data(n=1200, d=8, seed=0, scale2=30.0):
    """One draw of the two target functions, split 50/25/25 into train/validation/test
    (an earlier version drew the test set with a different seed, i.e. from different
    target functions - failure log F12)."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    B = rng.standard_normal((d, 2))
    y1 = np.sin(X @ B[:, 0]) + 0.05 * rng.standard_normal(n)              # values in [-1, 1]
    y2 = scale2 * (np.tanh(X @ B[:, 1] * 0.7) + 0.05 * rng.standard_normal(n))   # 30x larger
    Y = np.stack([y1, y2], 1)
    a, b = n // 2, 3 * n // 4
    return (X[:a], Y[:a]), (X[a:b], Y[a:b]), (X[b:], Y[b:])


class TwoTaskNet:
    def __init__(self, d, hidden=16, seed=0):
        rng = np.random.default_rng(seed)
        self.W = rng.standard_normal((d, hidden)) / np.sqrt(d)
        self.V = rng.standard_normal((hidden, 2)) / np.sqrt(hidden)

    def forward(self, X):
        H = np.tanh(X @ self.W)
        return H, H @ self.V

    def task_losses_grads(self, X, Y):
        """Per-task loss, gradient w.r.t. the SHARED W, and w.r.t. each head."""
        H, P = self.forward(X)
        n = len(X)
        out = []
        for k in range(2):
            r = P[:, k] - Y[:, k]
            loss = float(np.mean(r * r))
            dP = 2 * r / n
            gv = H.T @ dP
            gH = np.outer(dP, self.V[:, k])
            gW = X.T @ (gH * (1 - H * H))
            out.append((loss, gW, gv))
        return out


def mgda_alpha(g1, g2):
    """argmin_{a in [0,1]} ||a g1 + (1-a) g2||^2 (closed form)."""
    d = g1 - g2
    denom = float(np.sum(d * d))
    if denom < 1e-18:
        return 0.5
    return float(np.clip(np.sum((g2 - g1) * g2) / denom, 0.0, 1.0))


def normalized_mse(net, X, Y, var):
    _, P = net.forward(X)
    return [float(np.mean((P[:, k] - Y[:, k]) ** 2) / var[k]) for k in range(2)]


def train_multitask(method, train, test, steps=3000, lr=0.01, seed=0, gn_alpha=1.5, gn_lr=0.025):
    """Shared-layer update by `method`; each head is always updated by its own task's gradient.
    Returns the net, a history of normalised test MSE (MSE / target variance), and MGDA alphas."""
    X, Y = train
    var = Y.var(axis=0)
    net = TwoTaskNet(X.shape[1], seed=seed)
    w = np.ones(2)                                   # task weights (GradNorm learns them)
    L0 = None
    hist, alphas = [], []
    for t in range(steps):
        (l1, gW1, gv1), (l2, gW2, gv2) = net.task_losses_grads(X, Y)
        if L0 is None:
            L0 = np.array([l1, l2])
        gV = np.stack([gv1, gv2], 1)
        if method == "equal":
            gW = gW1 + gW2
        elif method == "mgda":
            # min-norm point of the convex hull of the (normalised) task gradients:
            # a direction that decreases both losses whenever one exists. For two unit
            # vectors the min-norm point is always the midpoint, so alpha = 0.5 exactly and
            # this reduces to 'average the normalised gradients' (see README).
            n1, n2 = np.linalg.norm(gW1) + 1e-12, np.linalg.norm(gW2) + 1e-12
            a = mgda_alpha(gW1 / n1, gW2 / n2)
            alphas.append(a)
            gW = a * gW1 / n1 + (1 - a) * gW2 / n2
        elif method == "mgda_raw":
            # textbook MGDA on the raw gradients (no normalisation)
            a = mgda_alpha(gW1, gW2)
            alphas.append(a)
            gW = a * gW1 + (1 - a) * gW2
        elif method == "gradnorm":
            raw = np.array([np.linalg.norm(gW1), np.linalg.norm(gW2)])
            G = w * raw                                      # weighted gradient norms
            ratio = np.array([l1, l2]) / L0
            target = G.mean() * (ratio / ratio.mean()) ** gn_alpha
            # gradient of sum_k |G_k - target_k| w.r.t. w_k (target held constant)
            w = w - gn_lr * np.sign(G - target) * raw / (raw.mean() + 1e-12)
            w = np.maximum(w, 1e-4)
            w = 2 * w / w.sum()
            gW = w[0] * gW1 + w[1] * gW2      # heads keep their own unweighted gradients
        else:
            raise ValueError(method)
        net.W -= lr * gW
        net.V -= lr * gV
        if not np.all(np.isfinite(net.W)):
            break                                     # diverged (too large a step)
        if t % 100 == 0 or t == steps - 1:
            te = normalized_mse(net, *test, var)
            hist.append(dict(step=t, train1=l1 / var[0], train2=l2 / var[1], test1=te[0], test2=te[1],
                             w1=w[0], w2=w[1],
                             cos_g1_g2=float(np.sum(gW1 * gW2) / (np.linalg.norm(gW1) * np.linalg.norm(gW2) + 1e-12))))
    return net, hist, alphas
