"""A two-layer neural network with HAND-WRITTEN backpropagation.

Backpropagation and the optimizer are different things:
  * backprop computes the gradient of the loss with respect to every parameter,
    by applying the chain rule backwards through the network (this file);
  * the optimizer (optimizers.py) decides how to use that gradient to update
    the parameters. Any optimizer can be paired with the same backprop.

Network:  h = tanh(X W1 + b1);  logits = h W2 + b2;  p = softmax(logits)
Loss:     mean cross-entropy + (lam/2)(||W1||^2 + ||W2||^2)   (L2 regularization)
All parameters are packed into one flat vector so the generic optimizers apply.
"""
from __future__ import annotations

import numpy as np


class MLP:
    def __init__(self, n_in, n_hidden, n_out, lam=1e-4, seed=0):
        self.shapes = [(n_in, n_hidden), (n_hidden,), (n_hidden, n_out), (n_out,)]
        self.sizes = [int(np.prod(s)) for s in self.shapes]
        self.lam = lam
        rng = np.random.default_rng(seed)
        W1 = rng.standard_normal((n_in, n_hidden)) / np.sqrt(n_in)
        W2 = rng.standard_normal((n_hidden, n_out)) / np.sqrt(n_hidden)
        self.x0 = self.pack(W1, np.zeros(n_hidden), W2, np.zeros(n_out))

    @property
    def n_params(self):
        return sum(self.sizes)

    def pack(self, *arrs):
        return np.concatenate([a.ravel() for a in arrs])

    def unpack(self, theta):
        out, k = [], 0
        for s, n in zip(self.shapes, self.sizes):
            out.append(theta[k:k + n].reshape(s))
            k += n
        return out

    def forward(self, theta, X):
        W1, b1, W2, b2 = self.unpack(theta)
        a1 = X @ W1 + b1
        h = np.tanh(a1)
        z = h @ W2 + b2
        z = z - z.max(axis=1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(axis=1, keepdims=True)
        return h, p

    def loss(self, theta, X, Y):
        W1, _, W2, _ = self.unpack(theta)
        _, p = self.forward(theta, X)
        ce = -np.mean(np.log(p[np.arange(len(Y)), Y] + 1e-300))
        return ce + 0.5 * self.lam * (np.sum(W1 ** 2) + np.sum(W2 ** 2))

    def backprop(self, theta, X, Y):
        """Gradient of `loss` by the chain rule, layer by layer, output to input."""
        W1, b1, W2, b2 = self.unpack(theta)
        n = len(Y)
        h, p = self.forward(theta, X)
        dz = p.copy()                         # d loss / d logits = (softmax - onehot) / n
        dz[np.arange(n), Y] -= 1
        dz /= n
        dW2 = h.T @ dz + self.lam * W2        # logits = h W2 + b2
        db2 = dz.sum(axis=0)
        dh = dz @ W2.T                        # back through the output layer
        da1 = dh * (1 - h * h)                # tanh'(a) = 1 - tanh(a)^2
        dW1 = X.T @ da1 + self.lam * W1
        db1 = da1.sum(axis=0)
        return self.pack(dW1, db1, dW2, db2)

    def accuracy(self, theta, X, Y):
        _, p = self.forward(theta, X)
        return float(np.mean(p.argmax(axis=1) == Y))


class MLPObjective:
    """Adapter so the generic first-order runner can train the MLP (full or mini-batch)."""

    def __init__(self, net: MLP, X, Y):
        self.net, self.X, self.Y = net, X, Y
        self.n = len(Y)
        self.x0 = net.x0
        self.name = "MLP"

    def f(self, th, idx=None):
        X, Y = (self.X, self.Y) if idx is None else (self.X[idx], self.Y[idx])
        return self.net.loss(th, X, Y)

    def grad(self, th, idx=None):
        X, Y = (self.X, self.Y) if idx is None else (self.X[idx], self.Y[idx])
        return self.net.backprop(th, X, Y)
