"""Objective functions with hand-derived gradients and Hessians.

Every gradient and Hessian here is checked against autograd in the tests.

Landscapes
  Quadratic(kappa)     f = 1/2 x^T A x, eigenvalues spaced from 1 to kappa:
                       strongly convex, condition number kappa, minimum 0 at x = 0.
  QuarticBowl          f = sum x_i^4: convex but NOT strongly convex (Hessian
                       vanishes at the minimum), so GD slows to a sublinear rate.
  Rosenbrock           f = (1-x)^2 + 100 (y-x^2)^2: non-convex curved valley,
                       single minimum at (1, 1).
  Saddle               f = x^2 - y^2 + y^4 / 4: saddle point at (0, 0), two global
                       minima at (0, +-sqrt 2) with f = -1.
  SixHumpCamel         two global minima (f = -1.0316) plus four local minima.

ML objective (finite sum, so stochastic methods apply)
  LogisticRegression   L(w) = mean_i log(1 + exp(-y_i x_i^T w)) + (lam/2)||w||^2,
                       labels y in {-1, +1}. Convex; strongly convex when lam > 0.
                       Parameters = w, loss = cross-entropy, regularization = L2.

Nonlinear least squares (for Gauss-Newton / Levenberg-Marquardt)
  CurveFit             residuals r_i(theta) = a exp(-b t_i) + c - y_i.
"""
from __future__ import annotations

import numpy as np


class Objective:
    name = "objective"
    x0 = None
    f_star = None

    def f(self, x): ...
    def grad(self, x): ...
    def hess(self, x): ...

    def hvp(self, x, v):
        return self.hess(x) @ v


# ------------------------------------------------------------------ convex
class Quadratic(Objective):
    def __init__(self, kappa=100.0, dim=20, seed=0):
        rng = np.random.default_rng(seed)
        Q, _ = np.linalg.qr(rng.standard_normal((dim, dim)))
        eig = np.linspace(1.0, kappa, dim)
        self.A = Q @ np.diag(eig) @ Q.T
        self.L, self.mu, self.kappa = kappa, 1.0, kappa
        self.x0 = rng.standard_normal(dim) * 3
        self.f_star = 0.0
        self.name = f"quadratic kappa={kappa:g}"

    def f(self, x):
        return 0.5 * x @ self.A @ x

    def grad(self, x):
        return self.A @ x

    def hess(self, x):
        return self.A


class QuarticBowl(Objective):
    name = "quartic sum x^4 (convex, not strongly convex)"

    def __init__(self, dim=10):
        self.x0 = np.linspace(0.5, 1.5, dim)
        self.f_star = 0.0

    def f(self, x):
        return np.sum(x ** 4)

    def grad(self, x):
        return 4 * x ** 3

    def hess(self, x):
        return np.diag(12 * x ** 2)


# ------------------------------------------------------------------ non-convex
class Rosenbrock(Objective):
    name = "Rosenbrock"
    x0 = np.array([-1.2, 1.0])
    f_star = 0.0
    x_star = np.array([1.0, 1.0])

    def f(self, z):
        x, y = z
        return (1 - x) ** 2 + 100 * (y - x * x) ** 2

    def grad(self, z):
        x, y = z
        return np.array([-2 * (1 - x) - 400 * x * (y - x * x), 200 * (y - x * x)])

    def hess(self, z):
        x, y = z
        return np.array([[2 - 400 * (y - 3 * x * x), -400 * x], [-400 * x, 200.0]])


class Saddle(Objective):
    name = "saddle x^2 - y^2 + y^4/4"
    x0 = np.array([1.0, 1e-6])            # almost exactly on the saddle's stable manifold
    f_star = -1.0
    minima = [np.array([0.0, np.sqrt(2)]), np.array([0.0, -np.sqrt(2)])]

    def f(self, z):
        x, y = z
        return x * x - y * y + y ** 4 / 4

    def grad(self, z):
        x, y = z
        return np.array([2 * x, -2 * y + y ** 3])

    def hess(self, z):
        x, y = z
        return np.array([[2.0, 0.0], [0.0, -2 + 3 * y * y]])


class SixHumpCamel(Objective):
    name = "six-hump camel"
    f_star = -1.031628453

    def f(self, z):
        x, y = z
        return (4 - 2.1 * x ** 2 + x ** 4 / 3) * x ** 2 + x * y + (-4 + 4 * y ** 2) * y ** 2

    def grad(self, z):
        x, y = z
        return np.array([8 * x - 8.4 * x ** 3 + 2 * x ** 5 + y, x - 8 * y + 16 * y ** 3])

    def hess(self, z):
        x, y = z
        return np.array([[8 - 25.2 * x ** 2 + 10 * x ** 4, 1.0], [1.0, -8 + 48 * y ** 2]])


# ------------------------------------------------------------------ ML: logistic regression
def make_classification(n=600, d=10, seed=0, noise=1.0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    X[:, 1:] *= np.linspace(1, 10, d - 1)       # features on different scales: ill-conditioned
    w_true = rng.standard_normal(d)
    logits = X @ w_true / np.linalg.norm(X @ w_true) * np.sqrt(n) * 2
    y = np.where(logits + noise * rng.standard_normal(n) > 0, 1.0, -1.0)
    X = np.hstack([X, np.ones((n, 1))])          # bias column
    return X, y


class LogisticRegression(Objective):
    def __init__(self, X, y, lam=1e-3):
        self.X, self.y, self.lam = X, y, lam
        self.n, self.d = X.shape
        self.x0 = np.zeros(self.d)
        self.name = f"logistic regression (lambda={lam:g})"
        self.f_star = None

    @staticmethod
    def _sigmoid(z):
        return 0.5 * (1 + np.tanh(0.5 * z))       # numerically stable

    def f(self, w, idx=None):
        X, y = (self.X, self.y) if idx is None else (self.X[idx], self.y[idx])
        m = y * (X @ w)
        return np.mean(np.logaddexp(0.0, -m)) + 0.5 * self.lam * w @ w

    def grad(self, w, idx=None):
        X, y = (self.X, self.y) if idx is None else (self.X[idx], self.y[idx])
        m = y * (X @ w)
        coef = -y * self._sigmoid(-m)              # d/dm log(1+e^-m) = -sigmoid(-m)
        return X.T @ coef / len(y) + self.lam * w

    def hess(self, w):
        p = self._sigmoid(self.X @ w)
        s = p * (1 - p)
        return (self.X * s[:, None]).T @ self.X / self.n + self.lam * np.eye(self.d)

    def hvp(self, w, v):
        """Hessian-vector product without forming the Hessian: O(n d) instead of O(n d^2)."""
        p = self._sigmoid(self.X @ w)
        return self.X.T @ (p * (1 - p) * (self.X @ v)) / self.n + self.lam * v

    def fisher(self, w):
        """Empirical Fisher: mean of per-example gradient outer products (+ lam I)."""
        m = self.y * (self.X @ w)
        G = (-self.y * self._sigmoid(-m))[:, None] * self.X
        return G.T @ G / self.n + self.lam * np.eye(self.d)

    def accuracy(self, w, X=None, y=None):
        X = self.X if X is None else X
        y = self.y if y is None else y
        return float(np.mean(np.sign(X @ w) == y))


# ------------------------------------------------------------------ nonlinear least squares
class CurveFit(Objective):
    """y = a * exp(-b t) + c, fitted by least squares: f = 1/2 ||r(theta)||^2."""

    def __init__(self, n=80, seed=0, noise=0.05):
        rng = np.random.default_rng(seed)
        self.t = np.linspace(0, 5, n)
        self.theta_true = np.array([2.5, 1.3, 0.5])
        self.y = self.model(self.theta_true) + noise * rng.standard_normal(n)
        self.x0 = np.array([1.0, 0.2, 0.0])
        self.name = "curve fit a exp(-b t) + c"
        self.f_star = None

    def model(self, th):
        a, b, c = th
        return a * np.exp(-b * self.t) + c

    def residuals(self, th):
        return self.model(th) - self.y

    def jac(self, th):
        a, b, c = th
        e = np.exp(-b * self.t)
        return np.stack([e, -a * self.t * e, np.ones_like(self.t)], axis=1)

    def f(self, th):
        r = self.residuals(th)
        return 0.5 * r @ r

    def grad(self, th):
        return self.jac(th).T @ self.residuals(th)

    def hess(self, th):
        a, b, c = th
        J, r = self.jac(th), self.residuals(th)
        e = np.exp(-b * self.t)
        # second derivatives of the model: d2/da db = -t e, d2/db2 = a t^2 e
        S = np.zeros((3, 3))
        S[0, 1] = S[1, 0] = np.sum(r * (-self.t * e))
        S[1, 1] = np.sum(r * (a * self.t ** 2 * e))
        return J.T @ J + S                          # Gauss-Newton drops S
