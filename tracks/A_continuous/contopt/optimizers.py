"""Optimizers, written from scratch in NumPy.

First-order (state + `step(x, gradfn)`; gradfn may be stochastic):
  GD / SGD / mini-batch SGD    x <- x - lr g                (the batch size decides which)
  Momentum (heavy ball)        v <- beta v + g;   x <- x - lr v
  Nesterov (NAG)               v <- beta v + g(x - lr beta v);   x <- x - lr v
  AdaGrad                      G <- G + g^2;      x <- x - lr g / (sqrt G + eps)
  RMSProp                      G <- rho G + (1-rho) g^2;   x <- x - lr g / (sqrt G + eps)
  Adam                         m, v moment estimates with bias correction

Second-order and curvature (full-batch, use obj.hess / obj.hvp / obj.fisher / obj.jac):
  newton           x <- x - H^-1 g                (pure; attracted to saddles!)
  newton_damped    Newton direction if H is positive definite, else -g; Armijo backtracking
  gauss_newton     x <- x - (J^T J)^-1 J^T r       (least squares: drops 2nd-order residual term)
  levenberg_marquardt  (J^T J + lambda I) step, lambda adapted
  lbfgs            low-rank inverse-Hessian approximation from the last m (s, y) pairs
  natural_gradient x <- x - lr F^-1 g, F = empirical Fisher
  diag_newton      x <- x - lr g / diag(H)         (diagonal curvature approximation)
  newton_cg        Hessian-free: solve H p = -g by conjugate gradient using only HVPs
"""
from __future__ import annotations

import time

import numpy as np


# ------------------------------------------------------------------ first-order
class GD:
    def __init__(self, lr):
        self.lr = lr

    def step(self, x, gradfn):
        return x - self.lr * gradfn(x)


class Momentum:
    def __init__(self, lr, beta=0.9):
        self.lr, self.beta, self.v = lr, beta, None

    def step(self, x, gradfn):
        g = gradfn(x)
        self.v = g if self.v is None else self.beta * self.v + g
        return x - self.lr * self.v


class Nesterov:
    def __init__(self, lr, beta=0.9):
        self.lr, self.beta, self.v = lr, beta, None

    def step(self, x, gradfn):
        if self.v is None:
            self.v = np.zeros_like(x)
        g = gradfn(x - self.lr * self.beta * self.v)      # gradient at the look-ahead point
        self.v = self.beta * self.v + g
        return x - self.lr * self.v


class AdaGrad:
    def __init__(self, lr, eps=1e-8):
        self.lr, self.eps, self.G = lr, eps, None

    def step(self, x, gradfn):
        g = gradfn(x)
        self.G = g * g if self.G is None else self.G + g * g
        return x - self.lr * g / (np.sqrt(self.G) + self.eps)


class RMSProp:
    def __init__(self, lr, rho=0.9, eps=1e-8):
        self.lr, self.rho, self.eps, self.G = lr, rho, eps, None

    def step(self, x, gradfn):
        g = gradfn(x)
        self.G = (1 - self.rho) * g * g if self.G is None else self.rho * self.G + (1 - self.rho) * g * g
        return x - self.lr * g / (np.sqrt(self.G) + self.eps)


class Adam:
    def __init__(self, lr, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = self.v = None
        self.t = 0

    def step(self, x, gradfn):
        g = gradfn(x)
        if self.m is None:
            self.m, self.v = np.zeros_like(x), np.zeros_like(x)
        self.t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * g
        self.v = self.b2 * self.v + (1 - self.b2) * g * g
        mhat = self.m / (1 - self.b1 ** self.t)
        vhat = self.v / (1 - self.b2 ** self.t)
        return x - self.lr * mhat / (np.sqrt(vhat) + self.eps)


FIRST_ORDER = {"GD": GD, "Momentum": Momentum, "Nesterov": Nesterov, "AdaGrad": AdaGrad,
               "RMSProp": RMSProp, "Adam": Adam}


def run_first_order(obj, method, lr, iters, batch=None, seed=0, x0=None, lr_decay=False,
                    noise=0.0, record_every=1):
    """Minimise obj with a first-order method. batch=None: full gradient (GD);
    batch=b: mini-batch SGD with b samples per step (b=1: plain SGD). noise>0 adds
    Gaussian noise to every gradient (a noisy objective). Returns a history dict."""
    rng = np.random.default_rng(seed)
    opt = FIRST_ORDER[method](lr)
    x = np.array(obj.x0 if x0 is None else x0, dtype=float)
    hist = dict(f=[obj.f(x)], gnorm=[np.linalg.norm(obj.grad(x))], t=[0.0], grads=[0], x=[x.copy()])
    evals = [0]
    base_lr = lr

    def gradfn(z):
        evals[0] += 1
        if batch is None:
            g = obj.grad(z)
        else:
            g = obj.grad(z, idx=rng.integers(0, obj.n, size=batch))
        if noise:
            g = g + noise * rng.standard_normal(g.shape)
        return g
    t0 = time.perf_counter()
    for k in range(1, iters + 1):
        if lr_decay:
            opt.lr = base_lr / np.sqrt(k)
        x = opt.step(x, gradfn)
        if not np.all(np.isfinite(x)) or np.linalg.norm(x) > 1e12:
            hist["diverged"] = k
            break
        if k % record_every == 0 or k == iters:
            hist["f"].append(obj.f(x)); hist["gnorm"].append(np.linalg.norm(obj.grad(x)))
            hist["t"].append(time.perf_counter() - t0); hist["grads"].append(evals[0])
            hist["x"].append(x.copy())
    hist["x_final"] = x
    hist["diverged"] = hist.get("diverged")
    return hist


# ------------------------------------------------------------------ second-order / curvature
def _record(hist, obj, x, t0, extra=1):
    hist["f"].append(obj.f(x)); hist["gnorm"].append(np.linalg.norm(obj.grad(x)))
    hist["t"].append(time.perf_counter() - t0); hist["x"].append(x.copy())


def _new_hist(obj, x):
    return dict(f=[obj.f(x)], gnorm=[np.linalg.norm(obj.grad(x))], t=[0.0], x=[x.copy()])


def _armijo(obj, x, p, g, a=1.0, c=1e-4, shrink=0.5, max_halvings=40):
    fx = obj.f(x)
    for _ in range(max_halvings):
        if obj.f(x + a * p) <= fx + c * a * (g @ p):
            return a
        a *= shrink
    return a


def newton(obj, iters=50, x0=None, damped=False, tol=1e-12):
    x = np.array(obj.x0 if x0 is None else x0, dtype=float)
    hist = _new_hist(obj, x)
    t0 = time.perf_counter()
    for _ in range(iters):
        g = obj.grad(x)
        if np.linalg.norm(g) < tol:
            break
        H = obj.hess(x)
        p = -np.linalg.solve(H, g)
        if damped:
            if np.any(np.linalg.eigvalsh(H) <= 0) or g @ p >= 0:
                p = -g                                    # not a descent direction: fall back
            p = _armijo(obj, x, p, g) * p
        x = x + p
        _record(hist, obj, x, t0)
    hist["x_final"] = x
    return hist


def gauss_newton(obj, iters=50, x0=None, tol=1e-12):
    x = np.array(obj.x0 if x0 is None else x0, dtype=float)
    hist = _new_hist(obj, x)
    t0 = time.perf_counter()
    for _ in range(iters):
        J, r = obj.jac(x), obj.residuals(x)
        if np.linalg.norm(J.T @ r) < tol:
            break
        x = x - np.linalg.lstsq(J, r, rcond=None)[0]     # (J^T J)^-1 J^T r, stably
        _record(hist, obj, x, t0)
    hist["x_final"] = x
    return hist


def levenberg_marquardt(obj, iters=50, x0=None, lam=1e-2, tol=1e-12):
    x = np.array(obj.x0 if x0 is None else x0, dtype=float)
    hist = _new_hist(obj, x)
    t0 = time.perf_counter()
    for _ in range(iters):
        J, r = obj.jac(x), obj.residuals(x)
        g = J.T @ r
        if np.linalg.norm(g) < tol:
            break
        A = J.T @ J
        while True:
            p = -np.linalg.solve(A + lam * np.diag(np.diag(A) + 1e-12), g)
            if obj.f(x + p) < obj.f(x):
                x, lam = x + p, lam / 3                   # accept, trust the model more
                break
            lam *= 4                                      # reject, move towards gradient descent
            if lam > 1e12:
                break
        _record(hist, obj, x, t0)
    hist["x_final"] = x
    return hist


def lbfgs(obj, iters=100, x0=None, m=10, tol=1e-12):
    x = np.array(obj.x0 if x0 is None else x0, dtype=float)
    hist = _new_hist(obj, x)
    S, Y = [], []
    g = obj.grad(x)
    t0 = time.perf_counter()
    for _ in range(iters):
        if np.linalg.norm(g) < tol:
            break
        q = g.copy()                                    # two-loop recursion
        alphas = []
        for s, y in reversed(list(zip(S, Y))):
            a = (s @ q) / (y @ s)
            alphas.append(a)
            q -= a * y
        gamma = (S[-1] @ Y[-1]) / (Y[-1] @ Y[-1]) if S else 1.0 / max(np.linalg.norm(g), 1e-12)
        r = gamma * q
        for (s, y), a in zip(zip(S, Y), reversed(alphas)):
            b = (y @ r) / (y @ s)
            r += s * (a - b)
        p = -r
        step = _armijo(obj, x, p, g)
        x_new = x + step * p
        g_new = obj.grad(x_new)
        s, y = x_new - x, g_new - g
        if y @ s > 1e-12:                               # keep the approximation positive definite
            S.append(s); Y.append(y)
            if len(S) > m:
                S.pop(0); Y.pop(0)
        x, g = x_new, g_new
        _record(hist, obj, x, t0)
    hist["x_final"] = x
    return hist


def natural_gradient(obj, iters=50, lr=1.0, x0=None, damping=1e-4):
    x = np.array(obj.x0 if x0 is None else x0, dtype=float)
    hist = _new_hist(obj, x)
    t0 = time.perf_counter()
    for _ in range(iters):
        F = obj.fisher(x) + damping * np.eye(len(x))
        x = x - lr * np.linalg.solve(F, obj.grad(x))
        _record(hist, obj, x, t0)
    hist["x_final"] = x
    return hist


def diag_newton(obj, iters=200, lr=1.0, x0=None, floor=1e-8):
    x = np.array(obj.x0 if x0 is None else x0, dtype=float)
    hist = _new_hist(obj, x)
    t0 = time.perf_counter()
    for _ in range(iters):
        d = np.maximum(np.abs(np.diag(obj.hess(x))), floor)
        x = x - lr * obj.grad(x) / d
        _record(hist, obj, x, t0)
    hist["x_final"] = x
    return hist


def newton_cg(obj, iters=30, x0=None, cg_iters=20, tol=1e-12):
    """Hessian-free Newton: never forms H, only Hessian-vector products."""
    x = np.array(obj.x0 if x0 is None else x0, dtype=float)
    hist = _new_hist(obj, x)
    hist["hvps"] = [0]
    hv = 0
    t0 = time.perf_counter()
    for _ in range(iters):
        g = obj.grad(x)
        if np.linalg.norm(g) < tol:
            break
        p, r = np.zeros_like(x), -g.copy()
        d = r.copy()
        for _ in range(cg_iters):
            Hd = obj.hvp(x, d); hv += 1
            curv = d @ Hd
            if curv <= 0:                           # negative curvature: stop CG
                if not p.any():
                    p = -g
                break
            a = (r @ r) / curv
            p += a * d
            r_new = r - a * Hd
            if np.linalg.norm(r_new) < 1e-10 * np.linalg.norm(g):
                break
            d = r_new + ((r_new @ r_new) / (r @ r)) * d
            r = r_new
        x = x + _armijo(obj, x, p, g) * p
        _record(hist, obj, x, t0)
        hist["hvps"].append(hv)
    hist["x_final"] = x
    return hist
