"""Correctness checks for Track A. Run from tracks/A_continuous: `pytest -q`.

Every hand-derived gradient, Hessian, Hessian-vector product and the MLP's
backpropagation are compared with autograd (an independent automatic
differentiation library) and with finite differences."""
import autograd.numpy as anp
import numpy as np
import pytest
from autograd import grad, hessian

from contopt import mlp as M, objectives as O, optimizers as Opt

RNG = np.random.default_rng(42)


# ------------------------------------------------------------------ autograd twins of the objectives
def ag_rosen(z):
    return (1 - z[0]) ** 2 + 100 * (z[1] - z[0] ** 2) ** 2


def ag_saddle(z):
    return z[0] ** 2 - z[1] ** 2 + z[1] ** 4 / 4


def ag_camel(z):
    x, y = z[0], z[1]
    return (4 - 2.1 * x ** 2 + x ** 4 / 3) * x ** 2 + x * y + (-4 + 4 * y ** 2) * y ** 2


def ag_logreg(X, y, lam):
    return lambda w: anp.mean(anp.logaddexp(0.0, -y * anp.dot(X, w))) + 0.5 * lam * anp.dot(w, w)


def ag_curve(c):
    return lambda th: 0.5 * anp.sum((th[0] * anp.exp(-th[1] * c.t) + th[2] - c.y) ** 2)


@pytest.mark.parametrize("obj,ag", [(O.Rosenbrock(), ag_rosen), (O.Saddle(), ag_saddle),
                                    (O.SixHumpCamel(), ag_camel)])
def test_2d_gradients_and_hessians(obj, ag):
    for _ in range(10):
        z = RNG.uniform(-2, 2, 2)
        assert np.allclose(obj.grad(z), grad(ag)(z))
        assert np.allclose(obj.hess(z), hessian(ag)(z))


def test_quadratic_and_quartic():
    q = O.Quadratic(50, dim=8)
    x = RNG.standard_normal(8)
    assert np.allclose(q.grad(x), grad(lambda z: 0.5 * anp.dot(z, anp.dot(q.A, z)))(x))
    assert np.allclose(np.linalg.eigvalsh(q.A)[[0, -1]], [1, 50])        # condition number 50
    b = O.QuarticBowl(5)
    assert np.allclose(b.hess(x[:5]), hessian(lambda z: anp.sum(z ** 4))(x[:5]))


def test_logistic_regression_gradient_hessian_hvp_fisher():
    X, y = O.make_classification(n=120, d=5)
    L = O.LogisticRegression(X, y, lam=0.01)
    ag = ag_logreg(X, y, 0.01)
    for _ in range(5):
        w = RNG.standard_normal(L.d) * 0.3
        assert np.allclose(L.grad(w), grad(ag)(w))
        H = hessian(ag)(w)
        assert np.allclose(L.hess(w), H)
        v = RNG.standard_normal(L.d)
        assert np.allclose(L.hvp(w, v), H @ v)                      # HVP without forming H
        assert np.all(np.linalg.eigvalsh(L.fisher(w)) > 0)          # Fisher is PSD (+ lam I)
    idx = np.array([0, 3, 7])                                       # mini-batch gradient
    assert np.allclose(L.grad(w, idx), grad(ag_logreg(X[idx], y[idx], 0.01))(w))


def test_curvefit_jacobian_and_hessian():
    c = O.CurveFit()
    th = np.array([2.0, 1.0, 0.3])
    eps = 1e-6
    J_fd = np.stack([(c.residuals(th + eps * e) - c.residuals(th - eps * e)) / (2 * eps) for e in np.eye(3)], 1)
    assert np.allclose(c.jac(th), J_fd, atol=1e-6)
    assert np.allclose(c.grad(th), grad(ag_curve(c))(th))
    assert np.allclose(c.hess(th), hessian(ag_curve(c))(th))


# ------------------------------------------------------------------ backprop
def test_backprop_matches_autograd_and_finite_differences():
    net = M.MLP(4, 7, 3, lam=0.01, seed=1)
    X = RNG.standard_normal((25, 4))
    Y = RNG.integers(0, 3, 25)
    th = net.x0 + 0.1 * RNG.standard_normal(net.n_params)

    def ag_loss(t):
        W1 = anp.reshape(t[:28], (4, 7)); b1 = t[28:35]
        W2 = anp.reshape(t[35:56], (7, 3)); b2 = t[56:59]
        z = anp.dot(anp.tanh(anp.dot(X, W1) + b1), W2) + b2
        logp = z - anp.log(anp.sum(anp.exp(z), axis=1, keepdims=True))
        ce = -anp.mean(logp[anp.arange(25), Y])
        return ce + 0.005 * (anp.sum(W1 ** 2) + anp.sum(W2 ** 2))
    assert net.n_params == 59
    assert np.isclose(net.loss(th, X, Y), ag_loss(th))
    g = net.backprop(th, X, Y)
    assert np.allclose(g, grad(ag_loss)(th), atol=1e-10)
    k = RNG.integers(0, 59, 8)                                       # spot-check with finite differences
    for i in k:
        e = np.zeros(59); e[i] = 1e-6
        fd = (net.loss(th + e, X, Y) - net.loss(th - e, X, Y)) / 2e-6
        assert abs(fd - g[i]) < 1e-6


# ------------------------------------------------------------------ optimizer behaviour
@pytest.mark.parametrize("method,lr", [("GD", 0.01), ("Momentum", 0.004), ("Nesterov", 0.004),
                                       ("AdaGrad", 2.0), ("RMSProp", 0.003), ("Adam", 0.03)])
def test_every_first_order_method_converges_on_a_quadratic(method, lr):
    q = O.Quadratic(100, dim=10)
    h = Opt.run_first_order(q, method, lr, 3000)
    assert h["f"][-1] < 1e-3 * h["f"][0] and not h["diverged"]


def test_gd_diverges_above_2_over_L():
    q = O.Quadratic(100, dim=10)                                     # L = 100
    assert not Opt.run_first_order(q, "GD", 1.9 / 100, 500)["diverged"]
    assert Opt.run_first_order(q, "GD", 2.1 / 100, 2000)["diverged"]


def test_newton_one_step_on_quadratic_and_quadratic_rate_on_rosenbrock():
    q = O.Quadratic(1000, dim=10)
    assert Opt.newton(q, 1)["f"][-1] < 1e-20
    h = Opt.newton(O.Rosenbrock(), 50, damped=True)
    assert np.allclose(h["x_final"], [1, 1]) and len(h["f"]) < 40


def test_newton_converges_to_saddle_but_gd_and_damped_newton_escape():
    s = O.Saddle()
    assert np.allclose(Opt.newton(s, 20)["x_final"], [0, 0], atol=1e-8)       # attracted to the saddle
    assert abs(abs(Opt.run_first_order(s, "GD", 0.1, 500)["x_final"][1]) - np.sqrt(2)) < 1e-6
    assert abs(Opt.newton(s, 100, damped=True)["f"][-1] - (-1)) < 1e-8


def test_second_order_methods_agree_on_logistic_regression():
    X, y = O.make_classification(n=300, d=6)
    L = O.LogisticRegression(X, y, lam=1e-2)
    ref = Opt.newton(L, 30)["x_final"]
    assert np.linalg.norm(L.grad(ref)) < 1e-10
    for h in [Opt.lbfgs(L, 300), Opt.newton_cg(L, 30), Opt.natural_gradient(L, 200)]:
        assert np.allclose(h["x_final"], ref, atol=1e-5)


def test_gauss_newton_and_lm_fit_the_curve():
    c = O.CurveFit()
    gn = Opt.gauss_newton(c, 30)["x_final"]
    lm = Opt.levenberg_marquardt(c, 100)["x_final"]
    assert np.allclose(gn, lm, atol=1e-6)
    assert np.linalg.norm(c.grad(gn)) < 1e-8
    assert np.allclose(gn, c.theta_true, atol=0.1)


def test_minibatch_sgd_reduces_loss():
    X, y = O.make_classification()
    L = O.LogisticRegression(X, y)
    h = Opt.run_first_order(L, "GD", 0.05, 2000, batch=16, seed=3)
    assert h["f"][-1] < 0.8 * h["f"][0]
