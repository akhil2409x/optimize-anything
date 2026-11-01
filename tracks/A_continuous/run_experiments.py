"""Track A experiments: continuous optimization.

    python run_experiments.py           # full run (~5 min)
    python run_experiments.py --quick

Exp 1  Convergence vs theory: strongly convex quadratics (kappa 10/100/1000),
       a convex-but-not-strongly-convex quartic, non-convex Rosenbrock.
Exp 2  Landscapes: saddle point (Newton is attracted, GD escapes), six-hump
       camel from 200 random starts (local vs global minima).
Exp 3  ML as optimization: L2-regularized logistic regression, all 13 methods,
       train loss gap, test accuracy, gradient evaluations, time; lambda sweep.
Exp 4  Nonlinear least squares: Gauss-Newton and Levenberg-Marquardt vs Newton,
       GD and Adam, from a near and a far start.
Exp 5  Backpropagation: MLP on two-moons trained by 6 first-order optimizers;
       backprop vs finite-difference gradient (accuracy and cost).
Exp 6  Step-size sensitivity: final loss vs learning rate for every first-order method.
Exp 7  Noisy objectives: mini-batch size, constant vs decaying step, added gradient noise.
"""
from __future__ import annotations

import argparse
import csv
import math
import platform
import statistics as st
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons

from contopt import mlp as M, objectives as O, optimizers as Opt

HERE = Path(__file__).parent
RESULTS = HERE / "results"
FIRST = list(Opt.FIRST_ORDER)
LR_GRID = [10.0 ** e for e in np.arange(-5, 1.01, 0.5)]


def write_csv(path, rows):
    if rows:
        fields = list(dict.fromkeys(k for r in rows for k in r))
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)


def md_table(rows, cols):
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join(str(r.get(c, "")) for c in cols) + " |" for r in rows]
    return "\n".join(out)


def iters_to(hist, target):
    for k, v in enumerate(hist["f"]):
        if v <= target:
            return k
    return None


def tune_lr(make_run, score=lambda h: h["f"][-1]):
    """Pick the learning rate from LR_GRID with the best final score (diverged = inf)."""
    best = None
    for lr in LR_GRID:
        h = make_run(lr)
        s = math.inf if h["diverged"] or not np.isfinite(h["f"][-1]) else score(h)
        if best is None or s < best[1]:
            best = (lr, s)
    return best[0]


# ---------------------------------------------------------------- Exp 1
def exp1(quick):
    rows, curves = [], {}
    for kappa in (10, 100, 1000):
        q = O.Quadratic(kappa, dim=50)
        L, mu = q.L, q.mu
        target = 1e-8 * q.f(q.x0)
        sk = math.sqrt(kappa)
        cfgs = {
            "GD (lr = 1/L)": lambda: Opt.run_first_order(q, "GD", 1 / L, 20000),
            "Momentum (Polyak optimal)": lambda: _run_mom(q, 4 / (math.sqrt(L) + math.sqrt(mu)) ** 2,
                                                          ((sk - 1) / (sk + 1)) ** 2, 20000),
            "Nesterov (lr = 1/L)": lambda: _run_nag(q, 1 / L, (sk - 1) / (sk + 1), 20000),
            "Newton": lambda: Opt.newton(q, 5),
        }
        for name, fn in cfgs.items():
            h = fn()
            k = iters_to(h, target)
            rows.append(dict(problem=f"quadratic kappa={kappa}", method=name, iters_to_1e8_reduction=k,
                             theory_iters=_theory(name, kappa)))
            curves[(kappa, name)] = h["f"]
    # convex, not strongly convex: sublinear rate
    b = O.QuarticBowl(10)
    h = Opt.run_first_order(b, "GD", 0.02, 20000)
    f = np.array(h["f"])
    ks = np.array([100, 1000, 10000])
    slope = np.polyfit(np.log(ks), np.log(f[ks]), 1)[0]
    rows.append(dict(problem="quartic (convex, not strongly)", method="GD (lr 0.02)",
                     iters_to_1e8_reduction=iters_to(h, 1e-8 * f[0]),
                     theory_iters=f"sublinear; fitted f(k) ~ k^{slope:.2f}"))
    curves[("quartic", "GD")] = h["f"]
    # non-convex Rosenbrock
    r = O.Rosenbrock()
    for m in FIRST:
        lr = tune_lr(lambda lr: Opt.run_first_order(r, m, lr, 5000))
        h = Opt.run_first_order(r, m, lr, 20000)
        rows.append(dict(problem="Rosenbrock (non-convex)", method=f"{m} (lr {lr:.0e})",
                         iters_to_1e8_reduction=iters_to(h, 1e-8), final_f=f"{h['f'][-1]:.1e}"))
        curves[("rosen", m)] = h["f"]
    for name, fn in [("Newton (damped)", lambda: Opt.newton(r, 100, damped=True)),
                     ("L-BFGS", lambda: Opt.lbfgs(r, 500)), ("Newton-CG (HVP)", lambda: Opt.newton_cg(r, 200))]:
        h = fn()
        rows.append(dict(problem="Rosenbrock (non-convex)", method=name, iters_to_1e8_reduction=iters_to(h, 1e-8),
                         final_f=f"{h['f'][-1]:.1e}"))
        curves[("rosen", name)] = h["f"]
    return rows, curves


def _run_mom(q, lr, beta, iters):
    opt = Opt.Momentum(lr, beta)
    return _loop(q, opt, iters)


def _run_nag(q, lr, beta, iters):
    opt = Opt.Nesterov(lr, beta)
    return _loop(q, opt, iters)


def _loop(q, opt, iters):
    x = q.x0.copy()
    f = [q.f(x)]
    for _ in range(iters):
        x = opt.step(x, q.grad)
        f.append(q.f(x))
        if f[-1] < 1e-30:
            break
    return dict(f=f, x_final=x)


def _theory(name, kappa):
    """Iterations the textbook rate predicts for shrinking f by 1e8.
      GD, lr = 1/L      : the slowest eigen-component of the error shrinks by (1 - 1/kappa)
                          per step, so f by (1 - 1/kappa)^2 -> worst-case bound
      heavy ball (Polyak): asymptotic error rate (sqrt k - 1)/(sqrt k + 1), f by its square;
                          ignores the k * rho^k transient, so real runs take longer
      Nesterov          : f - f* <= C (1 - 1/sqrt kappa)^k -> upper bound
    """
    eps = 1e-8
    if name.startswith("GD"):
        return f"≤ {math.ceil(math.log(eps) / (2 * math.log(1 - 1 / kappa)))} (bound)"
    if name.startswith("Momentum"):
        rho = (math.sqrt(kappa) - 1) / (math.sqrt(kappa) + 1)
        return f"≈ {math.ceil(math.log(eps) / (2 * math.log(rho)))} (asymptotic)"
    if name.startswith("Nesterov"):
        return f"≤ {math.ceil(math.log(eps) / math.log(1 - 1 / math.sqrt(kappa)))} (bound)"
    return "1 (exact on a quadratic)"


# ---------------------------------------------------------------- Exp 2
def exp2(quick):
    rows = []
    s = O.Saddle()
    h = Opt.newton(s, 50)
    rows.append(dict(landscape="saddle", method="Newton (pure)", outcome=f"stops at {np.round(h['x_final'], 6)}",
                     f_final=round(h["f"][-1], 6), iters_to_f_below_minus_0_99=iters_to(h, -0.99)))
    h = Opt.newton(s, 200, damped=True)
    rows.append(dict(landscape="saddle", method="Newton (damped: -g when H not PD)",
                     outcome=f"x = {np.round(h['x_final'], 4)}", f_final=round(h["f"][-1], 6),
                     iters_to_f_below_minus_0_99=iters_to(h, -0.99)))
    for m, lr in [("GD", 0.1), ("Momentum", 0.05), ("Nesterov", 0.05), ("Adam", 0.05), ("RMSProp", 0.05), ("AdaGrad", 0.5)]:
        h = Opt.run_first_order(s, m, lr, 2000)
        rows.append(dict(landscape="saddle", method=f"{m} (lr {lr})", outcome=f"x = {np.round(h['x_final'], 4)}",
                         f_final=round(h["f"][-1], 6), iters_to_f_below_minus_0_99=iters_to(h, -0.99)))
    # six-hump camel: many random starts
    c = O.SixHumpCamel()
    starts = np.random.default_rng(0).uniform([-2, -1], [2, 1], (60 if quick else 200, 2))
    methods = {"GD": lambda x0: Opt.run_first_order(c, "GD", 0.01, 3000, x0=x0),
               "Momentum": lambda x0: Opt.run_first_order(c, "Momentum", 0.005, 3000, x0=x0),
               "Adam": lambda x0: Opt.run_first_order(c, "Adam", 0.02, 3000, x0=x0),
               "Newton (pure)": lambda x0: Opt.newton(c, 100, x0=x0),
               "Newton (damped)": lambda x0: Opt.newton(c, 100, x0=x0, damped=True),
               "L-BFGS": lambda x0: Opt.lbfgs(c, 300, x0=x0)}
    for name, fn in methods.items():
        cls = {"global min": 0, "local min": 0, "saddle / max": 0, "not converged": 0}
        for x0 in starts:
            h = fn(x0)
            x = h["x_final"]
            if not np.all(np.isfinite(x)) or np.linalg.norm(c.grad(x)) > 1e-5:
                cls["not converged"] += 1
                continue
            eig = np.linalg.eigvalsh(c.hess(x))
            if np.all(eig > 0):
                cls["global min" if c.f(x) < c.f_star + 1e-6 else "local min"] += 1
            else:
                cls["saddle / max"] += 1
        rows.append(dict(landscape=f"six-hump camel, {len(starts)} random starts", method=name,
                         outcome=", ".join(f"{k} {v}" for k, v in cls.items())))
    return rows


# ---------------------------------------------------------------- Exp 3
def exp3(quick):
    Xall, yall = O.make_classification(n=1200, d=10, seed=0)
    Xtr, ytr, Xte, yte = Xall[:600], yall[:600], Xall[600:], yall[600:]
    lam = 1e-3
    L = O.LogisticRegression(Xtr, ytr, lam)
    ref = Opt.newton(L, 50)
    f_star = ref["f"][-1]
    epochs = 30 if quick else 100
    seeds = range(2) if quick else range(5)
    rows = []

    def summarise(name, hs, grads_per_iter=None, extra=""):
        gaps = [h["f"][-1] - f_star for h in hs]
        accs = [L.accuracy(h["x_final"], Xte, yte) for h in hs]
        rows.append(dict(method=name, runs=len(hs),
                         loss_gap=f"{st.mean(gaps):.1e}" + (f" ± {st.pstdev(gaps):.0e}" if len(hs) > 1 else ""),
                         test_acc=f"{st.mean(accs):.3f}",
                         epochs_or_iterations=round(st.mean(h.get('epochs', 0) for h in hs), 1),
                         seconds=f"{st.mean(h['t'][-1] for h in hs):.3f}", notes=extra))
    # full-batch first-order: budget = `epochs` full gradients
    for m in FIRST:
        lr = tune_lr(lambda lr: Opt.run_first_order(L, m, lr, epochs))
        h = Opt.run_first_order(L, m, lr, epochs)
        h["epochs"] = epochs
        summarise(f"{m} (full batch, lr {lr:.0e})", [h], extra="tuned lr")
    # stochastic: same number of epochs
    for b in (1, 32):
        iters = epochs * L.n // b
        lr = tune_lr(lambda lr: Opt.run_first_order(L, "GD", lr, iters, batch=b, seed=99, record_every=iters))
        hs = [Opt.run_first_order(L, "GD", lr, iters, batch=b, seed=s, record_every=iters) for s in seeds]
        for h in hs:
            h["epochs"] = epochs
        summarise(f"{'SGD' if b == 1 else 'mini-batch SGD'} (batch {b}, lr {lr:.0e})", hs, extra="tuned on seed 99")
        lr = tune_lr(lambda lr: Opt.run_first_order(L, "Adam", lr, iters, batch=b, seed=99, record_every=iters))
        hs = [Opt.run_first_order(L, "Adam", lr, iters, batch=b, seed=s, record_every=iters) for s in seeds]
        for h in hs:
            h["epochs"] = epochs
        summarise(f"Adam (batch {b}, lr {lr:.0e})", hs, extra="tuned on seed 99")
    # second-order / curvature
    for name, fn, cost in [
        ("Newton", lambda: Opt.newton(L, 20), "1 grad + 1 Hessian (d^2 n) per iter"),
        ("Newton-CG (Hessian-vector products)", lambda: Opt.newton_cg(L, 20), "HVPs, Hessian never formed"),
        ("L-BFGS (low-rank curvature)", lambda: Opt.lbfgs(L, 100), "1 grad (+ line search) per iter"),
        ("natural gradient (empirical Fisher)", lambda: Opt.natural_gradient(L, 100), "Fisher d^2 n per iter"),
        ("diagonal-Hessian scaled GD", lambda: Opt.diag_newton(L, epochs, lr=0.5), "diag(H) per iter"),
    ]:
        h = fn()
        h["epochs"] = len(h["f"]) - 1
        k = iters_to(h, f_star + 1e-8)
        summarise(name, [h], extra=f"{cost}; iterations to gap<1e-8: {k}" + (
            f"; {h['hvps'][-1]} HVPs" if "hvps" in h else ""))
    # regularization sweep
    reg = []
    for lam2 in (0.0, 1e-3, 1e-1, 1.0):
        L2 = O.LogisticRegression(Xtr, ytr, lam2)
        h = Opt.newton(L2, 50, damped=True) if lam2 == 0 else Opt.newton(L2, 50)
        w = h["x_final"]
        reg.append(dict(lam=lam2, weight_norm=round(float(np.linalg.norm(w)), 3),
                        train_acc=round(L2.accuracy(w), 3), test_acc=round(L2.accuracy(w, Xte, yte), 3),
                        hessian_min_eig=f"{np.linalg.eigvalsh(L2.hess(w))[0]:.2e}"))
    return rows, reg, f_star


# ---------------------------------------------------------------- Exp 4
def exp4():
    c = O.CurveFit()
    rows = []
    for start_name, x0 in [("near start (1, 0.2, 0)", np.array([1.0, 0.2, 0.0])),
                           ("far start (0.1, 4, -2)", np.array([0.1, 4.0, -2.0]))]:
        ref = Opt.levenberg_marquardt(c, 300, x0=np.array([1.0, 0.2, 0.0]))["f"][-1]
        for name, fn in [("Gauss-Newton", lambda: Opt.gauss_newton(c, 50, x0=x0)),
                         ("Levenberg-Marquardt", lambda: Opt.levenberg_marquardt(c, 100, x0=x0)),
                         ("Newton (damped, exact Hessian)", lambda: Opt.newton(c, 100, x0=x0, damped=True)),
                         ("GD (tuned)", lambda: Opt.run_first_order(c, "GD", tune_lr(
                             lambda lr: Opt.run_first_order(c, "GD", lr, 2000, x0=x0)), 20000, x0=x0)),
                         ("Adam (tuned)", lambda: Opt.run_first_order(c, "Adam", tune_lr(
                             lambda lr: Opt.run_first_order(c, "Adam", lr, 2000, x0=x0)), 20000, x0=x0))]:
            h = fn()
            x = h["x_final"]
            ok = np.all(np.isfinite(x)) and abs(c.f(x) - ref) < 1e-6
            rows.append(dict(start=start_name, method=name, iterations_to_optimum=iters_to(h, ref + 1e-6),
                             final_f=f"{h['f'][-1]:.6g}", reached_optimum=bool(ok),
                             theta=np.round(x, 3).tolist() if np.all(np.isfinite(x)) else "diverged"))
    return rows


# ---------------------------------------------------------------- Exp 5
def exp5(quick):
    X, Y = make_moons(n_samples=800, noise=0.25, random_state=0)
    Xtr, Ytr, Xte, Yte = X[:500], Y[:500], X[500:], Y[500:]
    epochs = 40 if quick else 150
    seeds = range(2) if quick else range(5)
    rows, curves = [], {}
    for m in FIRST:
        def run(lr, seed):
            net = M.MLP(2, 16, 2, lam=1e-4, seed=seed)
            obj = M.MLPObjective(net, Xtr, Ytr)
            iters = epochs * len(Ytr) // 32
            return Opt.run_first_order(obj, m, lr, iters, batch=32, seed=seed, record_every=len(Ytr) // 32), net
        lr = tune_lr(lambda lr: run(lr, 99)[0])
        accs, losses, hs = [], [], []
        for s in seeds:
            h, net = run(lr, s)
            accs.append(net.accuracy(h["x_final"], Xte, Yte))
            losses.append(h["f"][-1])
            hs.append(h)
        curves[m] = np.mean([h["f"] for h in hs], axis=0)
        rows.append(dict(optimizer=f"{m} (mini-batch 32)", lr=f"{lr:.0e}", seeds=len(accs),
                         train_loss=f"{st.mean(losses):.3f} ± {st.pstdev(losses):.3f}",
                         test_acc=f"{st.mean(accs):.3f} ± {st.pstdev(accs):.3f}"))
    # backprop vs finite differences
    net = M.MLP(2, 16, 2, seed=0)
    th = net.x0
    t0 = time.perf_counter(); g = net.backprop(th, Xtr, Ytr); t_bp = time.perf_counter() - t0
    t0 = time.perf_counter()
    fd = np.array([(net.loss(th + e, Xtr, Ytr) - net.loss(th - e, Xtr, Ytr)) / 2e-6
                   for e in np.eye(net.n_params) * 1e-6])
    t_fd = time.perf_counter() - t0
    grad_check = dict(parameters=net.n_params, max_abs_diff=f"{np.max(np.abs(g - fd)):.1e}",
                      backprop_ms=round(1000 * t_bp, 2), finite_diff_ms=round(1000 * t_fd, 1),
                      forward_passes_backprop="1 forward + 1 backward",
                      forward_passes_finite_diff=2 * net.n_params)
    return rows, curves, grad_check


# ---------------------------------------------------------------- Exp 6
def exp6():
    X, y = O.make_classification(n=600, d=10)
    L = O.LogisticRegression(X, y, 1e-3)
    f_star = Opt.newton(L, 50)["f"][-1]
    r = O.Rosenbrock()
    rows = []
    for prob, obj, iters, fstar in [("logistic", L, 200, f_star), ("Rosenbrock", r, 3000, 0.0)]:
        for m in FIRST:
            for lr in LR_GRID:
                h = Opt.run_first_order(obj, m, lr, iters, record_every=iters)
                gap = math.inf if h["diverged"] or not np.isfinite(h["f"][-1]) else h["f"][-1] - fstar
                rows.append(dict(problem=prob, method=m, lr=lr, gap=gap))
    summ = []
    for prob in ("logistic", "Rosenbrock"):
        for m in FIRST:
            rs = [r_ for r_ in rows if r_["problem"] == prob and r_["method"] == m]
            best = min(rs, key=lambda r_: r_["gap"])
            good = [r_["lr"] for r_ in rs if r_["gap"] <= max(10 * best["gap"], best["gap"] + 1e-6)]
            div = [r_["lr"] for r_ in rs if math.isinf(r_["gap"])]
            summ.append(dict(problem=prob, method=m, best_lr=f"{best['lr']:.0e}", best_gap=f"{best['gap']:.1e}",
                             good_lr_range=f"{min(good):.0e} – {max(good):.0e}",
                             good_range_decades=round(math.log10(max(good) / min(good)), 1),
                             smallest_diverging_lr=f"{min(div):.0e}" if div else "none ≤ 10"))
    return rows, summ


# ---------------------------------------------------------------- Exp 7
def exp7(quick):
    X, y = O.make_classification(n=600, d=10)
    L = O.LogisticRegression(X, y, 1e-3)
    f_star = Opt.newton(L, 50)["f"][-1]
    seeds = range(3) if quick else range(5)
    rows = []
    epochs = 20 if quick else 50
    for b in (1, 8, 64, None):
        for decay in (False, True):
            iters = epochs * (L.n // b if b else 1)
            lr = 0.05                       # the same step size for every batch size (fair comparison)
            gaps = []
            for s in seeds:
                h = Opt.run_first_order(L, "GD", lr, iters, batch=b, seed=s, lr_decay=decay, record_every=iters)
                gaps.append(h["f"][-1] - f_star)
            rows.append(dict(experiment="SGD batch size", setting=f"batch {b or 'full'}, lr {lr}"
                             + (", lr/sqrt(k)" if decay else ", constant lr"),
                             epochs=epochs, gap_mean=f"{st.mean(gaps):.2e}", gap_std=f"{st.pstdev(gaps):.1e}"))
    q = O.Quadratic(100, dim=20)
    for sigma in (0.0, 0.1, 1.0, 10.0):
        for m, lr in [("GD", 0.01), ("Momentum", 0.002), ("Adam", 0.05)]:
            fs = [Opt.run_first_order(q, m, lr, 3000, noise=sigma, seed=s, record_every=3000)["f"][-1] for s in seeds]
            rows.append(dict(experiment="additive gradient noise (quadratic kappa=100)",
                             setting=f"sigma {sigma}, {m} lr {lr}", epochs="3000 steps",
                             gap_mean=f"{st.mean(fs):.2e}", gap_std=f"{st.pstdev(fs):.1e}"))
    return rows


# ---------------------------------------------------------------- plots
def plots(c1, c5, e6):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.3))
    for name in ("GD (lr = 1/L)", "Momentum (Polyak optimal)", "Nesterov (lr = 1/L)"):
        for kappa, ls in ((100, "-"), (1000, "--")):
            f = np.array(c1[(kappa, name)]); f = f / f[0]
            axes[0].semilogy(np.maximum(f, 1e-30), ls, label=f"{name.split(' ')[0]} κ={kappa}")
    axes[0].set_xlim(0, 3000); axes[0].set_ylim(1e-12, 2)
    axes[0].set_xlabel("iteration"); axes[0].set_ylabel("f / f(x0)")
    axes[0].set_title("Strongly convex quadratics"); axes[0].legend(fontsize=7)
    for name in list(Opt.FIRST_ORDER) + ["Newton (damped)", "L-BFGS"]:
        f = c1.get(("rosen", name))
        if f is not None:
            axes[1].loglog(np.arange(1, len(f) + 1), np.maximum(f, 1e-20), label=name)
    axes[1].set_xlabel("iteration"); axes[1].set_ylabel("f"); axes[1].set_title("Rosenbrock (non-convex)")
    axes[1].legend(fontsize=7)
    for m in Opt.FIRST_ORDER:
        rs = [r for r in e6 if r["problem"] == "logistic" and r["method"] == m]
        axes[2].loglog([r["lr"] for r in rs], [r["gap"] if np.isfinite(r["gap"]) else np.nan for r in rs], "o-", label=m)
    axes[2].set_xlabel("learning rate"); axes[2].set_ylabel("loss gap after 200 iterations")
    axes[2].set_title("Step-size sensitivity (logistic regression)"); axes[2].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(RESULTS / "convergence_and_sensitivity.png", dpi=150)

    fig, ax = plt.subplots(figsize=(6.5, 4))
    for m, f in c5.items():
        ax.semilogy(f, label=m)
    ax.set_xlabel("epoch"); ax.set_ylabel("training loss (mean of seeds)")
    ax.set_title("MLP on two moons, gradients by backprop"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(RESULTS / "mlp_training.png", dpi=150)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    RESULTS.mkdir(exist_ok=True)
    print(f"Python {platform.python_version()}, numpy {np.__version__}")
    print("Exp 1"); r1, c1 = exp1(args.quick); write_csv(RESULTS / "exp1_convergence.csv", r1)
    print("Exp 2"); r2 = exp2(args.quick); write_csv(RESULTS / "exp2_landscapes.csv", r2)
    print("Exp 3"); r3, reg, f_star = exp3(args.quick)
    write_csv(RESULTS / "exp3_logistic.csv", r3); write_csv(RESULTS / "exp3_regularization.csv", reg)
    print("Exp 4"); r4 = exp4(); write_csv(RESULTS / "exp4_least_squares.csv", r4)
    print("Exp 5"); r5, c5, gc = exp5(args.quick); write_csv(RESULTS / "exp5_mlp.csv", r5)
    print("Exp 6"); e6, s6 = exp6(); write_csv(RESULTS / "exp6_stepsize_raw.csv", e6); write_csv(RESULTS / "exp6_stepsize.csv", s6)
    print("Exp 7"); r7 = exp7(args.quick); write_csv(RESULTS / "exp7_noise.csv", r7)
    plots(c1, c5, e6)
    (RESULTS / "SUMMARY.md").write_text("\n".join([
        "# Track A results (generated by run_experiments.py)\n",
        f"Python {platform.python_version()}, NumPy {np.__version__}, {platform.platform()}. "
        "All optimizers implemented from scratch in `contopt/optimizers.py`. Learning rates tuned from the "
        "grid 1e-5 … 10 (half-decade steps) where marked.\n",
        "## Exp 1 — convergence vs theory\n", md_table(r1, ["problem", "method", "iters_to_1e8_reduction", "theory_iters", "final_f"]),
        "\n![convergence](convergence_and_sensitivity.png)\n",
        "## Exp 2 — landscapes: saddle point and multiple minima\n",
        md_table(r2, ["landscape", "method", "outcome", "f_final", "iters_to_f_below_minus_0_99"]),
        f"\n## Exp 3 — ML as optimization: L2 logistic regression (f* = {f_star:.6f} from Newton)\n",
        md_table(r3, ["method", "runs", "loss_gap", "test_acc", "epochs_or_iterations", "seconds", "notes"]),
        "\n### Regularization\n", md_table(reg, list(reg[0])),
        "\n## Exp 4 — nonlinear least squares\n", md_table(r4, list(r4[0])),
        "\n## Exp 5 — backpropagation: MLP (2-16-2) on two moons\n", md_table(r5, list(r5[0])),
        "\nGradient check: " + ", ".join(f"{k} = {v}" for k, v in gc.items()) + "\n",
        "\n![mlp](mlp_training.png)\n",
        "## Exp 6 — step-size sensitivity\n", md_table(s6, list(s6[0])),
        "\n## Exp 7 — noisy objectives\n", md_table(r7, list(r7[0])), ""]) + "\n")
    print("Done. See results/")


if __name__ == "__main__":
    main()
