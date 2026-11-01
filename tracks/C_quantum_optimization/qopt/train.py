"""One training loop for every optimizer, with a budget in *circuit runs*.

Why circuit runs and not iterations: one iteration costs very different amounts
of quantum work per method.
  * parameter-shift gradient : 2 circuits per *gate* that uses a parameter, per step
                               (QAOA p=2 here: 58 gates -> 117 circuits per step)
  * SPSA                     : 2 circuits per step, whatever the number of parameters
  * QNG                      : a parameter-shift gradient + extra circuits for the
                               (block-diagonal) metric tensor
On hardware each circuit run costs time and money, so a fair comparison gives
every method the same budget of circuit runs.

Unit: qml.Tracker's "simulations" = distinct circuits run on the device. NOT
its "executions" field, which counts per measurement group (5 per circuit for
both problems here) and silently inflated every budget 5x in the first version
(docs/failure_log.md F5). With finite shots each circuit is measured in the
same 5 groups for every optimizer, so the grouping multiplies all methods'
hardware cost equally and does not change the comparison.

Optimizers
  gd    : plain gradient descent, gradients by the parameter-shift rule
  adam  : Adam, parameter-shift gradients
  spsa  : simultaneous perturbation stochastic approximation (gradient-free)
  qng   : quantum natural gradient (Fubini-Study metric, block-diagonal approx.)
"""
from __future__ import annotations

import pennylane as qml
from pennylane import numpy as pnp

METHODS = ["gd", "adam", "spsa", "qng"]


def make_optimizer(name: str, stepsize: float, budget_steps: int = 1000):
    if name == "gd":
        return qml.GradientDescentOptimizer(stepsize)
    if name == "adam":
        return qml.AdamOptimizer(stepsize)
    if name == "qng":
        return qml.QNGOptimizer(stepsize, approx="block-diag", lam=1e-3)
    if name == "spsa":
        # `a` scales the step, c is the perturbation size, maxiter sets the gain schedule
        return qml.SPSAOptimizer(maxiter=budget_steps, a=stepsize, c=0.1)
    raise ValueError(name)


def train(cost, dev, params, method: str, stepsize: float, max_circuits: int, eval_cost,
          record_every: int = 1):
    """Optimise `cost` until its device `dev` has run `max_circuits` circuits.

    eval_cost: an exact cost function on a *separate* device, used only to
    record progress. Its calls are not counted, and the recorded curve is free
    of shot noise even when `cost` uses finite shots.
    Returns final parameters and a history of (circuits used, exact cost).
    """
    opt = make_optimizer(method, stepsize, budget_steps=max(1, max_circuits // 2))
    params = pnp.array(params, requires_grad=True)
    history = []
    step = 0
    with qml.Tracker(dev) as tr:
        while True:
            used = int(tr.totals.get("simulations", 0))
            done = used >= max_circuits
            if done or step % record_every == 0:
                history.append((used, float(eval_cost(params))))
            if done:
                break
            params = opt.step(cost, params)
            step += 1
    return params, history
