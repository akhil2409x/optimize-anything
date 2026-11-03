# A · 001 · Continuous optimization: 13 methods, backprop, landscapes

- **Tool / model:** Claude (claude.ai), Claude Opus 5.5
- **Date:** 2026-10-29
- **Kind of work:** code generation, testing, experiments, documentation

## Original prompt

After Track F, Claude asked whether to continue with Track A. Reply:

```
yes continue
```

## AI response (summary)

Claude read the Track A and Chapter 6 requirements and wrote, from scratch in NumPy:

- `contopt/objectives.py`: quadratics with a chosen condition number, a convex
  but not strongly convex quartic, Rosenbrock, a saddle function, six-hump camel,
  L2 logistic regression (gradient, Hessian, HVP, empirical Fisher) and a
  nonlinear curve fit (Jacobian and full Hessian).
- `contopt/optimizers.py`: GD/SGD/mini-batch SGD, Momentum, Nesterov, AdaGrad,
  RMSProp and Adam; Newton (pure and damped), Gauss–Newton,
  Levenberg–Marquardt, L-BFGS, natural gradient (Fisher), diagonal-Hessian
  scaling, and Newton-CG using only Hessian-vector products.
- `contopt/mlp.py`: a 2-layer network with hand-written backpropagation.
- `tests/test_contopt.py`: 19 tests. Every gradient, Hessian, HVP and the
  backprop are checked against autograd.
- `run_experiments.py`: seven experiments.

## Revisions during the session

1. **A syntax error:** a hyphen in a keyword argument name.
2. **Wrong GD theory formula.** The convergence table first used the rate for
   step size 2/(L+μ) while the runs used 1/L. Corrected, with each theory
   column labelled as a bound or an asymptotic rate.

## Resulting code change

Commit `[AI:claude] Track A: continuous optimization from scratch` = local commit `c3dfcdc` (on GitHub: `https://github.com/akhil2409x/optimize-anything/commit/c3dfcdc`)

## Verification note

To do by the student:
- Derive the logistic-regression gradient and Hessian on paper and compare with
  `objectives.py`.
- Trace `MLP.backprop` line by line and explain why backprop is not an optimizer.
- Explain why pure Newton converges to the saddle point.
- Rerun `pytest -q` and `python run_experiments.py`.
