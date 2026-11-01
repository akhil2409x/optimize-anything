# Track A — Continuous optimization

Every optimizer is written from scratch in NumPy (`contopt/optimizers.py`).
Every gradient, Hessian, Hessian-vector product and the neural network's
backpropagation is checked against **autograd**, an independent automatic
differentiation library (`tests/test_contopt.py`, 19 tests).

| Chapter 6 item | Where it is implemented | Where it is shown |
|---|---|---|
| ML as an optimization problem: parameters, loss, regularization | `objectives.LogisticRegression` (weights w, cross-entropy loss, L2 term) | Exp 3, including a λ sweep |
| Convex vs non-convex; local/global minima; saddle points | `Quadratic`, `QuarticBowl`, `Rosenbrock`, `Saddle`, `SixHumpCamel` | Exp 1, 2 |
| GD, SGD, mini-batch SGD | `run_first_order(..., batch=None / 1 / b)` | Exp 3, 7 |
| Momentum, Nesterov (NAG), AdaGrad, RMSProp, Adam | `optimizers.py` | Exp 1, 3, 5, 6 |
| Newton | `newton` (pure and damped) | Exp 1, 2, 3, 4 |
| Gauss–Newton (+ Levenberg–Marquardt) | `gauss_newton`, `levenberg_marquardt` | Exp 4 |
| Backpropagation, distinct from the optimizer | `mlp.MLP.backprop`, hand-written chain rule | Exp 5 |
| Curvature: Fisher, diagonal, low-rank, Hessian-vector products | `natural_gradient` (empirical Fisher), `diag_newton`, `lbfgs` (low-rank), `newton_cg` (HVP only) | Exp 3 |
| Convergence: convex, strongly convex, non-convex | Measured iterations vs textbook rates | Exp 1 |
| Step-size and optimizer tuning | Learning-rate grid 10⁻⁵ … 10 | Exp 6 (and every tuned run) |
| Noisy objectives | Mini-batch noise, step-size decay, added gradient noise | Exp 7 |

## Results

Full tables and plots are in `results/SUMMARY.md`.

**1. Convergence rates follow theory.** The table shows iterations to shrink
f by 10⁸ on 50-dimensional quadratics.

| Condition number κ | GD (step 1/L) | Heavy-ball momentum | Nesterov | Newton |
|---|---|---|---|---|
| 10 | 49 (bound ≤ 88) | 17 (asymptotic ≈ 15) | 19 (bound ≤ 49) | 1 |
| 100 | 383 (≤ 917) | 63 (≈ 46) | 55 (≤ 175) | 1 |
| 1,000 | 2,695 (≤ 9,206) | 220 (≈ 146) | 138 (≤ 574) | 1 |

- **Scaling with κ:** GD's iterations grow about linearly in κ; momentum and
  Nesterov grow about with √κ. At κ = 1,000 that's 20× fewer iterations.
- **Newton is exact on a quadratic** in one step.
- **Convex but not strongly convex** (Σxᵢ⁴): GD slows to a *sublinear* rate.
  The fitted f(k) ∝ k^−1.96 matches the theoretical k^−2.
- **Non-convex Rosenbrock:**
  - Damped Newton and Newton-CG take 20 iterations, L-BFGS 44, Momentum 551,
    Adam 2,571 and AdaGrad 11,276.
  - Plain GD and RMSProp never reach 10⁻⁸ in 20,000 iterations.
  - Adam ends oscillating at its constant step size.

**2. The landscape decides the outcome.**
- **Saddle point** f = x² − y² + y⁴/4, starting 10⁻⁶ from the saddle's stable
  direction:
  - **Pure Newton converges *to* the saddle (0, 0)** and stops there. It
    follows wherever the gradient is zero, whatever the curvature.
  - GD, Momentum, Nesterov, Adam and AdaGrad all escape to a true minimum
    (0, ±√2) in 4–83 iterations.
  - Damped Newton (gradient step when the Hessian isn't positive definite)
    escapes in 14.
- **Six-hump camel, 200 random starts:**

| Method | Global min | Local min | Saddle / maximum |
|---|---|---|---|
| GD | 109 | 91 | 0 |
| Momentum | 115 | 85 | 0 |
| Adam | 108 | 85 | 0 (7 not converged) |
| L-BFGS | 127 | 73 | 0 |
| **Pure Newton** | 39 | 67 | **94** |
| Damped Newton | 115 | 85 | 0 |

Which minimum you reach depends mostly on where you start. Only pure Newton
ends at non-minima, in 47% of runs.

**3. ML as optimization: L2 logistic regression** (600 training, 600 test
samples, 11 features on scales 1–10).

| Method | Loss gap to optimum | Test accuracy | Budget |
|---|---|---|---|
| GD (tuned) | 1.1e-3 | 0.842 | 100 gradients |
| Momentum / Nesterov | 7.0e-6 / 2.3e-7 | 0.840 | 100 gradients |
| AdaGrad | 3.6e-12 | 0.842 | 100 gradients |
| RMSProp / Adam | 4.3e-4 / 5.9e-6 | 0.840 / 0.842 | 100 gradients |
| SGD (batch 1), 5 seeds | 3.0e-3 ± 1e-4 | 0.842 | 100 epochs |
| Mini-batch SGD (32), 5 seeds | 3.0e-3 ± 1e-4 | 0.842 | 100 epochs |
| Adam, mini-batch 32, 5 seeds | 8.4e-4 ± 1e-4 | 0.840 | 100 epochs |
| **Newton** | 0 | 0.842 | **7 iterations** |
| **Newton-CG** (HVPs only, Hessian never formed) | 0 | 0.842 | 7 iterations, 90 HVPs |
| L-BFGS (low-rank curvature) | 2.5e-14 | 0.842 | 31 iterations to gap < 1e-8 |
| Natural gradient (empirical Fisher) | ≈ 0 | 0.842 | 11 iterations to gap < 1e-8 |
| Diagonal-Hessian scaled GD | 2.8e-10 | 0.842 | 100 iterations |

- **Curvature pays off here.** Features on different scales make the problem
  ill-conditioned. Methods that use curvature or per-coordinate scaling
  (Newton, Fisher, diagonal Hessian, AdaGrad) win by orders of magnitude in
  loss.
- **Test accuracy barely moves (0.840–0.842).** Optimizing past a gap of about
  10⁻³ doesn't improve generalization, which is a useful reality check.
- **Regularization** (λ = 0 → 1): the weight norm shrinks from 0.85 to 0.35,
  and the Hessian's smallest eigenvalue grows from 0.077 to 1.14. The problem
  becomes more strongly convex and better conditioned, at a small cost in
  accuracy (0.842 → 0.830).

**4. Gauss–Newton is fast near the answer but unsafe far from it.** On the
fit a·e^(−bt) + c:
- **From a near start:** Gauss–Newton converges in 11 iterations,
  Levenberg–Marquardt in 4, and damped Newton in about 10. GD needs 302 and
  Adam 811.
- **From a far start:** Gauss–Newton **diverges** (b → −118), because the
  J^TJ model is poor far from the solution. Levenberg–Marquardt still
  converges in 12, by blending towards gradient descent when steps fail.

**5. Backpropagation computes the gradient; the optimizer uses it.**
- **Accuracy:** the hand-written backprop for a 2-16-2 network (82
  parameters) matches finite differences to 1.2 × 10⁻¹⁰, and autograd exactly.
- **Cost:** one gradient by backprop is 1 forward + 1 backward pass
  (0.3 ms). Finite differences need 164 forward passes (21 ms), about 65×
  slower, and the gap grows with network size.
- **Training on two moons** (mini-batch 32, 150 epochs, 5 seeds, step size
  tuned per optimizer): all six optimizers reach 0.947–0.953 test accuracy,
  within each other's spread (±0.005–0.011). AdaGrad's loss falls fastest
  early on (see `results/mlp_training.png`). After tuning, the choice of
  optimizer matters little for this small model.

**6. Step-size sensitivity.** The table shows the range of step sizes within
10× of each method's best loss.

| Problem | GD | Momentum | Nesterov | AdaGrad | RMSProp | Adam |
|---|---|---|---|---|---|---|
| Logistic regression | 0 decades | 1.0 | 0.5 | 0.5 | 0.5 | **1.5** |
| Rosenbrock | 0 (diverges at 1e-2) | 0.5 (diverges at 1e-2) | 0 (diverges at 3e-3) | 0.5 | 1.0 | **2.0** |

Adam is the most forgiving about its step size. GD, Momentum and Nesterov have
narrow windows and diverge just above them.

**7. Noisy objectives** (logistic regression, constant step size 0.05 for
every batch size, 50 epochs, 5 seeds):

| Batch | Constant step: final gap | Step ∝ 1/√k: final gap |
|---|---|---|
| 1 | 0.91 ± 0.24 | 0.0030 |
| 8 | 0.059 | 0.0034 |
| 64 | 0.0079 | 0.0090 |
| full | 0.0067 | 0.038 |

- **Smaller batches mean noisier gradients and a higher noise floor** with a
  constant step.
- **A decaying step removes the noise floor** (batch 1: 0.91 → 0.003), but it
  slows full-batch GD, which has no noise to average out.
- **Added Gaussian noise** (σ = 0.1, 1, 10) on a quadratic raises the final f
  roughly as σ² for GD and Momentum. At σ = 10, Adam's per-coordinate scaling
  kept it lower (2.1 vs 7.9 for GD).

## How to run

```bash
cd tracks/A_continuous
pytest -q                          # 19 tests, < 1 s
python run_experiments.py          # all 7 experiments (~4 min)
python run_experiments.py --quick  # ~1 min
```

## Limitations

- **Mostly small, synthetic problems,** chosen so the true optimum is known.
  The ML tasks are small (11 features; an 82-parameter network).
- **Step sizes are tuned on a coarse half-decade grid,** and stochastic
  methods on one held-out seed.
- **No momentum or step-size schedules** beyond 1/√k decay.
- **Natural gradient uses the *empirical* Fisher,** not the true Fisher.
