# Track B — Discrete optimization

Four requirements of the brief, one problem each:

| Requirement | Problem | Code |
|---|---|---|
| Compare greedy, DP, branch-and-bound and ILP on one problem | 0/1 knapsack | `discrete/knapsack.py` |
| Formulate a problem as QUBO / Ising | Max-Cut (unconstrained) and knapsack (with penalty) | `discrete/qubo.py` |
| Binary-to-spin conversion, penalty terms | x = (1 − s)/2; slack-bit penalty | `discrete/qubo.py` |
| Combinatorial application | feature selection | `discrete/feature_selection.py` |

Every track item: **objective · baseline · fair comparison · correctness check.**

## Knapsack

**Objective:** maximise Σ vᵢxᵢ subject to Σ wᵢxᵢ ≤ C, xᵢ ∈ {0,1}.

**Baseline:** greedy by value/weight ratio — what a person would do by hand.

| Method | Exact? | Complexity | Notes |
|---|---|---|---|
| `greedy_ratio` (baseline) | no | O(n log n) | no guarantee: 160 vs 220 on the textbook example |
| `greedy_half` | no | O(n log n) | max(greedy, best single item) ≥ OPT/2 |
| `dp` | yes | O(n·C) time and memory | pseudo-polynomial: depends on the size of C |
| `branch_and_bound` | yes | exponential worst case | own code, depth-first, LP-relaxation (fractional) bound, greedy incumbent |
| `ilp_highs` | yes | exponential worst case | PuLP model, HiGHS branch-and-cut, gap forced to 0 |

**Fair comparison:** same instances for every method; three instance classes
from Pisinger (2005) — uncorrelated, weakly correlated, strongly correlated;
3 seeds each; B&B capped at 10 s and ILP at 30 s; one CPU thread for all
Python methods. Quality is measured as the gap to the best value that some
method *proved* optimal.

**Correctness:** brute force over all 2¹⁴ subsets on 12 instances; the
textbook example by hand; the ½-approximation bound on 60 instances; edge cases
(zero capacity, nothing fits, everything fits); every solver's answer is
re-evaluated and an overweight answer raises an error; a regression test for
the solver-tolerance bug (failure log F4).

## QUBO / Ising

- **Max-Cut:** cut(x) = Σ wᵢⱼ(xᵢ + xⱼ − 2xᵢxⱼ), so the QUBO minimises −cut.
  Tests check f(x) = −cut(x) for all 2⁷ assignments, and that the Ising form
  gives the same energy as the QUBO for all 2⁶ assignments.
  The 10-node instance is saved to `results/maxcut_instance.json`, including
  all optimal bitstrings, as the classical baseline for **Track C (QAOA)**.
- **Knapsack:** inequality → equality with binary slack bits, then penalty
  P·(Σ wᵢxᵢ + Σ 2ᵏyₖ − C)². Experiment 3 sweeps P and shows the ground state
  breaks the constraint when P is too small. A 6-item problem already needs
  6 + 5 = 11 binary variables (qubits), which is why constrained problems are
  expensive on quantum hardware.

## Feature selection

Choose k of d features to maximise 5-fold CV accuracy of logistic regression
(breast-cancer data). Exhaustive search is the ground truth; greedy forward
selection is compared with random subsets given the **same number of model
evaluations**, so the comparison is fair.

## How to run

```bash
cd tracks/B_discrete
pytest -q                          # 31 tests, ~3 s
python run_experiments.py --quick  # ~40 s
python run_experiments.py          # ~6 min
```

## Results

Generated tables and plots: `results/SUMMARY.md`. Main findings:

1. **Greedy is close but rarely optimal.** Its gap shrinks as n grows (about
   0.3% at 50 items, under 0.02% at 800 for uncorrelated data) but is ~10×
   larger on strongly correlated instances (2.9% at 50 items).
2. **Instance class matters more than size for B&B.** It solves 800
   uncorrelated items in ~0.02 s, but on strongly correlated items it hits the
   5-million-node limit from n = 100. It usually *finds* the optimum early but
   cannot *prove* it: all item ratios are nearly equal, so the fractional
   bound prunes almost nothing.
3. **DP is pseudo-polynomial.** With n fixed at 60, DP time grows in step with
   capacity (≈15 s at C ≈ 3.2 million) while B&B and the ILP stay under 0.1 s.
4. **The ILP is the most robust exact method**, but strongly correlated
   instances with 400+ items still exceed its 30 s limit.
5. **QUBO penalty:** with P below about the largest item value, the ground
   state is often infeasible; at P ≥ 1 × max value it was correct on 30/30
   instances.
6. **Feature selection:** greedy forward selection used 42 evaluations instead
   of 495 and lost only ~0.2 percentage points of accuracy vs. exhaustive.

`greedy_half` returned the same value as `greedy_ratio` on every generated
instance; its single-item fallback only matters on adversarial cases
(see `test_greedy_half_rescues_bad_ratio_greedy`).

## Limitations

- B&B uses a simple depth-first search with the Dantzig bound; stronger bounds
  (Martello–Toth) or best-first search would help on correlated instances.
- When no method proves optimality (two of the three strongly correlated 800-item runs),
  the reference is the best value found, so reported gaps there are lower bounds
  on the true gap.
- Python implementations; absolute times would be much lower in C, but the
  growth rates are the point of the comparison.
