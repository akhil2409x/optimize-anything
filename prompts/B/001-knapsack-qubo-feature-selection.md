# B · 001 · Knapsack methods, QUBO/Ising, feature selection

- **Tool / model:** Claude (claude.ai), Claude Opus 5.5
- **Date:** 2026-10-28
- **Kind of work:** code generation, testing, debugging, documentation

## Original prompt

Claude offered: "I can build Track B now (knapsack with greedy, DP, my own
branch-and-bound and an ILP, plus the Max-Cut QUBO) while you do steps 1–3.
Want me to start?"

```
yes do it
```

## AI response (summary)

- `discrete/knapsack.py` — Pisinger instance classes; brute force, ratio greedy
  (baseline), ½-approximation greedy, DP, own depth-first branch-and-bound with
  fractional bound, ILP (PuLP + HiGHS)
- `discrete/qubo.py` — QUBO type, brute force, exact binary-to-spin conversion,
  Max-Cut QUBO and local-search baseline, knapsack QUBO with slack bits and penalty
- `discrete/feature_selection.py` — exhaustive vs greedy forward vs random subsets
- `tests/test_discrete.py` — 31 tests
- `run_experiments.py` — five experiments, CSVs, plots, `SUMMARY.md`,
  `maxcut_instance.json` for Track C

## Revisions during the session

1. Scaling probe showed the ILP one below the DP optimum while reporting
   "Optimal" → zero MIP gap, regression test (failure log F4). Same setting
   applied to Track E.
2. Penalty sweep refined around the feasibility transition (0.05, 0.2 added).
3. Max-Cut JSON extended with all optimal bitstrings, needed in Track C to
   measure QAOA's probability of sampling an optimal cut.

## Resulting code change

Commit: `[AI:claude] Track B: knapsack methods, QUBO/Ising, feature selection` (fill in hash after pushing)

## Verification note

To do by the student: rerun `pytest -q` and `python run_experiments.py`;
work through the textbook knapsack example (60/100/120, C=50) by hand for the
DP table and the B&B tree; derive the x = (1−s)/2 substitution on paper for one
edge; be ready to explain why B&B struggles on strongly correlated instances.
