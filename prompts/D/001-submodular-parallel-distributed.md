# D · 001 · Submodular greedy, parallel speed-up, FedAvg, MGDA / GradNorm

- **Tool / model:** Claude (claude.ai), Claude Opus 5.5
- **Date:** 2026-10-30 – 31
- **Kind of work:** code generation, testing, experiments, debugging, documentation

## Original prompt

After Track A, Claude asked whether to continue with Track D. Reply:

```
yes continue
```

## AI response (summary)

Claude read the Track D and Chapter 6 requirements and wrote:

- `distopt/submodular.py`: four objectives (coverage with an exact ILP,
  facility location, a supermodular "synergy" function, feature-selection R²),
  a random test of monotonicity and diminishing returns, greedy, lazy greedy,
  brute force and random sampling.
- `distopt/parallel.py`: real multiprocessing measurements (strong/weak
  scaling, load balancing, granularity, data shipping, locality, locks,
  data/model/pipeline parallelism), Amdahl/Gustafson fits, a scheduler
  simulator (block, cyclic, dynamic, LPT, work stealing) and an ILP for the
  optimal makespan.
- `distopt/learning.py`: FedAvg vs centralized softmax regression on digits
  (IID and non-IID clients), and a two-task network trained with equal
  weights, MGDA (normalised and raw) and GradNorm.
- `tests/test_distopt.py` (22 tests) and `run_experiments.py` (8 experiments).

## Revisions during the session

1. **Test set from a different task** (failure log F12). The multi-task test
   data was generated with another seed, which also drew new target
   functions. Fixed by splitting one draw into train/validation/test; a
   regression test was added.
2. **GradNorm weighted the heads.** The docstring said heads always use their
   own gradient, but the GradNorm branch also scaled the heads. When task 2's
   weight went to 0, its head stopped training. Made the code match the
   docstring.
3. **MGDA on normalised gradients always gives α = 0.5.** This was noticed
   from the results, then proved in a test. The raw-gradient variant was
   added so both are reported honestly.
4. **A wrong test expectation:** a test expected greedy to fall below
   1 − 1/e on the synergy function; it only reaches 78–84%. Replaced it with
   the R² case, where greedy really does fall below the bound (50%).
5. **Problem too small for speed-up.** The first strong-scaling problem was
   too small (process start-up dominated), so it was enlarged.

## Resulting code change

Commit `[AI:claude] Track D: submodular greedy, parallel speed-up, FedAvg, MGDA/GradNorm` (fill in hash after pushing)

## Verification note

To do by the student:
- Check one diminishing-returns counter-example from Exp 1 by hand.
- Derive the two-task MGDA closed form α* = (g₂−g₁)·g₂ / ‖g₁−g₂‖² and explain
  why it is 0.5 for unit vectors.
- Rerun Exp 3–6 on your own laptop (more cores) and compare the speed-ups
  with Amdahl's prediction.
- Rerun `pytest -q`.
