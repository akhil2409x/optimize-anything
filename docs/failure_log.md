# Failure log — AI-generated errors found and fixed

The brief requires at least three documented cases. Each entry: what the AI
produced, how the error was caught, the fix, and the lesson.

---

## F1 · Track E · ILP reported a cost four times higher than the real one

- **Tool:** Claude (claude.ai, Claude Opus 5.5), 2026-10-28
- **What happened:** the first version of `min_proximity` in `timetabling/ilp.py`
  returned the solver's own objective value. On a 30-exam instance with a
  60 s time limit it reported **200.59**, while the independent evaluator
  (`evaluate.proximity_cost`) gave **47.59** for the same timetable.
- **Cause:** the auxiliary variables `z[i,j,d]` are only bounded from below.
  At the optimum they sit on that bound, but a time-limited incumbent can leave
  them higher than needed, so the solver's objective over-states the true cost.
- **How it was caught:** comparing solver output with the evaluator on every
  run instead of trusting either one.
- **Fix:** `ILPResult.objective` is now always recomputed from the timetable by
  `evaluate.py`; the comment in the code explains why.
- **Lesson:** never report a solver's objective for a non-optimal run without
  re-evaluating the decision variables.

## F2 · Track E · Warm start silently ignored (two layers)

- **Tool:** Claude (claude.ai, Claude Opus 5.5), 2026-10-28
- **What happened:** the generated code called `setInitialValue` on the
  variables but never told the solver to use them. After adding
  `warmStart=True`, results with and without the warm start were still
  identical — on one instance the "warm-started" ILP (23.28) was even worse than
  the heuristic it started from (22.36), which is impossible if the start was used.
- **Cause:** PuLP 2.9's highspy-based `HiGHS` class accepts `warmStart` but
  only the command-line `HiGHS_CMD` actually passes it on. Also, a start
  that sets only the x variables violates the z constraints and is rejected.
- **How it was caught:** a sanity rule — a warm-started minimizer can never end
  worse than its start.
- **Fix:** `_HiGHSWarm` in `ilp.py` hands a complete solution (x **and** z) to
  highspy with `setSolution` before `run()`. The same instance now reaches
  20.05, better than the heuristic's 22.36.
- **Lesson:** check that an option has an observable effect; "accepted"
  parameters are not always used.

## F3 · Environment · Generated code broke on the installed library version

- **Tool:** Claude (claude.ai, Claude Opus 5.5), 2026-10-28
- **What happened:** the ILP code used `pulp.LpVariable(name, cat="Binary")`,
  which fails with `TypeError` on PuLP 4.2 (the latest release at the time),
  whose variable API was rewritten.
- **Fix:** pinned `pulp==2.9.0` in `requirements.txt`, the version the code
  and tests were written against.
- **Lesson:** AI tools write code for the library versions they were trained
  on. Pin versions and record them.

## F4 · Track B (also E) · ILP said "Optimal" but was off by one

- **Tool:** Claude (claude.ai, Claude Opus 5.5), 2026-10-28
- **What happened:** the knapsack ILP called HiGHS with default settings. On a
  strongly correlated instance with 100 items it returned value **34709** with
  status *Optimal*, while dynamic programming and branch-and-bound both found
  **34710**. Same on a weakly correlated 200-item instance (58745 vs 58746).
- **Cause:** HiGHS stops when the gap between incumbent and bound is below
  0.01% (`mip_rel_gap = 1e-4`) and still reports the run as optimal. On values
  around 35,000 that allows an error of up to 3.
- **How it was caught:** running three independent exact methods and checking
  they agree; the small brute-force tests (n = 14) were too small to show it.
- **Fix:** `gapRel=0.0, gapAbs=0.5` in `knapsack.py` (integer values, so a gap
  below 1 means truly optimal); `gapRel=0.0` added to Track E's ILP for
  consistency. Regression test `test_ilp_is_exactly_optimal_on_larger_instances`.
- **Lesson:** "Optimal" from a MIP solver means "optimal within tolerance".
  Cross-check exact methods against each other at realistic sizes, not only on
  toy instances.

## F5 · Track C · Quantum cost counted with the wrong metric (5× inflation)

- **Tool:** Claude (claude.ai, Claude Opus 5.5), 2026-10-28
- **What happened:** the first optimizer-comparison loop stopped each run
  after a budget of `qml.Tracker` *executions*. A single QAOA cost evaluation
  was counted as 5 executions and one gradient as 585, and the H₂ VQE run
  with the hardware-efficient ansatz stopped after about 4 steps at −0.14 Ha
  (true ground state −1.137 Ha). That looked like the method had failed.
- **Cause:** in PennyLane 0.45 the Tracker's `executions` field counts per
  measurement group, not per circuit. The number that matches theory is
  `simulations`: one forward pass = 1 circuit, one parameter-shift gradient of
  QAOA p=2 = 2 × 58 gate occurrences + 1 = 117 circuits.
- **How it was caught:** checking the counter against a hand calculation of
  what the parameter-shift rule must cost.
- **Fix:** budgets now use `simulations` (`qopt/train.py`, documented in the
  docstring); a test checks the budget is respected.
- **Lesson:** verify what a library's counter measures before using it as the
  basis of a "fair" comparison.

## F6 · Track C · Comment claimed the ansatz starts at Hartree–Fock

- **Tool:** Claude (claude.ai, Claude Opus 5.5), 2026-10-28
- **What happened:** `vqe.py` said "near-identity start: small angles keep the
  circuit close to the HF state". The measured starting energy was +0.89 Ha,
  far above Hartree–Fock (−1.117 Ha).
- **Cause:** the hardware-efficient ansatz ends every layer with a CNOT ladder,
  which changes |1100⟩ even when all rotation angles are zero.
- **Fix:** comment corrected; the README reports the real starting energy.
- **Lesson:** AI-written comments can state plausible but false facts. Test
  the claim (here: evaluate the circuit at zero angles).

## F7 · Track I · Trusted a library sampler that wasn't annealing

- **Tool:** Claude (claude.ai, Claude Opus 5.5), 2026-10-28
- **What happened:** the first version of the simulated-annealing baseline
  called `dwave.samplers.SimulatedAnnealingSampler` (v1.8.0). A test on a
  5-node cycle returned an average of only 0.82 of the optimal cut. Some of
  the returned states could be improved by flipping a single spin, which a
  finished anneal can never return.
- **How it was caught:** a unit test on a graph small enough to know the
  answer, followed by a check that the results did not change between 1 and
  1,000 sweeps, and that even a 5-spin ferromagnetic chain came back
  unaligned 4 times in 300.
- **Cause:** not identified. The library behaved as if it stopped sweeping
  after the first pass in this environment (Python 3.13, NumPy 2.5).
  It may be a binary-compatibility issue.
- **Fix:** a hand-written Metropolis simulated annealing (`qalgo/annealing.py`),
  about 25 lines and easy to explain, plus a regression test.
- **Lesson:** a heuristic baseline needs the same correctness checks as the
  method being studied. A broken baseline makes the other method look good.

## Note · Track I · Test expectation wrong, code right (Shor, a = 14)

The test first expected Shor's algorithm to factor 15 for every coprime a.
For a = 14 the order is 2 and 14¹ ≡ −1 (mod 15), the textbook case where
the post-processing yields only trivial factors. The test was corrected to
expect this failure. AI-written tests can encode wrong expectations as easily
as AI-written code.

## F8 · Track G · Benchmark gave the two schedulers different amounts of work

- **Tool:** Claude (claude.ai, Claude Opus 5.5), 2026-10-28
- **What happened:** `schedbench` sized every job in loop iterations, using
  a calibration it ran at the start of every run. The results said MLFQ made
  4 CPU-bound jobs 7–8% slower, in two separate runs.
- **How it was caught:**
  - **No idle time:** CPU utilization (CPU time ÷ makespan) was 99.6% for both
    kernels, so the slowdown wasn't idle CPU.
  - **Different calibration:** the MLFQ runs had calibrated to 40.6 million
    iterations per tick against 37.2 million for round robin.
  - **No real slowdown:** throughput in iterations per second was about the
    same, about 1% better for MLFQ.
- **Cause:** the calibration loop is itself preempted by the scheduler. MLFQ
  interrupts a CPU-bound process less often than round robin, so the loop
  measured about 9% more iterations per tick and every MLFQ job got 9% more
  work.
- **Fix:** `schedbench calib` measures once, on the original kernel, and the
  same iteration count is passed to every run on both kernels. The two earlier
  runs are kept in `tracks/G_xv6/results/superseded/` as evidence.
- **Lesson:** a benchmark's own measurement step must not depend on the thing
  being measured. Check that both sides did the same work before comparing times.

---

*(Add entries from Tracks A–J below as they happen — xv6 and SQL rewrites usually supply several.)*
