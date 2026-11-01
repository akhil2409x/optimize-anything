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

## F9 · Track H · ILP free to build useless indexes

- **Tool:** Claude (claude.ai, Claude Opus 5.5), 2026-10-29
- **What happened:** the index-selection ILP maximised Σ benefit·y with
  y ≤ x and a budget on x, but nothing in the objective charged for x itself.
  In a unit test with two overlapping indexes, it built both even though one
  could never be used, because the budget had room and the objective didn't
  change either way.
- **How it was caught:** a small hand-made test case where the right answer
  is known.
- **Fix:** a tiny storage penalty (ε · size, with ε about 10⁻⁶ × the smallest
  benefit), so among equally good solutions the smallest is chosen.
- **Impact on the reported results:** none. On TPC-H every index the ILP chose
  had positive benefit, and the selections were identical before and after the
  fix.
- **Lesson:** an optimization model gives you exactly what the objective asks
  for. Unpriced resources get used arbitrarily.

## Note · Track H · Noise larger than the effect

The first reading of the benchmark suggested a 4 MB index set made the
workload 13% faster (44.8 s vs 51.6 s). The default configuration's own three
passes ranged from 41.9 to 52.2 s, so the claim was dropped. Always look at
the spread between repeats before comparing medians.

## F10 · Track F · "0 mismatches" from a fuzzer that couldn't see two bugs

- **Tool:** Claude (claude.ai, Claude Opus 5.5), 2026-10-29
- **What happened:** the first random-program generator found 0 mismatches
  in 400 programs across 11 compiler configurations, which looked like strong
  evidence of correctness. Mutation testing (deliberately injecting known
  optimizer bugs) showed it was not:
  - **Missed entirely:** CSE that forgets to invalidate after an operand is
    redefined, and the classic `x / 2^k → x >> k` rewrite, which is wrong for
    negative x under truncating division.
  - **Caught:** only the dead-division bug.
  - **Another distortion:** 70% of the generated programs ended early on
    divide-by-zero.
- **Causes:**
  - **No repeated expressions:** random expressions almost never repeat after
    an operand changes, so stale-CSE cases never arose.
  - **Divisions folded away:** most divisions were constant-folded before
    strength reduction saw a variable dividend.
  - **A masked bug:** in the full pipeline, copy propagation runs before CSE
    and hides the CSE bug. It only shows when CSE runs alone or in another
    order, as in the pass-ordering experiment.
- **Fix:**
  - **Generator:** re-uses earlier expressions, uses a small variable pool,
    more negative values, and mostly safe divisors.
  - **Configurations:** tests run every pass alone and in reversed order.
  - **Targeted tests:** added for each hazard.
  - **Result:** `tests/mutation_check.py` now shows each of 5 real mutants
    caught by a targeted test. A sixth mutant is equivalent (it can't change
    any Tiny program's behaviour) and is documented as such.
- **Lesson:** a test suite that has never failed hasn't been shown to work.
  Inject known bugs and check they are caught.

## F11 · Track F · Ablation that couldn't remove the pass it claimed to remove

- **Tool:** Claude (claude.ai, Claude Opus 5.5), 2026-10-29
- **What happened:** the ablation reported "all passes except DCE" with
  exactly the same speed-up as "all passes" (1.42×), which would mean DCE
  contributes nothing.
- **Cause:** the pipeline always ran one final clean-up DCE after the chosen
  passes, so DCE was never actually absent.
- **Fix:** `optimize(..., cleanup=False)` for the ablation, then a rerun.
- **Lesson:** check that an ablation's configurations really differ in what
  they execute.

## F12 · Track D · Multi-task test set drawn from a different task

- **Tool:** Claude (claude.ai, Claude Opus 5.5), 2026-10-31
- **What happened:** task 2's training error fell to 2.2 while its test
  error stayed at 650. That was *worse* than predicting 0 (551), which
  suggested severe overfitting.
- **Cause:** not overfitting. The test set was made by calling the data
  generator with another seed. The seed also draws the target function's
  weights, so the test set came from a different task altogether.
- **How it was caught:** comparing test error with the trivial
  predict-zero baseline. A model that has learned anything should beat it.
- **Fix:** draw the data once and split it 50/25/25 into train, validation
  and test. A regression test now checks that a model fit on train beats
  predicting zero on test.
- **Lesson:** always print a trivial baseline next to a test score. Check
  that a seed controls only the sampling, not the problem itself.

## F13 · Track D · GradNorm code contradicted its own docstring

- **Tool:** Claude (claude.ai, Claude Opus 5.5), 2026-10-31
- **What happened:** the docstring said each task head always trains on its
  own gradient. The GradNorm branch scaled the heads by the learned weights
  anyway, so when task 2's weight went to 0, task 2's head froze.
- **Fix:** the weights now scale only the shared-layer gradient, as
  documented. The weight collapse itself is real GradNorm behaviour here and
  is reported in the README.
- **Lesson:** read generated code against its own comments, especially in
  the branches that are hardest to test.

## F14 · Track J · "Statement coverage" that contained no statements

- **Tool:** Claude (claude.ai, Claude Opus 5.5), 2026-10-31
- **What happened:** the first coverage collection reported 2,937
  requirements, all branch arcs and **zero statements**. The
  "statement-coverage" minimization would have been solved over an empty
  universe.
- **Cause:** with branch coverage on, coverage.py stores only arcs in its
  SQLite file; executed lines are derived from them. The generated reader
  queried the line table, which was empty.
- **How it was caught:** printing the requirement counts by kind before
  using them.
- **Fix:** derive statements from arc endpoints, as coverage.py does. A test
  now checks that every statement requirement is an endpoint of some arc.

## F15 · Track J · Mutation runs that never finished

- **Tool:** Claude (claude.ai, Claude Opus 5.5), 2026-10-31
- **What happened:** the first mutation-testing run used pytest-timeout
  (30 s per test). After 45 minutes it had finished fewer than 10% of the
  mutants, and one run on a single mutant didn't end within 10 minutes.
- **Cause:** a mutant such as `x * x % n` → `x * x // n` in the Miller–Rabin
  test makes the numbers double in length every iteration. The time goes
  into one huge C-level multiplication, which pytest-timeout's signal can't
  interrupt.
- **Fix:**
  - A wall-clock budget per mutant (3× the tests' normal time + 10 s),
    enforced by killing the process.
  - On overrun, each test is retried on its own to find which tests hang.
  - Progress is logged per mutant, so an interrupted run resumes.
  - The full run then took 60 minutes.
- **Lesson:** a timeout inside the process under test isn't a timeout.
  Enforce it from outside.

## F16 · Track D · Strided data shards that undid data parallelism

- **Tool:** Claude (claude.ai, Claude Opus 5.5), 2026-11-01
- **What happened:** rerunning Track D on a 4-core VM on the student's
  laptop, the data-parallel gradient with data resident in the workers was
  slower than serial.
- **Cause:** the generated code gave worker i the rows `X[i::p]`. That is a
  strided NumPy view, so every matrix product walked memory with a stride of
  p rows. On that machine, 20 products over a strided quarter took 0.26 s,
  against 0.03 s for a contiguous quarter. On the 2-core cloud machine the
  effect was small enough to go unnoticed.
- **Fix:** contiguous row blocks (`np.array_split`).
- **Remaining effect:** even with the fix, the 4-core VM shows no gain for
  these memory-bound products. That is reported as a negative result, not
  hidden.
- **Lesson:** a locality problem (Exp 5 measures exactly this) hid inside
  the parallel code itself. Benchmarks on a second machine catch what one
  machine hides.

---

*(Add entries from Tracks A–J below as they happen — xv6 and SQL rewrites usually supply several.)*
