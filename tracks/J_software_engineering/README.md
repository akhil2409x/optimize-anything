# Track J — Software-engineering optimization: test-suite minimization

**Subject:** [more-itertools](https://github.com/more-itertools/more-itertools)
v11.2.1 (commit `4aeb433`, MIT licence).
- **Code under test:** a public open-source library, ~7,300 lines of library
  code and ~9,200 lines of tests.
- **Suite size:** 765 test functions, plus ~21,000 subTests.
- **Not committed:** `setup.sh` clones the library; the checkout itself
  isn't in the repo.

## Formulation

Choose which tests to run in CI so that everything the full suite exercises is
still exercised, in the least time.

| | |
|---|---|
| Decision variables | x_t ∈ {0, 1}: run test t or not (765 variables) |
| Objective | min Σ_t c_t x_t, with c_t = the test's measured time + 1 ms overhead (or c_t = 1 to minimise the count) |
| Constraints | Σ_{t covers r} x_t ≥ 1 for every requirement r covered by the full suite |
| Requirements | 1,960 statements + 2,937 branch arcs of the library (coverage.py, per-test dynamic contexts) |

This is **weighted set cover**, an NP-hard problem.

| Method | What it is |
|---|---|
| Baseline 1 | run everything |
| Baseline 2 | random order until every requirement is covered (naive reduction) |
| Greedy | Chvátal's algorithm, an H(m) ≈ ln m approximation: most new requirements, or most per second, plus a redundancy-removal pass |
| Exact | the ILP solved with HiGHS (gap 0) |
| Lower bound | the LP relaxation |

**Prioritization** (order all tests so faults show early) is scored with APFD
(Average Percentage of Faults Detected) and APFDc (its time-weighted version).

**Correctness checks:**
- **Reduced suites, for real:** every reduced suite is rerun in a fresh
  process under coverage and compared with the full suite.
- **ILP and greedy on toy cases:** the ILP matches brute force, and greedy
  stays within its ln bound.
- **Mutation testing** measures what a reduced suite gives up: 1,116 seeded
  bugs, with each test's pass/fail recorded.
- **Unit tests:** 15 (`tests/test_tsmin.py`).

## Results

Full tables are in `results/SUMMARY.md`. Plots: `time_profile.png` and
`prioritization.png`.

**1. Profile: two tests are the bottleneck.** 96% of tests take under 10 ms,
but two tests take 73% of the 17 s of test time:
- `TestConcurrentTee::test_concurrent_consumers`: 7.3 s, 10⁵ items × 100
  threads.
- `PrimeFunctionTests::test_primes`: 5.2 s, checks every prime below 10⁵.

The threaded test also makes the wall time erratic: whole-suite runs took
12–45 s.

**2. Minimization (statements + branches).** Both slow tests are *essential*:
each is the only test covering some branch. So no cover can drop them.

| Method | Tests | Test time | Solve time |
|---|---|---|---|
| Full suite | 765 | 17.03 s | — |
| Random order until covered (10 seeds) | 367 (48%) | 15.67 s | — |
| Greedy, most new requirements | 302 (39%) | 14.66 s | 0.3 s |
| Greedy per second + redundancy removal | 304 | 14.42 s | 0.7 s |
| **ILP, minimise number of tests** | **296 (39%)** | 14.89 s | 0.07 s |
| **ILP, minimise time** | 300 | **14.42 s (−15%)** | 0.06 s |

- **Greedy is close to optimal:** within 2% of the ILP on count and equal on
  time.
- **Easy instance:** the LP bound equals the ILP optimum (14.721), so the
  relaxation already has an integral solution.
- **Only 15% saved:** 61% fewer tests, but only 15% less time. Minimization
  removes the many cheap tests; the time sits in the two it must keep.

**3. Verification.** Each reduced suite was rerun in a fresh process
under coverage, then timed 7 times, interleaved with the full suite:

| Suite | Tests | All pass | Missing statements/branches | Wall time min / median / max |
|---|---|---|---|---|
| Full suite | 765 | yes | — | 12.2 / 31.4 / 40.1 s |
| ILP, min count (statements + branches) | 296 | yes | 0 | 9.1 / 20.2 / 49.4 s |
| ILP, min time (statements + branches) | 300 | yes | 0 | 8.9 / 27.5 / 36.9 s |
| Greedy per second + redundancy removal | 304 | yes | 0 | 8.6 / 18.9 / 36.4 s |
| ILP, min time (statements only) | 263 | yes | **59 branches** | 8.5 / 23.4 / 25.2 s |

- **Coverage is preserved exactly** by every statement + branch suite. The
  statement-only suite really does lose 59 branches.
- **Wall time: honest negative result.** The best runs improve from 12.2 s
  to about 8.7 s (−29%). But the medians are swamped by the threaded test,
  whose time varies 3–4-fold from run to run. With noise this large, the
  15% saving can't be shown reliably in wall-clock time.
- **Next step:** this points to fixing the bottleneck itself (Exp 5).

**4. The cost of reducing (mutation testing).**
- **Baseline:** the full suite kills 960 of the 1,116 mutants (86%).
- **What the reduced suites kill** (share of those 960 the full suite kills):

| Reduced suite | Tests | Mutants killed | Random subsets of the same size |
|---|---|---|---|
| Greedy, most new requirements (statements + branches) | 302 | **92.8%** | 71.3% |
| ILP, min count (statements + branches) | 296 | 92.3% | 69.2% |
| ILP, min time (statements + branches) | 300 | 89.3% | 69.2% |
| ILP, min count (statements only) | 261 | 90.5% | 64.7% |

- **The price of reduction:** a coverage-preserving suite still misses 7–11%
  of detectable faults. Coverage says a line *ran*, not that its result was
  *checked*.
- **Random subsets of the same size do much worse** (65–72%), so coverage is
  a useful proxy.
- **Branch coverage beats statement coverage** at about +2 points.
- **Minimising time costs detection:** it swaps in cheap tests that assert
  less (−3 points vs minimising count).

**5. Fixing the bottleneck itself.** `patches/shrink-slow-tests.patch`
shrinks the two tests to the smallest inputs tried that keep **all** their
statements and branches:
- the threaded test to 10⁴ items × 10 threads;
- the primes test to primes below 10⁴.

10³ was too small. It lost the lock-contention branch in `concurrent_tee`
(the reason the stress test exists) and a Miller–Rabin loop arc. Coverage
caught that.

| | Before | After |
|---|---|---|
| `test_concurrent_consumers` | 7.27 s | 0.015 s |
| `test_primes` | 5.23 s | 0.19 s |
| Mutants they kill (of those they execute) | 10/12 and 43/74 | **10/12 and 43/74 (no loss)** |

| Suite | Tests | Test time | 2-worker CI makespan (LPT, simulated) | Measured wall time (7 runs) |
|---|---|---|---|---|
| Original full suite | 765 | 17.0 s | 8.5 s | 12.0–44.9 s (median 32.7) |
| ILP-minimised | 300 | 14.4 s | 7.3 s | 21.1–37.3 s (median 27.5) |
| Full suite, two tests shrunk | 765 | 4.7 s | 2.4 s | **6.0–6.4 s (median 6.1)** |
| ILP-minimised + shrunk | 300 | 2.1 s | 1.1 s | **2.9–3.3 s (median 3.1)** |

- **Shrinking the two tests is the real win:** a 3.6× shorter test time, a
  5× lower median wall time, and the 7-fold wall-time spread is gone. It
  costs nothing in coverage or in detection of these mutants.
- **Combined with the ILP:** a 3.1 s run, against a median of 32.7 s.
- **Why minimization alone can't do this:** it can only drop tests, and the
  time sat in essential ones. Once the bottleneck is fixed, minimization
  matters again (4.7 s → 2.1 s).
- **Parallel CI (Track D link):** 2 workers can't beat the longest test.
  With the original tests that floor is 7.3 s.

**6. Prioritization** (all 765 tests still run; seconds of test time until a
given share of the killable mutants is detected):

| Order | APFD | APFDc | 50% found | 90% found |
|---|---|---|---|---|
| Original file order | 0.537 | 0.762 | 1.92 s | 10.9 s |
| Random (10 seeds) | 0.701 | 0.726 | 3.09 s | 12.4 s |
| Total coverage | 0.810 | 0.408 | 14.2 s | 17.1 s |
| Additional coverage | **0.893** | 0.571 | 8.2 s | 14.7 s |
| **Additional coverage per second** | 0.849 | **0.898** | **0.06 s** | 14.9 s |

- **Count-based APFD misleads when test times vary 1,000-fold.**
  "Total coverage" puts the big, slow tests first. It looks good by test
  count, but takes 14 s to find half the faults.
- **The cost-aware order finds half the faults in 0.06 s.**
- **The last 10% always waits for the slow tests,** because some faults are
  only caught there.

## Trade-offs

- **Correctness:**
  - Minimization keeps 100% of coverage but loses 7–11% of fault detection.
  - Shrinking the two tests kept both on these mutants, but it weakens
    concurrency stress testing. 10 threads may miss a race that 100 would
    catch, and that isn't fully measured by single-line mutants.
  - A sensible policy: run the reduced, shrunk suite on every commit, and the
    full original suite nightly or before a release.
- **Maintainability:**
  - A reduced suite must be recomputed whenever code or tests change.
    Collecting coverage takes ~4 min, and the ILP solves in < 0.1 s.
  - The shrink patch changes two upstream tests and would need upstream
    agreement.
- **Readability:** no change to the library. The selection is a list of test
  IDs.

## How to run

```bash
cd tracks/J_software_engineering
./setup.sh                                # clone more-itertools v11.2.1
pytest -q                                 # 15 tests
python run_experiments.py                 # reuses results/measurements.json and kill_matrix.json (~30 min)
python run_experiments.py --remeasure     # redo timing, coverage and mutation testing (~1.5 h on 2 cores)
python run_experiments.py --only 3 5      # rerun some experiments
```

## Limitations

- **One subject on one 2-core machine;** wall times of the threaded test are
  very noisy. More repetitions would tighten the medians.
- **Simple mutants:** one AST node each. 156 mutants survived the full suite,
  some of them equivalent (not checked by hand). Faults that need two
  changes, or that sit in concurrency timing, are not modelled.
- **The coverage–fault link is specific to this suite.** A reduced suite
  chosen on today's coverage may age badly as the code changes; it wasn't
  evaluated across versions.
- **The shrink sizes were picked from a short list** (10³, 10⁴), not
  searched exhaustively.
