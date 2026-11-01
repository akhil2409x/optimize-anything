# Track J results

Machine: Linux-6.18.44-fc-v185-x86_64-with-glibc2.39, Python 3.13.18, 2 CPU cores. Last run 2026-10-31 06:56.

## Exp 1 · Profiling the baseline

Subject: more-itertools 4aeb4338ae (v11.2.1): 765 test functions (plus thousands of subTests), library ~7,300 lines, tests ~9,200 lines.
Sum of per-test times (median of 3 runs, no coverage): **17.03 s**; full `pytest` wall time per run: 40.3, 18.0, 12.3 s (includes start-up and collection; the spread comes mostly from one threaded test).

| rank | test | seconds | share | cumulative |
|---|---|---|---|---|
| 1 | TestConcurrentTee::test_concurrent_consumers | 7.272 | 42.7% | 42.7% |
| 2 | PrimeFunctionTests::test_primes | 5.229 | 30.7% | 73.4% |
| 3 | PrimeFunctionTests::test_nth_prime_approximate | 0.694 | 4.1% | 77.5% |
| 4 | ExtractTests::test_all_orderings | 0.552 | 3.2% | 80.7% |
| 5 | TestRandomOrderedRange::test_lcg_mitigation | 0.352 | 2.1% | 82.8% |
| 6 | TestSynchronized::test_concurrent_calls | 0.332 | 1.9% | 84.7% |
| 7 | PolynomialFromRootsTests::test_large | 0.318 | 1.9% | 86.6% |
| 8 | RunningMedianTests::test_vs_statistics_median | 0.274 | 1.6% | 88.2% |

- 735 of 765 tests (96%) take under 10 ms. The two slowest tests take 73% of all test time: **they are the bottleneck**.


## Exp 2 · Minimization as weighted set cover

- **statements**: 1960 requirements; 155 essential tests (sole cover of some requirement); both slow tests essential: True.
- **statements+branches**: 4897 requirements; 191 essential tests (sole cover of some requirement); both slow tests essential: True.

Full suite: 765 tests, 17.03 s of test time. Costs include 1 ms per test; the time ILP's objective counts that too.

| criterion | method | tests | tests_pct | test_time_s | time_pct | covers_all | solve_s |
|---|---|---|---|---|---|---|---|
| statements | random order until covered (10 seeds, mean) | 328 | 43% | 15.57 | 91.4% | True | 0.00 |
| statements | greedy, most new requirements | 266 | 35% | 14.63 | 85.9% | True | 0.12 |
| statements | greedy, new requirements per second | 284 | 37% | 14.57 | 85.6% | True | 0.10 |
| statements | greedy per second + redundancy removal | 268 | 35% | 14.39 | 84.5% | True | 0.20 |
| statements | ILP, minimise number of tests | 261 | 34% | 14.86 | 87.3% | True | 0.06 |
| statements | ILP, minimise time | 263 | 34% | 14.39 | 84.5% | True | 0.06 |
| statements | LP relaxation: lower bound 14.654 vs ILP objective 14.655 (both incl. 1 ms/test) |  |  |  |  |  |  |
| statements+branches | random order until covered (10 seeds, mean) | 367 | 48% | 15.67 | 92.0% | True | 0.00 |
| statements+branches | greedy, most new requirements | 302 | 39% | 14.66 | 86.1% | True | 0.27 |
| statements+branches | greedy, new requirements per second | 323 | 42% | 14.60 | 85.7% | True | 0.24 |
| statements+branches | greedy per second + redundancy removal | 304 | 40% | 14.42 | 84.7% | True | 0.71 |
| statements+branches | ILP, minimise number of tests | 296 | 39% | 14.89 | 87.4% | True | 0.06 |
| statements+branches | ILP, minimise time | 300 | 39% | 14.42 | 84.7% | True | 0.08 |
| statements+branches | LP relaxation: lower bound 14.721 vs ILP objective 14.721 (both incl. 1 ms/test) |  |  |  |  |  |  |


## Exp 3 · Verification: rerun each reduced suite for real

| suite | tests | wall_s_median | wall_s_min | wall_s_max | test_time_s_median | all_pass | missing_requirements | missing_branches |
|---|---|---|---|---|---|---|---|---|
| full suite | 765 | 31.35 | 12.18 | 40.08 | 30.19 | True | 0 | 0 |
| statements+branches: ILP, minimise number of tests | 296 | 20.21 | 9.14 | 49.35 | 19.55 | True | 0 | 0 |
| statements+branches: ILP, minimise time | 300 | 27.47 | 8.90 | 36.94 | 26.76 | True | 0 | 0 |
| statements+branches: greedy per second + redundancy removal | 304 | 18.86 | 8.60 | 36.41 | 18.13 | True | 0 | 0 |
| statements: ILP, minimise time | 263 | 23.40 | 8.54 | 25.17 | 22.74 | True | 59 | 59 |

7 timing runs per suite, interleaved. `missing_*` compares a fresh coverage run of the reduced suite with the requirements covered by the full suite. Wall time includes ~0.5 s of pytest start-up and collection; `test_time_s` is the sum of junit per-test times.


## Exp 4 · What the reduced suites give up: mutation testing

1116 mutants on lines the suite executes; each was run against every test that executes its line (60 min on 2 processes; a run over 3× its normal time + 10 s is retried test by test and a test that still overruns counts as failing). Killed by the full suite: **960** (86.0%); survived: 156 (missed faults or equivalent mutants).

| criterion | method | tests | kill_rate | random_same_size | missed | missed_ops |
|---|---|---|---|---|---|---|
| statements | ILP, minimise number of tests | 261 | 90.5% | 64.7% | 91 | arith: 18, bool: 8, cmp: 22, int: 43 |
| statements | ILP, minimise time | 263 | 86.6% | 66.2% | 129 | arith: 24, bool: 8, cmp: 33, int: 64 |
| statements | greedy per second + redundancy removal | 268 | 88.1% | 67.8% | 114 | arith: 15, bool: 7, cmp: 30, int: 62 |
| statements | greedy, most new requirements | 266 | 91.4% | 64.8% | 83 | arith: 11, bool: 7, cmp: 25, int: 40 |
| statements | greedy, new requirements per second | 284 | 88.4% | 67.3% | 111 | arith: 13, bool: 6, cmp: 30, int: 62 |
| statements+branches | ILP, minimise number of tests | 296 | 92.3% | 69.2% | 74 | arith: 13, bool: 9, cmp: 19, int: 33 |
| statements+branches | ILP, minimise time | 300 | 89.3% | 69.2% | 103 | arith: 17, bool: 8, cmp: 28, int: 50 |
| statements+branches | greedy per second + redundancy removal | 304 | 89.9% | 72.1% | 97 | arith: 13, bool: 7, cmp: 26, int: 51 |
| statements+branches | greedy, most new requirements | 302 | 92.8% | 71.3% | 69 | arith: 10, bool: 7, cmp: 22, int: 30 |
| statements+branches | greedy, new requirements per second | 323 | 90.5% | 72.2% | 91 | arith: 10, bool: 6, cmp: 24, int: 51 |

`kill_rate` = share of the full suite's killed mutants that the reduced suite also kills. `random_same_size` = mean over 20 random subsets with the same number of tests.

Examples of faults missed by `greedy, most new requirements` (statements+branches):

- `more_itertools/more.py:458` `index.stop is None` → `(index.stop is not None)`
- `more_itertools/more.py:464` `(start < 0) or (stop < 0)` → `(start < 0 and stop < 0)`
- `more_itertools/more.py:1068` `n - len(window)` → `(n + len(window))`
- `more_itertools/more.py:1074` `1` → `2`
- `more_itertools/more.py:1074` `step - 1` → `(step + 1)`


## Exp 5 · Fixing the bottleneck: shrink the two slowest tests

`patches/shrink-slow-tests.patch`: `test_concurrent_consumers` 10⁵ items × 100 threads → 10⁴ × 10; `test_primes` checks primes below 10⁵ → below 10⁴. Sizes were chosen as the smallest tried that keep each test's statement and branch coverage (10³ items lost the lock-contention branch `more.py:5571→5570` and the Miller–Rabin loop arc `recipes.py:1270→1269`).

| test | before_s | after_s | requirements_before | lost | gained |
|---|---|---|---|---|---|
| TestConcurrentTee::test_concurrent_consumers | 7.272 | 0.015 | 63 | 0 | 0 |
| PrimeFunctionTests::test_primes | 5.229 | 0.186 | 120 | 0 | 0 |

| test | mutants_executed | killed_before | killed_after | killed_only_by_this_test |
|---|---|---|---|---|
| TestConcurrentTee::test_concurrent_consumers | 12 | 10 | 10 | 10 |
| PrimeFunctionTests::test_primes | 74 | 43 | 43 | 10 |

| suite | tests | test_time_s | makespan_2_workers_lpt | measured_wall_s_median | measured_wall_s_min | measured_wall_s_max |
|---|---|---|---|---|---|---|
| original full suite | 765 | 17.03 | 8.51 | 32.67 | 11.96 | 44.86 |
| ILP-minimised (statements+branches), original tests | 300 | 14.42 | 7.27 | 27.52 | 21.09 | 37.25 |
| full suite, two tests shrunk | 765 | 4.73 | 2.36 | 6.14 | 6.00 | 6.39 |
| ILP-minimised + two tests shrunk | 300 | 2.12 | 1.06 | 3.11 | 2.94 | 3.26 |

`test_time_s` uses per-test medians; measured wall times are 7 interleaved runs each. `makespan_2_workers_lpt`: simulated CI time on 2 parallel workers with longest-first scheduling (Track D) - with the original tests it can never go below the 7 s threaded test.


## Exp 6 · Prioritization: find faults early (all tests still run)

| order | APFD | APFDc | s_to_50pct | s_to_90pct | s_to_100pct |
|---|---|---|---|---|---|
| original | 0.537 | 0.762 | 1.92 | 10.90 | 17.79 |
| random (10 seeds) | 0.701 | 0.726 | 3.09 | 12.38 | 17.75 |
| total | 0.810 | 0.408 | 14.16 | 17.10 | 17.79 |
| additional | 0.893 | 0.571 | 8.20 | 14.65 | 17.79 |
| additional/s | 0.849 | 0.898 | 0.06 | 14.92 | 17.78 |

APFD counts tests; APFDc weighs each test by its time (1 = every fault found instantly). `s_to_X` = seconds of test time until X% of the killable mutants are detected.

