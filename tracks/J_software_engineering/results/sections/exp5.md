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

