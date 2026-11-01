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

