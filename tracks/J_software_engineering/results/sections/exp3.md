## Exp 3 · Verification: rerun each reduced suite for real

| suite | tests | wall_s_median | wall_s_min | wall_s_max | test_time_s_median | all_pass | missing_requirements | missing_branches |
|---|---|---|---|---|---|---|---|---|
| full suite | 765 | 31.35 | 12.18 | 40.08 | 30.19 | True | 0 | 0 |
| statements+branches: ILP, minimise number of tests | 296 | 20.21 | 9.14 | 49.35 | 19.55 | True | 0 | 0 |
| statements+branches: ILP, minimise time | 300 | 27.47 | 8.90 | 36.94 | 26.76 | True | 0 | 0 |
| statements+branches: greedy per second + redundancy removal | 304 | 18.86 | 8.60 | 36.41 | 18.13 | True | 0 | 0 |
| statements: ILP, minimise time | 263 | 23.40 | 8.54 | 25.17 | 22.74 | True | 59 | 59 |

7 timing runs per suite, interleaved. `missing_*` compares a fresh coverage run of the reduced suite with the requirements covered by the full suite. Wall time includes ~0.5 s of pytest start-up and collection; `test_time_s` is the sum of junit per-test times.

