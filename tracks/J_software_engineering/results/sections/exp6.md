## Exp 6 · Prioritization: find faults early (all tests still run)

| order | APFD | APFDc | s_to_50pct | s_to_90pct | s_to_100pct |
|---|---|---|---|---|---|
| original | 0.537 | 0.762 | 1.92 | 10.90 | 17.79 |
| random (10 seeds) | 0.701 | 0.726 | 3.09 | 12.38 | 17.75 |
| total | 0.810 | 0.408 | 14.16 | 17.10 | 17.79 |
| additional | 0.893 | 0.571 | 8.20 | 14.65 | 17.79 |
| additional/s | 0.849 | 0.898 | 0.06 | 14.92 | 17.78 |

APFD counts tests; APFDc weighs each test by its time (1 = every fault found instantly). `s_to_X` = seconds of test time until X% of the killable mutants are detected.

