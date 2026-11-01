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

