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

