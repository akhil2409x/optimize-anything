# Track D results

Machine: Linux-6.18.44-fc-v185-x86_64-with-glibc2.39, Python 3.13.18, 2 CPU cores, full run, 2026-10-31 03:02.

## Exp 1 · Testing the assumptions (random A ⊆ B, e ∉ B)

| objective | trials | monotone_violations | submodular_violations | worst_violation |
|---|---|---|---|---|
| weighted coverage | 3000 | 0 | 0 | 0 |
| facility location | 3000 | 0 | 0 | 0 |
| synergy (supermodular) | 3000 | 0 | 2419 | 24.23 |
| feature-selection R^2 | 3000 | 0 | 970 | 0.5829 |

- Counter-example for synergy (supermodular): A=[1, 2], B=[0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11], e=9: gain(A)=5.82 < gain(B)=30
- Counter-example for feature-selection R^2: A=[], B=[1, 7], e=0: gain(A)=0.00705 < gain(B)=0.59

## Exp 2 · Greedy vs exact vs random sampling

| objective | k | exact_method | greedy_ratio | random_same_budget_ratio | greedy_evals | lazy_evals | lazy_same_answer | exact_seconds |
|---|---|---|---|---|---|---|---|---|
| weighted coverage | 2 | ILP | 1.000 | 0.965 | 79 | 44 | True | 0.02 |
| weighted coverage | 3 | ILP | 1.000 | 0.775 | 117 | 47 | True | 0.02 |
| weighted coverage | 4 | ILP | 1.000 | 0.917 | 154 | 52 | True | 0.02 |
| weighted coverage | 5 | ILP | 1.000 | 0.729 | 190 | 53 | True | 0.02 |
| weighted coverage | 6 | ILP | 1.000 | 0.767 | 225 | 60 | True | 0.02 |
| weighted coverage | 7 | ILP | 1.000 | 0.797 | 259 | 64 | True | 0.02 |
| weighted coverage | 8 | ILP | 1.000 | 0.753 | 292 | 71 | True | 0.02 |
| facility location | 2 | brute force | 0.942 | 1.000 | 59 | 61 | True | 0.00 |
| facility location | 3 | brute force | 0.952 | 0.982 | 87 | 62 | True | 0.01 |
| facility location | 4 | brute force | 0.967 | 0.975 | 114 | 75 | True | 0.12 |
| synergy (supermodular) | 2 | brute force | 0.994 | 0.994 | 23 | 14 | True | 0.00 |
| synergy (supermodular) | 3 | brute force | 0.814 | 0.942 | 33 | 15 | True | 0.00 |
| synergy (supermodular) | 4 | brute force | 0.778 | 0.884 | 42 | 16 | True | 0.00 |
| synergy (supermodular) | 5 | brute force | 0.800 | 0.939 | 50 | 17 | True | 0.00 |
| synergy (supermodular) | 6 | brute force | 0.835 | 0.932 | 57 | 18 | True | 0.00 |
| feature-selection R^2 | 2 | brute force | 0.504 | 1.000 | 19 | 13 | True | 0.00 |
| feature-selection R^2 | 3 | brute force | 0.554 | 0.800 | 27 | 14 | True | 0.00 |
| feature-selection R^2 | 4 | brute force | 0.493 | 0.876 | 34 | 17 | True | 0.01 |
| feature-selection R^2 | 5 | brute force | 1.000 | 0.779 | 40 | 19 | True | 0.01 |

Guarantee for monotone submodular f: greedy_ratio ≥ 1 − 1/e = 0.632.

## Exp 3 · Speed-up: strong and weak scaling (2 cores available)

| scaling | serial_part_s | n | p | seconds | speedup | efficiency | amdahl_prediction |
|---|---|---|---|---|---|---|---|
| strong | 0 | 200000 | 1 | 1.434 | 1.00 | 1.00 | 1.00 |
| strong | 0 | 200000 | 2 | 0.835 | 1.72 | 0.86 | 2.00 |
| strong | 0 | 200000 | 3 | 0.781 | 1.84 | 0.61 | 2.00 |
| strong | 0 | 200000 | 4 | 0.795 | 1.80 | 0.45 | 2.00 |
| strong | 0.6 | 200000 | 1 | 1.989 | 1.00 | 1.00 | 1.00 |
| strong | 0.6 | 200000 | 2 | 1.407 | 1.41 | 0.71 | 1.54 |
| strong | 0.6 | 200000 | 3 | 1.392 | 1.43 | 0.48 | 1.54 |
| strong | 0.6 | 200000 | 4 | 1.380 | 1.44 | 0.36 | 1.54 |
| weak | 0.6 | 100000 | 1 | 1.283 | 1.00 | 1.00 | 1.00 |
| weak | 0.6 | 200000 | 2 | 1.373 | 1.49 | 0.75 | 1.71 |
| weak | 0.6 | 300000 | 3 | 1.806 | 1.60 | 0.53 | 1.79 |
| weak | 0.6 | 400000 | 4 | 2.202 | 1.71 | 0.43 | 1.84 |

`amdahl_prediction` is Amdahl's law for strong rows and Gustafson's law for weak rows, using the parallel fraction measured from the serial timing and p capped at the core count. Measured speed-up for weak scaling = T(serial, same big problem) / T(p).

- serial part 0.0s: parallel fraction from timing f = 1.000, Amdahl fit on p ≤ 2: f = 0.836; Amdahl limit 1/(1−f) = unbounded
- serial part 0.6s: parallel fraction from timing f = 0.698, Amdahl fit on p ≤ 2: f = 0.585; Amdahl limit 1/(1−f) = 3.3×

## Exp 4 · Load balancing and makespan

48 tasks, measured serially: total 1.08s, shortest 0.6 ms, longest 55.1 ms.

Real runs on 2 processes (best of 3); imbalance = max busy / mean busy; `sim_makespan_s` is the simulator's prediction from the serial task times:

| policy | workers | makespan_s | imbalance | sim_makespan_s |
|---|---|---|---|---|
| block | 2 | 0.833 | 1.55 | 0.846 |
| cyclic | 2 | 0.593 | 1.06 | 0.552 |
| dynamic | 2 | 0.591 | 1.04 | 0.558 |
| dynamic-big | 2 | 0.791 | 1.56 | 0.846 |

Simulator on the same measured task times (zero scheduling cost); `vs_optimal` = makespan / ILP optimum:

| workers | block | cyclic | dynamic | lpt | steal | steals | ILP optimum (s) |
|---|---|---|---|---|---|---|---|
| 2 | 1.560 | 1.018 | 1.030 | 1.000 | 1.024 | 2 | 0.542 (Optimal) |
| 4 | 1.858 | 1.065 | 1.082 | 1.001 | 1.026 | 5 | 0.271 (Optimal) |
| 8 | 1.981 | 1.152 | 1.215 | 1.014 | 1.126 | 9 | 0.136 (Optimal) |
| 16 | 2.148 | 1.354 | 1.406 | 1.043 | 1.175 | 11 | 0.068 (Optimal) |

## Exp 5 · Overheads: granularity, communication, locality, synchronization

| test | setting | seconds |
|---|---|---|
| 200k tiny tasks | serial loop | 0.018 |
| 200k tiny tasks | 2 processes, chunksize 1 | 4.661 |
| 200k tiny tasks | 2 processes, chunksize 100 | 0.096 |
| 200k tiny tasks | 2 processes, chunksize 10000 | 0.032 |
| sum 8 arrays of 16 MB | serial (no copying) | 0.009 |
| sum 8 arrays of 16 MB | 2 processes (arrays pickled to workers) | 0.237 |

| access | seconds | ns_per_element |
|---|---|---|
| row by row (contiguous) | 0.009 | 1 |
| column by column (stride n*8 bytes) | 0.061 | 6.83 |

| method | total | seconds |
|---|---|---|
| shared counter, lock per increment | 80000 | 0.148 |
| local counts, one reduction at the end | 80000 | 0.007 |

## Exp 6 · Data, model and pipeline parallelism

20 steps each on 2 processes. 'shipped' sends the data/weights every step; 'resident' keeps them in the workers and sends only the small vector:

| kind | serial_s | shipped_s | resident_s | max_abs_diff |
|---|---|---|---|---|
| data parallel (gradient over data shards) | 0.571 | 8.807 | 0.361 | 3.9e-14 |
| model parallel (layer rows split) | 0.461 | 16.658 | 0.284 | 0.0e+00 |

- Data parallel moves 156 MB per step when shipped vs 1600 bytes when resident.
- Pipeline, 2 stages × 16 micro-batches: sequential 0.351s, pipelined 0.219s (ideal (M+S−1)·t = 0.188s, fill/drain bubble 6%); outputs in order: True.

## Exp 7 · FedAvg vs centralized (softmax regression on 8×8 digits, 10 clients)

| setting | data | local_epochs | rounds | epochs_of_work | test_acc | train_loss | comm_MB |
|---|---|---|---|---|---|---|---|
| centralized SGD | all |  |  | 20 | 0.967 | 0.121 | 0.00 |
| FedAvg (same work) | IID | 1 | 20 | 20 | 0.944 | 0.309 | 1.98 |
| FedAvg (same work) | IID | 2 | 10 | 20 | 0.946 | 0.311 | 0.99 |
| FedAvg (same rounds) | IID | 2 | 20 | 40 | 0.957 | 0.219 | 1.98 |
| FedAvg (same work) | IID | 5 | 4 | 20 | 0.946 | 0.316 | 0.40 |
| FedAvg (same rounds) | IID | 5 | 20 | 100 | 0.963 | 0.151 | 1.98 |
| FedAvg (same work) | IID | 10 | 2 | 20 | 0.948 | 0.324 | 0.20 |
| FedAvg (same rounds) | IID | 10 | 20 | 200 | 0.969 | 0.123 | 1.98 |
| FedAvg (same work) | non-IID | 1 | 20 | 20 | 0.922 | 0.600 | 1.98 |
| FedAvg (same work) | non-IID | 2 | 10 | 20 | 0.900 | 0.781 | 0.99 |
| FedAvg (same rounds) | non-IID | 2 | 20 | 40 | 0.930 | 0.499 | 1.98 |
| FedAvg (same work) | non-IID | 5 | 4 | 20 | 0.869 | 1.100 | 0.40 |
| FedAvg (same rounds) | non-IID | 5 | 20 | 100 | 0.933 | 0.395 | 1.98 |
| FedAvg (same work) | non-IID | 10 | 2 | 20 | 0.837 | 1.387 | 0.20 |
| FedAvg (same rounds) | non-IID | 10 | 20 | 200 | 0.926 | 0.342 | 1.98 |

Communication = rounds × 10 clients × 2 transfers × model size (5.1 KB). Centralized training moves the raw data once instead (0.62 MB), which FedAvg avoids for privacy.

## Exp 8 · Multi-task learning: equal weights vs MGDA vs GradNorm

Two regression tasks sharing a 16-unit tanh layer; task 2's targets are 30× larger. nMSE = test MSE / target variance (1.0 = predicting the mean). 3 seed(s), 3000 full-batch steps.

| method | lr | task1_test_nmse | task1_sd | task2_test_nmse | task2_sd | mean_alpha | final_weights | cos_g1_g2_final |
|---|---|---|---|---|---|---|---|---|
| equal | 0.01 | 1.037 | 0.010 | 0.0043 | 0.0000 |  |  | -0.59 |
| mgda | 0.1 | 0.237 | 0.021 | 0.0044 | 0.0001 | 0.500 |  | -0.01 |
| mgda_raw | 0.1 | 0.229 | 0.017 | 0.1236 | 0.0145 | 1.000 |  | -0.26 |
| gradnorm | 0.03 | 0.284 | 0.020 | 0.0048 | 0.0004 |  | (2.00, 0.00) | -0.10 |

Total run time 214s.

## Exp 6 rerun after the F16 fix (contiguous shards), cloud container, 3 runs

| run | data parallel: serial / shipped / resident (s) | model parallel: serial / shipped / resident (s) | pipeline: sequential / pipelined (s) |
|---|---|---|---|
| 1 | 0.531 / 8.920 / 0.221 | 0.378 / 16.681 / 0.284 | 0.413 / 0.194 |
| 2 | 0.322 / 8.992 / 0.269 | 0.367 / 17.827 / 0.265 | 0.435 / 0.279 |
| 3 | 0.421 / 9.158 / 0.320 | 0.520 / 16.003 / 0.251 | 0.351 / 0.216 |

The 4-core rerun on the student's laptop is in `mac_4core/` (CSV files; machine details in `mac_4core/machine.txt`).
