# Track D — Submodular, parallel and distributed optimization

Three parts, each with its own baseline and correctness check:

| Part | Objective | Baseline | Correctness check |
|---|---|---|---|
| Submodular (`distopt/submodular.py`) | pick k items to maximise a set function | exact optimum (ILP or brute force); random sampling with the same number of evaluations | assumptions tested numerically; exhaustive diminishing-returns test on a small instance; ILP = brute force; greedy ≥ (1−1/e)·OPT |
| Parallel (`distopt/parallel.py`) | minimise wall-clock time / makespan | the serial program; static block scheduling; the ILP-optimal makespan | every parallel result equals the serial result (sums, gradients to 4e-14, layer outputs exactly); simulator checked by hand calculation and Graham's LPT bound |
| Distributed / multi-task (`distopt/learning.py`) | train a model across clients; train two tasks on a shared layer | centralized SGD with the same epochs; equal task weights | gradients vs finite differences; FedAvg with one client = centralized SGD exactly; MGDA's closed form vs a grid search |

22 tests (`tests/test_distopt.py`). **The cloud machine has only 2 CPU
cores**, so every speed-up there is capped at 2×. Exp 3–6 were therefore
rerun on a 4-core VM on the student's laptop (`results/mac_4core/`). All
scripts read the core count.

| Chapter 6 item | Where |
|---|---|
| Diminishing returns test; greedy; when the guarantee holds | Exp 1, 2 |
| Speed-up, efficiency, Amdahl, Gustafson | Exp 3 |
| Load balancing, makespan, work stealing | Exp 4 |
| Locality, synchronization, communication overhead | Exp 5 |
| Data / model / pipeline parallelism | Exp 6 |
| FedAvg vs centralized | Exp 7 |
| MGDA, GradNorm | Exp 8 |

## Results

Full tables: `results/SUMMARY.md` (and one CSV per experiment). Plots:
`greedy_ratio.png`, `speedup.png`, `fedavg.png`, `multitask.png`.

**1. Test the assumption before trusting greedy.** Random checks of
f(A+e) − f(A) ≥ f(B+e) − f(B) for A ⊆ B (3,000 trials each):

| Objective | Monotone violations | Diminishing-returns violations |
|---|---|---|
| Weighted coverage (40 sensors) | 0 | 0 (also exhaustive on a 7-set instance) |
| Facility location | 0 | 0 |
| Synergy (supermodular) | 0 | 2,419 |
| Feature-selection R² | 0 | 970 |

A single counter-example is enough. For R²: x₀ alone explains 0.7% of the
variance, but added to {x₁, x₇} it explains 59%, because x₀ and x₁ are only
useful together (y depends on x₀ − x₁).

**2. Greedy vs the optimum.**
- **Submodular (coverage):** greedy hit the exact ILP optimum for every k = 2–8,
  far above the 63.2% guarantee. Random sampling with the same number of
  evaluations reached 73–97%.
- **Lazy greedy** returned the same sets with far fewer evaluations
  (53 vs 190 at k = 5, 71 vs 292 at k = 8). Its shortcut is valid *only* for submodular f.
- **Facility location:** greedy reached 94–97% of optimal, above the bound.
  Random sampling with the same budget did as well or better (97.5–100%).
  On a 30-item instance that budget covers a large share of all subsets, so
  the guarantee is a floor, not a ranking.
- **R² (not submodular):** greedy fell **below** the guarantee: 50% of the
  optimum at k = 2, 49% at k = 4. It never picks x₀ or x₁ first, since each is
  useless alone. Random sampling did better (80–100%).
- **Synergy:** greedy took the decoy item first and reached 78–84% for k ≥ 3
  (99% at k = 2).
- **Conclusion:** without submodularity, greedy carries no guarantee.

**3. Speed-up** (Collatz workload, median of 3 runs). First on the 2-core cloud container:

| Setting | p = 2 | p = 3 | p = 4 | Prediction |
|---|---|---|---|---|
| Strong, no serial part | 1.72× | 1.84× | 1.80× | Amdahl 2.00× |
| Strong, 0.6 s serial (f = 0.70) | 1.41× | 1.43× | 1.44× | Amdahl 1.54× |
| Weak, 0.6 s serial | 1.49× | 1.60× | 1.71× | Gustafson 1.71× / 1.79× / 1.84× |

- **Below the formulas:** measured speed-ups fall short of both formulas
  because of process start-up and load imbalance, which they ignore. With no
  serial code at all, fitting Amdahl to the measurements gives an *effective*
  f of only 0.84.
- **Strong scaling flattens at 2 cores.** Extra processes (p = 3, 4) can't
  add compute. The slight gain at p = 3 comes from more, smaller chunks
  balancing better.
- **Weak scaling keeps improving** as the problem grows, because the serial
  part becomes a smaller share. That's Gustafson's point.
- **Amdahl limit:** with f = 0.70, no core count can beat 1/(1−f) = 3.3×.

Then rerun on a 4-core Linux VM on the student's MacBook Air (arm64,
Python 3.10; `results/mac_4core/`):

| Setting | p = 2 | p = 4 | p = 8 (oversubscribed) | Prediction at p = 4 |
|---|---|---|---|---|
| Strong, no serial part | 1.89× | **3.40×** | 3.28× | Amdahl 4.00× |
| Strong, 0.6 s serial (f = 0.61) | 1.41× | 1.76× | 1.77× | Amdahl 1.84× |
| Weak, 0.6 s serial | 1.40× | 2.20× | 2.63× | Gustafson 3.30× |

- **Near-linear with no serial code:** 85% efficiency at 4 cores. The
  effective Amdahl f is 0.94 (0.84 on the cloud machine).
- **Serial code caps it:** with 39% serial time, 4 cores give only 1.76×,
  close to Amdahl's 1.84×. Doubling to 8 processes adds nothing.
- **Same load-balancing pattern on 4 cores:**
  - Real runs: block 0.34 s (imbalance 1.83) vs cyclic/dynamic 0.21 s.
  - The simulator's ratios to the ILP optimum match the 2-core run within
    0.04.

**4. Load balancing.** 48 tasks whose sizes grow along the list (0.6–55 ms):

| Policy (2 real processes) | Makespan | Imbalance (max / mean busy) |
|---|---|---|
| Block (contiguous) | 0.83 s | 1.55 |
| Cyclic | 0.59 s | 1.06 |
| Dynamic (shared queue) | 0.59 s | 1.04 |
| Dynamic, big chunks | 0.79 s | 1.56 |

- **Simulator accuracy:** the simulator, fed the measured task times,
  predicted each real makespan to within 8%.
- **Scaling to more workers** (makespan / ILP optimum):

| Workers | Block | Cyclic | Dynamic | LPT | Work stealing |
|---|---|---|---|---|---|
| 2 | 1.56 | 1.02 | 1.03 | 1.00 | 1.02 |
| 4 | 1.86 | 1.07 | 1.08 | 1.00 | 1.03 |
| 8 | 1.98 | 1.15 | 1.22 | 1.01 | 1.13 |
| 16 | 2.15 | 1.35 | 1.41 | 1.04 | 1.18 |

- **Static block scheduling gets worse** as workers are added.
- **Dynamic scheduling suffers** when big tasks come last.
- **Work stealing stays within 18% of optimal** with 2–11 steals, and needs
  no task-time estimates.
- **LPT is near-optimal,** but needs the task times in advance.

**5. Overheads.**
- **Granularity:** 200,000 tiny tasks take 4.7 s with one task per message,
  0.03 s with chunks of 10,000, and 0.02 s serially. Here the overhead
  dominates and parallelism can't win.
- **Data shipping:** summing 128 MB across processes took 26× longer than
  serially, because pickling costs more than adding.
- **Locality:** column-wise traversal is 6.8× slower per element than
  row-wise (stride vs contiguous memory).
- **Synchronization:** a lock on every shared-counter increment is 21× slower
  than counting locally and reducing once. The totals are identical, so the
  locks cost time without changing the answer.

**6. Data, model and pipeline parallelism** (20 steps; after fixing
failure log F16):

| Kind | Machine | Serial | Data shipped every step | Data resident in workers |
|---|---|---|---|---|
| Data parallel (gradient over shards of 400k×50) | cloud, 2 processes, 3 runs | 0.32–0.53 s | 8.9–9.2 s | **0.22–0.32 s** |
| Model parallel (6000×6000 layer split by rows) | cloud, 2 processes, 3 runs | 0.37–0.52 s | 16.0–17.8 s | **0.25–0.28 s** |
| Data parallel | Mac VM, 4 processes | 0.32–0.45 s | 4.0–4.5 s | 0.71–0.80 s (slower) |
| Model parallel | Mac VM, 4 processes | 0.08 s | 6.1–6.2 s | 0.60–0.63 s (slower) |

- **Correctness:** results match serial to 6e-14 (data parallel) and exactly
  (model parallel) on both machines.
- **Communication volume:** each data-parallel step ships 156 MB in the
  naive version, but only 1.6–3.2 KB when data stays put. Keeping shards
  resident is what makes the cloud runs 1.2–2.4× faster than serial.
- **The 4-core VM is an honest negative result.** These kernels are
  memory-bandwidth-bound matrix–vector products: serial model-parallel takes
  only 4 ms per step there. Four processes add no memory bandwidth but do add
  a process round trip per step, so the parallel version loses.
  - *Lesson:* parallelism pays for compute-bound work (Exp 3: 3.40×), not
    for memory-bound work this small.
- **A bug the Mac run exposed (failure log F16):** the first version gave
  each worker every p-th row (`X[i::p]`). A strided view made each product
  about 9× slower on the Mac (0.26 s vs 0.03 s for 20 products). Shards are
  now contiguous row blocks.
- **Pipeline:** the 2-stage pipeline took 0.19–0.28 s vs 0.35–0.44 s
  sequentially on the cloud machine (fill/drain bubble 1/17), and 0.13 s vs
  0.23 s on the Mac.

**7. FedAvg vs centralized** (softmax regression on 8×8 digits, 10 clients):

| | IID clients | Non-IID (≈2 digits each) |
|---|---|---|
| Centralized SGD, 20 epochs | 0.967 | — |
| FedAvg, E = 1, 20 rounds | 0.944 | 0.922 |
| FedAvg, E = 10, 2 rounds (same work, 10× less traffic) | 0.948 | **0.837** |
| FedAvg, E = 10, 20 rounds | **0.969** | 0.926 |

- **IID clients:** more local epochs cut communication almost for free.
- **Non-IID clients:** they cost 2–4 points of accuracy. Long local training
  then causes **client drift**: each model overfits its own two digits. With
  the same total work, E = 10 is 8.5 points worse than E = 1.
- **Communication:** FedAvg sends 0.2–2 MB of model weights. Centralized
  training would need the raw data moved instead, which is what FedAvg
  avoids.

**8. Multi-task: equal weights vs MGDA vs GradNorm.** Two regression tasks
share one tanh layer; task 2's targets are 30× larger. Error is reported as
normalized MSE (1.0 = predicting the mean); step sizes are tuned on a
validation split, 3 seeds.

| Method | Task 1 nMSE | Task 2 nMSE | Note |
|---|---|---|---|
| Equal weights | 1.04 ± 0.01 | 0.004 | task 1 not learned at all |
| MGDA (normalised gradients) | **0.24 ± 0.02** | **0.004** | α = 0.5 exactly (see below) |
| MGDA (raw gradients) | 0.23 ± 0.02 | 0.124 | α ≈ 1, shared layer follows task 1 only |
| GradNorm (α = 1.5) | 0.28 ± 0.02 | 0.005 | weights collapsed to (2, 0) by step 500 |

- **Equal weighting fails:** task 2's gradient is about 30× larger, so the
  shared layer serves task 2 only. The tasks' gradients point in opposing
  directions (cosine −0.6).
- **MGDA works here, with two caveats:**
  - *Normalised gradients:* for two unit vectors the min-norm point is
    always the midpoint, so this is effectively "average the normalised
    gradients".
  - *Raw gradients:* textbook MGDA picks α ≈ 1, i.e. the smaller gradient.
    Task 2's accuracy drops 30-fold, because MGDA balances directions, not
    progress.
- **GradNorm fixed the imbalance**, but its learned weights went to the
  extreme: task 2 learns fast, so its weight is driven to 0. Task 2 holds its
  accuracy only because its own head keeps training. That's a known
  sensitivity of GradNorm to its α and weight step size.

## How to run

```bash
cd tracks/D_submodular_parallel_distributed
pytest -q                            # 22 tests, ~4 s
python run_experiments.py --quick    # ~1.5 min
python run_experiments.py            # ~4 min on 2 cores
python run_experiments.py --only 3 4 # rerun the speed-up / load-balance parts
```

On a laptop with more cores, rerun Exp 3–6: the process counts follow the
core count automatically. `results/mac_4core/` holds such a rerun, done
from this session on a 4-core VM on the student's MacBook Air.

## Limitations

- **At most 4 cores:** the 4-core run is a Linux VM on a laptop, not bare
  metal. The load-balance simulator covers up to 16 workers, but it ignores
  scheduling and cache costs.
- **Distributed work is simulated in one process.** FedAvg clients run one
  after another, so communication is counted in bytes, not measured over a
  network.
- **Small, synthetic models:** a linear softmax model and a 16-unit
  two-task network.
- **The R² and synergy counter-examples were built on purpose** to break
  submodularity. Real feature-selection problems are often only mildly
  non-submodular.
- **Simplified GradNorm:** it uses a sign-based weight update on full-batch
  gradients, not the paper's separate optimizer.
