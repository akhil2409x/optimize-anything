# Track I — Quantum algorithms and QML advantage

> **Simulation, not hardware.** All quantum results come from classical
> simulation in PennyLane 0.45.1 on a 2-core cloud CPU. "Noise" is a
> simulated depolarizing model, not a calibrated device. Quantum annealing is
> simulated exactly on a state vector; no D-Wave machine was used. The
> embedding study uses D-Wave's published hardware graphs, but nothing ran on
> them. A query-count advantage is a **theoretical** result reproduced in
> simulation; no **measured** quantum speedup is claimed.

| Requirement in the brief | Where |
|---|---|
| Every algorithm taught, on a simulator | `qalgo/algorithms.py`, `qalgo/annealing.py`, Exp 1 and 4 |
| At least two written by hand | All of DJ, BV, Grover, QFT, QPE and Shor's order finding are gate-by-gate. Tests compare QFT and Grover with `qml.QFT`, `qml.GroverOperator` and `qml.FlipSign`. |
| At least one experiment with a hardware noise model | Exp 3 (Grover, BV) |
| QML vs an equivalent classical model: accuracy, time, parameters, circuit evaluations | `qalgo/qml_models.py`, Exp 6 |
| A known complexity advantage vs its classical counterpart, by queries, gates and wall-clock | Exp 2 (Grover vs linear search) |
| Limits of embedding QUBOs on annealing hardware | Exp 5 |

## Results

Full tables and plots are in `results/SUMMARY.md`.

**1. Every algorithm gives the answer the theory guarantees (23/23 checks).**
- **Deutsch–Jozsa and Bernstein–Vazirani:** one oracle query each. The best
  deterministic classical algorithm needs 2ⁿ⁻¹ + 1 queries for DJ and n for BV.
- **QFT:** matches `qml.QFT` to 10⁻¹⁶.
- **QPE:** exact for phases with an n-bit expansion, and within 1/2ᵗ otherwise.
- **Shor (N = 15):**
  - For a = 2, 7, 11 and 13, half of the measurement outcomes give 3 × 5.
  - For a = 14 it fails by design: the order r = 2 and 14¹ ≡ −1 (mod 15), so
    the post-processing only gives trivial factors.
  - The modular multiplication is a compiled black-box unitary. Building it from
    reversible arithmetic is where most of Shor's real gate cost lies.

**2. Grover's query advantage is real, but gate count and simulation time grow fast.**

| N | Grover queries | Linear search (mean) | Grover gates, decomposed | Grover sim time | Linear search time |
|---|---|---|---|---|---|
| 16 | 3 | 8.4 | 442 | 4.6 ms | 0.4 µs |
| 256 | 12 | 127.6 | 22,544 | 46 ms | 4.6 µs |
| 4096 | 50 | 2,019 | — | 38 s | 89 µs |

- **Queries:** Grover's count follows (π/4)√N exactly, a 40× reduction at
  N = 4096.
- **Gates:** each multi-controlled Z, decomposed without helper qubits, costs
  far more than one gate. At N = 256 a single oracle call is already about
  1,900 gates.
- **Wall-clock:** simulating 12 qubits takes 38 s, against 89 µs for a Python
  linear search. Simulation costs about 2ⁿ, so the advantage exists only in
  oracle queries on real hardware, never in simulated wall-clock time.

**3. Noise erases Grover's advantage faster than BV's.**

| Noise per gate | Grover n = 6 | BV n = 8 |
|---|---|---|
| 0 | 0.997 | 1.000 |
| 1% | 0.213 | 0.847 |
| 5% | 0.018 (random guess: 0.016) | 0.439 |

- **Why the difference:** Grover's circuit is deep (repeated oracle +
  diffuser) while BV's is shallow, so Grover collects far more errors.
- **Noise changes the best iteration count:** for n = 5, the noise-free best
  is k = 4 iterations (P = 0.999). At 1% noise it drops to k = 3 (0.44), and
  at 3% to k = 2 (0.15).
- **This underestimates the damage:** noise is applied per *logical* gate, but
  hardware would decompose the multi-controlled Z into many gates.

**4. Quantum annealing obeys the adiabatic theorem, but classical annealing is far cheaper.**

| Method | Setting | P(optimal cut) | Cost |
|---|---|---|---|
| Quantum annealing (simulated) | T = 100 / 1,000 / 3,000 | 0.06 / 0.45 / 0.87 | 0.6 / 6.5 / 20 s of simulation |
| Simulated annealing (classical) | 30 / 100 sweeps | 0.97 / 1.00 | 0.6 / 2 ms per read |
| Brute force | all 2¹⁰ states | 1.00 | 0.2 ms |

- **The slow anneal times are expected.** The smallest spectral gap is 0.044,
  just before the end of the anneal (s = 0.99), where the optimum (cut 14) has
  to separate from the cut-13 states. The adiabatic theorem asks for anneal
  times well beyond 1/gap² ≈ 520, and that is where success climbs.
- **A cheap classical method wins easily.** Simulated annealing reaches 100%
  success in 100 sweeps.

**5. Embedding needs far more physical qubits than the problem has variables.**

| Problem | Variables | Pegasus P6 qubits | Chimera C16 qubits |
|---|---|---|---|
| Track B Max-Cut (sparse) | 10 | 11 | 19 |
| Fully connected K40 | 40 | 212 (5.3×) | 521 (13×) |
| Knapsack QUBO, 20 items | 29 | 109 | 268 |

- **Why the overhead:** each hardware qubit couples to only about 15 others
  (Pegasus) or 6 (Chimera), so one variable becomes a chain of qubits.
- **Constraints make it worse:** the knapsack penalty term couples every pair
  of variables, so a 20-item knapsack needs 109–268 physical qubits.
- **Chains add error:** longer chains (up to 6 qubits for K30 on Pegasus) can
  break during the anneal.

**6. QML is no better than classical here (5 seeds, same splits).**

| Dataset | Best classical | VQC | Quantum kernel | Simple classical |
|---|---|---|---|---|
| Moons (2 features) | RBF SVM 0.950 ± 0.011 | 0.877 ± 0.025 | 0.893 ± 0.025 | Logistic 0.870, MLP 0.860 |
| Cancer (PCA, 4 features) | MLP 0.967 ± 0.038 | 0.890 ± 0.040 | 0.853 ± 0.032 | Logistic 0.940, RBF SVM 0.937 |

- **Accuracy:** the quantum models are roughly on a par with logistic
  regression and the MLP on moons, where the spreads overlap. They are clearly
  below the RBF SVM there, and below every classical model on the cancer data.
- **Training cost:** the quantum models take 15–83 s to train in simulation,
  against 1–300 ms for the classical ones.
- **Hardware cost:**
  - Training the VQC would take 35,000–70,000 circuit runs with
    parameter-shift gradients.
  - The quantum kernel needs a circuit for every pair of points: 9,730 for
    140 training points, plus 8,400 more to classify 60 test points.

## When quantum approaches help, and when they don't

**Where they help**
- **Query complexity, in theory:** Grover (√N), DJ and BV (one query) and Shor
  (an exponential speedup for factoring) are real advantages with oracle access.
- **Physics-native problems:** quantum annealing works when the problem's own
  energy landscape is the computation (Exp 4).

**Where they don't, in these experiments**
- **Simulation overhead:** every quantum result here costs about 2ⁿ classical
  work. Simulated Grover is about 400,000× slower than linear search at N = 4096.
- **Noise:**
  - Deep circuits like Grover fall to random guessing at about 5% error per gate.
  - The best iteration count shrinks as noise grows.
  - Track C showed the same effect for QAOA depth.
- **Barren plateaus:** in Track C, gradients shrank about 1,000× from 2 to 12
  qubits with a global cost function.
- **Data loading:**
  - Each kernel entry is a separate circuit, so cost grows with n².
  - Each VQC prediction re-encodes the input features.
  - Classical models read the data once.
- **Hardware connectivity:** embedding overhead multiplies qubit needs by
  5–13× for dense QUBOs (Exp 5).
- **Strong classical baselines:** simulated annealing, an RBF SVM and plain
  local search (Track B) match or beat every quantum method at these sizes.

**Contrast with parallel computing (Track D):**

- **What is measured.** A parallel speed-up is wall-clock time on real cores.
  Track D measured 1.72× on 2 cores and 3.40× on 4 cores with no serial
  code, but only 1.41× (2 cores) to 1.76× (4 cores) with 30–40% serial
  code.
- **Amdahl's limit.** That second case can never beat 1/(1−f) = 3.3×,
  however many cores are added.
- **The quantum advantages here are query counts.** Examples: one oracle
  call instead of 2ⁿ⁻¹+1 (Deutsch–Jozsa), and √N Grover iterations instead of
  N/2.
- **No wall-clock gain on a simulator.** Simulating them classically costs
  2ⁿ memory and time, so they show up as fewer queries, never as less time.
- **Parallelism is a constant-factor gain** (at most the number of cores, and
  less after overheads) that any problem can get. A quantum advantage is
  asymptotic, but only for specific problem structures, and needs fault-
  tolerant hardware to become a time saving.

## How to run

```bash
cd tracks/I_quantum_algorithms_qml
pytest -q                               # 33 tests, ~10 s
python run_experiments.py --quick       # ~4 min
python run_experiments.py               # ~25 min
python run_experiments.py --only 2 3    # rerun some; SUMMARY.md is rebuilt from CSVs
```

Requires Track B's `results/maxcut_instance.json`, which is committed in the repo.

## Correctness checks (`tests/test_qalgo.py`, 33 tests)

- **Against library implementations:** the hand-written QFT and its inverse
  equal `qml.QFT` for 2–5 qubits, and the hand-written Grover equals the
  library's `FlipSign` + `GroverOperator` on 4 targets.
- **Against theory:**
  - Grover's success matches sin²((2k+1)θ).
  - DJ is correct for every balanced function on up to 4 bits.
  - BV recovers 6 different secrets.
  - QPE is exact on 3 phases.
  - Shor works for every coprime a, including the a = 14 failure case.
- **Annealing:**
  - The hand-written RX layer equals a Kronecker product of RX matrices.
  - Cut values agree with Track B's optimal assignments.
  - Annealing gives the uniform answer when instantaneous and the ground state
    when slow.
  - Simulated annealing solves small graphs (a regression test for failure log F7).
  - Embeddings use at least one physical qubit per variable.
- **QML:**
  - The quantum kernel is symmetric, has a unit diagonal and is positive semi-definite.
  - Preprocessing is fitted on training data only.
  - The VQC learns.

## Limitations

- **Simplified noise:** depolarizing only, applied per logical gate. There is
  no readout error, connectivity limit or gate decomposition, so real hardware
  would do worse.
- **Shor's black box:** a QubitUnitary hides the cost of modular arithmetic,
  and N = 15 is the smallest interesting case.
- **Tiny QML data:** 140 training points and 2–4 features. Quantum kernels are
  argued to help only for specially structured data, which this is not.
- **Annealing gap estimate:** the spectral gap is computed in the bit-flip
  symmetric sector on an 80-point grid, so it is an estimate.
