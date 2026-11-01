# Track C — Quantum optimization

> **Simulation, not hardware.** Every number in this track comes from classical
> simulation in PennyLane 0.45.1 on a laptop-class CPU: `default.qubit` (exact
> state vector), `default.mixed` (density matrix, for the noise study) and
> `lightning.qubit` (barren plateaus). No real quantum computer was used, and
> no speedup is claimed: simulating n qubits costs 2ⁿ memory, so these runs
> are *slower* than the classical solvers they are compared with. The noise
> study uses a simplified depolarizing model, not a calibrated device.

| Requirement in the brief | Where |
|---|---|
| QAOA on the Track B QUBO (Max-Cut) | `qopt/maxcut.py`, Exp 1 |
| VQE | `qopt/vqe.py`, Exp 2 and 6 |
| PQCs / QNNs | VQE hardware-efficient ansatz (Exp 2), random layered PQCs (Exp 5); classifiers are in Track I |
| Parameter-shift, SPSA, quantum natural gradient | `qopt/train.py`, Exp 2 |
| Classical baseline | brute force + local search (Track B), Hartree–Fock + exact diagonalisation |
| Finite measurement shots | Exp 3 (and shots = 1000 runs in Exp 2) |
| Hardware noise | Exp 4 |
| Barren plateaus as circuits grow | Exp 5 |

## Problems

**Max-Cut (from Track B).** 10 nodes, 19 edges, optimum cut 14 (8 optimal
bitstrings). The cost Hamiltonian is built directly from the Ising coefficients
Track B saved in `B_discrete/results/maxcut_instance.json`, so both tracks solve
the same model: ⟨z|H_C|z⟩ = −cut(z) for every bitstring z (tested).

**QAOA.** Start in |+⟩ⁿ and apply p layers of exp(−iγₖH_C), which is one
IsingZZ(2γJᵢⱼ) per edge, then exp(−iβₖΣX), which is RX(2βₖ) on every qubit.
Classical optimizers tune the 2p angles to minimise ⟨H_C⟩.

**VQE for H₂.** STO-3G basis, 4 qubits. The baselines are Hartree–Fock and
exact diagonalisation in the 2-electron sector (FCI). The target is chemical
accuracy: within 1.6 mHa of exact. Two ansätze:
- **UCC:** one DoubleExcitation gate, which conserves particle number.
- **Hardware-efficient (HEA):** RY/RZ on every qubit plus a CNOT ladder,
  16 parameters.

## Fair comparison

- **Budget in circuit runs, not iterations.** A parameter-shift gradient
  costs 2 circuits per *gate* that uses a parameter: 117 circuits per step for
  QAOA p = 2, 33 for the HEA. SPSA costs 2 per step whatever the size, and QNG
  costs a gradient plus metric-tensor circuits. Each method gets the same
  number of circuits (4,000 for QAOA, 2,500 for VQE), counted by `qml.Tracker`.
  The counting unit was a real bug, logged as F5 in `docs/failure_log.md`.
- **Tuning.** Each optimizer's step size is picked from a grid of 3 values on a
  held-out seed (99), then fixed for seeds 0–2.
- **Measurement.** Progress is measured with exact expectations on a separate,
  uncounted device, so shot noise doesn't blur the curves.
- **Variability.** Results are reported as mean ± std over seeds, not the best run.

## Results

Full tables and plots are in `results/SUMMARY.md`.

**1. QAOA does improve with depth, but loses to a simple classical heuristic.**

| Method | Approx. ratio | P(optimal cut) | Shots to see optimum (99%) |
|---|---|---|---|
| QAOA p = 1 | 0.838 | 0.076 | 59 |
| QAOA p = 2 | 0.886 ± 0.006 | 0.165 | 26 |
| QAOA p = 3 | 0.906 ± 0.017 | 0.227 | 18 |
| QAOA p = 4 | 0.932 ± 0.003 | 0.334 | 12 |
| Local search (Track B, classical) | 0.949 | 78/200 runs exact | — |
| Random cut | 0.679 | 0.0078 | 588 |
| Brute force (classical) | 1.000 | 1 | — |

Sampling a few dozen bitstrings from QAOA p ≥ 2 almost surely contains an
optimal cut. Brute force over 2¹⁰ states is still trivial for a classical
computer, though, so this is a demonstration of the method, not an advantage.

**2. No optimizer wins everywhere at equal cost.**
- **QAOA p = 2:** all four methods end within about 0.3 of each other (−12.2 to
  −12.5, target −14). SPSA has the best mean and gets there in a few hundred
  circuits, while gradient methods spend 117 circuits per step.
- **VQE with the HEA:** results are spread widely across seeds (std ≈ 0.28 Ha),
  and with 3 seeds the ranking isn't reliable. Plain gradient descent clearly
  made the least progress within the budget.
- **One run looks too good.** SPSA with 1,000 shots reached −1.1354 ± 0.0013,
  better than its own exact-expectation run. With 3 seeds that's most likely
  chance (one seed stuck in the exact run), not evidence that shot noise helps.
- **UCC is the better choice for H₂.** It reaches chemical accuracy with 1
  parameter (result 6), so the HEA is the harder, less suitable ansatz here.

**3. Shot noise falls as 1/√shots.** The spread of the cost estimate × √shots
stays between 1.4 and 1.7 from 10 to 10,000 shots. At 100 shots the gradient
estimate has std 0.67 around a true value of 5.55; near an optimum, where the
gradient is small, that much noise would swamp it.

**4. Noise cancels the benefit of depth.** On the 8-node sub-graph, without
noise, p = 3 > p = 2 > p = 1 (0.899, 0.844, 0.785). At 2% depolarizing noise
per gate they are nearly equal (0.77). At 5% the order reverses: p = 1 is best
(0.714) and p = 3 worst (0.677), all heading towards the random-cut ratio 0.611.
Deeper circuits have more gates, so they collect more errors.

**5. Barren plateaus appear.** The variance of ∂C/∂θ₀ over random parameters:
- **Global cost** (depth = n): drops from 1.2 × 10⁻¹ at 2 qubits to
  1.2 × 10⁻⁴ at 12, about 1,000× smaller.
- **Local cost** (depth = n): decays more slowly, to 9.3 × 10⁻⁴.
- **Local cost, depth 2:** stays flat at about 0.1.

This matches Cerezo et al. (2021): local cost functions with shallow circuits
avoid the plateau.

**6. VQE solves H₂.** With the UCC ansatz it matches the exact energy to
within 0.0002 mHa at all 7 bond lengths from 0.3 to 2.5 Å. Hartree–Fock's
error grows from 8 mHa to 233 mHa as the bond stretches. A classical computer
diagonalises this 4-qubit problem instantly; H₂ is a correctness benchmark,
not a case for quantum advantage.

## How to run

```bash
cd tracks/C_quantum_optimization
pytest -q                               # 13 tests, ~20 s
python run_experiments.py --quick       # ~5 min
python run_experiments.py               # ~45 min
python run_experiments.py --only 4 5    # rerun some experiments; SUMMARY.md is rebuilt from CSVs
```

Requires Track B's `results/maxcut_instance.json`, which is committed in the repo.

## Correctness checks (`tests/test_qopt.py`)

- **Max-Cut model:**
  - ⟨z|H_C|z⟩ = −cut(z) on 27 bitstrings, including optimal ones.
  - The probability-vector bit order decodes back to Track B's optimal assignment.
  - At γ = β = 0 the expected cut is exactly half the edges.
  - Brute force on the 5-cycle finds max cut 4.
- **Gradients and devices:**
  - Parameter-shift gradients equal backprop gradients.
  - The noise-free density-matrix device matches the state-vector device.
  - Added noise moves the cost towards random guessing.
  - 20,000-shot estimates are unbiased.
- **H₂:**
  - The Hartree–Fock (−1.1167) and FCI (−1.1373) energies match literature values.
  - A UCC scan never goes below the exact energy (variational principle) and
    reaches chemical accuracy.
- **Training and barren-plateau code:**
  - The training loop respects its circuit budget and improves the cost.
  - The parameter-shift gradient used in the barren-plateau study equals autodiff.

## Limitations

- **Small seed count:** 3 seeds in Exp 2 are too few to rank optimizers
  confidently on VQE. Use 10 or more if time allows.
- **Simplified noise:** the noise model is depolarizing only. It has no readout
  error, no connectivity limits or SWAP overhead, and no T1/T2 decay. Real
  devices would do worse, especially for QAOA on a non-planar graph, which
  needs SWAPs.
- **Tiny problems:** they are chosen so they can be simulated exactly. They say
  nothing about QAOA's behaviour at sizes where classical solvers struggle.
- **Measurement grouping:** PennyLane measures each Hamiltonian in 5 groups
  when shots are finite. Max-Cut only needs 1 (all terms are Z-products), so
  real hardware cost per circuit would be lower for QAOA than PennyLane's
  default.
