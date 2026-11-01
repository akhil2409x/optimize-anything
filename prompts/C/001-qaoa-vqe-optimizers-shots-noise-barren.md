# C · 001 · QAOA, VQE, optimizers, shots, noise, barren plateaus

- **Tool / model:** Claude (claude.ai), Claude Opus 5.5
- **Date:** 2026-10-28
- **Kind of work:** planning, code generation, testing, debugging, documentation

## Original prompt

Claude explained why it suggested Track J next and offered C as the
"hardest work first" alternative. Reply:

```
lets finish the hardest work first & do C next
```

## AI response (summary)

Claude read the Track C requirements from the brief and built, with PennyLane 0.45:

- `qopt/maxcut.py` — QAOA built from Track B's saved Ising coefficients
- `qopt/vqe.py` — H₂ Hamiltonian (STO-3G), UCC and hardware-efficient ansätze,
  Hartree–Fock and exact (FCI) baselines
- `qopt/train.py` — one training loop for GD / Adam (parameter-shift), SPSA and
  QNG with a budget in circuit runs
- `qopt/barren.py` — gradient-variance study for random PQCs
- `tests/test_qopt.py` — 13 tests
- `run_experiments.py` — six experiments, CSVs, plots, `SUMMARY.md`

## Revisions during the session

1. Wrong cost counter (`executions`) → `simulations` (failure log F5).
2. False comment about the ansatz starting at Hartree–Fock (F6).
3. Recording calls moved to a separate, uncounted device so they do not eat
   the optimizer's budget.
4. The density-matrix simulator needed 16 s per evaluation at 10 qubits, so the
   noise study uses the 8-node induced sub-graph (0.6 s per evaluation).
5. CSV writer fixed for rows with different columns.

## Resulting code change

Commit: `[AI:claude] Track C: QAOA, VQE, optimizers, shots, noise, barren plateaus`
= local commit `7bd080d` (on GitHub: `<repo URL>/commit/7bd080d`)

## Verification note

To do by the student: rerun `pytest -q`; derive on paper why
exp(−iγJZZ) = IsingZZ(2γJ); work out the parameter-shift rule for one RX gate;
explain why SPSA's cost per step does not grow with the number of gates; be
ready to say clearly that every result is a classical simulation.
