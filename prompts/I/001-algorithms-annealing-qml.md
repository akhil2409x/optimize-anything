# I · 001 · Quantum algorithms, annealing, QML vs classical

- **Tool / model:** Claude (claude.ai), Claude Opus 5.5
- **Date:** 2026-10-28
- **Kind of work:** planning, code generation, testing, debugging, documentation

## Original prompt

After Track C, Claude suggested Track I next (same PennyLane setup). Reply:

```
yes start it, after this how many more tracks are left
```

## AI response (summary)

Claude read the Track I requirements from the brief and built:

- `qalgo/algorithms.py` — hand-written Deutsch–Jozsa, Bernstein–Vazirani,
  Grover (oracle + diffuser), QFT, QPE, and Shor order finding for N = 15
- `qalgo/annealing.py` — exact simulation of quantum annealing (Trotterised
  state vector), hand-written simulated annealing, spectral gap, minor
  embedding into Pegasus/Chimera with minorminer
- `qalgo/qml_models.py` — variational classifier and quantum-kernel SVM vs
  logistic regression, RBF SVM and MLP
- `tests/test_qalgo.py` — 33 tests, including hand-written circuits vs
  PennyLane library operators (qml.QFT, qml.GroverOperator, qml.FlipSign)
- `run_experiments.py` — six experiments, CSVs, plots, `SUMMARY.md`

Answer to the question: after Track I, six tracks remain (A, D, F, G, H, J).

## Revisions during the session

1. dwave-samplers' simulated annealing returned unconverged states →
   replaced by a hand-written version (failure log F7).
2. Test for Shor with a = 14 expected success; the theory says it must fail →
   test corrected (failure log note).
3. Random Bernstein–Vazirani / Deutsch–Jozsa secrets could be all-zero, which
   proves nothing → generation forces at least one 1.

## Resulting code change

Commit: `[AI:claude] Track I: quantum algorithms, annealing, QML vs classical`
= local commit `7803599` (on GitHub: `<repo URL>/commit/7803599`)

## Verification note

To do by the student: rerun `pytest -q`; trace Grover for n = 2 by hand (one
iteration gives the target with probability 1); work through Shor's classical
post-processing for a = 7 (measurements 0, 64, 128, 192 out of 256); explain why
a = 14 fails; explain why the quantum kernel needs n(n−1)/2 circuits.
