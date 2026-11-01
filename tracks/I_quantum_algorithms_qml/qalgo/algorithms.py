"""Quantum algorithms, written gate by gate (the brief asks for at least two by hand;
here all of DJ, BV, Grover, QFT, QPE and Shor's order finding are hand-written,
and the tests check them against PennyLane's own library operators).

Conventions: wire 0 is the most significant bit of every register, which matches
PennyLane's probability ordering, so `int(bitstring, 2)` indexes `qml.probs`.

All functions build circuits inside a QNode. `noise` > 0 adds a depolarizing
channel after every gate on the wires it touches (density-matrix device only),
which is how the hardware-noise experiment is run.
"""
from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import pennylane as qml


# ------------------------------------------------------------------ helpers
def _noisy(noise: float, wires):
    if noise:
        for w in wires:
            qml.DepolarizingChannel(noise, wires=w)


def H(w, noise=0.0):
    qml.Hadamard(w); _noisy(noise, [w])


def X(w, noise=0.0):
    qml.PauliX(w); _noisy(noise, [w])


def CNOT(c, t, noise=0.0):
    qml.CNOT([c, t]); _noisy(noise, [c, t])


def MCZ(wires, noise=0.0):
    """Z on the last wire controlled on all others = phase -1 on |1...1>."""
    if len(wires) == 1:
        qml.PauliZ(wires[0])
    else:
        qml.ctrl(qml.PauliZ(wires[-1]), control=wires[:-1])
    _noisy(noise, wires)


def device(n_wires, noise=0.0, shots=None):
    name = "default.mixed" if noise else "default.qubit"
    return qml.device(name, wires=n_wires, shots=shots) if shots else qml.device(name, wires=n_wires)


# ------------------------------------------------------------------ Deutsch–Jozsa
def dj_oracle(kind: str, n: int, noise=0.0):
    """f: {0,1}^n -> {0,1} on inputs 0..n-1, output (ancilla) n.
    'constant0': f=0, 'constant1': f=1, 'balanced:<s>': f(x) = s.x mod 2 (s != 0)."""
    if kind == "constant0":
        return
    if kind == "constant1":
        X(n, noise)
        return
    s = kind.split(":")[1]
    for i, bit in enumerate(s):
        if bit == "1":
            CNOT(i, n, noise)


def deutsch_jozsa(kind: str, n: int, noise=0.0):
    """Returns P(all inputs measure 0): 1 for constant f, 0 for balanced f (one query)."""
    dev = device(n + 1, noise)

    @qml.qnode(dev)
    def circuit():
        X(n, noise)
        for w in range(n + 1):
            H(w, noise)
        dj_oracle(kind, n, noise)
        for w in range(n):
            H(w, noise)
        return qml.probs(wires=range(n))
    return float(circuit()[0])


# ------------------------------------------------------------------ Bernstein–Vazirani
def bernstein_vazirani(secret: str, noise=0.0):
    """One oracle query recovers the n-bit secret s of f(x) = s.x mod 2.
    Returns the probability distribution over n-bit outputs."""
    n = len(secret)
    dev = device(n + 1, noise)

    @qml.qnode(dev)
    def circuit():
        X(n, noise)
        for w in range(n + 1):
            H(w, noise)
        for i, bit in enumerate(secret):        # oracle: phase kickback onto |->
            if bit == "1":
                CNOT(i, n, noise)
        for w in range(n):
            H(w, noise)
        return qml.probs(wires=range(n))
    return np.array(circuit())


# ------------------------------------------------------------------ Grover
def grover_oracle(target: str, noise=0.0):
    """Phase flip on |target>: X where target bit is 0, multi-controlled Z, undo."""
    n = len(target)
    zeros = [i for i, b in enumerate(target) if b == "0"]
    for i in zeros:
        X(i, noise)
    MCZ(list(range(n)), noise)
    for i in zeros:
        X(i, noise)


def grover_diffuser(n: int, noise=0.0):
    """2|s><s| - I  (up to a global phase): H X MCZ X H."""
    for w in range(n):
        H(w, noise)
    for w in range(n):
        X(w, noise)
    MCZ(list(range(n)), noise)
    for w in range(n):
        X(w, noise)
    for w in range(n):
        H(w, noise)


def grover_iterations(n: int) -> int:
    return max(1, math.floor(math.pi / 4 * math.sqrt(2 ** n)))


def grover_circuit(target: str, iterations: int | None = None, noise=0.0):
    n = len(target)
    k = grover_iterations(n) if iterations is None else iterations
    for w in range(n):
        H(w, noise)
    for _ in range(k):
        grover_oracle(target, noise)
        grover_diffuser(n, noise)


def grover(target: str, iterations: int | None = None, noise=0.0):
    """Returns P(measuring the target)."""
    n = len(target)
    dev = device(n, noise)

    @qml.qnode(dev)
    def circuit():
        grover_circuit(target, iterations, noise)
        return qml.probs(wires=range(n))
    return float(circuit()[int(target, 2)])


def grover_gate_counts(n: int):
    """Gate counts after decomposing to 1- and 2-qubit gates (what hardware runs)."""
    dev = qml.device("default.qubit", wires=n + max(0, n - 2))   # spare wires not used by default decomposition

    @qml.qnode(dev)
    def circuit():
        grover_circuit("1" * n)
        return qml.probs(wires=range(n))
    tape = qml.workflow.construct_tape(circuit)()
    basic = qml.transforms.decompose(tape, gate_set={"Hadamard", "PauliX", "PauliZ", "CNOT", "RZ", "RY", "RX",
                                                    "PhaseShift", "T", "Adjoint(T)", "S", "Adjoint(S)",
                                                    "GlobalPhase", "CZ"})[0][0]
    ops = [op for op in basic.operations if op.name != "GlobalPhase"]
    two_q = sum(1 for op in ops if len(op.wires) == 2)
    return dict(oracle_queries=grover_iterations(n), total_gates=len(ops), two_qubit_gates=two_q)


# ------------------------------------------------------------------ QFT and QPE
def qft(wires, inverse=False):
    """Textbook QFT: H and controlled phases R_k = diag(1, e^{2 pi i / 2^k}), then swaps.
    Matches qml.QFT (wire 0 = most significant bit)."""
    wires = list(wires)
    n = len(wires)
    if not inverse:
        for i in range(n):
            qml.Hadamard(wires[i])
            for j in range(i + 1, n):
                qml.ControlledPhaseShift(2 * np.pi / 2 ** (j - i + 1), wires=[wires[j], wires[i]])
        for i in range(n // 2):
            qml.SWAP([wires[i], wires[n - 1 - i]])
    else:
        for i in range(n // 2):
            qml.SWAP([wires[i], wires[n - 1 - i]])
        for i in reversed(range(n)):
            for j in reversed(range(i + 1, n)):
                qml.ControlledPhaseShift(-2 * np.pi / 2 ** (j - i + 1), wires=[wires[j], wires[i]])
            qml.Hadamard(wires[i])


def qpe(unitary: np.ndarray, eigvec_prep, n_counting: int, n_target: int):
    """Phase estimation: counting register 0..t-1 (wire 0 = MSB), target t..t+m-1.
    Counting qubit i controls U^(2^(t-1-i)). Returns probabilities of the counting register;
    peak at round(phase * 2^t)."""
    t = n_counting
    target = list(range(t, t + n_target))
    dev = qml.device("default.qubit", wires=t + n_target)

    @qml.qnode(dev)
    def circuit():
        eigvec_prep(target)
        for i in range(t):
            qml.Hadamard(i)
        for i in range(t):
            power = 2 ** (t - 1 - i)
            Up = np.linalg.matrix_power(unitary, power)
            qml.ctrl(qml.QubitUnitary(Up, wires=target), control=i)
        qft(range(t), inverse=True)
        return qml.probs(wires=range(t))
    return np.array(circuit())


# ------------------------------------------------------------------ Shor (order finding)
def modmul_matrix(a: int, N: int, n_work: int) -> np.ndarray:
    """Permutation |x> -> |a x mod N> for x < N, identity for N <= x < 2^n.
    NB: this is a compiled 'black box'. A real device must build it from
    reversible arithmetic, which is where almost all of Shor's gate cost lies."""
    dim = 2 ** n_work
    U = np.zeros((dim, dim))
    for x in range(dim):
        U[(a * x) % N if x < N else x, x] = 1
    return U


def order_finding_probs(a: int, N: int = 15, n_counting: int = 8):
    n_work = math.ceil(math.log2(N))
    U = modmul_matrix(a, N, n_work)

    def prep_one(wires):                   # work register starts in |1>
        qml.PauliX(wires[-1])
    return qpe(U, prep_one, n_counting, n_work), n_counting


def order_from_measurement(y: int, t: int, N: int) -> int | None:
    """Continued-fraction step: y / 2^t ~ s / r  ->  candidate r."""
    if y == 0:
        return None
    return Fraction(y, 2 ** t).limit_denominator(N).denominator


def shor_factor(N: int = 15, a: int = 7, n_counting: int = 8):
    """Run order finding, then classical post-processing over the exact outcome
    distribution. Returns per-outcome results and the overall success probability."""
    if math.gcd(a, N) != 1:
        raise ValueError("pick a coprime to N (otherwise gcd already gives a factor)")
    probs, t = order_finding_probs(a, N, n_counting)
    outcomes, p_success = [], 0.0
    for y, p in enumerate(probs):
        if p < 1e-9:
            continue
        r = order_from_measurement(y, t, N)
        factors = None
        if r and r % 2 == 0 and pow(a, r, N) == 1:
            x = pow(a, r // 2, N)
            f1, f2 = math.gcd(x - 1, N), math.gcd(x + 1, N)
            if 1 < f1 < N:
                factors = (f1, N // f1)
            elif 1 < f2 < N:
                factors = (f2, N // f2)
        outcomes.append(dict(y=y, prob=float(p), r=r, factors=factors))
        if factors:
            p_success += float(p)
    true_r = next(r for r in range(1, N) if pow(a, r, N) == 1)
    return dict(N=N, a=a, true_order=true_r, outcomes=outcomes, p_success=p_success)
