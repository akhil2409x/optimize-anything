"""Correctness checks for Track I. Run from tracks/I_quantum_algorithms_qml: `pytest -q`.

Hand-written circuits are checked against PennyLane's library operators and
against the answers the theory guarantees."""
import itertools
import json

import numpy as np
import pennylane as qml
import pytest

from qalgo import algorithms as A, annealing as AN, qml_models as QM


# ------------------------------------------------------------------ algorithms
@pytest.mark.parametrize("n", [2, 3, 4, 5])
def test_hand_written_qft_equals_library(n):
    mine = qml.matrix(lambda: A.qft(range(n)), wire_order=range(n))()
    lib = qml.matrix(qml.QFT(wires=range(n)))
    assert np.allclose(mine, lib)
    inv = qml.matrix(lambda: A.qft(range(n), inverse=True), wire_order=range(n))()
    assert np.allclose(inv @ mine, np.eye(2 ** n))


@pytest.mark.parametrize("n", [2, 3, 4])
def test_deutsch_jozsa(n):
    assert A.deutsch_jozsa("constant0", n) == pytest.approx(1)
    assert A.deutsch_jozsa("constant1", n) == pytest.approx(1)
    for s in itertools.product("01", repeat=n):
        s = "".join(s)
        if "1" in s:
            assert A.deutsch_jozsa(f"balanced:{s}", n) == pytest.approx(0, abs=1e-9)


def test_bernstein_vazirani_recovers_every_secret():
    for s in ["1", "01", "101", "1101", "10110", "011011"]:
        p = A.bernstein_vazirani(s)
        assert p[int(s, 2)] == pytest.approx(1)


@pytest.mark.parametrize("target", ["01", "110", "1011", "00000"])
def test_grover_matches_library_grover_operator(target):
    n = len(target)
    k = A.grover_iterations(n)
    dev = qml.device("default.qubit", wires=n)

    @qml.qnode(dev)
    def library():
        for w in range(n):
            qml.Hadamard(w)
        for _ in range(k):
            qml.FlipSign([int(b) for b in target], wires=range(n))   # library oracle
            qml.GroverOperator(wires=range(n))                     # library diffuser
        return qml.probs(wires=range(n))
    assert A.grover(target) == pytest.approx(float(library()[int(target, 2)]))


def test_grover_success_probability_formula():
    # P = sin^2((2k+1) theta), sin(theta) = 1/sqrt(N)
    for n in [2, 3, 4, 5, 6]:
        N, k = 2 ** n, A.grover_iterations(n)
        theta = np.arcsin(1 / np.sqrt(N))
        assert A.grover("1" * n) == pytest.approx(np.sin((2 * k + 1) * theta) ** 2)


@pytest.mark.parametrize("t,k", [(3, 5), (4, 3), (5, 19)])
def test_qpe_exact_phase(t, k):
    phase = k / 2 ** t
    U = np.diag([1, np.exp(2j * np.pi * phase)])
    probs = A.qpe(U, lambda w: qml.PauliX(w[0]), n_counting=t, n_target=1)
    assert np.argmax(probs) == k and probs[k] == pytest.approx(1)


def test_modular_multiplication_is_a_permutation():
    for a in [2, 4, 7, 8, 11, 13]:
        U = A.modmul_matrix(a, 15, 4)
        assert np.allclose(U @ U.T, np.eye(16)) and set(U.sum(axis=0)) == {1}


@pytest.mark.parametrize("a", [2, 4, 7, 8, 11, 13, 14])
def test_shor_factors_15(a):
    r = A.shor_factor(15, a)
    assert pow(a, r["true_order"], 15) == 1
    if a == 14:
        # order 2 and 14^1 = -1 (mod 15): the textbook failure case, only trivial factors
        assert r["true_order"] == 2 and r["p_success"] == 0
    else:
        assert r["p_success"] > 0
    for o in r["outcomes"]:
        if o["factors"]:
            assert sorted(o["factors"]) == [3, 5]


def test_noise_degrades_algorithms():
    assert A.deutsch_jozsa("constant0", 3, noise=0.02) < 0.999
    assert A.bernstein_vazirani("101", noise=0.02)[5] < 0.999


# ------------------------------------------------------------------ annealing
def test_rx_layer_matches_kronecker_product():
    n, theta = 3, 0.7
    rng = np.random.default_rng(0)
    psi = rng.normal(size=8) + 1j * rng.normal(size=8)
    rx = qml.matrix(qml.RX(theta, wires=0))
    full = np.kron(np.kron(rx, rx), rx)
    assert np.allclose(AN._apply_rx_all(psi.copy(), n, theta), full @ psi)


def test_cut_values_agree_with_track_b():
    d = json.loads(AN.TRACK_B_JSON.read_text())
    n, e = AN.load_maxcut()
    cut = AN.cut_values(n, e)
    assert cut.max() == d["optimal_cut"]
    for bits in d["all_optimal_assignments_bits"]:
        assert cut[int("".join(map(str, bits)), 2)] == d["optimal_cut"]


def test_quantum_anneal_limits():
    n, e = 4, [(0, 1, 1), (1, 2, 1), (2, 3, 1), (3, 0, 1)]        # 4-cycle: 2 optimal cuts of 16
    fast = AN.quantum_anneal(n, e, total_time=1e-3)
    slow = AN.quantum_anneal(n, e, total_time=200)
    assert fast["p_optimal"] == pytest.approx(2 / 16, abs=0.01)     # sudden: stays uniform
    assert slow["p_optimal"] > 0.99                                # adiabatic: ground state


def test_simulated_annealing_small_graph():
    # regression test for failure log F7 (library sampler returned unconverged states)
    n, e = 5, [(i, (i + 1) % 5, 1) for i in range(5)]
    r = AN.simulated_annealing(n, e, num_reads=50, num_sweeps=200)
    assert r["approx_ratio"] == pytest.approx(1.0)
    n, e = AN.load_maxcut()
    assert AN.simulated_annealing(n, e, num_reads=20, num_sweeps=300)["p_optimal"] > 0.5


def test_embedding_overhead():
    hw = AN.hardware_graph("pegasus_p6")
    r = AN.embed(AN.complete_graph_edges(8), hw, timeout=10, tries=1)
    assert r and r["physical_qubits"] >= 8 and r["max_chain"] >= 1


# ------------------------------------------------------------------ QML
def test_quantum_kernel_is_a_valid_kernel():
    Xtr, _, _, _ = QM.load("moons", 40, seed=0)
    qk = QM.QuantumKernelSVM(2)
    G, evals = qk.gram(Xtr[:8], Xtr[:8], symmetric=True)
    assert np.allclose(G, G.T) and np.allclose(np.diag(G), 1)
    assert np.linalg.eigvalsh(G).min() > -1e-8                       # positive semi-definite
    assert evals == 8 * 7 // 2
    assert qk.k(Xtr[0], Xtr[1]) == pytest.approx(qk.k(Xtr[1], Xtr[0]))


def test_data_preprocessing_fitted_on_train_only():
    Xtr, Xte, ytr, yte = QM.load("cancer", 200, seed=1)
    assert Xtr.shape[1] == 4 and Xte.shape[1] == 4
    assert Xtr.min() == pytest.approx(0) and Xtr.max() == pytest.approx(np.pi)
    assert Xte.min() >= 0 and Xte.max() <= np.pi + 1e-12
    assert len(Xtr) + len(Xte) == 200


def test_vqc_learns_something():
    Xtr, Xte, ytr, yte = QM.load("moons", 80, seed=0)
    m = QM.VQC(2, layers=2, steps=25).fit(Xtr, ytr)
    assert m.n_params == 2 * 2 * 3 + 1
    assert (m.predict(Xtr) == ytr).mean() > 0.7
