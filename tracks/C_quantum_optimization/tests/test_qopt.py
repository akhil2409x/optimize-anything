"""Correctness checks for Track C. Run from tracks/C_quantum_optimization: `pytest -q`."""
import numpy as np
import pennylane as qml
import pytest
from pennylane import numpy as pnp

from qopt import barren, maxcut as M, train as T, vqe as V


@pytest.fixture(scope="module")
def mc():
    return M.MaxCut.from_track_b()


def basis_energy(mc, bits):
    """<z|H_C|z> computed by preparing |z> on the simulator."""
    dev = qml.device("default.qubit", wires=mc.n)

    @qml.qnode(dev)
    def f():
        qml.BasisState(np.array(bits), wires=range(mc.n))
        return qml.expval(mc.hamiltonian())
    return float(f())


def test_track_b_link(mc):
    # the instance really is Track B's: optimum and optimal bitstrings agree
    assert mc.n == 10 and mc.optimal_cut == 14
    for bits in mc.optimal_set:
        assert mc.cut(bits) == 14


def test_hamiltonian_equals_minus_cut(mc):
    rng = np.random.default_rng(0)
    samples = [tuple(rng.integers(0, 2, mc.n)) for _ in range(25)] + list(mc.optimal_set)[:2]
    for bits in samples:
        assert basis_energy(mc, bits) == pytest.approx(-mc.cut(bits))


def test_bit_order_of_probabilities(mc):
    # prepare one optimal basis state and check the probability vector decodes back to it
    bits = sorted(mc.optimal_set)[0]
    dev = qml.device("default.qubit", wires=mc.n)

    @qml.qnode(dev)
    def f():
        qml.BasisState(np.array(bits), wires=range(mc.n))
        return qml.probs(wires=range(mc.n))
    s = mc.stats_from_probs(f())
    assert s["p_optimal"] == pytest.approx(1.0) and s["approx_ratio"] == pytest.approx(1.0)


def test_from_edges_matches_brute_force():
    edges = [(i, (i + 1) % 5, 1) for i in range(5)]         # 5-cycle: max cut 4
    small = M.MaxCut.from_edges(5, edges)
    assert small.optimal_cut == 4 and len(small.optimal_set) == 10
    assert basis_energy(small, (0, 1, 0, 1, 1)) == pytest.approx(-small.cut((0, 1, 0, 1, 1)))


def test_qaoa_uniform_superposition_at_zero_angles(mc):
    # gamma = beta = 0 leaves |+>^n: expected cut = half the edges
    cost, _ = M.cost_qnode(mc)
    assert -float(cost(pnp.zeros((1, 2)))) == pytest.approx(mc.total_weight / 2)


def test_parameter_shift_equals_backprop(mc):
    p = M.init_params(2, seed=3)
    ps, _ = M.cost_qnode(mc, diff_method="parameter-shift")
    bp, _ = M.cost_qnode(mc, diff_method="backprop")
    assert np.allclose(qml.grad(ps)(p), qml.grad(bp)(p), atol=1e-8)


def test_noise_free_mixed_device_matches_statevector():
    small = M.MaxCut.from_track_b().induced(6)
    p = M.init_params(2, seed=1)
    a, _ = M.cost_qnode(small, device="default.mixed", noise=0.0, diff_method="backprop")
    b, _ = M.cost_qnode(small, diff_method="backprop")
    assert float(a(p)) == pytest.approx(float(b(p)))


def test_noise_moves_cost_towards_random_guessing():
    small = M.MaxCut.from_track_b().induced(6)
    p = M.init_params(1, seed=1)
    clean, _ = M.cost_qnode(small, device="default.mixed", noise=0.0, diff_method="backprop")
    noisy, _ = M.cost_qnode(small, device="default.mixed", noise=0.05, diff_method="backprop")
    random_guess = -small.total_weight / 2
    assert abs(float(noisy(p)) - random_guess) < abs(float(clean(p)) - random_guess)


def test_shot_estimate_is_unbiased(mc):
    p = M.init_params(1, seed=0)
    exact, _ = M.cost_qnode(mc, diff_method="backprop")
    noisy, _ = M.cost_qnode(mc, shots=20000, seed=7)
    assert float(noisy(p)) == pytest.approx(float(exact(p)), abs=0.1)


# ------------------------------------------------------------------ VQE
def test_h2_reference_energies():
    H, n = V.h2_hamiltonian(0.7414)
    assert n == 4
    assert V.exact_energy(H, n) == pytest.approx(-1.1373, abs=5e-4)   # FCI / STO-3G literature value
    e, _ = V.energy_qnode(H, n, "ucc")
    assert float(e(pnp.array([0.0]))) == pytest.approx(-1.1167, abs=5e-4)  # Hartree-Fock


def test_ucc_vqe_reaches_chemical_accuracy_and_respects_variational_bound():
    H, n = V.h2_hamiltonian(0.7414)
    exact = V.exact_energy(H, n)
    e, _ = V.energy_qnode(H, n, "ucc", diff_method="backprop")
    thetas = np.linspace(-np.pi, np.pi, 721)
    energies = [float(e(pnp.array([t]))) for t in thetas]
    assert min(energies) >= exact - 1e-9                     # variational principle
    assert min(energies) - exact < 1.6e-3                    # chemical accuracy reachable


# ------------------------------------------------------------------ training loop
def test_training_respects_budget_and_improves(mc):
    cost, dev = M.cost_qnode(mc)
    ev, _ = M.cost_qnode(mc, diff_method="backprop")
    for method, lr in [("spsa", 0.2), ("adam", 0.05)]:
        _, hist = T.train(cost, dev, M.init_params(1, 0), method, lr, 300, ev)
        used = [u for u, _ in hist]
        assert used == sorted(used) and used[0] == 0
        # the last step may overshoot by at most one step's worth of circuits
        assert used[-2] < 300 <= used[-1]
        assert hist[-1][1] < hist[0][1]


# ------------------------------------------------------------------ barren plateaus
def test_barren_gradient_matches_autodiff():
    rng = np.random.default_rng(0)
    for cost in ["global", "local"]:
        f = barren.cost_qnode(4, 3, cost, device="default.qubit")
        theta = pnp.array(rng.uniform(0, 2 * np.pi, barren.n_params(4, 3)), requires_grad=True)
        auto = qml.grad(f)(theta)[0]
        assert barren.first_gradient(f, theta) == pytest.approx(float(auto), abs=1e-9)
