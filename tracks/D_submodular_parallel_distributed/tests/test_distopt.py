"""Correctness checks for Track D. Run from tracks/D_submodular_parallel_distributed: `pytest -q`."""
import itertools
import math

import numpy as np
import pytest

from distopt import learning as L, parallel as P, submodular as S


# ------------------------------------------------------------------ submodular
def test_property_tests_separate_submodular_from_not():
    assert S.test_properties(S.Coverage(), 500)["submodular_violations"] == 0
    assert S.test_properties(S.FacilityLocation(), 500)["submodular_violations"] == 0
    assert S.test_properties(S.Coverage(), 500)["monotone_violations"] == 0
    assert S.test_properties(S.Synergy(), 500)["submodular_violations"] > 0
    assert S.test_properties(S.R2Selection(), 500)["submodular_violations"] > 0


def test_diminishing_returns_exhaustive_on_small_coverage():
    """Every (A ⊆ B, e) triple on a 7-set instance, not just random ones."""
    obj = S.Coverage(n_sets=7, n_points=60, seed=3)
    for r in range(7):
        for B in itertools.combinations(range(7), r):
            for a in range(r + 1):
                for A in itertools.combinations(B, a):
                    for e in set(range(7)) - set(B):
                        ga = obj.f(set(A) | {e}) - obj.f(set(A))
                        gb = obj.f(set(B) | {e}) - obj.f(set(B))
                        assert ga >= gb - 1e-9


def test_lazy_greedy_matches_greedy_with_fewer_evaluations():
    for obj in (S.Coverage(), S.FacilityLocation()):
        g, vg, eg = S.greedy(obj, 6)
        l, vl, el = S.lazy_greedy(obj, 6)
        assert g == l and math.isclose(vg, vl)
        assert el < eg


@pytest.mark.parametrize("k", [2, 3, 4])
def test_greedy_guarantee_against_exact(k):
    obj = S.Coverage()
    _, vg, _ = S.greedy(obj, k)
    _, vx, status = obj.exact(k)
    assert status == "Optimal"
    assert vx >= vg - 1e-6                         # ILP really is optimal
    assert vg >= (1 - 1 / math.e) * vx             # Nemhauser-Wolsey-Fisher bound


def test_ilp_matches_brute_force_on_small_instance():
    obj = S.Coverage(n_sets=12, n_points=80, seed=5)
    _, vb = S.brute_force(obj, 3)
    _, vx, _ = obj.exact(3)
    assert math.isclose(vb, vx, rel_tol=1e-9)


def test_greedy_can_fail_without_submodularity():
    # supermodular: greedy is fooled by the decoy (suboptimal, though not below 63% here)
    obj = S.Synergy()
    assert S.greedy(obj, 4)[1] < 0.8 * S.brute_force(obj, 4)[1]
    # R^2 with a cancelling pair: greedy falls BELOW the (1 - 1/e) guarantee
    obj = S.R2Selection()
    assert S.greedy(obj, 2)[1] < (1 - 1 / math.e) * S.brute_force(obj, 2)[1]


# ------------------------------------------------------------------ parallel
def test_amdahl_and_gustafson_formulas():
    assert P.amdahl(1, 0.7) == pytest.approx(1)
    assert P.amdahl(1e9, 0.9) == pytest.approx(10, rel=1e-6)
    assert P.gustafson(4, 0.9) == pytest.approx(3.7)
    ps = [1, 2, 4, 8]
    assert P.fit_parallel_fraction(ps, [P.amdahl(p, 0.83) for p in ps]) == pytest.approx(0.83, abs=1e-4)


def test_parallel_result_equals_serial():
    assert P.run_parallel(3000, 1)[0] == P.run_parallel(3000, 2)[0] == P.work(3000)


def test_simulator_matches_hand_calculation():
    d = [5, 1, 1, 1, 4, 4]
    assert P.simulate(d, 2, "block")[0] == 9            # [5,1,1]=7  [1,4,4]=9
    assert P.simulate(d, 2, "cyclic")[0] == 10          # [5,1,4]=10 [1,1,4]=6
    assert P.simulate(d, 2, "lpt")[0] == 8              # 5,4,4,1,1,1 -> {5,1,1,1}=8 {4,4}=8
    assert P.optimal_makespan(d, 2)[0] == pytest.approx(8)


def test_schedules_respect_lower_bounds_and_lpt_bound():
    rng = np.random.default_rng(0)
    for _ in range(5):
        d = rng.uniform(1, 10, 14).tolist()
        opt, status = P.optimal_makespan(d, 3)
        assert status == "Optimal"
        assert opt >= max(sum(d) / 3, max(d)) - 1e-6
        for pol in ("block", "cyclic", "dynamic", "lpt", "steal"):
            assert P.simulate(d, 3, pol)[0] >= opt - 1e-6
        assert P.simulate(d, 3, "lpt")[0] <= (4 / 3 - 1 / 9) * opt + 1e-6   # Graham's bound


def test_work_stealing_does_all_the_work():
    d = list(range(1, 30))
    m, info = P.simulate(d, 4, "steal")
    assert m >= sum(d) / 4 and info["steals"] > 0
    assert m < P.simulate(d, 4, "block")[0]


def test_data_and_model_parallel_give_the_serial_answer():
    assert P.data_parallel(n=4000, d=5, steps=2)["max_abs_diff"] < 1e-10
    assert P.model_parallel(n=300, steps=2)["max_abs_diff"] < 1e-10
    assert P.pipeline(micro_batches=5, stage_work=200)["in_order"]


def test_synchronization_counts_are_exact():
    rows = P.synchronization(n=2000)
    assert rows[0]["total"] == rows[1]["total"] == 4000


# ------------------------------------------------------------------ FedAvg
def test_softmax_gradient_vs_finite_differences():
    rng = np.random.default_rng(0)
    X, Y, W = rng.standard_normal((20, 5)), rng.integers(0, 10, 20), rng.standard_normal((5, 10)) * 0.1
    _, g = L.loss_grad(W, X, Y)
    eps = 1e-6
    for i, j in [(0, 0), (2, 3), (4, 9)]:
        Wp, Wm = W.copy(), W.copy()
        Wp[i, j] += eps
        Wm[i, j] -= eps
        assert g[i, j] == pytest.approx((L.loss_grad(Wp, X, Y)[0] - L.loss_grad(Wm, X, Y)[0]) / (2 * eps), abs=1e-7)


def test_fedavg_with_one_client_is_centralized_sgd():
    Xtr, Xte, Ytr, Yte = L.digits_data()
    Wf, _ = L.fedavg(Xtr, Ytr, Xte, Yte, [np.arange(len(Ytr))], rounds=2, local_epochs=1)
    Wc, _ = L.centralized(Xtr, Ytr, Xte, Yte, epochs=2)
    assert np.allclose(Wf, Wc)


def test_client_splits_partition_the_data():
    _, _, Ytr, _ = L.digits_data()
    for iid in (True, False):
        parts = L.split_clients(Ytr, 10, iid)
        allidx = np.sort(np.concatenate(parts))
        assert np.array_equal(allidx, np.arange(len(Ytr)))
    noniid = L.split_clients(Ytr, 10, False)
    assert max(len(set(Ytr[c])) for c in noniid) <= 4
    assert min(len(set(Ytr[c])) for c in L.split_clients(Ytr, 10, True)) == 10


# ------------------------------------------------------------------ multi-task
def test_mgda_alpha_closed_form_vs_grid():
    rng = np.random.default_rng(1)
    for _ in range(20):
        g1, g2 = rng.standard_normal(6), rng.standard_normal(6) * rng.uniform(0.1, 10)
        a = L.mgda_alpha(g1, g2)
        grid = np.linspace(0, 1, 20001)
        best = grid[np.argmin([np.linalg.norm(t * g1 + (1 - t) * g2) for t in grid])]
        assert a == pytest.approx(best, abs=1e-4)
        d = a * g1 + (1 - a) * g2                  # min-norm point: d.g_k >= ||d||^2 for both tasks
        assert d @ g1 >= d @ d - 1e-9 and d @ g2 >= d @ d - 1e-9


def test_mgda_on_unit_vectors_is_the_midpoint():
    rng = np.random.default_rng(2)
    g1, g2 = rng.standard_normal(5), rng.standard_normal(5)
    assert L.mgda_alpha(g1 / np.linalg.norm(g1), g2 / np.linalg.norm(g2)) == pytest.approx(0.5)


def test_multitask_gradients_vs_finite_differences():
    (X, Y), _, _ = L.two_task_data(n=40)
    net = L.TwoTaskNet(X.shape[1], seed=0)
    grads = net.task_losses_grads(X, Y)
    eps = 1e-6
    for k in range(2):
        for (i, j) in [(0, 0), (3, 7)]:
            W0 = net.W.copy()
            net.W[i, j] += eps
            lp = net.task_losses_grads(X, Y)[k][0]
            net.W[i, j] -= 2 * eps
            lm = net.task_losses_grads(X, Y)[k][0]
            net.W = W0
            assert grads[k][1][i, j] == pytest.approx((lp - lm) / (2 * eps), rel=1e-5, abs=1e-6)


def test_train_and_test_come_from_the_same_task():
    """Regression test for failure log F12: a model fit on train must beat predicting 0 on test."""
    (X, Y), _, (Xt, Yt) = L.two_task_data()
    A = np.hstack([np.tanh(X @ np.eye(8) * 0.5), np.ones((len(X), 1))])
    At = np.hstack([np.tanh(Xt @ np.eye(8) * 0.5), np.ones((len(Xt), 1))])
    c, *_ = np.linalg.lstsq(A, Y[:, 1], rcond=None)
    assert np.mean((At @ c - Yt[:, 1]) ** 2) < 0.5 * np.mean(Yt[:, 1] ** 2)
