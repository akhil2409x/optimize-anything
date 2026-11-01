"""Correctness checks for Track B. Run from tracks/B_discrete with `pytest -q`."""
import itertools

import pytest

from discrete import knapsack as K
from discrete import qubo as Q

KINDS = ["uncorrelated", "weakly_correlated", "strongly_correlated"]


# ------------------------------------------------------------- knapsack
def test_textbook_instance():
    # values 60,100,120 / weights 10,20,30 / C=50: optimum 220 (items 1,2);
    # ratio greedy takes items 0,1 for 160 — the classic counter-example.
    k = K.Knapsack([60, 100, 120], [10, 20, 30], 50)
    assert K.greedy_ratio(k).value == 160
    for name in ["dp", "branch_and_bound", "ilp_highs"]:
        s = K.METHODS[name](k)
        assert (s.value, s.items) == (220, [1, 2]), name


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("seed", range(4))
def test_exact_methods_match_brute_force(kind, seed):
    k = K.generate(14, kind, R=100, seed=seed)
    opt = K.brute_force(k).value
    for name in ["dp", "branch_and_bound", "ilp_highs"]:
        assert K.METHODS[name](k).value == opt, name


@pytest.mark.parametrize("kind,n,seed", [("strongly_correlated", 100, 0),
                                         ("weakly_correlated", 200, 0)])
def test_ilp_is_exactly_optimal_on_larger_instances(kind, n, seed):
    # regression test for failure-log F4: ILP was off by one under the default MIP gap
    k = K.generate(n, kind, R=1000, seed=seed)
    assert K.ilp(k).value == K.dp(k).value


@pytest.mark.parametrize("kind", KINDS)
def test_greedy_half_guarantee(kind):
    for seed in range(20):
        k = K.generate(12, kind, R=100, seed=seed)
        opt = K.brute_force(k).value
        g = K.greedy_half(k)
        assert 2 * g.value >= opt
        assert g.value <= opt


def test_greedy_half_rescues_bad_ratio_greedy():
    # tiny high-ratio item blocks a huge item: ratio greedy gets 2, best single item 100
    k = K.Knapsack([2, 100], [1, 100], 100)
    assert K.greedy_ratio(k).value == 2
    assert K.greedy_half(k).value == 100


def test_edge_cases():
    assert K.dp(K.Knapsack([5], [10], 0)).value == 0                # zero capacity
    assert K.branch_and_bound(K.Knapsack([5], [10], 9)).value == 0  # nothing fits
    k = K.Knapsack([1, 2, 3], [1, 1, 1], 10)                        # everything fits
    assert K.dp(k).value == K.branch_and_bound(k).value == 6
    with pytest.raises(ValueError):
        K.Knapsack([1], [0], 5)
    with pytest.raises(ValueError):
        K.Knapsack([1, 2], [1], 5)


def test_overweight_solution_is_rejected():
    k = K.Knapsack([1, 1], [5, 5], 6)
    with pytest.raises(AssertionError):
        K._solution(k, [0, 1], 0.0)


# ------------------------------------------------------------- QUBO / Ising
def test_maxcut_qubo_equals_minus_cut_for_every_assignment():
    edges = Q.random_graph(7, 0.6, seed=3, weighted=True)
    q = Q.maxcut_qubo(7, edges)
    for x in itertools.product([0, 1], repeat=7):
        assert q.energy(x) == pytest.approx(-Q.cut_value(edges, x))


def test_qubo_to_ising_is_exact():
    edges = Q.random_graph(6, 0.7, seed=1, weighted=True)
    q = Q.maxcut_qubo(6, edges)
    q.add(2, 2, 3.5)            # add linear terms so h and offset are exercised
    J, h, off = Q.qubo_to_ising(q)
    for x in itertools.product([0, 1], repeat=6):
        assert Q.ising_energy(J, h, off, Q.bits_to_spins(x)) == pytest.approx(q.energy(x))


def test_maxcut_known_graphs():
    # complete graph K4: best cut 4 (2+2 split); 5-cycle: best cut 4
    k4 = [(i, j, 1) for i, j in itertools.combinations(range(4), 2)]
    c5 = [(i, (i + 1) % 5, 1) for i in range(5)]
    for n, edges, best in [(4, k4, 4), (5, c5, 4)]:
        x, e, _ = Q.brute_force(Q.maxcut_qubo(n, edges))
        assert -e == best == Q.cut_value(edges, x)


def test_maxcut_local_search_half_guarantee():
    for seed in range(10):
        edges = Q.random_graph(10, 0.5, seed=seed)
        x = Q.maxcut_greedy(10, edges, seed=seed)
        assert 2 * Q.cut_value(edges, x) >= sum(w for *_, w in edges)


@pytest.mark.parametrize("seed", range(5))
def test_knapsack_qubo_with_large_penalty_matches_optimum(seed):
    k = K.generate(6, "uncorrelated", R=12, seed=seed)
    opt = K.brute_force(k).value
    q, n, _ = Q.knapsack_qubo(k.values, k.weights, k.capacity, penalty=sum(k.values) + 1)
    x, e, _ = Q.brute_force(q)
    v, w, ok = K.evaluate(k, [i for i in range(n) if x[i]])
    assert ok and v == opt and e == pytest.approx(-opt)


def test_knapsack_qubo_small_penalty_violates_constraint():
    # with a tiny penalty, taking every item is "cheaper" than respecting capacity
    k = K.Knapsack([10, 10, 10], [5, 5, 5], 5)
    q, n, _ = Q.knapsack_qubo(k.values, k.weights, k.capacity, penalty=0.01)
    x, _, _ = Q.brute_force(q)
    assert not K.evaluate(k, [i for i in range(n) if x[i]])[2]
