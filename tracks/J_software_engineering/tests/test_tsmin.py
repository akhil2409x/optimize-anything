"""Correctness checks for Track J. Run from tracks/J_software_engineering: `pytest -q`.

Fast tests use small hand-made coverage maps. Tests marked `subject` need the
more-itertools checkout (./setup.sh) and the measurements in results/."""
import itertools
import math
from pathlib import Path

import pytest

from tsmin import collect as C, mutate as M, select as S

HERE = Path(__file__).resolve().parent.parent
MEAS = HERE / "results" / "measurements.json"
needs_subject = pytest.mark.skipif(not (C.SUBJECT.exists() and MEAS.exists()),
                                   reason="run ./setup.sh and run_experiments.py first")

COV = {"a": {1, 2, 3}, "b": {3, 4}, "c": {4, 5, 6}, "d": {1, 6}, "e": {2, 5}, "z": set()}
COST = {"a": 3.0, "b": 1.0, "c": 3.0, "d": 1.0, "e": 1.0, "z": 0.1}


def brute(cov, cost):
    U = S.universe(cov)
    tests = sorted(t for t in cov if cov[t])
    best = None
    for k in range(1, len(tests) + 1):
        for sub in itertools.combinations(tests, k):
            if S.is_cover(sub, cov, U):
                c = sum(cost[t] for t in sub) if cost else len(sub)
                if best is None or c < best[0] - 1e-12:
                    best = (c, sub)
    return best


@pytest.mark.parametrize("cost", [None, COST])
def test_ilp_equals_brute_force(cost):
    sel, val, status, _ = S.ilp(COV, cost)
    assert status == "Optimal"
    assert S.is_cover(sel, COV)
    assert val == pytest.approx(brute(COV, cost)[0])


def test_greedy_is_a_cover_within_the_ln_bound():
    for cost in (None, COST):
        sel = S.greedy(COV, cost)
        assert S.is_cover(sel, COV)
        opt = brute(COV, cost)[0]
        val = sum(cost[t] for t in sel) if cost else len(sel)
        m = max(len(v) for v in COV.values())
        assert val <= sum(1 / i for i in range(1, m + 1)) * opt + 1e-9    # H(max set size) bound


def test_lp_relaxation_is_a_lower_bound():
    lb = S.ilp(COV, COST, relax=True)[0]
    assert lb <= brute(COV, COST)[0] + 1e-9


def test_remove_redundant_keeps_cover_and_never_grows():
    sel = sorted(COV)
    red = S.remove_redundant(sel, COV, COST)
    assert S.is_cover(red, COV) and len(red) < len(sel)


def test_essential_tests():
    cov = {"a": {1, 2}, "b": {2, 3}, "c": {3}}
    assert S.essential_tests(cov) == ["a"]
    assert "a" in S.ilp(cov)[0]


def test_random_until_cover_is_a_cover():
    for seed in range(5):
        assert S.is_cover(S.random_until_cover(COV, seed), COV)


def test_prioritization_orders_are_permutations():
    for m in ("original", "random", "total", "additional", "additional/s"):
        order = S.prioritize(COV, m, COST)
        assert sorted(order) == sorted(COV)


def test_additional_greedy_first_picks():
    order = S.prioritize(COV, "additional")
    assert order[0] == "a" and order[1] == "c"          # a: 3 new, then c: {4,5,6} 3 new


def test_apfd_hand_calculation():
    # 4 tests, 2 faults first found at positions 1 and 3: APFD = 1 - (1+3)/(4*2) + 1/8 = 0.625
    kills = {"f1": {"t1"}, "f2": {"t3", "t4"}}
    assert S.apfd(["t1", "t2", "t3", "t4"], kills) == pytest.approx(0.625)
    cost = {t: 1.0 for t in ("t1", "t2", "t3", "t4")}
    assert S.apfd_c(["t1", "t2", "t3", "t4"], kills, cost) == pytest.approx(0.625)
    assert S.time_to_find(["t1", "t2", "t3", "t4"], kills, cost, 1.0) == 3


def test_apfdc_rewards_cheap_tests_first():
    kills = {"f": {"slow", "fast"}}
    cost = {"slow": 10.0, "fast": 1.0, "x": 1.0}
    assert S.apfd_c(["fast", "slow", "x"], kills, cost) > S.apfd_c(["slow", "fast", "x"], kills, cost)


# ------------------------------------------------------------------ mutation operators
def test_mutants_change_exactly_one_spot_and_compile(tmp_path):
    src = "def f(a, b):\n    if a < b and not a:\n        return a + 1\n    return b * 2\n"
    (tmp_path / "m.py").write_text(src)
    ms = M.generate(tmp_path, ["m.py"], {"m.py": {2, 3, 4}})
    ops = {m.operator for m in ms}
    assert {"cmp Lt->LtE", "bool And->Or", "not x->x", "arith Add->Sub", "int k->k+1",
            "arith Mult->FloorDiv"} <= ops
    for m in ms:
        M.apply(tmp_path, m, src)
        new = (tmp_path / "m.py").read_text()
        diff = [i for i, (x, y) in enumerate(zip(src.splitlines(), new.splitlines())) if x != y]
        assert diff == [m.line - 1]
        ns = {}
        exec(new, ns)
        assert "f" in ns
    (tmp_path / "m.py").write_text(src)


def test_mutant_behaviour_differs():
    src = "def f(a, b):\n    return a < b\n"
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        p = Path(td)
        (p / "m.py").write_text(src)
        (m,) = M.generate(p, ["m.py"], {"m.py": {2}})
        M.apply(p, m, src)
        ns = {}
        exec((p / "m.py").read_text(), ns)
        assert ns["f"](1, 1) is True                 # a <= b
        assert m.replacement == "(a <= b)"


# ------------------------------------------------------------------ against the real subject
@needs_subject
def test_measured_coverage_is_consistent():
    d = C.load(MEAS)
    assert len(d["times"]) == len(d["coverage"]) > 700
    U = S.universe(d["coverage"])
    assert any(r[0] == "A" for r in U) and any(r[0] == "L" for r in U)
    # every statement requirement is an endpoint of some arc requirement
    arc_lines = {(r[1], x) for r in U if r[0] == "A" for x in r[2:] if x > 0}
    assert {(r[1], r[2]) for r in U if r[0] == "L"} <= arc_lines


@needs_subject
def test_unmutated_subject_passes_the_tests_that_cover_a_line():
    d = C.load(MEAS)
    t = sorted(d["coverage"])[:5]
    _, ok, _ = C.run_junit(t)
    assert ok and all(ok.values())
