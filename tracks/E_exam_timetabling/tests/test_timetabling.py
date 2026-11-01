"""Correctness checks for Track E. Run from tracks/E_exam_timetabling with `pytest -q`.

Every solver's output is checked by evaluate.py, which is written independently
of the solvers, against hand-computed values and brute force on tiny instances.
"""
import itertools

import pytest

from timetabling import data, evaluate, greedy, ilp, local_search


def tiny():
    # 4 exams; student lists chosen so the conflict graph is a path 0-1-2-3
    # with weights w01=2, w12=1, w23=3.
    students = [[0, 1], [0, 1], [1, 2], [2, 3], [2, 3], [2, 3]]
    return data.Instance("tiny", 4, students, n_slots=3)


def test_conflict_weights():
    inst = tiny()
    assert inst.conflicts == {(0, 1): 2, (1, 2): 1, (2, 3): 3}
    assert inst.n_students == 6


def test_proximity_cost_by_hand():
    inst = tiny()
    slots = [0, 1, 0, 2]   # gaps: |0-1|=1 -> 2*16, |1-0|=1 -> 1*16, |0-2|=2 -> 3*8
    assert evaluate.proximity_cost(inst, slots) == pytest.approx((32 + 16 + 24) / 6)
    assert evaluate.is_feasible(inst, slots)


def test_clash_detected():
    inst = tiny()
    assert evaluate.clashes(inst, [0, 0, 1, 2]) == 2
    assert not evaluate.is_feasible(inst, [0, 0, 1, 2])
    assert not evaluate.is_feasible(inst, [0, 1, 2, 3])      # slot 3 out of range
    assert not evaluate.is_feasible(inst, None)


def test_exam_cost_matches_total():
    inst = data.synthetic(12, 60, 6, seed=3)
    s = greedy.dsatur(inst)
    total = sum(evaluate.exam_cost(inst, s, e, s[e]) for e in range(inst.n_exams))
    # each conflicting pair is counted from both ends
    assert total / 2 / inst.n_students == pytest.approx(evaluate.proximity_cost(inst, s))


def brute_force_best(inst):
    best = None
    for slots in itertools.product(range(inst.n_slots), repeat=inst.n_exams):
        slots = list(slots)
        if evaluate.is_feasible(inst, slots):
            c = evaluate.proximity_cost(inst, slots)
            best = c if best is None else min(best, c)
    return best


@pytest.mark.parametrize("solver", [greedy.first_fit, greedy.largest_degree,
                                    greedy.dsatur, greedy.dsatur_proximity])
def test_greedy_feasible_on_easy_instances(solver):
    for seed in range(5):
        inst = data.synthetic(25, 120, 14, seed=seed)
        s = solver(inst)
        assert s is not None and evaluate.is_feasible(inst, s), (solver.__name__, seed)


def test_hill_climb_keeps_feasibility_and_never_worsens():
    for seed in range(5):
        inst = data.synthetic(30, 200, 12, seed=seed)
        s = greedy.dsatur(inst)
        h = local_search.hill_climb(inst, s, seed=seed)
        assert evaluate.is_feasible(inst, h)
        assert evaluate.proximity_cost(inst, h) <= evaluate.proximity_cost(inst, s) + 1e-9


def test_ilp_min_slots_matches_chromatic_number():
    # odd cycle of 5 exams needs 3 slots
    students = [[i, (i + 1) % 5] for i in range(5)]
    inst = data.Instance("c5", 5, students, n_slots=5)
    r = ilp.min_slots(inst, time_limit=20)
    assert r.status == "Optimal" and r.objective == 3
    assert evaluate.is_feasible(inst, r.slots)


@pytest.mark.parametrize("seed", range(6))
def test_ilp_proximity_matches_brute_force(seed):
    # covers both feasible and infeasible instances (brute force decides which)
    inst = data.synthetic(6, 25, 4, n_programmes=2, seed=seed)
    expected = brute_force_best(inst)
    r = ilp.min_proximity(inst, time_limit=30)
    if expected is None:
        assert r.slots is None and "Infeasible" in r.status
    else:
        assert r.status == "Optimal"
        assert evaluate.is_feasible(inst, r.slots)
        assert r.objective == pytest.approx(expected)


def test_load_carter_format(tmp_path):
    p = tmp_path / "toy.stu"
    p.write_text("1 2\n2 3\n\n1 3 4\n")
    inst = data.load_carter(p, n_slots=4)
    assert inst.n_exams == 4 and inst.n_students == 3
    assert inst.conflicts == {(0, 1): 1, (1, 2): 1, (0, 2): 1, (0, 3): 1, (2, 3): 1}


def test_bad_input_rejected():
    with pytest.raises(ValueError):
        data.Instance("bad", 2, [[0, 5]], n_slots=2)
    with pytest.raises(ValueError):
        data.Instance("bad", 2, [[0, 1]], n_slots=0)
