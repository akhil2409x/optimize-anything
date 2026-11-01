"""Objective and constraint checks, computed independently of any solver.

Hard constraint : no student sits two exams in the same slot (no clashes),
                  and every exam gets exactly one slot in 0..n_slots-1.
Objective       : Carter proximity cost
                      C = (1/S) * sum_{i<j} w_ij * 2^(5 - |t_i - t_j|)   for 1 <= |t_i - t_j| <= 5
                  i.e. exams one slot apart cost 16 per shared student, two apart 8,
                  ... five apart 1, six or more apart 0. Lower is better.
"""
from __future__ import annotations

from .data import Instance

PROXIMITY_WEIGHTS = {1: 16, 2: 8, 3: 4, 4: 2, 5: 1}


def clashes(inst: Instance, slots: list[int]) -> int:
    """Number of students affected by same-slot clashes (0 for a feasible timetable)."""
    return sum(w for (i, j), w in inst.conflicts.items() if slots[i] == slots[j])


def is_feasible(inst: Instance, slots: list[int] | None) -> bool:
    if slots is None or len(slots) != inst.n_exams:
        return False
    if any(not (0 <= s < inst.n_slots) for s in slots):
        return False
    return clashes(inst, slots) == 0


def proximity_cost(inst: Instance, slots: list[int]) -> float:
    total = 0
    for (i, j), w in inst.conflicts.items():
        total += w * PROXIMITY_WEIGHTS.get(abs(slots[i] - slots[j]), 0)
    return total / inst.n_students


def slots_used(slots: list[int]) -> int:
    return len(set(slots))


def exam_cost(inst: Instance, slots: list[int], exam: int, slot: int) -> int:
    """Unnormalised proximity cost contributed by `exam` if it sat in `slot`."""
    cost = 0
    for nb in inst.neighbours[exam]:
        if nb == exam:
            continue
        key = (exam, nb) if exam < nb else (nb, exam)
        s = slots[nb]
        if s is None or s < 0:
            continue
        cost += inst.conflicts[key] * PROXIMITY_WEIGHTS.get(abs(slot - s), 0)
    return cost
