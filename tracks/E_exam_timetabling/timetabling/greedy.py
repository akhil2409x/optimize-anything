"""Greedy constructive methods (Chapter 6: greedy).

All methods place exams one at a time and never undo a placement, so they run
in roughly O(E * T * degree). They can fail to find a clash-free timetable even
when one exists; the result then has `slots=None`.

  * first_fit        - BASELINE: exams in id order, earliest clash-free slot.
                       This mimics a naive manual process ("fill the calendar
                       from the start").
  * largest_degree   - Carter's classic ordering: most-conflicted exams first.
  * dsatur           - Brelaz's DSatur: always schedule the exam with the fewest
                       remaining feasible slots next (ties: largest degree).
  * dsatur_proximity - DSatur ordering, but pick the feasible slot that adds the
                       least proximity cost instead of the earliest one.
"""
from __future__ import annotations

import random

from .data import Instance
from .evaluate import exam_cost


def _feasible_slots(inst: Instance, slots: list[int], exam: int) -> list[int]:
    blocked = {slots[nb] for nb in inst.neighbours[exam] if slots[nb] >= 0}
    return [t for t in range(inst.n_slots) if t not in blocked]


def _place_in_order(inst: Instance, order: list[int], cost_aware: bool) -> list[int] | None:
    slots = [-1] * inst.n_exams
    for e in order:
        options = _feasible_slots(inst, slots, e)
        if not options:
            return None
        if cost_aware:
            slots[e] = min(options, key=lambda t: (exam_cost(inst, slots, e, t), t))
        else:
            slots[e] = options[0]
    return slots


def first_fit(inst: Instance) -> list[int] | None:
    return _place_in_order(inst, list(range(inst.n_exams)), cost_aware=False)


def largest_degree(inst: Instance, cost_aware: bool = False) -> list[int] | None:
    order = sorted(range(inst.n_exams), key=lambda e: (-len(inst.neighbours[e]), e))
    return _place_in_order(inst, order, cost_aware)


def dsatur(inst: Instance, cost_aware: bool = False, seed: int | None = None) -> list[int] | None:
    """DSatur. With `seed`, remaining ties are broken randomly (for repeated runs)."""
    rng = random.Random(seed)
    tiebreak = [rng.random() if seed is not None else 0.0 for _ in range(inst.n_exams)]
    slots = [-1] * inst.n_exams
    unplaced = set(range(inst.n_exams))
    while unplaced:
        # fewest feasible slots first, then most conflicts, then tiebreak, then id
        e = min(unplaced, key=lambda x: (len(_feasible_slots(inst, slots, x)),
                                         -len(inst.neighbours[x]), tiebreak[x], x))
        options = _feasible_slots(inst, slots, e)
        if not options:
            return None
        if cost_aware:
            slots[e] = min(options, key=lambda t: (exam_cost(inst, slots, e, t), t))
        else:
            slots[e] = options[0]
        unplaced.remove(e)
    return slots


def dsatur_proximity(inst: Instance, seed: int | None = None) -> list[int] | None:
    return dsatur(inst, cost_aware=True, seed=seed)
