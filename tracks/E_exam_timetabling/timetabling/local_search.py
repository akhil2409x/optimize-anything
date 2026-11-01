"""Hill-climbing improvement of a feasible timetable.

Neighbourhood: move one exam to another clash-free slot. Only improving moves
are accepted (first-improvement), so the search stops at a local minimum. The
timetable stays feasible at every step, which the tests check.
"""
from __future__ import annotations

import random
import time

from .data import Instance
from .evaluate import exam_cost


def hill_climb(inst: Instance, slots: list[int], *, seed: int = 0,
               max_passes: int = 50, time_limit: float = 30.0) -> list[int]:
    rng = random.Random(seed)
    slots = list(slots)
    start = time.perf_counter()
    exams = list(range(inst.n_exams))
    for _ in range(max_passes):
        improved = False
        rng.shuffle(exams)
        for e in exams:
            current = exam_cost(inst, slots, e, slots[e])
            if current == 0:
                continue
            blocked = {slots[nb] for nb in inst.neighbours[e]}
            best_t, best_c = slots[e], current
            for t in range(inst.n_slots):
                if t == slots[e] or t in blocked:
                    continue
                c = exam_cost(inst, slots, e, t)
                if c < best_c:
                    best_t, best_c = t, c
            if best_t != slots[e]:
                slots[e] = best_t
                improved = True
            if time.perf_counter() - start > time_limit:
                return slots
        if not improved:
            break
    return slots
