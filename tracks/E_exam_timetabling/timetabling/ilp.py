"""Integer Linear Programming models (Chapter 6: ILP, solved by branch-and-bound
inside the HiGHS MIP solver).

Decision variables
    x[e,t] in {0,1}   exam e is held in slot t

Model 1 - minimum number of slots (graph colouring):
    min  sum_t u[t]
    s.t. sum_t x[e,t] = 1                 every exam scheduled once
         x[i,t] + x[j,t] <= u[t]          conflicting exams never share a slot,
                                          and a slot counts only if used
         u[t] >= u[t+1]                   symmetry breaking: use slots in order

Model 2 - minimum Carter proximity cost with the slot count fixed:
    z[i,j,d] >= x[i,t] + x[j,t-d] + x[j,t+d] - 1    for each conflict (i,j),
                                                     slot t and distance d=1..5
    min  (1/S) * sum w_ij * 2^(5-d) * z[i,j,d]
    Because exam j sits in exactly one slot, x[j,t-d] + x[j,t+d] <= 1, so the
    right-hand side is 1 exactly when the two exams are d slots apart.

Model 2 grows as |conflicts| * 5 * T constraints, so it is only practical for
small instances; that size limit is one of the findings of this track.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import pulp

from .data import Instance
from .evaluate import PROXIMITY_WEIGHTS, proximity_cost, slots_used


@dataclass
class ILPResult:
    slots: list[int] | None
    status: str            # "Optimal", "TimeLimit-Feasible", "TimeLimit-NoSolution", ...
    objective: float | None    # re-computed by evaluate.py from `slots`, NOT the solver's value
    bound: float | None    # best lower bound proven by the solver (None if unknown)
    seconds: float
    n_vars: int
    n_constraints: int


class _HiGHSWarm(pulp.HiGHS):
    """PuLP 2.9's highspy-based HiGHS wrapper accepts `warmStart=True` but never
    passes the start to the solver (only the command-line HiGHS_CMD does). This
    subclass hands the initial values of *all* columns to highspy before solving.
    A start that leaves any constraint violated is rejected by HiGHS, so every
    variable (including the z auxiliaries) must be given a consistent value."""

    def callSolver(self, lp):
        import highspy
        sol = highspy.HighsSolution()
        sol.col_value = [float(v.varValue or 0.0) for v in lp.variables()]
        sol.value_valid = True
        lp.solverModel.setSolution(sol)
        lp.solverModel.run()


def _solver(time_limit: float, msg: bool, warm: bool = False):
    cls = _HiGHSWarm if warm else pulp.HiGHS
    # gapRel=0: HiGHS otherwise stops at a 0.01% gap and still says "Optimal" (failure log F4)
    return cls(msg=msg, timeLimit=time_limit, gapRel=0.0)


def _bound(prob) -> float | None:
    try:
        return float(prob.solverModel.getInfo().mip_dual_bound)
    except Exception:          # other solvers / versions
        return None


def _extract(inst: Instance, x) -> list[int] | None:
    slots = []
    for e in range(inst.n_exams):
        chosen = [t for t in range(len(x[e])) if (x[e][t].value() or 0) > 0.5]
        if len(chosen) != 1:
            return None
        slots.append(chosen[0])
    return slots


def _status(prob, slots) -> str:
    raw = pulp.LpStatus[prob.status]
    if raw == "Optimal":
        # HiGHS through PuLP reports "Optimal" for some time-limited runs; the
        # gap check below (sol_status / bound) is what we trust.
        return "Optimal" if prob.sol_status == pulp.LpSolutionOptimal else "TimeLimit-Feasible"
    if slots is not None:
        return "TimeLimit-Feasible"
    return f"{raw}-NoSolution"


def min_slots(inst: Instance, max_slots: int | None = None, time_limit: float = 60.0,
              msg: bool = False) -> ILPResult:
    T = max_slots or inst.n_slots
    prob = pulp.LpProblem("min_slots", pulp.LpMinimize)
    x = [[pulp.LpVariable(f"x_{e}_{t}", cat="Binary") for t in range(T)] for e in range(inst.n_exams)]
    u = [pulp.LpVariable(f"u_{t}", cat="Binary") for t in range(T)]
    prob += pulp.lpSum(u)
    for e in range(inst.n_exams):
        prob += pulp.lpSum(x[e]) == 1
    for (i, j) in inst.conflicts:
        for t in range(T):
            prob += x[i][t] + x[j][t] <= u[t]
    for t in range(T - 1):
        prob += u[t] >= u[t + 1]
    # isolated exams still need a used slot
    for e in range(inst.n_exams):
        for t in range(T):
            prob += x[e][t] <= u[t]
    start = time.perf_counter()
    prob.solve(_solver(time_limit, msg))
    secs = time.perf_counter() - start
    slots = _extract(inst, x)
    obj = float(slots_used(slots)) if slots else None
    return ILPResult(slots, _status(prob, slots), obj, _bound(prob), secs,
                     len(prob.variables()), len(prob.constraints))


def min_proximity(inst: Instance, time_limit: float = 120.0, warm_start: list[int] | None = None,
                  msg: bool = False) -> ILPResult:
    T = inst.n_slots
    prob = pulp.LpProblem("min_proximity", pulp.LpMinimize)
    x = [[pulp.LpVariable(f"x_{e}_{t}", cat="Binary") for t in range(T)] for e in range(inst.n_exams)]
    z = {}
    terms = []
    for (i, j), w in inst.conflicts.items():
        for d, weight in PROXIMITY_WEIGHTS.items():
            if d >= T:
                continue
            z[i, j, d] = pulp.LpVariable(f"z_{i}_{j}_{d}", lowBound=0, upBound=1)
            terms.append(w * weight * z[i, j, d])
    prob += pulp.lpSum(terms) * (1.0 / inst.n_students)
    for e in range(inst.n_exams):
        prob += pulp.lpSum(x[e]) == 1
    for (i, j) in inst.conflicts:
        for t in range(T):
            prob += x[i][t] + x[j][t] <= 1
            for d in PROXIMITY_WEIGHTS:
                if (i, j, d) not in z:
                    continue
                near = [x[j][s] for s in (t - d, t + d) if 0 <= s < T]
                prob += z[i, j, d] >= x[i][t] + pulp.lpSum(near) - 1
    if warm_start is not None:
        for e, s in enumerate(warm_start):
            for t in range(T):
                x[e][t].setInitialValue(1 if t == s else 0)
        for (i, j, d), var in z.items():
            var.setInitialValue(1 if abs(warm_start[i] - warm_start[j]) == d else 0)
    start = time.perf_counter()
    prob.solve(_solver(time_limit, msg, warm=warm_start is not None))
    secs = time.perf_counter() - start
    slots = _extract(inst, x)
    # A time-limited incumbent can leave slack in the continuous z variables, so the
    # solver's objective over-states the true cost. Always re-evaluate independently.
    obj = proximity_cost(inst, slots) if slots else None
    return ILPResult(slots, _status(prob, slots), obj, _bound(prob), secs,
                     len(prob.variables()), len(prob.constraints))
