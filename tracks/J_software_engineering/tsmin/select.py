"""Test-suite minimization (weighted set cover) and prioritization.

Formulation (minimization)
  decision variables  x_t ∈ {0, 1}          run test t or not
  objective           min  Σ_t c_t x_t       c_t = measured seconds (or 1 to minimise the count)
  constraints         Σ_{t covers r} x_t ≥ 1   for every requirement r covered by the full suite
Requirements r are statements and/or branch arcs of the library (see collect.py).
This is the (weighted) set-cover problem: NP-hard; greedy is an H(m) ≈ ln m + 1
approximation (Chvátal 1979); the ILP is exact.

Prioritization orders ALL tests so that faults are found early:
  total      : most requirements first
  additional : most not-yet-covered requirements first (reset when everything is covered)
  additional per second : as above, divided by test time
Quality is measured by APFD / cost-aware APFDc on a mutant kill matrix (mutate.py).
"""
from __future__ import annotations

import math
import random

import pulp


def requirements(cov, kinds=("L", "A")):
    return {t: {r for r in rs if r[0] in kinds} for t, rs in cov.items()}


def universe(cov):
    return set().union(*cov.values()) if cov else set()


def is_cover(selected, cov, U=None):
    U = universe(cov) if U is None else U
    got = set().union(*(cov[t] for t in selected)) if selected else set()
    return U <= got


def essential_tests(cov):
    """Tests that are the ONLY test covering some requirement: every cover must contain them."""
    owner = {}
    for t, rs in cov.items():
        for r in rs:
            owner[r] = t if r not in owner else None
    return sorted({t for t in owner.values() if t is not None})


def greedy(cov, cost=None):
    """Chvátal's greedy: repeatedly take the test with the most new requirements
    (per unit cost if `cost` is given); ties broken by name for determinism."""
    U = universe(cov)
    left, chosen = set(U), []
    cand = {t: set(rs) for t, rs in cov.items() if rs}
    while left:
        def score(t):
            g = len(cand[t] & left)
            return g / max(cost[t], 1e-6) if cost else g
        best = max(sorted(cand), key=score)          # first name wins ties
        if not cand[best] & left:
            break
        chosen.append(best)
        left -= cand[best]
    return chosen


def remove_redundant(selected, cov, cost=None):
    """Reverse-delete pass: drop any chosen test whose requirements are all covered by
    the others (most expensive first). Never breaks coverage."""
    sel = list(selected)
    order = sorted(sel, key=lambda t: -(cost[t] if cost else len(cov[t])))
    for t in order:
        others = [u for u in sel if u != t]
        if is_cover(others, cov, universe({u: cov[u] for u in sel})):
            sel = others
    return sel


def ilp(cov, cost=None, time_limit=120, relax=False):
    """Exact weighted set cover with HiGHS (gap 0). relax=True solves the LP relaxation,
    a lower bound on any cover's cost."""
    U = sorted(universe(cov))
    tests = sorted(t for t in cov if cov[t])
    prob = pulp.LpProblem("tsm", pulp.LpMinimize)
    cat = "Continuous" if relax else "Binary"
    x = {t: pulp.LpVariable(f"x{i}", lowBound=0, upBound=1, cat=cat) for i, t in enumerate(tests)}
    prob += pulp.lpSum((cost[t] if cost else 1) * x[t] for t in tests)
    by_req = {r: [] for r in U}
    for t in tests:
        for r in cov[t]:
            by_req[r].append(x[t])
    seen = set()
    for r in U:                                   # identical constraints added once
        key = tuple(sorted(v.name for v in by_req[r]))
        if key in seen:
            continue
        seen.add(key)
        prob += pulp.lpSum(by_req[r]) >= 1
    prob.solve(pulp.HiGHS(msg=False, timeLimit=time_limit, gapRel=0.0))
    status = pulp.LpStatus[prob.status]
    val = pulp.value(prob.objective)
    if relax:
        return val, status, len(seen)
    return sorted(t for t in tests if (x[t].value() or 0) > 0.5), val, status, len(seen)


def random_until_cover(cov, seed=0):
    """Naive baseline: run tests in random order until coverage is complete."""
    U = universe(cov)
    order = sorted(cov)
    random.Random(seed).shuffle(order)
    got, chosen = set(), []
    for t in order:
        if cov[t] - got:
            chosen.append(t)
            got |= cov[t]
        if got >= U:
            break
    return chosen


# ------------------------------------------------------------------ prioritization
def prioritize(cov, method, cost=None, seed=0):
    tests = sorted(cov)
    if method == "original":
        return tests
    if method == "random":
        r = list(tests)
        random.Random(seed).shuffle(r)
        return r
    if method == "total":
        return sorted(tests, key=lambda t: (-len(cov[t]), t))
    if method in ("additional", "additional/s"):
        left_tests, order = set(tests), []
        U = universe(cov)
        uncovered = set(U)
        while left_tests:
            if not uncovered:
                uncovered = set(U)                       # everything covered: start a new round
            def score(t):
                g = len(cov[t] & uncovered)
                return g / max(cost[t], 1e-6) if method == "additional/s" else g
            best = max(sorted(left_tests), key=score)
            if score(best) == 0 and uncovered == U:      # remaining tests cover nothing
                order += sorted(left_tests, key=lambda t: cost[t] if cost else 0)
                break
            if score(best) == 0:
                uncovered = set()
                continue
            order.append(best)
            left_tests.discard(best)
            uncovered -= cov[best]
        return order
    raise ValueError(method)


def apfd(order, kills):
    """Average Percentage of Faults Detected. kills: {mutant: set of tests that kill it}
    (only mutants killed by at least one test). 1 is best."""
    pos = {t: i + 1 for i, t in enumerate(order)}
    n, m = len(order), len(kills)
    tf = [min(pos[t] for t in ks) for ks in kills.values()]
    return 1 - sum(tf) / (n * m) + 1 / (2 * n)


def apfd_c(order, kills, cost):
    """Cost-cognizant APFD (Elbaum et al. 2001) with equal fault severities:
    the area under 'fraction of faults found' vs 'fraction of test time spent'."""
    total = sum(cost[t] for t in order)
    pos = {t: i for i, t in enumerate(order)}
    cum = []
    acc = 0.0
    for t in order:
        acc += cost[t]
        cum.append(acc)
    s = 0.0
    for ks in kills.values():
        i = min(pos[t] for t in ks)
        s += (total - cum[i]) + 0.5 * cost[order[i]]
    return s / (total * len(kills))


def time_to_find(order, kills, cost, frac):
    """Seconds of test time until `frac` of the killable mutants are detected."""
    pos = {t: i for i, t in enumerate(order)}
    first = sorted(min(pos[t] for t in ks) for ks in kills.values())
    need = first[max(0, math.ceil(frac * len(first)) - 1)]
    return sum(cost[t] for t in order[:need + 1])
