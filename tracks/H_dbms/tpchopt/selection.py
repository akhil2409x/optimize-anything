"""Index selection under a storage budget (Chapter 6: greedy, DP, ILP).

Problem
    Variables  x_i in {0,1}: build candidate index i (size s_i MB).
    Constraint sum_i s_i x_i <= B (storage budget).
    Objective  minimise the workload's estimated cost  C(S) = sum_q cost_q(S),
               S = {i : x_i = 1}, as estimated by the PostgreSQL planner (hypopg).

C(S) is not additive: two indexes that help the same query do not help it
twice. Each method below simplifies that differently.

  default         : no secondary indexes (the system as shipped: primary keys only)
  knapsack_dp     : 0/1 knapsack by dynamic programming with ADDITIVE values
                    v_i = sum_q [cost_q({}) - cost_q({i})]. Exact for the additive
                    model, which over-counts when indexes overlap.
  greedy_ratio    : add candidates in order of v_i / s_i (same additive values).
  greedy_whatif   : AutoAdmin-style greedy. Each step re-costs the WHOLE workload
                    with every remaining candidate added to the current set and
                    takes the best (C(S) - C(S + i)) / s_i. Sees interactions;
                    no optimality guarantee.
  ilp             : y_{q,i} = "query q uses index i". At most one index per
                    query per table, y_{q,i} <= x_i, budget on x, maximise
                    sum b_{q,i} y_{q,i} (b = single-index benefit for that query).
                    Removes the double counting within a table. Solved exactly
                    by HiGHS branch-and-bound. A tiny storage penalty breaks ties
                    so that indexes with no benefit are never built just because
                    the budget has room (failure log F9).
"""
from __future__ import annotations

import time

import pulp

from .whatif import Candidate, WhatIf


def single_benefits(wi: WhatIf, cands: list[Candidate]):
    base = wi.costs(frozenset())
    b = {}
    for c in cands:
        cost = wi.costs(frozenset([c]))
        b[c] = {q: max(0.0, base[q] - cost[q]) for q in base}
    return base, b


def knapsack_dp(cands, sizes, values, budget_mb, unit=0.1):
    """Exact 0/1 knapsack, capacity discretised to `unit` MB (sizes rounded UP,
    so a chosen set never exceeds the true budget)."""
    import math
    W = int(budget_mb / unit)
    w = [math.ceil(sizes[c] / unit) for c in cands]
    v = [values[c] for c in cands]
    best = [0.0] * (W + 1)
    keep = []
    for k in range(len(cands)):
        row = bytearray(W + 1)
        for cap in range(W, w[k] - 1, -1):
            cand = best[cap - w[k]] + v[k]
            if cand > best[cap]:
                best[cap] = cand
                row[cap] = 1
        keep.append(row)
    chosen, cap = [], W
    for k in range(len(cands) - 1, -1, -1):
        if keep[k][cap]:
            chosen.append(cands[k])
            cap -= w[k]
    return chosen


def greedy_ratio(cands, sizes, values, budget_mb):
    chosen, used = [], 0.0
    for c in sorted(cands, key=lambda c: -values[c] / sizes[c]):
        if values[c] > 0 and used + sizes[c] <= budget_mb:
            chosen.append(c)
            used += sizes[c]
    return chosen


def greedy_whatif(wi: WhatIf, cands, sizes, budget_mb):
    chosen, used = [], 0.0
    current = wi.total(chosen)
    while True:
        best, best_ratio, best_cost = None, 0.0, None
        for c in cands:
            if c in chosen or used + sizes[c] > budget_mb:
                continue
            cost = wi.total(chosen + [c])
            ratio = (current - cost) / sizes[c]
            if ratio > best_ratio + 1e-9:
                best, best_ratio, best_cost = c, ratio, cost
        if best is None:
            return chosen
        chosen.append(best)
        used += sizes[best]
        current = best_cost


def ilp(cands, sizes, b, budget_mb, time_limit=60):
    queries = sorted(next(iter(b.values())).keys())
    prob = pulp.LpProblem("index_selection", pulp.LpMaximize)
    x = {c: pulp.LpVariable(f"x_{k}", cat="Binary") for k, c in enumerate(cands)}
    y = {(q, c): pulp.LpVariable(f"y_{q}_{k}", lowBound=0, upBound=1)
         for q in queries for k, c in enumerate(cands) if b[c][q] > 0}
    # Without the second term the solver is indifferent to building indexes that
    # no query uses (x_i = 1 with every y_{q,i} = 0) and may do so when the budget
    # has slack. The penalty is far smaller than any real benefit difference.
    eps = 1e-6 * max(1.0, min((v for bq in b.values() for v in bq.values() if v > 0), default=1.0))
    prob += (pulp.lpSum(b[c][q] * v for (q, c), v in y.items())
             - eps * pulp.lpSum(sizes[c] * x[c] for c in cands))
    prob += pulp.lpSum(sizes[c] * x[c] for c in cands) <= budget_mb
    for (q, c), v in y.items():
        prob += v <= x[c]
    tables = {c.table for c in cands}
    for q in queries:
        for t in tables:
            terms = [v for (qq, c), v in y.items() if qq == q and c.table == t]
            if len(terms) > 1:
                prob += pulp.lpSum(terms) <= 1
    prob.solve(pulp.HiGHS(msg=False, timeLimit=time_limit, gapRel=0.0))
    status = "Optimal" if prob.sol_status == pulp.LpSolutionOptimal else "Limit"
    return [c for c in cands if (x[c].value() or 0) > 0.5], status


def select_all(wi: WhatIf, cands, budget_mb):
    """Run every method for one budget. Returns {method: dict(...)}."""
    sizes = {c: wi.size_mb(c) for c in cands}
    base, b = single_benefits(wi, cands)
    values = {c: sum(b[c].values()) for c in cands}
    out = {"default (PK only)": dict(indexes=[], seconds=0.0, note="")}
    t0 = time.perf_counter(); s = knapsack_dp(cands, sizes, values, budget_mb)
    out["knapsack DP (additive)"] = dict(indexes=s, seconds=time.perf_counter() - t0, note="exact for additive model")
    t0 = time.perf_counter(); s = greedy_ratio(cands, sizes, values, budget_mb)
    out["greedy ratio (additive)"] = dict(indexes=s, seconds=time.perf_counter() - t0, note="")
    calls0 = wi.calls
    t0 = time.perf_counter(); s = greedy_whatif(wi, cands, sizes, budget_mb)
    out["greedy what-if"] = dict(indexes=s, seconds=time.perf_counter() - t0,
                                 note=f"{wi.calls - calls0} EXPLAINs")
    t0 = time.perf_counter(); s, st = ilp(cands, sizes, b, budget_mb)
    out["ILP (one index per query per table)"] = dict(indexes=s, seconds=time.perf_counter() - t0, note=st)
    for m, r in out.items():
        r["size_mb"] = sum(sizes[c] for c in r["indexes"])
        r["est_cost"] = wi.total(r["indexes"])
        r["model_value"] = sum(values[c] for c in r["indexes"])
    return out, sizes, base, b
