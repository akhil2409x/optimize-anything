"""0/1 knapsack solved five ways (Chapter 6: discrete optimization).

    maximise   sum_i v_i x_i
    subject to sum_i w_i x_i <= C,   x_i in {0,1}

Methods
  brute_force      - enumerate all 2^n subsets (ground truth for tests, n <= 20)
  greedy_ratio     - BASELINE: sort by value/weight, take what fits. No guarantee.
  greedy_half      - max(greedy_ratio, best single item): a 1/2-approximation
  dp               - dynamic programming over capacity, O(n*C) time, exact
  branch_and_bound - own implementation: depth-first, items in ratio order,
                     fractional (LP-relaxation) upper bound for pruning, exact
  ilp              - PuLP model solved by HiGHS (which runs its own B&B), exact

Every solver returns a Solution whose value/weight are recomputed from the chosen
items, so a solver cannot misreport its own result.

Instance classes follow Pisinger (2005), "Where are the hard knapsack problems?":
  uncorrelated         v, w independent           - easy, greedy does well
  weakly_correlated    v close to w                - harder
  strongly_correlated  v = w + R/10                - hard for B&B: all ratios similar
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from itertools import combinations

import pulp


@dataclass
class Knapsack:
    values: list[int]
    weights: list[int]
    capacity: int
    name: str = "knapsack"

    def __post_init__(self):
        if len(self.values) != len(self.weights):
            raise ValueError("values and weights must have the same length")
        if self.capacity < 0 or any(w <= 0 for w in self.weights) or any(v < 0 for v in self.values):
            raise ValueError("need capacity >= 0, weights > 0, values >= 0")

    @property
    def n(self) -> int:
        return len(self.values)


@dataclass
class Solution:
    items: list[int]
    value: int
    weight: int
    seconds: float = 0.0
    nodes: int = 0                       # B&B nodes explored (0 if not applicable)
    status: str = "Optimal"
    extra: dict = field(default_factory=dict)


def evaluate(k: Knapsack, items) -> tuple[int, int, bool]:
    items = sorted(set(items))
    v = sum(k.values[i] for i in items)
    w = sum(k.weights[i] for i in items)
    return v, w, w <= k.capacity


def _solution(k: Knapsack, items, t0, **kw) -> Solution:
    v, w, ok = evaluate(k, items)
    if not ok:
        raise AssertionError(f"solver returned an overweight solution ({w} > {k.capacity})")
    return Solution(sorted(set(items)), v, w, time.perf_counter() - t0, **kw)


# ---------------------------------------------------------------- instances
def generate(n: int, kind: str = "uncorrelated", R: int = 1000, seed: int = 0,
             capacity_ratio: float = 0.5) -> Knapsack:
    rng = random.Random(seed)
    weights = [rng.randint(1, R) for _ in range(n)]
    if kind == "uncorrelated":
        values = [rng.randint(1, R) for _ in range(n)]
    elif kind == "weakly_correlated":
        values = [max(1, w + rng.randint(-R // 10, R // 10)) for w in weights]
    elif kind == "strongly_correlated":
        values = [w + R // 10 for w in weights]
    else:
        raise ValueError(f"unknown kind {kind!r}")
    cap = int(capacity_ratio * sum(weights))
    return Knapsack(values, weights, cap, f"{kind}-n{n}-r{seed}")


# ---------------------------------------------------------------- methods
def brute_force(k: Knapsack) -> Solution:
    if k.n > 22:
        raise ValueError("brute force limited to n <= 22")
    t0 = time.perf_counter()
    best, best_v = [], 0
    for r in range(1, k.n + 1):
        for combo in combinations(range(k.n), r):
            v, w, ok = evaluate(k, combo)
            if ok and v > best_v:
                best, best_v = list(combo), v
    return _solution(k, best, t0)


def _ratio_order(k: Knapsack) -> list[int]:
    return sorted(range(k.n), key=lambda i: (-k.values[i] / k.weights[i], i))


def greedy_ratio(k: Knapsack) -> Solution:
    t0 = time.perf_counter()
    items, load = [], 0
    for i in _ratio_order(k):
        if load + k.weights[i] <= k.capacity:
            items.append(i)
            load += k.weights[i]
    return _solution(k, items, t0, status="Heuristic")


def greedy_half(k: Knapsack) -> Solution:
    """Guarantee: value >= OPT/2, because OPT <= greedy prefix + next item."""
    t0 = time.perf_counter()
    g = greedy_ratio(k)
    fits = [i for i in range(k.n) if k.weights[i] <= k.capacity]
    single = max(fits, key=lambda i: k.values[i], default=None)
    items = g.items if single is None or g.value >= k.values[single] else [single]
    return _solution(k, items, t0, status="Heuristic")


def dp(k: Knapsack) -> Solution:
    """Table best[i][c] = best value using items < i with capacity c.

    Kept as a full 2-D table (bytearray of take-flags) so the chosen items can be
    recovered; memory is O(n*C), which is what limits DP in Experiment 2.
    """
    t0 = time.perf_counter()
    C = k.capacity
    best = [0] * (C + 1)
    take = []
    for i in range(k.n):
        w, v = k.weights[i], k.values[i]
        row = bytearray(C + 1)
        for c in range(C, w - 1, -1):          # backwards: each item used at most once
            cand = best[c - w] + v
            if cand > best[c]:
                best[c] = cand
                row[c] = 1
        take.append(row)
    items, c = [], C
    for i in range(k.n - 1, -1, -1):
        if take[i][c]:
            items.append(i)
            c -= k.weights[i]
    return _solution(k, items, t0, extra={"table_cells": k.n * (C + 1)})


def _fractional_bound(k: Knapsack, order, start, room, value) -> float:
    """LP relaxation of the remaining sub-problem: fill greedily, last item fractional."""
    bound = value
    for idx in range(start, len(order)):
        i = order[idx]
        if k.weights[i] <= room:
            room -= k.weights[i]
            bound += k.values[i]
        else:
            return bound + k.values[i] * room / k.weights[i]
    return bound


def branch_and_bound(k: Knapsack, node_limit: int = 5_000_000, time_limit: float = 60.0) -> Solution:
    """Depth-first B&B, 'take' branch first, starting from the greedy incumbent.

    A node is pruned when its fractional bound cannot beat the incumbent by at
    least 1 (values are integers). Iterative with an explicit stack to avoid
    Python's recursion limit.
    """
    t0 = time.perf_counter()
    order = _ratio_order(k)
    inc = greedy_half(k)
    best_v, best_items = inc.value, list(inc.items)
    nodes = 0
    stack = [(0, k.capacity, 0, ())]          # (depth, room left, value, chosen items)
    status = "Optimal"
    while stack:
        depth, room, value, chosen = stack.pop()
        nodes += 1
        if nodes > node_limit or (nodes % 4096 == 0 and time.perf_counter() - t0 > time_limit):
            status = "Limit-Feasible"
            break
        if value > best_v:
            best_v, best_items = value, list(chosen)
        if depth == k.n:
            continue
        if _fractional_bound(k, order, depth, room, value) < best_v + 1:
            continue
        i = order[depth]
        stack.append((depth + 1, room, value, chosen))                       # skip i
        if k.weights[i] <= room:
            stack.append((depth + 1, room - k.weights[i], value + k.values[i], chosen + (i,)))  # take i (popped first)
    return _solution(k, best_items, t0, nodes=nodes, status=status)


def ilp(k: Knapsack, time_limit: float = 60.0) -> Solution:
    t0 = time.perf_counter()
    prob = pulp.LpProblem("knapsack", pulp.LpMaximize)
    x = [pulp.LpVariable(f"x_{i}", cat="Binary") for i in range(k.n)]
    prob += pulp.lpSum(k.values[i] * x[i] for i in range(k.n))
    prob += pulp.lpSum(k.weights[i] * x[i] for i in range(k.n)) <= k.capacity
    # HiGHS stops at a 0.01% relative gap by default and still reports "Optimal";
    # on n=100 that returned 34709 instead of 34710 (docs/failure_log.md F4).
    # Values are integers, so an absolute gap below 1 forces true optimality.
    prob.solve(pulp.HiGHS(msg=False, timeLimit=time_limit, gapRel=0.0, gapAbs=0.5))
    items = [i for i in range(k.n) if (x[i].value() or 0) > 0.5]
    status = "Optimal" if prob.sol_status == pulp.LpSolutionOptimal else "Limit-Feasible"
    return _solution(k, items, t0, status=status)


METHODS = {
    "greedy_ratio (baseline)": greedy_ratio,
    "greedy_half": greedy_half,
    "dp": dp,
    "branch_and_bound": branch_and_bound,
    "ilp_highs": ilp,
}
