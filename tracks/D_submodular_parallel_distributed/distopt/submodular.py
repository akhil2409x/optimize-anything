"""Submodular maximization: test the assumptions, then use greedy.

A set function f: 2^V -> R is
  monotone     if A ⊆ B  =>  f(A) <= f(B)
  submodular   if A ⊆ B, e ∉ B  =>  f(A + e) - f(A) >= f(B + e) - f(B)   (diminishing returns)
For monotone submodular f with f(∅) = 0, greedy selection of k elements is
guaranteed to reach at least (1 - 1/e) ≈ 63.2% of the best possible value
(Nemhauser, Wolsey & Fisher, 1978). The guarantee needs BOTH properties; this
module checks them numerically instead of assuming them.

Objectives
  coverage        sensors (sets) covering weighted points: f(S) = weight of points covered
                  -> monotone submodular
  facility        facility location: f(S) = sum_i max_{j in S} sim(i, j)
                  -> monotone submodular
  synergy         f(S) = (sum_{i in S} w_i)^2 + pairs bonus: SUPERmodular, violates the
                  assumption; used to show greedy has no guarantee then
  r2              R^2 of a least-squares fit on the chosen features: monotone but NOT
                  submodular in general (features can be useless alone, useful together)
"""
from __future__ import annotations

import itertools
import math

import numpy as np
import pulp


class Coverage:
    name = "weighted coverage"

    def __init__(self, n_sets=40, n_points=300, seed=0):
        rng = np.random.default_rng(seed)
        self.weights = rng.uniform(1, 10, n_points)
        # each set covers a random disc of points: overlapping, uneven sizes
        pts = rng.uniform(0, 1, (n_points, 2))
        centres = rng.uniform(0, 1, (n_sets, 2))
        radii = rng.uniform(0.05, 0.25, n_sets)
        d = np.linalg.norm(pts[None, :, :] - centres[:, None, :], axis=2)
        self.sets = [frozenset(np.flatnonzero(d[j] <= radii[j]).tolist()) for j in range(n_sets)]
        self.n = n_sets

    def f(self, S):
        covered = set().union(*(self.sets[j] for j in S)) if S else set()
        return float(sum(self.weights[i] for i in covered))

    def exact(self, k, time_limit=60):
        """Maximum weighted coverage as an ILP (exact): choose k sets, y_i = point covered."""
        prob = pulp.LpProblem("maxcov", pulp.LpMaximize)
        x = [pulp.LpVariable(f"x{j}", cat="Binary") for j in range(self.n)]
        npts = len(self.weights)
        y = [pulp.LpVariable(f"y{i}", lowBound=0, upBound=1) for i in range(npts)]
        prob += pulp.lpSum(self.weights[i] * y[i] for i in range(npts))
        prob += pulp.lpSum(x) == k
        cover = [[] for _ in range(npts)]
        for j, s in enumerate(self.sets):
            for i in s:
                cover[i].append(x[j])
        for i in range(npts):
            prob += y[i] <= pulp.lpSum(cover[i])
        prob.solve(pulp.HiGHS(msg=False, timeLimit=time_limit, gapRel=0.0))
        S = [j for j in range(self.n) if (x[j].value() or 0) > 0.5]
        return S, self.f(S), pulp.LpStatus[prob.status]


class FacilityLocation:
    name = "facility location"

    def __init__(self, n=30, n_clients=200, seed=0):
        rng = np.random.default_rng(seed)
        cl = rng.uniform(0, 1, (n_clients, 2))
        fac = rng.uniform(0, 1, (n, 2))
        self.sim = np.exp(-4 * np.linalg.norm(cl[:, None] - fac[None], axis=2))
        self.n = n

    def f(self, S):
        return float(self.sim[:, list(S)].max(axis=1).sum()) if S else 0.0


class Synergy:
    """Supermodular: items are worth more together. Violates diminishing returns."""
    name = "synergy (supermodular)"

    def __init__(self, n=12, seed=0):
        rng = np.random.default_rng(seed)
        self.w = rng.uniform(0.5, 1.5, n)
        self.n = n
        # one item is individually attractive but contributes nothing to synergy
        self.decoy = 0
        self.w[self.decoy] = 0.0
        self.decoy_value = 6.0

    def f(self, S):
        S = set(S)
        base = self.decoy_value if self.decoy in S else 0.0
        return base + float(sum(self.w[i] for i in S if i != self.decoy) ** 2)


class R2Selection:
    """Explained variance of a least-squares fit: monotone, generally NOT submodular."""
    name = "feature-selection R^2"

    def __init__(self, n=10, m=300, seed=0):
        rng = np.random.default_rng(seed)
        X = rng.standard_normal((m, n))
        X[:, 1] = X[:, 0] + 0.05 * rng.standard_normal(m)       # x0, x1 nearly collinear
        y = 40 * (X[:, 0] - X[:, 1]) + X[:, 2:5].sum(axis=1) + 0.5 * rng.standard_normal(m)
        self.X, self.y, self.n = X, y - y.mean(), n

    def f(self, S):
        if not S:
            return 0.0
        A = self.X[:, list(S)]
        coef, *_ = np.linalg.lstsq(A, self.y, rcond=None)
        r = self.y - A @ coef
        return float(1 - (r @ r) / (self.y @ self.y))


# ------------------------------------------------------------------ assumption tests
def test_properties(obj, trials=2000, seed=0, tol=1e-9):
    """Random tests of monotonicity and diminishing returns. Returns violation counts
    and the worst violation found (a single violation disproves submodularity)."""
    rng = np.random.default_rng(seed)
    mono_viol = sub_viol = 0
    worst = 0.0
    example = None
    for _ in range(trials):
        B = set(rng.choice(obj.n, size=rng.integers(1, obj.n), replace=False).tolist())
        A = set(x for x in B if rng.random() < 0.5)
        rest = [e for e in range(obj.n) if e not in B]
        if not rest:
            continue
        e = int(rng.choice(rest))
        fa, fb = obj.f(A), obj.f(B)
        if fa > fb + tol:
            mono_viol += 1
        gain_a = obj.f(A | {e}) - fa
        gain_b = obj.f(B | {e}) - fb
        if gain_b > gain_a + tol:
            sub_viol += 1
            if gain_b - gain_a > worst:
                worst, example = gain_b - gain_a, (sorted(A), sorted(B), e, gain_a, gain_b)
    return dict(trials=trials, monotone_violations=mono_viol, submodular_violations=sub_viol,
                worst_violation=worst, example=example)


# ------------------------------------------------------------------ algorithms
def greedy(obj, k):
    S, evals = [], 0
    cur = obj.f(S)
    for _ in range(k):
        best, best_gain = None, -math.inf
        for e in range(obj.n):
            if e in S:
                continue
            g = obj.f(S + [e]) - cur
            evals += 1
            if g > best_gain:
                best, best_gain = e, g
        S.append(best)
        cur += best_gain
    return S, cur, evals


def lazy_greedy(obj, k):
    """Minoux's accelerated greedy: stale gains are upper bounds (valid ONLY if f is
    submodular), so most re-evaluations are skipped. Same answer as greedy for
    submodular f."""
    import heapq
    S, cur, evals = [], obj.f([]), 0
    heap = []
    for e in range(obj.n):
        heap.append((-(obj.f([e]) - cur), e))
        evals += 1
    heapq.heapify(heap)
    while len(S) < k:
        neg, e = heapq.heappop(heap)
        g = obj.f(S + [e]) - cur
        evals += 1
        if not heap or g >= -heap[0][0] - 1e-12:
            S.append(e)
            cur += g
        else:
            heapq.heappush(heap, (-g, e))
    return S, cur, evals


def brute_force(obj, k):
    best = max(itertools.combinations(range(obj.n), k), key=lambda S: obj.f(S))
    return list(best), obj.f(best)


def random_best(obj, k, tries, seed=0):
    rng = np.random.default_rng(seed)
    return max(obj.f(rng.choice(obj.n, k, replace=False).tolist()) for _ in range(tries))
