"""QUBO and Ising formulations (Chapter 6: QUBO/Ising, binary-to-spin, penalties).

QUBO  : minimise  f(x) = x^T Q x + c,          x in {0,1}^n   (Q upper-triangular)
Ising : minimise  E(s) = sum_{i<j} J_ij s_i s_j + sum_i h_i s_i + offset,   s in {-1,+1}^n

Binary-to-spin substitution x_i = (1 - s_i)/2 maps one onto the other exactly;
`qubo_to_ising` does it and the tests check f(x) == E(s) for every assignment.
(Convention: x=0 <-> s=+1, x=1 <-> s=-1, the one used by Qiskit/PennyLane for
|0>, |1> eigenvalues of Z.)

Problems encoded
  * Max-Cut: maximise number (weight) of edges crossing the cut.
        cut(x) = sum_{(i,j)} w_ij (x_i + x_j - 2 x_i x_j)
    so the QUBO minimises -cut(x). No constraints, so no penalty is needed.
    This instance is saved to results/maxcut_instance.json as the classical
    baseline for Track C (QAOA).
  * Knapsack: a constrained problem. The inequality is turned into an equality
    with binary slack bits y (sum_i w_i x_i + sum_k 2^k y_k = C), then added
    as a penalty  P * (sum w x + sum 2^k y - C)^2.  Too small a P lets the
    minimiser cheat by breaking the constraint; Experiment 3 measures that.
"""
from __future__ import annotations

import itertools
import json
import math
import random
from collections import defaultdict

import numpy as np


# ------------------------------------------------------------- core QUBO type
class QUBO:
    def __init__(self, n: int):
        self.n = n
        self.Q: dict[tuple[int, int], float] = defaultdict(float)   # i <= j
        self.offset = 0.0

    def add(self, i: int, j: int, coef: float):
        if i > j:
            i, j = j, i
        self.Q[i, j] += coef

    def energy(self, x) -> float:
        return self.offset + sum(c * x[i] * x[j] for (i, j), c in self.Q.items())

    def matrix(self) -> np.ndarray:
        M = np.zeros((self.n, self.n))
        for (i, j), c in self.Q.items():
            M[i, j] += c
        return M


def brute_force(q: QUBO):
    """Exhaustive minimum (vectorised over all 2^n states). n <= 24."""
    if q.n > 24:
        raise ValueError("brute force limited to n <= 24")
    states = ((np.arange(2 ** q.n)[:, None] >> np.arange(q.n)) & 1).astype(np.int8)
    e = np.full(len(states), q.offset)
    for (i, j), c in q.Q.items():
        e += c * states[:, i] * states[:, j]
    k = int(np.argmin(e))
    return states[k].tolist(), float(e[k]), e


# ------------------------------------------------------------- binary -> spin
def qubo_to_ising(q: QUBO):
    """Return (J, h, offset) with x_i = (1 - s_i)/2."""
    J: dict[tuple[int, int], float] = defaultdict(float)
    h = [0.0] * q.n
    offset = q.offset
    for (i, j), c in q.Q.items():
        if i == j:                    # c x_i = c (1 - s_i)/2      (x_i^2 = x_i)
            offset += c / 2
            h[i] -= c / 2
        else:                         # c x_i x_j = c/4 (1 - s_i - s_j + s_i s_j)
            offset += c / 4
            h[i] -= c / 4
            h[j] -= c / 4
            J[i, j] += c / 4
    return dict(J), h, offset


def ising_energy(J, h, offset, s) -> float:
    return offset + sum(c * s[i] * s[j] for (i, j), c in J.items()) + sum(hi * si for hi, si in zip(h, s))


def bits_to_spins(x):
    return [1 - 2 * b for b in x]


# ------------------------------------------------------------- Max-Cut
def random_graph(n: int, p: float = 0.5, seed: int = 0, weighted: bool = False):
    rng = random.Random(seed)
    edges = []
    for i, j in itertools.combinations(range(n), 2):
        if rng.random() < p:
            edges.append((i, j, rng.randint(1, 5) if weighted else 1))
    return edges


def cut_value(edges, x) -> int:
    return sum(w for i, j, w in edges if x[i] != x[j])


def maxcut_qubo(n: int, edges) -> QUBO:
    q = QUBO(n)
    for i, j, w in edges:            # minimise -w (x_i + x_j - 2 x_i x_j)
        q.add(i, i, -w)
        q.add(j, j, -w)
        q.add(i, j, 2 * w)
    return q


def maxcut_greedy(n: int, edges, seed: int = 0):
    """Classical heuristic baseline: one-flip local search from a random cut.
    Any local optimum cuts at least half of the total edge weight."""
    rng = random.Random(seed)
    x = [rng.randint(0, 1) for _ in range(n)]
    improved = True
    while improved:
        improved = False
        for v in range(n):
            before = cut_value(edges, x)
            x[v] ^= 1
            if cut_value(edges, x) > before:
                improved = True
            else:
                x[v] ^= 1
    return x


def save_maxcut(path, n, edges, best_x, best_cut, ising, all_optimal=None):
    J, h, off = ising
    path.write_text(json.dumps({
        "description": "Max-Cut instance for Track C (QAOA). Brute-force optimum included as classical baseline.",
        "n_nodes": n, "edges": [[i, j, w] for i, j, w in edges],
        "optimal_cut": best_cut, "optimal_assignment_bits": best_x,
        "all_optimal_assignments_bits": all_optimal,
        "bit_order": "list index i = node i (little-endian w.r.t. the integer state index)",
        "ising": {"J": {f"{i},{j}": c for (i, j), c in J.items()}, "h": h, "offset": off,
                  "convention": "x_i = (1 - s_i)/2, energy = -cut"},
    }, indent=2))


# ------------------------------------------------------------- Knapsack QUBO
def knapsack_qubo(values, weights, capacity, penalty: float):
    """Variables: n item bits, then K slack bits encoding 0..capacity."""
    n = len(values)
    K = max(1, math.ceil(math.log2(capacity + 1)))
    coeffs = list(weights) + [2 ** k for k in range(K)]
    # last slack bit coefficient trimmed so slack can't exceed capacity
    coeffs[-1] = capacity + 1 - 2 ** (K - 1)
    q = QUBO(n + K)
    for i in range(n):
        q.add(i, i, -values[i])                      # maximise value -> minimise -value
    # P (sum a_i z_i - C)^2 = P [ sum a_i^2 z_i + 2 sum_{i<j} a_i a_j z_i z_j - 2C sum a_i z_i + C^2 ]
    m = len(coeffs)
    for i in range(m):
        q.add(i, i, penalty * (coeffs[i] ** 2 - 2 * capacity * coeffs[i]))
        for j in range(i + 1, m):
            q.add(i, j, penalty * 2 * coeffs[i] * coeffs[j])
    q.offset += penalty * capacity ** 2
    return q, n, K
