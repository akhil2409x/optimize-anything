"""Quantum annealing, its classical cousin, and the hardware-embedding limit.

1. Exact simulation of quantum annealing (written by hand, numpy state vector):
       H(s) = -(1 - s) * sum_i X_i  +  s * H_P / scale,     s = t / T,  t in [0, T]
   starting in the ground state of the transverse field, |+>^n. H_P is the
   Track B Max-Cut Hamiltonian (diagonal, energy = -cut). If the anneal is slow
   compared with 1/gap^2 (adiabatic theorem) the state ends in the ground
   space, i.e. an optimal cut. Evolution uses a 2nd-order Trotter splitting:
   the X part is a product of single-qubit rotations, the problem part is a
   diagonal phase, so n = 10 qubits is cheap.

2. Simulated annealing: a CLASSICAL Monte-Carlo heuristic on the same Ising
   model, written by hand (Metropolis single-spin flips, geometric temperature
   schedule). dwave-samplers 1.8.0 was tried first but returned states that a
   single flip would improve, identically after 1 or 1000 sweeps, in this
   environment (failure log F7), so it is not used.

3. Minor embedding (minorminer) of QUBO interaction graphs into D-Wave hardware
   graphs (Pegasus P6, Chimera C16). Hardware qubits have only ~15 couplers
   each, so one logical variable becomes a *chain* of physical qubits. Dense
   QUBOs (e.g. knapsack with a squared penalty, which couples every pair of
   variables) need far more physical qubits than logical ones.
"""
from __future__ import annotations

import itertools
import json
import time
from pathlib import Path

import numpy as np

TRACK_B_JSON = Path(__file__).resolve().parents[2] / "B_discrete" / "results" / "maxcut_instance.json"


def load_maxcut(path=TRACK_B_JSON, nodes: int | None = None):
    d = json.loads(Path(path).read_text())
    n = d["n_nodes"] if nodes is None else nodes
    edges = [(i, j, w) for i, j, w in d["edges"] if i < n and j < n]
    return n, edges


def cut_values(n, edges):
    """Cut value of every basis state, index k with bit i = (k >> (n-1-i)) & 1 (wire 0 = MSB)."""
    k = np.arange(2 ** n)
    bits = (k[:, None] >> (n - 1 - np.arange(n))) & 1
    cut = np.zeros(2 ** n)
    for i, j, w in edges:
        cut += w * (bits[:, i] != bits[:, j])
    return cut


# ------------------------------------------------------------------ 1. exact quantum annealing
def _apply_rx_all(psi, n, theta):
    """exp(-i theta/2 X) on every qubit."""
    c, s = np.cos(theta / 2), -1j * np.sin(theta / 2)
    psi = psi.reshape([2] * n)
    for q in range(n):
        a = np.take(psi, 0, axis=q)
        b = np.take(psi, 1, axis=q)
        psi = np.stack([c * a + s * b, s * a + c * b], axis=q)
    return psi.reshape(-1)


def quantum_anneal(n, edges, total_time: float, steps: int | None = None):
    """Return P(ending in an optimal cut) and expected cut / optimum."""
    cut = cut_values(n, edges)
    best = cut.max()
    hp = -cut / best                         # problem Hamiltonian scaled to [-1, 0]
    steps = steps or max(200, int(20 * total_time))
    dt = total_time / steps
    psi = np.full(2 ** n, 2 ** (-n / 2), dtype=complex)     # |+>^n
    for k in range(steps):
        s = (k + 0.5) / steps
        # exp(-i dt (1-s)(-sum X)) = prod_i exp(+i dt (1-s) X_i) = RX(-2 dt (1-s))
        psi = _apply_rx_all(psi, n, -dt * (1 - s))          # half step
        psi *= np.exp(-1j * dt * s * hp)
        psi = _apply_rx_all(psi, n, -dt * (1 - s))          # half step
    p = np.abs(psi) ** 2
    return dict(p_optimal=float(p[cut == best].sum()),
                approx_ratio=float((p * cut).sum() / best))


def spectral_gap(n, edges, points: int = 60):
    """Minimum gap between the ground state and the first excited level that does
    not belong to the final ground space, in the bit-flip-symmetric sector
    (both H_X and H_P commute with flipping every bit, and |+>^n is symmetric)."""
    cut = cut_values(n, edges)
    best = cut.max()
    hp = np.diag(-cut / best)
    hx = np.zeros((2 ** n, 2 ** n))
    for q in range(n):
        mask = 1 << (n - 1 - q)
        idx = np.arange(2 ** n)
        hx[idx, idx ^ mask] -= 1
    # symmetric sector basis: (|k> + |~k>)/sqrt2 for k with top bit 0
    half = np.arange(2 ** (n - 1))
    comp = (2 ** n - 1) - half
    B = np.zeros((2 ** n, len(half)))
    B[half, np.arange(len(half))] = B[comp, np.arange(len(half))] = 1 / np.sqrt(2)
    n_ground = int(((cut == best).sum()) // 2)            # degenerate optimal pairs
    gaps = []
    for s in np.linspace(0, 1, points):
        Hs = B.T @ ((1 - s) * hx + s * hp) @ B
        e = np.linalg.eigvalsh(Hs)
        gaps.append(e[n_ground] - e[0])   # first level outside the final ground manifold
    return float(min(gaps)), int(np.argmin(gaps)) / (points - 1)


# ------------------------------------------------------------------ 2. simulated annealing (classical)
def simulated_annealing(n, edges, num_reads=100, num_sweeps=1000, beta_range=(0.1, 5.0), seed=0):
    """Minimise E(s) = sum w_ij s_i s_j over s in {-1,+1}^n (antiferromagnetic
    Ising = max-cut). Each sweep proposes a flip of every spin once; a flip is
    accepted with probability min(1, exp(-beta * dE)). beta grows geometrically
    from beta_range[0] (hot, random) to beta_range[1] (cold, greedy)."""
    rng = np.random.default_rng(seed)
    nbrs = [[] for _ in range(n)]
    for i, j, w in edges:
        nbrs[i].append((j, w)); nbrs[j].append((i, w))
    betas = np.geomspace(beta_range[0], beta_range[1], num_sweeps)
    best = cut_values(n, edges).max()
    t0 = time.perf_counter()
    cuts = []
    for _ in range(num_reads):
        s = rng.choice([-1, 1], size=n)
        for beta in betas:
            for i in rng.permutation(n):
                dE = -2 * s[i] * sum(w * s[j] for j, w in nbrs[i])   # energy change if s_i flips
                if dE <= 0 or rng.random() < np.exp(-beta * dE):
                    s[i] = -s[i]
        cuts.append(sum(w for i, j, w in edges if s[i] != s[j]))
    secs = time.perf_counter() - t0
    return dict(p_optimal=float(np.mean([c == best for c in cuts])),
                approx_ratio=float(np.mean(cuts) / best), seconds=secs, reads=num_reads,
                sweeps=num_sweeps)


# ------------------------------------------------------------------ 3. minor embedding
def hardware_graph(name: str):
    import dwave_networkx as dnx
    if name == "pegasus_p6":
        return dnx.pegasus_graph(6)
    if name == "chimera_c16":
        return dnx.chimera_graph(16)
    raise ValueError(name)


def embed(logical_edges, hw, seed=0, timeout=30, tries=3):
    """Find a minor embedding; return physical qubit count and chain stats (None if not found)."""
    import minorminer
    best = None
    for k in range(tries):
        emb = minorminer.find_embedding(logical_edges, hw.edges, random_seed=seed + k,
                                        timeout=timeout)
        if emb:
            chains = [len(c) for c in emb.values()]
            r = dict(physical_qubits=sum(chains), max_chain=max(chains), mean_chain=float(np.mean(chains)))
            if best is None or r["physical_qubits"] < best["physical_qubits"]:
                best = r
    return best


def complete_graph_edges(n):
    return list(itertools.combinations(range(n), 2))


def knapsack_qubo_edges(n_items: int, capacity: int):
    """Interaction graph of the Track B knapsack QUBO: the squared penalty couples
    every pair among items + slack bits, so it is a complete graph."""
    import math
    m = n_items + max(1, math.ceil(math.log2(capacity + 1)))
    return complete_graph_edges(m), m
