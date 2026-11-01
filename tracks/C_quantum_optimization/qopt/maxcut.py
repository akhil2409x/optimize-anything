"""QAOA for the Track B Max-Cut instance.

The cost Hamiltonian is built from the Ising coefficients that Track B wrote to
tracks/B_discrete/results/maxcut_instance.json, so Track C solves *exactly*
the problem Track B formulated:

    H_C = offset * I + sum_{i<j} J_ij Z_i Z_j + sum_i h_i Z_i,     <z|H_C|z> = -cut(z)

Convention (same as Track B): bit x_i = 0  <->  |0>  <->  Z eigenvalue +1  <->  s_i = +1.

QAOA with p layers:
    |psi(gamma, beta)> = prod_{k=1..p} exp(-i beta_k H_M) exp(-i gamma_k H_C) |+>^n,
    H_M = sum_i X_i
    exp(-i gamma J Z_i Z_j) = IsingZZ(2 gamma J),   exp(-i beta X_i) = RX(2 beta)
The classical optimizer minimises <H_C> = -(expected cut).

Bit order: PennyLane's probability vector treats wire 0 as the MOST significant
bit, so index k corresponds to bits[i] = (k >> (n-1-i)) & 1. `index_to_bits`
does this and a test checks it against Track B's optimal assignments.
"""
from __future__ import annotations

import json
from itertools import product
from pathlib import Path

import pennylane as qml
from pennylane import numpy as pnp

TRACK_B_JSON = (Path(__file__).resolve().parents[2] / "B_discrete" / "results" / "maxcut_instance.json")


class MaxCut:
    def __init__(self, n: int, edges, J: dict, h: list, offset: float, optimal_cut=None,
                 optimal_bits=None, name="maxcut"):
        self.n, self.edges, self.J, self.h, self.offset = n, edges, J, h, offset
        self.name = name
        if optimal_cut is None:              # brute force (n <= ~20)
            cuts = {bits: self.cut(bits) for bits in product((0, 1), repeat=n)}
            optimal_cut = max(cuts.values())
            optimal_bits = [list(b) for b, c in cuts.items() if c == optimal_cut]
        self.optimal_cut = optimal_cut
        self.optimal_set = {tuple(b) for b in optimal_bits}
        self.total_weight = sum(w for *_, w in edges)

    # --------------------------------------------------------- constructors
    @classmethod
    def from_track_b(cls, path=TRACK_B_JSON):
        d = json.loads(Path(path).read_text())
        J = {tuple(map(int, k.split(","))): v for k, v in d["ising"]["J"].items()}
        return cls(d["n_nodes"], [tuple(e) for e in d["edges"]], J, d["ising"]["h"],
                   d["ising"]["offset"], d["optimal_cut"], d["all_optimal_assignments_bits"],
                   name=f"trackB-maxcut-{d['n_nodes']}")

    @classmethod
    def from_edges(cls, n, edges, name="maxcut"):
        """Max-Cut Ising: -w(x_i + x_j - 2x_i x_j) = w/2 (Z_i Z_j - 1)."""
        J, offset = {}, 0.0
        for i, j, w in edges:
            J[(i, j)] = J.get((i, j), 0.0) + w / 2
            offset -= w / 2
        return cls(n, list(edges), J, [0.0] * n, offset, name=name)

    def induced(self, nodes: int):
        """Sub-problem on nodes 0..nodes-1 (used for the slow noisy simulations)."""
        sub = [(i, j, w) for i, j, w in self.edges if i < nodes and j < nodes]
        return MaxCut.from_edges(nodes, sub, name=f"{self.name}-induced{nodes}")

    # --------------------------------------------------------- classical side
    def cut(self, bits) -> int:
        return sum(w for i, j, w in self.edges if bits[i] != bits[j])

    def hamiltonian(self):
        coeffs, ops = [self.offset], [qml.Identity(0)]
        for (i, j), c in self.J.items():
            coeffs.append(c); ops.append(qml.Z(i) @ qml.Z(j))
        for i, c in enumerate(self.h):
            if c:
                coeffs.append(c); ops.append(qml.Z(i))
        return qml.Hamiltonian(coeffs, ops)

    def index_to_bits(self, k: int):
        return tuple((k >> (self.n - 1 - i)) & 1 for i in range(self.n))

    def stats_from_probs(self, probs):
        """Expected cut, approximation ratio and P(sampling an optimal cut)."""
        exp_cut = p_opt = 0.0
        for k, p in enumerate(probs):
            bits = self.index_to_bits(k)
            exp_cut += float(p) * self.cut(bits)
            if bits in self.optimal_set:
                p_opt += float(p)
        return dict(expected_cut=exp_cut, approx_ratio=exp_cut / self.optimal_cut, p_optimal=p_opt)


# ------------------------------------------------------------------ circuits
def qaoa_layers(mc: MaxCut, params, noise: float = 0.0):
    """params shape (p, 2): rows (gamma_k, beta_k). Optional depolarizing noise
    after every gate (density-matrix device only)."""
    for w in range(mc.n):
        qml.Hadamard(w)
        if noise:
            qml.DepolarizingChannel(noise, w)
    for gamma, beta in params:
        for (i, j), c in mc.J.items():
            qml.IsingZZ(2 * gamma * c, wires=[i, j])
            if noise:
                qml.DepolarizingChannel(noise, i); qml.DepolarizingChannel(noise, j)
        for i, c in enumerate(mc.h):
            if c:
                qml.RZ(2 * gamma * c, wires=i)
        for w in range(mc.n):
            qml.RX(2 * beta, wires=w)
            if noise:
                qml.DepolarizingChannel(noise, w)


def cost_qnode(mc: MaxCut, shots=None, diff_method="parameter-shift", device="default.qubit",
               noise: float = 0.0, seed=None):
    dev = qml.device(device, wires=mc.n, seed=seed)
    H = mc.hamiltonian()

    @qml.set_shots(shots)
    @qml.qnode(dev, diff_method=diff_method)
    def cost(params):
        qaoa_layers(mc, params, noise)
        return qml.expval(H)
    return cost, dev


def probs_qnode(mc: MaxCut, device="default.qubit", noise: float = 0.0):
    dev = qml.device(device, wires=mc.n)

    @qml.qnode(dev)
    def probs(params):
        qaoa_layers(mc, params, noise)
        return qml.probs(wires=range(mc.n))
    return probs


def init_params(p: int, seed: int = 0, scale: float = 0.1):
    """Linear-ramp start (an adiabatic-like schedule) plus small random noise."""
    rng = pnp.random.default_rng(seed)
    ramp = (pnp.arange(p) + 0.5) / p
    base = pnp.stack([0.4 * ramp, 0.4 * (1 - ramp)], axis=1)
    return pnp.array(base + scale * rng.standard_normal((p, 2)), requires_grad=True)
