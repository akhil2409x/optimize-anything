"""Barren plateaus (McClean et al. 2018; Cerezo et al. 2021).

Random hardware-efficient PQC on n qubits with L layers:
    each layer: RY(theta) RZ(theta) on every qubit, then a CNOT ladder.
Two cost functions on the same circuit:
    global : <Z_0 Z_1 ... Z_{n-1}>   (acts on every qubit)
    local  : <Z_0 Z_1>               (acts on two qubits)
We sample random parameter vectors uniformly from [0, 2*pi) and measure the
variance of dC/dtheta_0 across samples. A barren plateau means this variance
shrinks exponentially in n, so a gradient-based optimizer started at random sees
an essentially flat landscape and would need exponentially many shots to
estimate the gradient's sign.

The gradient is computed with the parameter-shift rule (two circuit runs), the
same way hardware would do it:
    dC/dtheta_0 = [C(theta + pi/2 e_0) - C(theta - pi/2 e_0)] / 2
"""
from __future__ import annotations

import numpy as np
import pennylane as qml


def n_params(n: int, layers: int) -> int:
    return layers * n * 2


def cost_qnode(n: int, layers: int, cost: str = "global", device: str = "lightning.qubit"):
    dev = qml.device(device, wires=n)
    if cost == "global":
        obs = qml.prod(*[qml.Z(w) for w in range(n)])
    elif cost == "local":
        obs = qml.Z(0) @ qml.Z(1)
    else:
        raise ValueError(cost)

    @qml.qnode(dev, diff_method="best")
    def f(theta):
        k = 0
        for _ in range(layers):
            for w in range(n):
                qml.RY(theta[k], wires=w); qml.RZ(theta[k + 1], wires=w)
                k += 2
            for w in range(n - 1):
                qml.CNOT(wires=[w, w + 1])
        return qml.expval(obs)
    return f


def first_gradient(f, theta) -> float:
    plus, minus = np.array(theta, dtype=float), np.array(theta, dtype=float)
    plus[0] += np.pi / 2
    minus[0] -= np.pi / 2
    return (float(f(plus)) - float(f(minus))) / 2


def gradient_variance(n: int, layers: int, cost: str, samples: int = 200, seed: int = 0):
    rng = np.random.default_rng(seed)
    f = cost_qnode(n, layers, cost)
    grads = [first_gradient(f, rng.uniform(0, 2 * np.pi, n_params(n, layers)))
             for _ in range(samples)]
    return float(np.var(grads)), float(np.mean(grads))
