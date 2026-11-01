"""VQE for the hydrogen molecule (H2, STO-3G basis, 4 qubits).

Classical baselines:
  * Hartree-Fock energy (mean-field; what VQE starts from)
  * exact diagonalisation of the qubit Hamiltonian restricted to the 2-electron
    sector (= full configuration interaction in this basis). This is the true
    answer VQE is trying to reach; chemical accuracy is 1.6 mHa from it.

Ansatze (both are parameterized quantum circuits, PQCs):
  * ucc  : HF state + one DoubleExcitation gate (1 parameter). Chemistry-inspired,
           conserves particle number.
  * hea  : hardware-efficient ansatz, layers of RY/RZ on every qubit + CNOT
           ladder, started from the HF state. Many parameters, hardware-friendly
           gates, does NOT conserve particle number.
"""
from __future__ import annotations

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from pennylane import qchem

ANGSTROM_TO_BOHR = 1.8897259886
N_ELECTRONS = 2


def h2_hamiltonian(bond_angstrom: float = 0.7414):
    d = bond_angstrom * ANGSTROM_TO_BOHR
    coords = np.array([0.0, 0.0, -d / 2, 0.0, 0.0, d / 2])
    H, n_qubits = qchem.molecular_hamiltonian(["H", "H"], coords)
    return H, n_qubits


def exact_energy(H, n_qubits: int, n_electrons: int = N_ELECTRONS) -> float:
    """Lowest eigenvalue within the n-electron sector (basis states of Hamming weight n)."""
    M = qml.matrix(H, wire_order=range(n_qubits))
    idx = [k for k in range(2 ** n_qubits) if bin(k).count("1") == n_electrons]
    return float(np.linalg.eigvalsh(M[np.ix_(idx, idx)].real)[0])


def hf_state(n_qubits: int):
    return qchem.hf_state(N_ELECTRONS, n_qubits)


def ucc_circuit(params, n_qubits: int):
    qml.BasisState(hf_state(n_qubits), wires=range(n_qubits))
    qml.DoubleExcitation(params[0], wires=list(range(n_qubits)))


def hea_circuit(params, n_qubits: int):
    """params shape (layers, n_qubits, 2)."""
    qml.BasisState(hf_state(n_qubits), wires=range(n_qubits))
    for layer in params:
        for w in range(n_qubits):
            qml.RY(layer[w, 0], wires=w)
            qml.RZ(layer[w, 1], wires=w)
        for w in range(n_qubits - 1):
            qml.CNOT(wires=[w, w + 1])


def energy_qnode(H, n_qubits: int, ansatz: str = "ucc", shots=None,
                 diff_method="parameter-shift", seed=None):
    dev = qml.device("default.qubit", wires=n_qubits, seed=seed)
    circuit = {"ucc": ucc_circuit, "hea": hea_circuit}[ansatz]

    @qml.set_shots(shots)
    @qml.qnode(dev, diff_method=diff_method)
    def energy(params):
        circuit(params, n_qubits)
        return qml.expval(H)
    return energy, dev


def init_params(ansatz: str, n_qubits: int, layers: int = 2, seed: int = 0):
    rng = np.random.default_rng(seed)
    if ansatz == "ucc":
        return pnp.array([0.0], requires_grad=True)
    # Small random angles. NB: the start is NOT the HF state even at zero angles,
    # because the CNOT ladder still acts on |1100> (start energy ~ +0.89 Ha).
    return pnp.array(0.1 * rng.standard_normal((layers, n_qubits, 2)), requires_grad=True)
