"""Eigen-solution of the assembled envelope Hamiltonian."""
from __future__ import annotations

import numpy as np

from .hamiltonian import assemble_hamiltonian


def solve_subbands(p, z, kx=0.0, ky=0.0, potential=None, n_states=8,
                   strain=None):
    """Solve for the top valence states (the ones holes occupy).

    Returns (energies, envelopes):
      energies : (n_states,) eV, sorted descending (highest first).
      envelopes : (n_states, 6, N) complex envelope functions, normalized
          such that sum_m integral |F_m|^2 dz = 1.
    strain : optional symmetric 3 x 3 tensor, passed to the assembly
        (requires cited D1..D6 on the parameter set).
    """
    z = np.asarray(z, dtype=float)
    H = assemble_hamiltonian(p, z, kx, ky, potential, strain=strain)
    vals, vecs = np.linalg.eigh(H)
    order = np.argsort(vals)[::-1][:n_states]
    dz = z[1] - z[0]
    energies = vals[order]
    envelopes = np.empty((len(order), 6, z.size), dtype=complex)
    for j, o in enumerate(order):
        v = vecs[:, o].reshape(z.size, 6).T  # (6, N)
        norm = np.sqrt((np.abs(v) ** 2).sum() * dz)
        envelopes[j] = v / norm
    return energies, envelopes
