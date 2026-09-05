"""Finite barriers: the envelope Hamiltonian for layered heterostructures.

The uniform-material assembly puts hard walls at the grid ends, and a
hard wall forces the envelope to vanish at the interface, pushing the gas
outward and shifting subband energies; the package README names finite
barriers the gate before research-grade use.  This module closes that
gap: material parameters and the valence band edge become functions of z,
and the kinetic terms are discretized in the symmetrized
(Ben Daniel-Duke) form,

    kz^2 term:  -d/dz [H2(z) d/dz]  ->  midpoint averages
                H2_{i+1/2} = (H2_i + H2_{i+1}) / 2,
    kz term:    (1/2) { H1(z), kz }  ->  arithmetic-mean symmetrized
                first difference,

so the assembled matrix is exactly Hermitian for arbitrary profiles, and
reduces exactly (to machine precision, asserted in the tests) to the
uniform-material assembly when every point carries the same material.

The valence band offset enters as ``band_edge(z)``, the local shift of
the valence band maximum in the valence-electron energy convention used
throughout the package (holes occupy the highest eigenvalues, so a hole
barrier is a region with a lower band edge).  No default offset value is
shipped on purpose: band alignments are material- and strain-specific and
must be supplied with a citation, like every other number in this
package.

The test suite validates the finite-barrier physics against closed forms:
the decoupled single-band well against the textbook transcendental
equation for a finite square well, the exponential envelope decay in the
barrier against the analytic decay constant, and the deep-barrier limit
against the hard-wall solver.
"""
from __future__ import annotations

import numpy as np

from .hamiltonian import assemble_hamiltonian, bulk_blocks


def layered_profile(z, layers):
    """Build per-point (params, band_edge) profiles from a layer stack.

    layers: sequence of (thickness_nm, params, band_edge_eV) from the
    left end of the grid; thicknesses must tile the grid span.  Points
    are assigned by their coordinate; the total thickness must cover the
    grid within half a grid spacing.

    Returns (params_list, band_edge_array).
    """
    z = np.asarray(z, dtype=float)
    dz = z[1] - z[0]
    total = sum(t for t, _, _ in layers)
    span = z[-1] - z[0]
    if abs(total - span) > 0.51 * dz:
        raise ValueError(
            f"layer thicknesses sum to {total} nm but the grid spans "
            f"{span} nm; they must agree to within half a grid spacing")
    edges = np.cumsum([0.0] + [t for t, _, _ in layers]) + z[0]
    params_list = []
    band_edge = np.empty(z.size)
    for i, zi in enumerate(z):
        k = int(np.searchsorted(edges[1:-1], zi, side="right"))
        params_list.append(layers[k][1])
        band_edge[i] = layers[k][2]
    return params_list, band_edge


def assemble_heterostructure(z, params_list, band_edge=None,
                             kx: float = 0.0, ky: float = 0.0,
                             potential=None, strain_list=None):
    """Dense 6N x 6N envelope Hamiltonian with position-dependent
    material parameters and band edge (Ben Daniel-Duke symmetrized).

    z: uniform grid (nm); params_list: one WurtziteParameters per grid
    point; band_edge: valence band edge shift per point (eV, valence
    electron convention), optional; potential: optional hole potential
    V_h(z) in eV, entering as -V_h on the diagonal exactly as in the
    uniform assembly; strain_list: optional per-point symmetric 3 x 3
    strain tensors (or one tensor for all points) -- pseudomorphic
    stacks strain each layer differently, which is why this is
    per-point. Each strained point's parameter set must carry cited
    D1..D6.
    """
    z = np.asarray(z, dtype=float)
    n = z.size
    dz = z[1] - z[0]
    if not np.allclose(np.diff(z), dz):
        raise ValueError("z grid must be uniform")
    if len(params_list) != n:
        raise ValueError("one parameter set per grid point required")
    if band_edge is not None:
        band_edge = np.asarray(band_edge, dtype=float)
        if band_edge.shape != (n,):
            raise ValueError("band_edge must have one value per grid point")

    blocks = [bulk_blocks(p, kx, ky) for p in params_list]
    if strain_list is not None:
        from .strain import strain_blocks
        s = np.asarray(strain_list, dtype=float)
        if s.shape == (3, 3):
            strains = [s] * n
        elif s.shape == (n, 3, 3):
            strains = list(s)
        else:
            raise ValueError("strain_list must be one 3x3 tensor or one "
                             "per grid point")
        blocks = [(H0 + strain_blocks(p, e), H1, H2)
                  for (H0, H1, H2), p, e in zip(blocks, params_list,
                                                strains)]
    H = np.zeros((6 * n, 6 * n), dtype=complex)
    idx = lambda i: slice(6 * i, 6 * i + 6)
    inv_dz2 = 1.0 / (dz * dz)
    inv_2dz = 1.0 / (2.0 * dz)

    # midpoint kz^2 coefficients
    half = [0.5 * (blocks[i][2] + blocks[i + 1][2]) for i in range(n - 1)]
    for i in range(n):
        H0_i, H1_i, H2_i = blocks[i]
        diag = H0_i.copy()
        left = half[i - 1] if i > 0 else blocks[i][2]
        right = half[i] if i + 1 < n else blocks[i][2]
        diag += (left + right) * inv_dz2
        if band_edge is not None:
            diag += band_edge[i] * np.eye(6)
        if potential is not None:
            diag -= potential[i] * np.eye(6)
        H[idx(i), idx(i)] = diag
        if i + 1 < n:
            H1_half = 0.5 * (H1_i + blocks[i + 1][1])
            off = -half[i] * inv_dz2 + (-1j) * H1_half * inv_2dz
            H[idx(i), idx(i + 1)] = off
            H[idx(i + 1), idx(i)] = off.conj().T
    return H


def solve_heterostructure(z, params_list, band_edge=None, kx: float = 0.0,
                          ky: float = 0.0, potential=None, n_states: int = 8,
                          strain_list=None):
    """Top valence states of a layered heterostructure.

    Same return convention as :func:`~kpenvelope.solver.solve_subbands`:
    (energies descending in eV, envelopes (n_states, 6, N) normalized to
    unit total probability).
    """
    z = np.asarray(z, dtype=float)
    H = assemble_heterostructure(z, params_list, band_edge, kx, ky,
                                 potential, strain_list=strain_list)
    vals, vecs = np.linalg.eigh(H)
    order = np.argsort(vals)[::-1][:n_states]
    dz = z[1] - z[0]
    envelopes = np.empty((len(order), 6, z.size), dtype=complex)
    for j, o in enumerate(order):
        v = vecs[:, o].reshape(z.size, 6).T
        norm = np.sqrt((np.abs(v) ** 2).sum() * dz)
        envelopes[j] = v / norm
    return vals[order], envelopes
