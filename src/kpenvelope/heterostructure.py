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

from .hamiltonian import (assemble_hamiltonian, bulk_blocks, check_grid,
                          check_n_states, check_potential, place_blocks)
from .solver import envelopes_from_vectors, top_eigenpairs


def layered_profile(z, layers) -> tuple:
    """Build per-point (params, band_edge) profiles from a layer stack.

    layers: sequence of (thickness_nm, params, band_edge_eV) from the
    left end of the grid; thicknesses must tile the grid span.  Points
    are assigned by their coordinate; the total thickness must cover the
    grid within half a grid spacing.

    A grid point that sits on an interface belongs to the layer on its
    right. "On" allows for rounding: a point within 1e-9 grid steps of
    an interface counts as on it (the tolerance `poisson.sheet_mask`
    uses for the same reason). Before v0.13.0 the comparison was exact,
    so whenever the running sum of the thicknesses rounded up past the
    grid point (for example 2.7 + 3.4 = 6.1000000000000005 against the
    point 6.1), that point went to the left layer instead, and a layer
    gained or lost one grid point depending on where it sat in the
    stack.

    Returns (params_list, band_edge_array).
    """
    z = check_grid(z)
    dz = z[1] - z[0]
    try:
        thicknesses = [float(t) for t, _, _ in layers]
    except (TypeError, ValueError):
        raise ValueError("layers must be a sequence of (thickness_nm, "
                         "params, band_edge_eV) triples") from None
    if not thicknesses:
        raise ValueError("layers must contain at least one layer")
    if not all(np.isfinite(t) and t >= 0.0 for t in thicknesses):
        raise ValueError("every layer thickness must be finite and "
                         f"not negative (nm); got {thicknesses}")
    total = sum(thicknesses)
    span = z[-1] - z[0]
    if abs(total - span) > 0.51 * dz:
        raise ValueError(
            f"layer thicknesses sum to {total} nm but the grid spans "
            f"{span} nm; they must agree to within half a grid spacing")
    edges = np.cumsum([0.0] + thicknesses) + z[0]
    tol = 1e-9 * dz
    params_list = []
    band_edge = np.empty(z.size)
    for i, zi in enumerate(z):
        k = int(np.searchsorted(edges[1:-1], zi + tol, side="right"))
        params_list.append(layers[k][1])
        band_edge[i] = layers[k][2]
    return params_list, band_edge


def assemble_heterostructure(z, params_list, band_edge=None,
                             kx: float = 0.0, ky: float = 0.0,
                             potential=None,
                             strain_list=None) -> np.ndarray:
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
    z = check_grid(z)
    n = z.size
    dz = z[1] - z[0]
    potential = check_potential(potential, n)
    if len(params_list) != n:
        raise ValueError("one parameter set per grid point required")
    if band_edge is not None:
        band_edge = np.asarray(band_edge, dtype=float)
        if band_edge.shape != (n,):
            raise ValueError("band_edge must have one value per grid point")
        if not np.all(np.isfinite(band_edge)):
            raise ValueError("band_edge must be finite (eV)")

    # one bulk evaluation per distinct parameter set (v0.15.0; the same
    # object gives the same blocks, so this only saves repeated work)
    cache = {}
    blocks = []
    for p in params_list:
        key = id(p)
        if key not in cache:
            cache[key] = bulk_blocks(p, kx, ky)
        blocks.append(cache[key])
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
    inv_dz2 = 1.0 / (dz * dz)
    inv_2dz = 1.0 / (2.0 * dz)
    H0s = np.array([b[0] for b in blocks])
    H1s = np.array([b[1] for b in blocks])
    H2s = np.array([b[2] for b in blocks])

    # midpoint kz^2 coefficients; at the two ends the point's own
    # (the operations, element by element, are those of the per-point
    # loop of v0.14.0 and earlier)
    half = 0.5 * (H2s[:-1] + H2s[1:])
    left = np.concatenate([H2s[:1], half])
    right = np.concatenate([half, H2s[-1:]])
    diags = H0s.copy()
    diags += (left + right) * inv_dz2
    if band_edge is not None:
        diags += band_edge[:, None, None] * np.eye(6)
    if potential is not None:
        diags -= potential[:, None, None] * np.eye(6)
    H1_half = 0.5 * (H1s[:-1] + H1s[1:])
    off = -half * inv_dz2 + (-1j) * H1_half * inv_2dz
    return place_blocks(n, diags, off, off.conj().transpose(0, 2, 1))


def solve_heterostructure(z, params_list, band_edge=None, kx: float = 0.0,
                          ky: float = 0.0, potential=None, n_states: int = 8,
                          strain_list=None) -> tuple:
    """Top valence states of a layered heterostructure.

    Same return convention as :func:`~kpenvelope.solver.solve_subbands`:
    (energies descending in eV, envelopes (n_states, 6, N) normalized to
    unit total probability). Since v0.14.0 the matrix is diagonalized in
    exact independent blocks where it has them, as in `solve_subbands`
    (energies equal to rounding, several times faster).
    """
    z = check_grid(z)
    n_states = check_n_states(n_states, z.size)
    H = assemble_heterostructure(z, params_list, band_edge, kx, ky,
                                 potential, strain_list=strain_list)
    vals, vecs = top_eigenpairs(H, z.size, n_states, kx, ky)
    return vals, envelopes_from_vectors(vecs, z.size, z[1] - z[0])


def heterostructure_energies(z, params_list, band_edge=None,
                             kx: float = 0.0, ky: float = 0.0,
                             potential=None, n_states: int = 8,
                             strain_list=None) -> np.ndarray:
    """The energies of `solve_heterostructure` (bit for bit) without
    building the envelopes (new in v0.14.0)."""
    z = check_grid(z)
    n_states = check_n_states(n_states, z.size)
    H = assemble_heterostructure(z, params_list, band_edge, kx, ky,
                                 potential, strain_list=strain_list)
    return top_eigenpairs(H, z.size, n_states, kx, ky, vectors=False)[0]
