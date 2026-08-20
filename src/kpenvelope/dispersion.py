"""In-plane dispersion and effective-mass utilities.

A hole mass in a wurtzite heterostructure is not one number: it depends
on which branch is followed, at which in-plane momentum, along which
direction, and at which carrier density the confining potential was
solved. These utilities make that explicit instead of hiding it. They
compute subband dispersions along an in-plane path and convert any
dispersion, subband or bulk, into a local (finite-difference) effective
mass along the path.

Exact facts the test suite asserts, rather than states:

* For the decoupled demo set (E = A c k^2 exactly), the local mass is
  1/|A| m0 at every momentum and for every well level, to 1e-10.
* Applied to the bulk GaN bands of the Rinke 2008 set, the local mass at
  large k_t reproduces the closed-form quasi-cubic asymptotic masses
  m0/|A2+A4-A5| = 1.89 m0 and m0/|A2+A4+A5| = 0.180 m0.
* The dispersion path solver at a single k point equals solve_subbands
  exactly.
"""
from __future__ import annotations

import numpy as np

from .hamiltonian import HBAR2_OVER_2M0
from .solver import solve_subbands


def subband_dispersion(p, z, kts, theta=0.0, potential=None, n_states=4):
    """Subband energies along an in-plane path k_t >= 0 at angle theta.

    kts : array of in-plane momentum magnitudes (nm^-1).
    theta : in-plane direction in radians (kx = kt cos, ky = kt sin).
    potential : optional confining potential on z (eV), e.g. from
        solve_self_consistent; None solves the bare problem.

    Returns an array of shape (len(kts), n_states) of valence-electron
    energies in eV, each row sorted descending (the hole ground state
    first), exactly as solve_subbands returns them.
    """
    kts = np.asarray(kts, dtype=float)
    if np.any(kts < 0.0):
        raise ValueError("kts must be non-negative magnitudes")
    ct, st = np.cos(theta), np.sin(theta)
    out = np.empty((kts.size, n_states), dtype=float)
    for i, kt in enumerate(kts):
        energies, _ = solve_subbands(p, z, kx=kt * ct, ky=kt * st,
                                     potential=potential, n_states=n_states)
        out[i] = energies
    return out


def local_mass(kts, energies):
    """Local (finite-difference) in-plane effective mass along a path.

    Converts any dispersion E(k_t), subband or bulk, into the local mass
    m*(k_t)/m0 defined through E = (hbar^2 k^2 / 2 m0) / (m*/m0), i.e.
    m*/m0 = (hbar^2/2m0) d(k_t^2)/dE evaluated between neighboring path
    points. The sign is dropped: valence-electron energies fall with k_t
    while the hole energy rises, and the magnitude is the same.

    kts : (M,) in-plane momenta (nm^-1), strictly increasing.
    energies : (M,) or (M, n) energies (eV); a 2D array is treated as one
        branch per column.

    Returns (k_mid, masses) with k_mid the midpoints (M-1,) and masses of
    shape (M-1,) or (M-1, n). A branch that is locally flat between two
    points yields inf there, deliberately: a diverging mass is physics
    (a band extremum or inflection), not an error.
    """
    kts = np.asarray(kts, dtype=float)
    if kts.ndim != 1 or kts.size < 2:
        raise ValueError("kts must be a 1D array of at least two points")
    if np.any(np.diff(kts) <= 0.0):
        raise ValueError("kts must be strictly increasing")
    E = np.asarray(energies, dtype=float)
    if E.shape[0] != kts.size:
        raise ValueError("energies must have one row per kts point")
    dk2 = np.diff(kts ** 2)
    dE = np.diff(E, axis=0)
    k_mid = 0.5 * (kts[1:] + kts[:-1])
    with np.errstate(divide="ignore"):
        masses = np.abs(HBAR2_OVER_2M0 * (dk2 if E.ndim == 1 else
                                          dk2[:, None]) / dE)
    return k_mid, masses
