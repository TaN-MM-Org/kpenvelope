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
* `cyclotron_mass` (new in v0.14.0), the mass a Shubnikov-de Haas
  measurement sees at the Fermi contour of a branch, is exact for
  parabolic branches, second order in the path spacing against a
  closed-form nonparabolic branch, and agrees with a direct difference
  of two six-band solves around k_F.
"""
from __future__ import annotations

import numpy as np

from .hamiltonian import HBAR2_OVER_2M0
from .solver import solve_subbands, subband_energies  # noqa: F401


def subband_dispersion(p, z, kts, theta: float = 0.0, potential=None,
                       n_states: int = 4) -> np.ndarray:
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
        out[i] = subband_energies(p, z, kx=kt * ct, ky=kt * st,
                                  potential=potential, n_states=n_states)
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


def spin_splitting(energies):
    """Pairwise splittings of Kramers-related subband pairs.

    energies : descending subband energies at one k (as returned
    everywhere in this package). Returns E[0]-E[1], E[2]-E[3], ... --
    zero for every pair at k_t = 0 (time reversal) and at every k in a
    structure with inversion-symmetric confinement, nonzero at finite
    k_t under an asymmetric potential (Rashba-type splitting of the
    2DHG); both statements are asserted in the tests rather than
    stated.
    """
    e = np.asarray(energies, dtype=float)
    n_pairs = e.size // 2
    return e[0:2 * n_pairs:2] - e[1:2 * n_pairs:2]


def splitting_vs_k(p, z, kts, theta=0.0, potential=None, n_pairs=2):
    """Spin splitting of the top ``n_pairs`` subband pairs along an
    in-plane path: `subband_dispersion` + `spin_splitting` in one call.
    Returns shape (len(kts), n_pairs)."""
    disp = subband_dispersion(p, z, kts, theta=theta, potential=potential,
                              n_states=2 * n_pairs)
    return np.stack([spin_splitting(row) for row in disp])


def cyclotron_mass(kts, energies, n_branch) -> dict:
    """Cyclotron mass of a computed subband at the Fermi contour that
    holds ``n_branch`` holes per nm^2 (new in v0.14.0).

    This is the number a Shubnikov-de Haas temperature series measures
    (`fit_sdh_mass`), computed from the dispersion instead, so the two
    can be compared directly. For a branch that depends only on |k_t|
    and lies above the Fermi energy exactly inside k_F, the occupied
    region is a disk of radius k_F = sqrt(4 pi n_branch) (one
    spin-resolved branch has 1/(2 pi)^2 states per unit k-area), and
    the cyclotron mass
    m_c = (hbar^2 / 2 pi) dA/dE with A = pi k^2 is

        m_c / m0 = (hbar^2/2m0) |d(k_t^2)/dE|  at k_F,

    the same definition as `local_mass`, here at k_F itself: E is
    interpolated by a parabola in q = k_t^2 through the three path
    points nearest to k_F^2 and differentiated there. That is exact for
    a parabolic branch at any spacing and second order in the step
    otherwise (both asserted in the tests).

    kts : (M,) in-plane momenta (nm^-1), M >= 3, strictly increasing
        and starting at 0. energies : (M,) or (M, n) energies (eV) of
        each branch along the path, e.g. from `subband_dispersion`.
    n_branch : holes per nm^2 in each branch, scalar or (n,); for the
        package's per-state occupations, pass them as they are (one
        spin-resolved branch each).

    Returns dict(mass_ratio, k_fermi), scalars for a single branch.
    Refuses a path that does not start at k_t = 0, a k_F beyond the
    path, and a branch whose region above the Fermi energy E(k_F) is
    not the disk k_t < k_F on the path (every point inside must lie
    above it, every point outside below it, and the branch must fall
    at k_F): then the occupied region is not one disk and this formula
    does not apply.
    """
    kts = np.asarray(kts, dtype=float)
    if kts.ndim != 1 or kts.size < 3:
        raise ValueError("kts must be a 1D array of at least three "
                         "points")
    if not np.all(np.isfinite(kts)) or np.any(np.diff(kts) <= 0.0):
        raise ValueError("kts must be finite and strictly increasing")
    if kts[0] != 0.0:
        raise ValueError("kts must start at 0: the occupied disk is "
                         "only known if the whole branch from k_t = 0 "
                         "is on the path")
    E = np.asarray(energies, dtype=float)
    single = E.ndim == 1
    E2 = E[:, None] if single else E
    if E2.ndim != 2 or E2.shape[0] != kts.size:
        raise ValueError("energies must have one row per kts point")
    if not np.all(np.isfinite(E2)):
        raise ValueError("energies must be finite")
    n = np.atleast_1d(np.asarray(n_branch, dtype=float))
    if n.size == 1 and E2.shape[1] > 1:
        n = np.full(E2.shape[1], n[0])
    if n.shape != (E2.shape[1],):
        raise ValueError("one n_branch per branch (column of energies)")
    if not (np.all(np.isfinite(n)) and np.all(n > 0.0)):
        raise ValueError("n_branch must be finite and > 0 (nm^-2; "
                         "1e13 cm^-2 = 0.1 nm^-2)")
    q = kts ** 2
    kf = np.sqrt(4.0 * np.pi * n)
    masses = np.empty(n.size)
    for b in range(n.size):
        qf = kf[b] ** 2
        if qf > q[-1]:
            raise ValueError(
                f"branch {b}: k_F = {kf[b]:.6g} nm^-1 lies beyond the "
                f"path (kts[-1] = {kts[-1]:.6g} nm^-1); extend the path")
        # three nodes centered on the node nearest to q_F
        i0 = min(max(int(np.argmin(np.abs(q - qf))) - 1, 0), q.size - 3)
        e = E2[:, b]
        x0, x1, x2 = q[i0:i0 + 3]
        y0, y1, y2 = e[i0:i0 + 3]
        # value and derivative at qf of the parabola through the nodes
        l0 = (qf - x1) * (qf - x2) / ((x0 - x1) * (x0 - x2))
        l1 = (qf - x0) * (qf - x2) / ((x1 - x0) * (x1 - x2))
        l2 = (qf - x0) * (qf - x1) / ((x2 - x0) * (x2 - x1))
        ef = y0 * l0 + y1 * l1 + y2 * l2
        d = (y0 * ((qf - x1) + (qf - x2)) / ((x0 - x1) * (x0 - x2))
             + y1 * ((qf - x0) + (qf - x2)) / ((x1 - x0) * (x1 - x2))
             + y2 * ((qf - x0) + (qf - x1)) / ((x2 - x0) * (x2 - x1)))
        # the holes fill {E > E_F}: that is the disk k < k_F only if
        # every path point inside lies above E_F and every one outside
        # below it, and the branch falls through E_F at k_F
        if not (d < 0.0 and np.all(e[q < qf] > ef)
                and np.all(e[q > qf] < ef)):
            raise ValueError(
                f"branch {b}: the region where it lies above the Fermi "
                f"energy ({ef:.6g} eV) is not the disk k_t < k_F on this "
                "path (the branch does not fall through the Fermi "
                "energy once), so this formula does not apply; use the "
                "full dispersion")
        masses[b] = HBAR2_OVER_2M0 / abs(d)
    if single:
        return {"mass_ratio": float(masses[0]), "k_fermi": float(kf[0])}
    return {"mass_ratio": masses, "k_fermi": kf}


def group_velocity(kts, energies):
    """dE/dk along the path, by central differences (eV nm; divide by
    hbar for a velocity). Input as from `subband_dispersion`:
    energies shape (len(kts), n_states). Exactly 2 c A k for the
    parabolic demo set, asserted in the tests."""
    kts = np.asarray(kts, dtype=float)
    E = np.asarray(energies, dtype=float)
    return np.gradient(E, kts, axis=0, edge_order=2)


def dos_from_dispersion(kts, energies, e_grid) -> np.ndarray:
    """Two-dimensional density of states of isotropic, monotone
    subband dispersions (states nm^-2 eV^-1 per spin-resolved branch).

    For a hole subband E_n(k_t) decreasing from its edge, the exact 2D
    relation is rho_n(E) = k/(2 pi) |dk/dE|; evaluated here by
    differentiating the occupied-area function N_n(E) = k*(E)^2/(4 pi)
    on ``e_grid``.  A non-monotone dispersion is refused (its DOS needs
    the full k-grid machinery of `fill_subbands_kgrid`, not this
    shortcut) -- stated plainly rather than silently interpolated.
    Exactly (1/|A|)/(4 pi hbar^2/2m0) for the parabolic demo set,
    asserted in the tests.

    Energies below the end of the path (below E_n(kts[-1])) are outside
    what the path describes, so the result there is NaN, as it is at a
    grid point whose difference stencil reaches such an energy (new in
    v0.13.0). Before, the occupied area was held at its value at
    kts[-1], which returned a DOS of exactly 0 there: a plausible
    looking gap where the band in fact continues. Above the subband
    edge the DOS is 0, correctly.

    The same holds at the start of a path that does not begin at
    k_t = 0 (new in v0.14.0): energies above E_n(kts[0]) are outside
    what the path describes, so they give NaN too, as does any grid
    point whose stencil reaches them. Before, the occupied area there
    was taken as 0, which returned a DOS of 0 where the band continues
    and a spike at E_n(kts[0]) (2.6 instead of 1.04 nm^-2 eV^-1 for the
    demo set on a path from 0.2 nm^-1).

    Returns shape (len(e_grid), n_states).
    """
    kts = np.asarray(kts, dtype=float)
    E = np.asarray(energies, dtype=float)
    e_grid = np.asarray(e_grid, dtype=float)
    if kts.ndim != 1 or kts.size < 2:
        raise ValueError("kts must be a 1D array of at least two points")
    if np.any(np.diff(kts) <= 0.0) or kts[0] < 0.0:
        raise ValueError("kts must be non-negative and strictly "
                         "increasing")
    if E.ndim != 2 or E.shape[0] != kts.size:
        raise ValueError(
            f"energies must have shape (len(kts), n_states) = "
            f"({kts.size}, n), as from subband_dispersion; got {E.shape}")
    if e_grid.ndim != 1 or e_grid.size < 2:
        raise ValueError("e_grid must be a 1D array of at least two "
                         "energies")
    out = np.zeros((e_grid.size, E.shape[1]))
    for n in range(E.shape[1]):
        en = E[:, n]
        if np.any(np.diff(en) > 1e-12):
            raise ValueError(
                f"subband {n} is not monotone decreasing along the "
                "path; use fill_subbands_kgrid for its thermodynamics")
        kstar = np.interp(-e_grid, -en, kts, left=0.0, right=kts[-1])
        N = kstar ** 2 / (4.0 * np.pi)
        out[:, n] = -np.gradient(N, e_grid)
        beyond = e_grid < en[-1]
        if kts[0] > 0.0:
            beyond = beyond | (e_grid > en[0])
        if np.any(beyond):
            reach = beyond.copy()
            reach[1:] |= beyond[:-1]
            reach[:-1] |= beyond[1:]
            out[reach, n] = np.nan
    return out


def dipole_matrix(z, envelopes):
    """Intersubband dipole matrix z_fi = <f| z |i> (nm), summed over
    the six band components with the solver's normalization,

        z_fi = sum_m integral F_m^f(z)* z F_m^i(z) dz,

    by the same uniform-grid quadrature the solver normalizes with.
    Hermitian by construction (asserted). Exact facts the test suite
    pins rather than states: for the decoupled hard-wall well the
    off-diagonal element is the infinite-well closed form
    z_12 = 16 L / (9 pi^2), the diagonal is the well center L/2, and
    same-parity transitions vanish; in a symmetric six-band well the
    parity selection rule survives the band mixing.
    """
    z = np.asarray(z, dtype=float)
    env = np.asarray(envelopes)
    dz = z[1] - z[0]
    n = env.shape[0]
    out = np.empty((n, n), dtype=complex)
    for f in range(n):
        for i in range(f, n):
            val = (np.conj(env[f]) * z[None, :] * env[i]).sum() * dz
            out[f, i] = val
            out[i, f] = np.conj(val)
    return out


def oscillator_strengths(energies, dip, mass_ratio):
    """Intersubband oscillator strengths from the ground subband, in
    the hole convention of this package (energies descending):

        f_(1 -> i) = (m*/m0) (E_1 - E_i) |z_1i|^2 / c,

    with c = hbar^2/2m0 in eV nm^2, so the Thomas-Reiche-Kuhn sum rule
    reads sum_i f_(1->i) = 1 for a single parabolic band of mass
    m* = mass_ratio * m0 -- asserted in the tests against the
    hard-wall demo set, together with the closed-form
    f_12 = 256/(27 pi^2) of the infinite well. The demo set is
    six-fold degenerate, so individual elements within a degenerate
    multiplet are basis-dependent; the MULTIPLET-SUMMED strengths are
    the invariant quantities and are what the closed forms pin.

    energies : (n,) descending eV (as returned everywhere here);
    dip : (n, n) dipole matrix from `dipole_matrix`;
    mass_ratio : the confinement-direction effective mass ratio
        m*/m0 of the band (for the decoupled demo set,
        1/|A|). For coupled six-band states no single mass exists and
        the sum rule is not 1; the strengths remain well-defined
        matrix elements. Returns (n,) with f_11 = 0.
    """
    e = np.asarray(energies, dtype=float)
    d = np.asarray(dip)
    f = float(mass_ratio) * (e[0] - e) * np.abs(d[0, :]) ** 2 \
        / HBAR2_OVER_2M0
    f[0] = 0.0
    return f
