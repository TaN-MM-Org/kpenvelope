"""Self-consistent solution of the coupled k.p and Poisson problem."""
from __future__ import annotations

import dataclasses

import numpy as np

from .poisson import hole_potential
from .solver import solve_subbands
from .hamiltonian import HBAR2_OVER_2M0


@dataclasses.dataclass
class SelfConsistentResult:
    z: np.ndarray
    potential: np.ndarray          # hole potential energy V_h(z), eV
    energies: np.ndarray           # subband edges at k_t = 0 (valence eV, descending)
    envelopes: np.ndarray          # (n, 6, N)
    density: np.ndarray            # hole volume density (nm^-3)
    occupations: np.ndarray        # sheet density per subband (nm^-2)
    masses: np.ndarray             # numeric in-plane edge masses (units of m0)
    iterations: int
    residual: float


def _inplane_masses(p, z, potential, energies0, n_states, dk=0.02):
    """Numeric in-plane effective masses at the subband edge along kx.

    m*/m0 = (hbar^2/2m0) * (2 dk^2) / (E(0) - E(dk)) for a band curving
    downward from the valence edge (hole mass positive).
    """
    e_k, _ = solve_subbands(p, z, kx=dk, ky=0.0, potential=potential,
                            n_states=n_states)
    masses = []
    for i in range(n_states):
        curv = energies0[i] - e_k[i]
        if curv <= 1e-12:
            masses.append(np.inf)
        else:
            masses.append(HBAR2_OVER_2M0 * dk * dk / curv)
    return np.asarray(masses)


def _fill_subbands(energies, masses, ps):
    """T = 0 filling of hole subbands with parabolic in-plane dispersion.

    Holes fill from the highest valence energy downward. Each state is a
    single spin-resolved branch with 2D DOS m/(2 pi hbar^2) = m/(m0) /
    (4 pi (hbar^2/2m0)). Solves for E_F by bisection so occupations sum
    to ps. Returns per-subband sheet densities (nm^-2).
    """
    dos = (masses / HBAR2_OVER_2M0) / (4.0 * np.pi)   # states nm^-2 eV^-1
    finite = np.isfinite(dos)

    def total(ef):
        occ = np.where(finite, dos * np.clip(energies - ef, 0.0, None), 0.0)
        return occ.sum()

    lo, hi = energies.min() - 5.0, energies.max()
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if total(mid) > ps:
            lo = mid
        else:
            hi = mid
    ef = 0.5 * (lo + hi)
    occ = np.where(finite, dos * np.clip(energies - ef, 0.0, None), 0.0)
    if occ.sum() > 0:
        occ = occ * (ps / occ.sum())      # exact neutrality after bisection
    return occ


def solve_self_consistent(p, z, ps, n_states=6, mixing=0.3, tol=1e-5,
                          max_iter=80):
    """Iterate k.p and Poisson to self-consistency at sheet density ps.

    p : WurtziteParameters.  z : uniform grid from the interface (nm).
    ps : hole sheet density (nm^-2). Note 1e13 cm^-2 = 0.1 nm^-2.

    Returns SelfConsistentResult. Convergence is measured as the maximum
    absolute change of V_h between iterations (eV).
    """
    if not (p.eps_r == p.eps_r):   # NaN check without importing math
        raise ValueError(
            "this parameter set carries no vetted permittivity (eps_r is "
            "NaN; barrier-only set). Supply a cited eps_r before running "
            "a self-consistent Poisson solve."
        )
    z = np.asarray(z, dtype=float)
    vh = np.zeros_like(z)
    density = np.zeros_like(z)
    residual = np.inf
    for it in range(1, max_iter + 1):
        energies, envelopes = solve_subbands(p, z, potential=vh,
                                             n_states=n_states)
        masses = _inplane_masses(p, z, vh, energies, n_states)
        occ = _fill_subbands(energies, masses, ps)
        new_density = np.zeros_like(z)
        for i in range(n_states):
            prob = (np.abs(envelopes[i]) ** 2).sum(axis=0)
            new_density += occ[i] * prob
        density = (1 - mixing) * density + mixing * new_density
        new_vh = hole_potential(z, density, ps, p.eps_r)
        residual = float(np.max(np.abs(new_vh - vh)))
        vh = (1 - mixing) * vh + mixing * new_vh
        if residual < tol:
            break
    # final consistent (unmixed) state at the converged potential
    energies, envelopes = solve_subbands(p, z, potential=vh, n_states=n_states)
    masses = _inplane_masses(p, z, vh, energies, n_states)
    occ = _fill_subbands(energies, masses, ps)
    density = np.zeros_like(z)
    for i in range(n_states):
        density += occ[i] * (np.abs(envelopes[i]) ** 2).sum(axis=0)
    return SelfConsistentResult(z=z, potential=vh, energies=energies,
                                envelopes=envelopes, density=density,
                                occupations=occ, masses=masses,
                                iterations=it, residual=residual)


def fill_subbands_kgrid(solve_at_k, ps, n_states, kmax, nk=16, ntheta=4):
    """T = 0 subband filling from the full in-plane dispersion -- no
    parabolic assumption.

    solve_at_k : callable (kx, ky) -> energies (descending, eV), e.g. a
        closure over `solve_subbands` at the converged potential.
    ps : target sheet density (nm^-2); kmax, nk, ntheta: polar k-grid
        (trapezoid in k, uniform in theta over the full circle).

    Fills holes from the highest valence energy down: E_F solves
    ps = sum_n (1/(2 pi)^2) integral d^2k of Theta(E_n(k) > E_F), by
    bisection, then per-subband occupations are normalized to exact
    neutrality (as in the parabolic filler).  Refuses a kmax that the
    Fermi disc reaches (states outside the grid would be silently
    dropped otherwise).  On an exactly parabolic model this must agree
    with the closed-form parabolic filling -- asserted in the tests.

    Returns (occupations nm^-2 per subband, E_F eV).
    """
    kts = np.linspace(0.0, float(kmax), int(nk))
    thetas = np.arange(int(ntheta)) * 2.0 * np.pi / int(ntheta)
    E = np.empty((n_states, kts.size, thetas.size))
    for i, kt in enumerate(kts):
        for j, th in enumerate(thetas):
            e = solve_at_k(kt * np.cos(th), kt * np.sin(th))
            E[:, i, j] = e[:n_states]
    dtheta = 2.0 * np.pi / thetas.size

    integrate = getattr(np, "trapezoid", getattr(np, "trapz", None))

    def occ_per_band(ef):
        occ = np.empty(n_states)
        for n in range(n_states):
            f = (E[n] > ef).astype(float) * kts[:, None]
            occ[n] = integrate(f, kts, axis=0).sum() * dtheta \
                / (4.0 * np.pi ** 2)
        return occ

    lo, hi = E.min() - 5.0, E.max()
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if occ_per_band(mid).sum() > ps:
            lo = mid
        else:
            hi = mid
    ef = 0.5 * (lo + hi)
    if np.any(E[:, -1, :] > ef):
        raise ValueError(
            "the Fermi disc reaches the edge of the k-grid (a subband "
            "is still occupied at kmax); increase kmax")
    occ = occ_per_band(ef)
    if occ.sum() > 0:
        occ = occ * (ps / occ.sum())
    return occ, ef


def solve_self_consistent_hetero(z, params_list, band_edge, ps, eps_r,
                                 n_states=6, mixing=0.3, tol=1e-5,
                                 max_iter=80, strain_list=None):
    """Self-consistent k.p + Poisson loop on the *finite-barrier*
    heterostructure assembly -- the hard-wall restriction of
    `solve_self_consistent`, lifted.

    params_list / band_edge : per-point profiles as in
        `assemble_heterostructure` (e.g. from `layered_profile`).
    eps_r : permittivity used by the Poisson solve (one number: the
        Poisson solver is single-medium; a spatially varying
        permittivity is deliberately not smuggled in here -- supply the
        well material's cited value).
    strain_list : optional per-point strain, passed to the assembly.

    Anchors in the tests: a uniform stack with zero offset reproduces
    the hard-wall solver's self-consistent state (the assemblies are
    identical there, asserted at machine precision elsewhere), the
    deep-barrier limit approaches it, and occupations sum to exact
    neutrality.
    """
    from .heterostructure import solve_heterostructure
    if not (eps_r == eps_r):
        raise ValueError("eps_r is NaN; supply a cited permittivity")
    z = np.asarray(z, dtype=float)

    def solve(vh, kx=0.0, ky=0.0):
        return solve_heterostructure(z, params_list, band_edge, kx, ky,
                                     potential=vh, n_states=n_states,
                                     strain_list=strain_list)

    dk = 0.02
    vh = np.zeros_like(z)
    density = np.zeros_like(z)
    residual = np.inf
    for it in range(1, max_iter + 1):
        energies, envelopes = solve(vh)
        e_k, _ = solve(vh, kx=dk)
        curv = energies - e_k
        masses = np.where(curv > 1e-12,
                          HBAR2_OVER_2M0 * dk * dk / np.maximum(curv, 1e-30),
                          np.inf)
        occ = _fill_subbands(energies, masses, ps)
        new_density = np.zeros_like(z)
        for i in range(n_states):
            new_density += occ[i] * (np.abs(envelopes[i]) ** 2).sum(axis=0)
        density = (1 - mixing) * density + mixing * new_density
        new_vh = hole_potential(z, density, ps, eps_r)
        residual = float(np.max(np.abs(new_vh - vh)))
        vh = (1 - mixing) * vh + mixing * new_vh
        if residual < tol:
            break
    energies, envelopes = solve(vh)
    e_k, _ = solve(vh, kx=dk)
    curv = energies - e_k
    masses = np.where(curv > 1e-12,
                      HBAR2_OVER_2M0 * dk * dk / np.maximum(curv, 1e-30),
                      np.inf)
    occ = _fill_subbands(energies, masses, ps)
    density = np.zeros_like(z)
    for i in range(n_states):
        density += occ[i] * (np.abs(envelopes[i]) ** 2).sum(axis=0)
    return SelfConsistentResult(z=z, potential=vh, energies=energies,
                                envelopes=envelopes, density=density,
                                occupations=occ, masses=masses,
                                iterations=it, residual=residual)
