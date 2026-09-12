"""Self-consistent solution of the coupled k.p and Poisson problem.

Subband filling supports the temperature every experiment actually has
(new in v0.8): for 2D parabolic subbands the Fermi-Dirac sheet density
has the closed form

    n_i = dos_i * kT * ln(1 + exp((E_i - E_F) / kT)),

which reduces exactly to the T = 0 expression dos_i * (E_i - E_F)_+ as
T -> 0. The closed form is anchored in the tests against direct
numerical integration of the Fermi-Dirac occupation over the constant
2D density of states -- two independent code paths -- and the T -> 0
limit against the zero-temperature filler. The Boltzmann constant in
eV/K is computed from the two exact SI defining constants (k_B =
1.380649e-23 J/K and e = 1.602176634e-19 C, both exact since the 2019
SI redefinition), not typed by hand.
"""
from __future__ import annotations

import dataclasses

import numpy as np

from .poisson import hole_potential
from .solver import solve_subbands
from .hamiltonian import HBAR2_OVER_2M0

# exact SI defining constants (2019 redefinition): both are exact
KB_EV_PER_K = 1.380649e-23 / 1.602176634e-19   # eV per kelvin


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
    converged: bool = True         # residual < tol at exit (new in v0.8)


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
    dos_f = np.where(finite, dos, 0.0)

    def total(ef):
        return (dos_f * np.clip(energies - ef, 0.0, None)).sum()

    lo, hi = energies.min() - 5.0, energies.max()
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if total(mid) > ps:
            lo = mid
        else:
            hi = mid
    ef = 0.5 * (lo + hi)
    occ = dos_f * np.clip(energies - ef, 0.0, None)
    if occ.sum() > 0:
        occ = occ * (ps / occ.sum())      # exact neutrality after bisection
    return occ


def fill_subbands_thermal(energies, masses, ps, temperature_K):
    """Fermi-Dirac filling of parabolic hole subbands at temperature T.

    For a 2D parabolic subband the sheet density is available in closed
    form: n_i = dos_i * kT * ln(1 + exp((E_i - E_F)/kT)) (holes fill
    downward from the highest valence energy; dos_i is the constant 2D
    density of states of one spin-resolved branch). Solves for E_F by
    bisection so occupations sum to ps, then normalizes to exact
    neutrality exactly as the T = 0 filler does. At temperature_K = 0
    this IS the T = 0 filler (same code path).

    energies : subband edges (eV, valence convention, descending).
    masses : in-plane edge masses (units of m0); non-finite masses are
        excluded, as in the T = 0 filler.
    ps : total sheet density (nm^-2).  temperature_K : kelvin, >= 0.

    Returns per-subband sheet densities (nm^-2).
    """
    T = float(temperature_K)
    if not (T >= 0.0) or not np.isfinite(T):
        raise ValueError("temperature_K must be finite and >= 0")
    energies = np.asarray(energies, dtype=float)
    masses = np.asarray(masses, dtype=float)
    if T == 0.0:
        return _fill_subbands(energies, masses, ps)
    kT = KB_EV_PER_K * T
    dos = (masses / HBAR2_OVER_2M0) / (4.0 * np.pi)
    finite = np.isfinite(dos)

    dos_f = np.where(finite, dos, 0.0)

    def occ_of(ef):
        # kT * ln(1 + exp((E - ef)/kT)), computed stably
        x = (energies - ef) / kT
        return dos_f * kT * np.logaddexp(0.0, x)

    lo = energies.min() - 5.0 - 40.0 * kT
    hi = energies.max() + 40.0 * kT
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if occ_of(mid).sum() > ps:
            lo = mid
        else:
            hi = mid
    occ = occ_of(0.5 * (lo + hi))
    if occ.sum() > 0:
        occ = occ * (ps / occ.sum())
    return occ


def solve_self_consistent(p, z, ps, n_states=6, mixing=0.3, tol=1e-5,
                          max_iter=80, temperature_K=0.0):
    """Iterate k.p and Poisson to self-consistency at sheet density ps.

    p : WurtziteParameters.  z : uniform grid from the interface (nm).
    ps : hole sheet density (nm^-2). Note 1e13 cm^-2 = 0.1 nm^-2 (see
    `kpenvelope.units` for the converters).
    temperature_K : lattice temperature for the subband filling (new in
    v0.8; 0 reproduces the historical T = 0 filling exactly).

    Returns SelfConsistentResult; its `converged` flag records whether
    the residual dropped below tol within max_iter iterations.
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
        occ = fill_subbands_thermal(energies, masses, ps, temperature_K)
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
    occ = fill_subbands_thermal(energies, masses, ps, temperature_K)
    density = np.zeros_like(z)
    for i in range(n_states):
        density += occ[i] * (np.abs(envelopes[i]) ** 2).sum(axis=0)
    return SelfConsistentResult(z=z, potential=vh, energies=energies,
                                envelopes=envelopes, density=density,
                                occupations=occ, masses=masses,
                                iterations=it, residual=residual,
                                converged=bool(residual < tol))


def fill_subbands_kgrid(solve_at_k, ps, n_states, kmax, nk=16, ntheta=4,
                        temperature_K=0.0):
    """Subband filling from the full in-plane dispersion -- no
    parabolic assumption -- at T = 0 or finite temperature.

    solve_at_k : callable (kx, ky) -> energies (descending, eV), e.g. a
        closure over `solve_subbands` at the converged potential.
    ps : target sheet density (nm^-2); kmax, nk, ntheta: polar k-grid
        (trapezoid in k, uniform in theta over the full circle).
    temperature_K : lattice temperature (new in v0.8); at 0 the
        occupation factor is the step Theta(E_n(k) > E_F), the
        historical behavior, and at T > 0 it is the Fermi-Dirac hole
        factor 1 / (1 + exp((E_F - E)/kT)).

    Fills holes from the highest valence energy down: E_F solves
    ps = sum_n (1/(2 pi)^2) integral d^2k f_h(E_n(k)), by bisection,
    then per-subband occupations are normalized to exact neutrality
    (as in the parabolic filler).  Refuses a kmax that the occupied
    region reaches (states outside the grid would be silently dropped
    otherwise): at T = 0 that is a subband still occupied at kmax, at
    T > 0 a hole occupation factor above 1e-4 on the outermost ring.
    On an exactly parabolic model this must agree with the closed-form
    parabolic filling at the same temperature -- asserted in the tests.

    Returns (occupations nm^-2 per subband, E_F eV).
    """
    T = float(temperature_K)
    if not (T >= 0.0) or not np.isfinite(T):
        raise ValueError("temperature_K must be finite and >= 0")
    kts = np.linspace(0.0, float(kmax), int(nk))
    thetas = np.arange(int(ntheta)) * 2.0 * np.pi / int(ntheta)
    E = np.empty((n_states, kts.size, thetas.size))
    for i, kt in enumerate(kts):
        for j, th in enumerate(thetas):
            e = solve_at_k(kt * np.cos(th), kt * np.sin(th))
            E[:, i, j] = e[:n_states]
    dtheta = 2.0 * np.pi / thetas.size

    integrate = getattr(np, "trapezoid", getattr(np, "trapz", None))

    def f_hole(ef):
        if T == 0.0:
            return (E > ef).astype(float)
        kT = KB_EV_PER_K * T
        # stable hole Fermi factor 1/(1 + exp((ef - E)/kT))
        x = np.clip((ef - E) / kT, -700.0, 700.0)
        return 1.0 / (1.0 + np.exp(x))

    def occ_per_band(ef):
        fh = f_hole(ef)
        occ = np.empty(n_states)
        for n in range(n_states):
            f = fh[n] * kts[:, None]
            occ[n] = integrate(f, kts, axis=0).sum() * dtheta \
                / (4.0 * np.pi ** 2)
        return occ

    pad = 40.0 * KB_EV_PER_K * T
    lo, hi = E.min() - 5.0 - pad, E.max() + pad
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if occ_per_band(mid).sum() > ps:
            lo = mid
        else:
            hi = mid
    ef = 0.5 * (lo + hi)
    if np.any(f_hole(ef)[:, -1, :] > 1e-4):
        raise ValueError(
            "the occupied region reaches the edge of the k-grid (a "
            "subband is still occupied at kmax); increase kmax")
    occ = occ_per_band(ef)
    if occ.sum() > 0:
        occ = occ * (ps / occ.sum())
    return occ, ef


def solve_self_consistent_hetero(z, params_list, band_edge, ps, eps_r,
                                 n_states=6, mixing=0.3, tol=1e-5,
                                 max_iter=80, strain_list=None,
                                 temperature_K=0.0):
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
    temperature_K : lattice temperature for the subband filling (new
        in v0.8; 0 reproduces the historical T = 0 filling exactly).

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
        occ = fill_subbands_thermal(energies, masses, ps, temperature_K)
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
    occ = fill_subbands_thermal(energies, masses, ps, temperature_K)
    density = np.zeros_like(z)
    for i in range(n_states):
        density += occ[i] * (np.abs(envelopes[i]) ** 2).sum(axis=0)
    return SelfConsistentResult(z=z, potential=vh, energies=energies,
                                envelopes=envelopes, density=density,
                                occupations=occ, masses=masses,
                                iterations=it, residual=residual,
                                converged=bool(residual < tol))
