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

Momentum-grid filling (v0.12.0). `fill_subbands_kgrid` and the
``filling="kgrid"`` option of both self-consistent loops integrate the
occupation over the computed in-plane dispersion. Along each ray the
energy is interpolated linearly in k_t^2 between grid points and the
Fermi-Dirac factor is then integrated over each segment in closed form
(its antiderivative is kT ln(1 + exp((E - E_F)/kT)), or (E - E_F)_+ at
T = 0). A parabolic subband is exactly linear in k_t^2, so for it the
result is exact at any grid spacing; for a general dispersion the error
is second order in the grid step. Before v0.12.0 a trapezoid rule over
the step-like occupation was used, which is only first order at T = 0.
"""
from __future__ import annotations

import dataclasses

import numpy as np

from .poisson import hole_potential, _check_eps
from .solver import solve_subbands
from .hamiltonian import HBAR2_OVER_2M0

# exact SI defining constants (2019 redefinition): both are exact
KB_EV_PER_K = 1.380649e-23 / 1.602176634e-19   # eV per kelvin

# momentum step of the parabolic-filling edge masses (nm^-1)
_DK = 0.02


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
    filling: str = "parabolic"     # "parabolic" or "kgrid" (new in v0.12)
    fermi_level: float = float("nan")  # eV; kgrid filling only (v0.12)


def _masses_from_step(energies0, e_k, dk=_DK):
    """Edge masses m*/m0 = (hbar^2/2m0) dk^2 / (E(0) - E(dk)); inf for a
    branch that does not fall over the step."""
    curv = np.asarray(energies0) - np.asarray(e_k)
    return np.where(curv > 1e-12,
                    HBAR2_OVER_2M0 * dk * dk / np.maximum(curv, 1e-30),
                    np.inf)


def _inplane_masses(p, z, potential, energies0, n_states, dk=_DK):
    """Numeric in-plane effective masses at the subband edge along kx.

    m*/m0 = (hbar^2/2m0) * dk^2 / (E(0) - E(dk)) for a band curving
    downward from the valence edge (hole mass positive).
    """
    e_k, _ = solve_subbands(p, z, kx=dk, ky=0.0, potential=potential,
                            n_states=n_states)
    return _masses_from_step(energies0[:n_states], e_k[:n_states], dk)


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


# --------------------------------------------------------------------
# momentum-grid filling
# --------------------------------------------------------------------

def _check_temperature(temperature_K):
    T = float(temperature_K)
    if not (T >= 0.0) or not np.isfinite(T):
        raise ValueError("temperature_K must be finite and >= 0")
    return T


def _hole_factor(E, ef, T):
    """Hole occupation 1 / (1 + exp((ef - E)/kT)); the step E > ef at
    T = 0."""
    if T == 0.0:
        return (E > ef).astype(float)
    kT = KB_EV_PER_K * T
    x = np.clip((ef - E) / kT, -700.0, 700.0)
    return 1.0 / (1.0 + np.exp(x))


def _hole_antiderivative(E, ef, T):
    """G(E) with dG/dE = hole factor: kT ln(1 + exp((E - ef)/kT)), or
    (E - ef)_+ at T = 0."""
    if T == 0.0:
        return np.maximum(E - ef, 0.0)
    kT = KB_EV_PER_K * T
    return kT * np.logaddexp(0.0, (E - ef) / kT)


def _segment_integrals(E, kts, ef, T):
    """Occupied area of each (branch, segment, direction), in nm^-2.

    E : (n, nk, ntheta) energies on the polar grid. The energy is taken
    linear in q = k_t^2 on each segment, so the segment integral
    (1/(4 pi ntheta)) int f(E(q)) dq equals
    (1/(4 pi ntheta)) dq (G(E_b) - G(E_a)) / (E_b - E_a),
    exact for a parabolic branch. A segment whose end energies differ
    by less than 1e-7 eV uses the factor at its midpoint instead (the
    divided difference would lose digits there).
    Returns (n, nk - 1, ntheta).
    """
    q = kts ** 2
    dq = np.diff(q)[None, :, None]
    Ea, Eb = E[:, :-1, :], E[:, 1:, :]
    dE = Eb - Ea
    flat = np.abs(dE) <= 1e-7
    safe = np.where(flat, 1.0, dE)
    mean_f = np.where(
        flat, _hole_factor(0.5 * (Ea + Eb), ef, T),
        (_hole_antiderivative(Eb, ef, T)
         - _hole_antiderivative(Ea, ef, T)) / safe)
    return dq * mean_f / (4.0 * np.pi * E.shape[2])


def _kgrid_fill(E, kts, ps, T):
    """Fermi level and per-node weights for the polar momentum grid.

    Returns (weights (n, nk, ntheta) in nm^-2 summing to ps, ef).
    Each segment's occupied area is split equally between its two end
    nodes, so a node weight times the state's |F(z)|^2 gives its share
    of the hole density. Refuses a grid whose outer ring is still
    occupied (occupation factor above 1e-4).
    """
    pad = 40.0 * KB_EV_PER_K * T
    lo, hi = E.min() - 5.0 - pad, E.max() + pad
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _segment_integrals(E, kts, mid, T).sum() > ps:
            lo = mid
        else:
            hi = mid
    ef = 0.5 * (lo + hi)
    if np.any(_hole_factor(E[:, -1, :], ef, T) > 1e-4):
        raise ValueError(
            "the occupied region reaches the edge of the k-grid (a "
            "subband is still occupied at kmax); increase kmax")
    seg = _segment_integrals(E, kts, ef, T)
    w = np.zeros_like(E)
    w[:, :-1, :] += 0.5 * seg
    w[:, 1:, :] += 0.5 * seg
    total = w.sum()
    if total > 0:
        w = w * (ps / total)
    return w, ef


def _polar_grid(kmax, nk, ntheta):
    kmax = float(kmax)
    if not (np.isfinite(kmax) and kmax > 0.0):
        raise ValueError("kmax must be finite and positive (nm^-1)")
    if int(nk) < 2 or int(ntheta) < 1:
        raise ValueError("need nk >= 2 and ntheta >= 1")
    kts = np.linspace(0.0, kmax, int(nk))
    thetas = np.arange(int(ntheta)) * 2.0 * np.pi / int(ntheta)
    return kts, thetas


def fill_subbands_kgrid(solve_at_k, ps, n_states, kmax, nk=16, ntheta=4,
                        temperature_K=0.0):
    """Subband filling from the full in-plane dispersion -- no
    parabolic assumption -- at T = 0 or finite temperature.

    solve_at_k : callable (kx, ky) -> energies (descending, eV), e.g. a
        closure over `solve_subbands` at the converged potential.
    ps : target sheet density (nm^-2); kmax, nk, ntheta: polar k-grid
        (nk points uniform in k_t from 0 to kmax, ntheta directions
        uniform over the full circle).
    temperature_K : lattice temperature (new in v0.8); at 0 the
        occupation factor is the step Theta(E_n(k) > E_F), and at T > 0
        the Fermi-Dirac hole factor 1 / (1 + exp((E_F - E)/kT)).

    Fills holes from the highest valence energy down: E_F solves
    ps = sum_n (1/(2 pi)^2) integral d^2k f_h(E_n(k)), by bisection,
    then per-subband occupations are normalized to exact neutrality
    (as in the parabolic filler). Since v0.12.0 the k_t integral treats
    each branch as linear in k_t^2 between grid points and integrates
    the occupation factor exactly on each segment, so a parabolic
    branch is integrated exactly at any nk (asserted against the
    closed-form filler to 1e-9 nm^-2 in the tests). Branches are
    labelled by energy order at each k.

    The six-band subbands of a structure without in-plane anisotropic
    strain depend only on |k_t| (asserted in the tests), so ntheta=1 is
    then exact and four times cheaper than the default 4.

    Refuses a kmax that the occupied region reaches (states outside
    the grid would be silently dropped otherwise): at T = 0 that is a
    subband still occupied at kmax, at T > 0 a hole occupation factor
    above 1e-4 on the outermost ring.

    Returns (occupations nm^-2 per subband, E_F eV).
    """
    T = _check_temperature(temperature_K)
    kts, thetas = _polar_grid(kmax, nk, ntheta)
    E = np.empty((n_states, kts.size, thetas.size))
    for i, kt in enumerate(kts):
        for j, th in enumerate(thetas):
            e = solve_at_k(kt * np.cos(th), kt * np.sin(th))
            E[:, i, j] = e[:n_states]
    w, ef = _kgrid_fill(E, kts, ps, T)
    return w.sum(axis=(1, 2)), ef


# --------------------------------------------------------------------
# self-consistent loops
# --------------------------------------------------------------------

def _strain_is_inplane_isotropic(strain):
    """True when no strain is given or every tensor has eps_xx = eps_yy
    and no shear: then the subbands depend only on |k_t|."""
    if strain is None:
        return True
    s = np.asarray(strain, dtype=float)
    s = s.reshape(-1, 3, 3)
    return bool(np.all(s[:, 0, 0] == s[:, 1, 1])
                and np.all(s[:, 0, 1] == 0.0) and np.all(s[:, 0, 2] == 0.0)
                and np.all(s[:, 1, 2] == 0.0) and np.all(s[:, 1, 0] == 0.0)
                and np.all(s[:, 2, 0] == 0.0) and np.all(s[:, 2, 1] == 0.0))


def _check_filling(filling, kmax, nk, ntheta, strain):
    if filling not in ("parabolic", "kgrid"):
        raise ValueError('filling must be "parabolic" or "kgrid"')
    if filling == "parabolic":
        return None
    if kmax is None:
        raise ValueError(
            'filling="kgrid" needs kmax (nm^-1): the largest in-plane '
            "momentum of the grid. It must exceed the Fermi wavevector "
            "of every occupied subband; too small a value is refused")
    if ntheta is None:
        if not _strain_is_inplane_isotropic(strain):
            raise ValueError(
                "the strain breaks in-plane isotropy, so one momentum "
                "direction is not enough: pass ntheta (e.g. 8)")
        ntheta = 1
    return _polar_grid(kmax, nk, ntheta)


def _parabolic_state(solve, z, vh, ps, n_states, T):
    """Edge energies, envelopes, step masses, parabolic occupations and
    the density they make (the historical filling model)."""
    energies, envelopes = solve(vh, 0.0, 0.0)
    e_k, _ = solve(vh, _DK, 0.0)
    masses = _masses_from_step(energies, e_k)
    occ = fill_subbands_thermal(energies, masses, ps, T)
    density = np.zeros_like(z)
    for i in range(n_states):
        density += occ[i] * (np.abs(envelopes[i]) ** 2).sum(axis=0)
    return energies, envelopes, masses, occ, density, float("nan")


def _kgrid_state(solve, z, vh, ps, n_states, T, grid):
    """As `_parabolic_state`, but occupations and density come from the
    states on the whole polar momentum grid: each state's |F(z)|^2 is
    weighted by its share of the occupied momentum area."""
    kts, thetas = grid
    energies, envelopes = solve(vh, 0.0, 0.0)
    E = np.empty((n_states, kts.size, thetas.size))
    P = np.empty((n_states, kts.size, thetas.size, z.size))
    prob0 = (np.abs(envelopes) ** 2).sum(axis=1)
    for i, kt in enumerate(kts):
        for j, th in enumerate(thetas):
            if kt == 0.0:
                e, prob = energies, prob0
            else:
                e, F = solve(vh, kt * np.cos(th), kt * np.sin(th))
                prob = (np.abs(F) ** 2).sum(axis=1)
            E[:, i, j] = e
            P[:, i, j, :] = prob
    w, ef = _kgrid_fill(E, kts, ps, T)
    occ = w.sum(axis=(1, 2))
    density = np.einsum("nij,nijz->z", w, P)
    e_k, _ = solve(vh, _DK, 0.0)
    masses = _masses_from_step(energies, e_k)
    return energies, envelopes, masses, occ, density, float(ef)


def _loop(solve, z, ps, eps_r, n_states, mixing, tol, max_iter, T,
          filling, grid, sheet_z):
    if grid is None:
        def state(vh):
            return _parabolic_state(solve, z, vh, ps, n_states, T)
    else:
        def state(vh):
            return _kgrid_state(solve, z, vh, ps, n_states, T, grid)
    vh = np.zeros_like(z)
    density = np.zeros_like(z)
    residual = np.inf
    for it in range(1, max_iter + 1):
        new_density = state(vh)[4]
        density = (1 - mixing) * density + mixing * new_density
        new_vh = hole_potential(z, density, ps, eps_r, sheet_z=sheet_z)
        residual = float(np.max(np.abs(new_vh - vh)))
        vh = (1 - mixing) * vh + mixing * new_vh
        if residual < tol:
            break
    # final consistent (unmixed) state at the converged potential
    energies, envelopes, masses, occ, density, ef = state(vh)
    return SelfConsistentResult(z=z, potential=vh, energies=energies,
                                envelopes=envelopes, density=density,
                                occupations=occ, masses=masses,
                                iterations=it, residual=residual,
                                converged=bool(residual < tol),
                                filling=filling, fermi_level=ef)


def solve_self_consistent(p, z, ps, n_states=6, mixing=0.3, tol=1e-5,
                          max_iter=80, temperature_K=0.0,
                          filling="parabolic", kmax=None, nk=24,
                          ntheta=None):
    """Iterate k.p and Poisson to self-consistency at sheet density ps.

    p : WurtziteParameters.  z : uniform grid from the interface (nm).
    ps : hole sheet density (nm^-2). Note 1e13 cm^-2 = 0.1 nm^-2 (see
    `kpenvelope.units` for the converters).
    temperature_K : lattice temperature for the subband filling (new in
    v0.8; 0 reproduces the historical T = 0 filling exactly).
    filling : "parabolic" (default, the historical model: each state a
        parabola with the mass from one 0.02 nm^-1 step) or "kgrid"
        (new in v0.12.0): occupations AND the hole density come from
        the states on a polar momentum grid (``kmax`` in nm^-1, ``nk``
        points; see `fill_subbands_kgrid`), so the non-parabolic
        dispersion and the change of the envelopes with momentum both
        enter. ``ntheta`` defaults to 1, which is exact here because
        the unstrained subbands depend only on |k_t|. The result then
        also carries ``fermi_level``.

    Returns SelfConsistentResult; its `converged` flag records whether
    the residual dropped below tol within max_iter iterations. The
    `masses` field is always the one-step edge mass, whatever the
    filling.
    """
    if int(max_iter) < 1:
        raise ValueError("max_iter must be at least 1")
    if not (p.eps_r == p.eps_r):   # NaN check without importing math
        raise ValueError(
            "this parameter set carries no vetted permittivity (eps_r is "
            "NaN; barrier-only set). Supply a cited eps_r before running "
            "a self-consistent Poisson solve."
        )
    T = _check_temperature(temperature_K)
    grid = _check_filling(filling, kmax, nk, ntheta, None)
    z = np.asarray(z, dtype=float)

    def solve(vh, kx, ky):
        return solve_subbands(p, z, kx=kx, ky=ky, potential=vh,
                              n_states=n_states)

    return _loop(solve, z, ps, p.eps_r, n_states, mixing, tol,
                 int(max_iter), T, filling, grid, None)


def solve_self_consistent_hetero(z, params_list, band_edge, ps, eps_r,
                                 n_states=6, mixing=0.3, tol=1e-5,
                                 max_iter=80, strain_list=None,
                                 temperature_K=0.0, filling="parabolic",
                                 kmax=None, nk=24, ntheta=None,
                                 sheet_z=None):
    """Self-consistent k.p + Poisson loop on the *finite-barrier*
    heterostructure assembly -- the hard-wall restriction of
    `solve_self_consistent`, lifted.

    params_list / band_edge : per-point profiles as in
        `assemble_heterostructure` (e.g. from `layered_profile`).
    eps_r : permittivity used by the Poisson solve: one cited number,
        or (new in v0.12.0) one value per grid point for a stack whose
        layers differ. NaN anywhere is refused.
    strain_list : optional per-point strain, passed to the assembly.
    temperature_K : lattice temperature for the subband filling (new
        in v0.8; 0 reproduces the historical T = 0 filling exactly).
    filling, kmax, nk, ntheta : as in `solve_self_consistent`. With a
        strain that breaks in-plane isotropy (eps_xx != eps_yy or any
        shear) ``ntheta`` must be given for ``filling="kgrid"``.
    sheet_z : position (nm) of the fixed negative sheet charge that
        binds the gas, normally the interface the gas sits against
        (new in v0.12.0). None (default) keeps the historical choice,
        the first grid point, which is only right when the grid starts
        at that interface: if the grid starts inside a barrier, the
        default puts the full sheet field across that barrier.

    Anchors in the tests: a uniform stack with zero offset reproduces
    the hard-wall solver's self-consistent state (the assemblies are
    identical there, asserted at machine precision elsewhere), the
    deep-barrier limit approaches it, and occupations sum to exact
    neutrality.
    """
    from .heterostructure import solve_heterostructure
    if int(max_iter) < 1:
        raise ValueError("max_iter must be at least 1")
    z = np.asarray(z, dtype=float)
    if np.ndim(eps_r) == 0 and not (eps_r == eps_r):
        raise ValueError("eps_r is NaN; supply a cited permittivity")
    _check_eps(eps_r, z.size)
    T = _check_temperature(temperature_K)
    grid = _check_filling(filling, kmax, nk, ntheta, strain_list)

    def solve(vh, kx, ky):
        return solve_heterostructure(z, params_list, band_edge, kx, ky,
                                     potential=vh, n_states=n_states,
                                     strain_list=strain_list)

    return _loop(solve, z, ps, eps_r, n_states, mixing, tol,
                 int(max_iter), T, filling, grid, sheet_z)
