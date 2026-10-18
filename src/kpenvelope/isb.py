"""Collective intersubband physics: what the spectrometer actually
sees (new in v0.9).

The measured intersubband resonance of a doped well does not sit at
the subband spacing: the oscillating charge screens itself, and the
resonance is pushed up by the depolarization shift -- the many-body
effect every intersubband absorption experiment must subtract before
comparing with a band-structure calculation (S. J. Allen, D. C. Tsui
and B. Vinter, Solid State Commun. 20, 425 (1976); T. Ando, A. B.
Fowler and F. Stern, Rev. Mod. Phys. 54, 437 (1982); M. Helm, in
Intersubband Transitions in Quantum Wells, 2000). In the standard
form the shifted energy is

    E_tilde = E_ij sqrt(1 + alpha),
    alpha   = 2 e^2 n_s S / (eps0 eps_r E_ij),
    S       = integral dz | integral_{-inf}^{z} lambda(z') dz' |^2,

with lambda(z) the spinor-summed overlap density of the two envelope
states and n_s the sheet density in the lower subband (for real
envelopes the modulus squared is the plain square of the textbook
form; the modulus, new in v0.13.0, makes S independent of each
eigenvector's arbitrary phase). S carries the
whole geometry; no effective mass enters this form. The final-state
(excitonic) correction beta is deliberately omitted, stated rather
than hidden: it requires an exchange-correlation model this package
does not ship.

Anchors asserted in the tests rather than stated: the geometry
integral S computed on the grid from the analytic infinite-well
wavefunctions matches an independent adaptive quadrature of the same
functions (to 1e-3 relative); S is invariant under a shift of the
coordinate origin (orthogonality makes the inner integral vanish at
both ends; asserted to 1e-12); S doubles when the well width doubles
(to 1e-3 relative); alpha is linear in the sheet density (to 1e-12
relative) with the zero-density limit exact; and the lineshape helper
integrates to the oscillator-strength sum by quadrature (to 1e-3).

Levels (new in v0.15.0): `level_transitions` gives |z|^2, oscillator
strengths and S between whole degenerate levels (summed over the final
level, averaged over the initial one), which do not depend on the basis
the solver picks inside a Kramers pair; checked against the closed-form
eigenvectors of the discrete demo well and for invariance under any
unitary mixing inside each level.
"""
from __future__ import annotations

import numpy as np

from .hamiltonian import HBAR2_OVER_2M0

# e^2 / (4 pi eps0) = 1.4399645 eV nm (as in kpenvelope.poisson)
_E2_OVER_4PI_EPS0 = 1.4399645


__all__ = ["overlap_geometry_integral", "depolarization_shift",
           "isb_lineshape", "degenerate_levels", "level_transitions"]


def overlap_geometry_integral(z, env_i, env_j) -> float:
    """The depolarization geometry integral S (nm):
    S = int dz | int_{-inf}^{z} lambda |^2 with
    lambda(z) = sum_m F_m^i(z)* F_m^j(z) the spinor-summed
    overlap (transition) density on the solver's own grid.

    For real envelopes this is the textbook
    S = int dz [ int_{-inf}^{z} psi_i psi_j ]^2, bit for bit. The
    modulus makes S independent of the arbitrary global phase of each
    eigenvector, as a measurable shift must be. Before v0.13.0 the real
    part Re[F^i* F^j] was used instead, so multiplying one state by a
    phase factor (an equally valid eigenvector) changed S: by i turned
    S, and with it the shift, into exactly 0.

    The standard form was written for real envelope functions, and it
    applies unchanged to a transition density with one phase along z,
    as for the solver's states at k_t = 0 (they are real). A density
    whose phase varies along z (states at finite in-plane momentum can
    have one) is outside what this form is checked for here; use
    k_t = 0 states.
    """
    z = np.asarray(z, dtype=float)
    a = np.asarray(env_i)
    b = np.asarray(env_j)
    if a.shape != b.shape or a.ndim != 2 or a.shape[1] != z.size:
        raise ValueError("envelopes must both be (n_components, N) on "
                         "the given grid")
    if z.size < 2:
        raise ValueError("z must have at least two grid points")
    dz = z[1] - z[0]
    lam = (np.conj(a) * b).sum(axis=0)
    if np.iscomplexobj(lam):
        inner = np.cumsum(lam) * dz
        return float((np.abs(inner) ** 2).sum() * dz)
    inner = np.cumsum(lam) * dz
    return float((inner ** 2).sum() * dz)


def depolarization_shift(z, env_i, env_j, e_i, e_j, n_sheet: float,
                         eps_r: float) -> dict:
    """Depolarization-shifted intersubband energy (eV).

    z : uniform grid (nm). env_i, env_j : (6, N) envelope spinors of
    the two subband states (any solver here). e_i, e_j : their
    energies (eV). n_sheet : sheet density in the lower subband
    (nm^-2). eps_r : cited static permittivity.

    Returns dict(E_bare, E_shifted, alpha, S): the bare spacing
    |e_i - e_j|, the collective resonance E_bare sqrt(1 + alpha),
    the dimensionless depolarization parameter and the geometry
    integral. The exciton correction is deliberately not included
    (module docstring).
    """
    ns = float(n_sheet)
    er = float(eps_r)
    if ns < 0.0 or not np.isfinite(ns):
        raise ValueError("n_sheet must be finite and >= 0")
    if not (er > 0.0 and np.isfinite(er)):
        raise ValueError("eps_r must be finite and positive (a cited "
                         "value, as everywhere in this package)")
    dE = abs(float(e_i) - float(e_j))
    if dE <= 0.0:
        raise ValueError("the two states are degenerate; no "
                         "intersubband resonance to shift")
    S = overlap_geometry_integral(z, env_i, env_j)
    # alpha = 2 e^2 n_s S / (eps0 eps_r dE); e^2/eps0 = 4 pi * 1.4399645 eV nm
    alpha = 2.0 * (4.0 * np.pi * _E2_OVER_4PI_EPS0) * ns * S / (er * dE)
    return dict(E_bare=dE, E_shifted=dE * np.sqrt(1.0 + alpha),
                alpha=float(alpha), S=S)


def isb_lineshape(transition_energies, strengths, gamma, e_grid):
    """Oscillator-strength-weighted Lorentzian absorption lineshape.

    A(E) = sum_i f_i * (gamma / (2 pi)) / ((E - E_i)^2 + (gamma/2)^2)

    with unit-area Lorentzians of FWHM `gamma` (eV, YOUR measured
    linewidth), so integral A dE = sum_i f_i within tail truncation
    -- asserted by quadrature in the tests. Absolute 2D absorbance
    requires geometry (polarization angle, passes, index) this
    package does not guess; this is the shape and weight, stated
    plainly.
    """
    E = np.asarray(e_grid, dtype=float)
    Ei = np.atleast_1d(np.asarray(transition_energies, dtype=float))
    fi = np.atleast_1d(np.asarray(strengths, dtype=float))
    g = float(gamma)
    if g <= 0.0:
        raise ValueError("gamma must be positive")
    if Ei.shape != fi.shape:
        raise ValueError("one strength per transition energy")
    half = 0.5 * g
    out = np.zeros_like(E)
    for e0, f0 in zip(Ei, fi):
        out += f0 * (half / np.pi) / ((E - e0) ** 2 + half ** 2)
    return out


# --------------------------------------------------------------------
# levels instead of single states (new in v0.15.0)
# --------------------------------------------------------------------

def degenerate_levels(energies, tol_ev: float = 1e-9) -> list:
    """Group states into levels: runs of descending energies in which
    each energy lies within ``tol_ev`` (eV) of the one before (new in
    v0.15.0). Returns a list of index arrays, highest level first. At
    k_t = 0 every level of the solvers is a Kramers pair (two states
    with exactly equal energies); the synthetic demo set has six-fold
    levels."""
    e = np.asarray(energies, dtype=float)
    if e.ndim != 1 or e.size < 1 or not np.all(np.isfinite(e)):
        raise ValueError("energies must be a 1D array of finite "
                         "energies (eV)")
    t = float(tol_ev)
    if not (np.isfinite(t) and t >= 0.0):
        raise ValueError("tol_ev must be finite and >= 0 (eV)")
    if np.any(np.diff(e) > 0.0):
        raise ValueError("energies must be sorted from the highest down, "
                         "as the solvers return them")
    out = [[0]]
    for i in range(1, e.size):
        if e[i - 1] - e[i] <= t:
            out[-1].append(i)
        else:
            out.append([i])
    return [np.asarray(c) for c in out]


def level_transitions(z, energies, envelopes, initial: int = 0,
                      mass_ratio=None, tol_ev: float = 1e-9) -> dict:
    """Intersubband quantities between whole levels, which do not
    depend on the basis the solver picks inside a degenerate level (new
    in v0.15.0).

    The two states of a Kramers pair (and the six states of a demo-set
    level) are any orthonormal pair of the level: which pair the solver
    returns is a convention, and a number computed from one member,
    such as |z_fi|^2 or the depolarization integral S, depends on it
    (for HH1 -> HH2 of a tilted 6 nm GaN well, one HH2 member gives
    S = 0.10378 nm and the other 0). Summed over the states of the
    final level and averaged over those of the initial level, as a
    transition rate between levels is, the numbers no longer depend on
    the convention: the sums below are unchanged by any unitary mixing
    of the states inside each level (asserted in the tests).

    z : uniform grid (nm). energies : (n,) descending (eV), as the
    solvers return them. envelopes : (n, 6, N) from the same solve.
    initial : index of the initial level (0 = the highest, where the
        holes sit). mass_ratio : optional m*/m0 for the oscillator
        strength (as in `oscillator_strengths`). tol_ev : the largest
        spacing within a level (eV).

    For every level L, with I the initial level and g_I its number of
    states:
      transition_energy[L] = E_I - E_L (eV; positive for the levels
        below I, which a hole in I absorbs into),
      z2[L] = (1/g_I) sum_{i in I} sum_{f in L} |<f|z|i>|^2 (nm^2),
      oscillator_strength[L] = (m*/m0) (E_I - E_L) z2[L] / (hbar^2/2m0)
        (None without mass_ratio), per hole in level I,
      S[L] = (1/g_I) sum_{i in I} sum_{f in L} S_fi (nm), with S_fi of
        `overlap_geometry_integral`.
    The entries of the initial level itself are 0 (no transition).

    For the solver's states at k_t = 0 without a strain that couples
    the spin groups, each member of a pair lies in one group of
    components; each member of the initial pair then couples to at
    most one member of the final pair, both members with the same S,
    and S[L] is that S (as for the two spin channels of a
    spin-degenerate subband in the textbook form). Pass complete levels: a level cut by
    ``n_states`` gives sums that again depend on the convention.
    Returns dict(levels, level_energies, degeneracy, initial,
    transition_energy, z2, oscillator_strength, S).
    """
    z = np.asarray(z, dtype=float)
    e = np.asarray(energies, dtype=float)
    F = np.asarray(envelopes)
    if F.ndim != 3 or F.shape[1] != 6 or F.shape[0] != e.size:
        raise ValueError("envelopes must have shape (n_states, 6, N), "
                         "one state per energy")
    if z.ndim != 1 or z.size != F.shape[2] or z.size < 2:
        raise ValueError("z must be the grid of the envelopes")
    levels = degenerate_levels(e, tol_ev)
    I = int(initial)
    if I != initial or not 0 <= I < len(levels):
        raise ValueError(f"initial must be a level index from 0 to "
                         f"{len(levels) - 1}; got {initial!r}")
    mr = None
    if mass_ratio is not None:
        mr = float(mass_ratio)
        if not (np.isfinite(mr) and mr > 0.0):
            raise ValueError("mass_ratio must be finite and > 0 (m*/m0)")
    dz = z[1] - z[0]
    e_lev = np.array([e[c].mean() for c in levels])
    g_i = levels[I].size
    z2 = np.zeros(len(levels))
    S = np.zeros(len(levels))
    for L, c in enumerate(levels):
        if L == I:
            continue
        for i in levels[I]:
            for f in c:
                zfi = (np.conj(F[f]) * z[None, :] * F[i]).sum() * dz
                z2[L] += abs(zfi) ** 2
                S[L] += overlap_geometry_integral(z, F[i], F[f])
    z2 /= g_i
    S /= g_i
    de = e_lev[I] - e_lev
    de[I] = 0.0
    osc = None if mr is None else mr * de * z2 / HBAR2_OVER_2M0
    return {"levels": levels, "level_energies": e_lev,
            "degeneracy": np.array([c.size for c in levels]),
            "initial": I, "transition_energy": de, "z2": z2,
            "oscillator_strength": osc, "S": S}
