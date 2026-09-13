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
    S       = integral dz [ integral_{-inf}^{z} lambda(z') dz' ]^2,

with lambda(z) the spinor-summed overlap density of the two envelope
states and n_s the sheet density in the lower subband. S carries the
whole geometry; no effective mass enters this form. The final-state
(excitonic) correction beta is deliberately omitted, stated rather
than hidden: it requires an exchange-correlation model this package
does not ship.

Anchors asserted in the tests rather than stated: the geometry
integral S computed from solver envelopes matches an independent
adaptive quadrature over the analytic infinite-well wavefunctions;
S is exactly invariant under a shift of the coordinate origin
(orthogonality makes the inner integral vanish at both ends); S
scales exactly linearly with the well width; alpha is exactly linear
in the sheet density with the zero-density limit exact; and the
lineshape helper integrates to the oscillator-strength sum by
quadrature.
"""
from __future__ import annotations

import numpy as np

# e^2 / (4 pi eps0) = 1.4399645 eV nm (as in kpenvelope.poisson)
_E2_OVER_4PI_EPS0 = 1.4399645


__all__ = ["overlap_geometry_integral", "depolarization_shift",
           "isb_lineshape"]


def overlap_geometry_integral(z, env_i, env_j):
    """The depolarization geometry integral S (nm):
    S = int dz [ int_{-inf}^{z} lambda ]^2 with
    lambda(z) = sum_m Re[F_m^i(z)* F_m^j(z)] the spinor-summed
    overlap density on the solver's own grid."""
    z = np.asarray(z, dtype=float)
    a = np.asarray(env_i)
    b = np.asarray(env_j)
    if a.shape != b.shape or a.ndim != 2 or a.shape[1] != z.size:
        raise ValueError("envelopes must both be (n_components, N) on "
                         "the given grid")
    dz = z[1] - z[0]
    lam = np.real(np.conj(a) * b).sum(axis=0)
    inner = np.cumsum(lam) * dz
    return float((inner ** 2).sum() * dz)


def depolarization_shift(z, env_i, env_j, e_i, e_j, n_sheet, eps_r):
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
