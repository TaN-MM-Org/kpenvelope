"""Calibrate the numbers this package refuses to ship, from your own
measured spectra.

Two of the quantities every kpenvelope calculation needs are exactly
the ones no package should invent: the valence band offset of YOUR
heterostructure (alignments are material- and strain-specific, so
`layered_profile` takes them from you, with a citation), and the
sheet density actually sitting in YOUR well. Both are routinely
inferred from the same measurement -- the intersubband resonance --
and this module closes that loop with the package's own forward
models, pure NumPy, refusing rather than guessing when the
measurement cannot decide:

* `fit_band_offset`: the offset that makes the calculated subband
  spacing match your measured resonance, by bracketing bisection on
  the package's own `solve_heterostructure`, with an error bar from
  the exact sensitivity dE/d(offset) -- and a refusal when the
  transition barely feels the offset (a deep-well measurement cannot
  calibrate a barrier height it never probes).
* `offset_sensitivity`: that same sensitivity on its own, so the
  refusal can be anticipated before the measurement.
* `sheet_density_from_shift`: the depolarization relation
  E_meas = E_bare sqrt(1 + C n_s) is exactly linear in the sheet
  density, so the inversion is a closed form, not a fit -- with the
  exact error propagation, and a refusal when the measured resonance
  lies below the bare spacing (the depolarization shift only pushes
  up).

Shubnikov-de Haas (SdH) analysis (new in v0.13.0). Magnetotransport
gives two numbers per subband that the package also computes: the
density, from the oscillation frequency, and the mass, from how the
oscillation amplitude dies away with temperature.

* `sheet_density_from_sdh` and `sdh_frequency`: the Onsager relation
  F = hbar A / (2 pi e) (L. Onsager, Phil. Mag. 43, 1006 (1952))
  between the frequency F (in tesla, of oscillations periodic in 1/B)
  and the k-space area A of the Fermi contour. A spin-resolved branch
  holding n holes per unit area encloses A = 4 pi^2 n (two-dimensional
  states have density 1/(2 pi)^2 in k-space), whatever the shape of
  the contour, so F = (h/e) n exactly. `branches` says how many
  spin-resolved branches share the contour that oscillates: 2 for
  the usual spin-degenerate reading, 1 for one branch of a split pair
  (the package's per-state occupations are per branch).
* `fit_sdh_mass`: the cyclotron mass from the temperature damping of
  the oscillation amplitude at a fixed field, with the
  Lifshitz-Kosevich thermal factor R_T = X / sinh(X),
  X = 2 pi^2 k_B T / (hbar omega_c) = 2 pi^2 k_B T m* / (hbar e B)
  (I. M. Lifshitz and A. M. Kosevich, Sov. Phys. JETP 2, 636 (1956);
  D. Shoenberg, Magnetic Oscillations in Metals, Cambridge University
  Press (1984)). It is a weighted least-squares fit of
  A(T) = A0 R_T(T) with honest error bars, and it refuses data that do
  not determine the mass. The mass it measures is the cyclotron mass
  of the Fermi contour, m_c = (hbar^2 / 2 pi) dA/dE. For a dispersion
  that depends only on |k_t| this is hbar^2 k / |dE/dk| at k_F, which
  is exactly what `local_mass` returns between two momenta around k_F,
  so a measured mass can be set against the computed dispersion.

The constants are computed, not typed: h/e from the exact 2019 SI
values of h and e, and 2 pi^2 k_B m0 / (hbar e) (14.69 T/K) from the
exact k_B, h and e and the package's own hbar^2/2m0 (m0 = hbar^2 /
(2 * HBAR2_OVER_2M0)), so it is consistent with every mass the
package computes.

Units as everywhere in this package: energies in eV, lengths in nm,
sheet densities in nm^-2 (1e13 cm^-2 = 0.1 nm^-2); fields in tesla,
temperatures in kelvin, masses in units of m0.
"""
from __future__ import annotations

import math

import numpy as np

from .hamiltonian import HBAR2_OVER_2M0
from .heterostructure import solve_heterostructure
from .isb import depolarization_shift

__all__ = ["fit_band_offset", "offset_sensitivity",
           "sheet_density_from_shift", "sheet_density_from_sdh",
           "sdh_frequency", "fit_sdh_mass", "lk_thermal_factor",
           "H_OVER_E_T_NM2", "LK_TESLA_PER_KELVIN"]

# exact 2019 SI defining constants
_H = 6.62607015e-34          # J s
_E = 1.602176634e-19         # C
_KB = 1.380649e-23           # J/K

#: h/e in T nm^2 (exact): an SdH frequency F of one spin-resolved
#: branch corresponds to a sheet density n = F / H_OVER_E_T_NM2
H_OVER_E_T_NM2 = _H / _E * 1e18

#: 2 pi^2 k_B m0 / (hbar e) in tesla per kelvin, with m0 from the
#: package's own HBAR2_OVER_2M0 (= hbar^2 / 2 m0): X = this * (m*/m0)
#: * T / B in the Lifshitz-Kosevich factor
LK_TESLA_PER_KELVIN = (2.0 * math.pi ** 2 * _KB / (_H / (2.0 * math.pi) * _E)
                       * (_H / (2.0 * math.pi)) ** 2
                       / (2.0 * HBAR2_OVER_2M0 * _E * 1e-18))


def _spacing(z, params_list, band_edge_unit, offset, states, kwargs):
    energies, _ = solve_heterostructure(
        z, params_list, band_edge=offset * band_edge_unit,
        n_states=max(states) + 1, **kwargs)
    i, j = states
    return abs(float(energies[i]) - float(energies[j]))


def fit_band_offset(z, params_list, band_edge_unit, e_meas_ev,
                    bracket, states=(0, 2), sigma_e_ev=None, tol=1e-10,
                    **kwargs):
    """Calibrate the band-offset scale from a measured subband spacing.

    The offset profile is `offset * band_edge_unit`: you supply the
    SHAPE (which layers are barriers, in the package's valence-electron
    convention, where a hole barrier has a LOWER band edge -- e.g. -1.0
    on barrier points and 0.0 in the well, so that a positive offset
    scale confines the holes in the well) and the measured spacing;
    this returns the offset scale in eV.

    z : uniform grid (nm). params_list : per-point parameter sets, as
        `solve_heterostructure` takes them.
    band_edge_unit : (N,) the offset profile at unit scale.
    e_meas_ev : the measured (bare) subband spacing between `states`
        (eV) -- subtract the depolarization shift first for a doped
        well (`sheet_density_from_shift` works the other way around).
    bracket : (lo, hi) offset scales (eV) known to straddle the
        answer; the calculated spacing must differ from the measured
        one with opposite signs at the two ends, or the fit refuses.
    states : which two subbands (indices into the solver's
        ordering). At k = 0 the solver's states come in exactly
        degenerate Kramers pairs -- (0, 1) is one doublet -- so
        the first intersubband spacing is (0, 2), the default.
    sigma_e_ev : optional 1-sigma error of the measured spacing; the
        offset error bar follows from the exact sensitivity,
        sigma_offset = sigma_E / |dE/d(offset)|.

    Returns dict(offset_ev, sigma_offset_ev, e_model_ev,
    sensitivity, n_iter). Refuses a bracket that does not straddle
    the answer, and an offset the transition is insensitive to
    (|dE/d(offset)| so small the calibration would amplify the
    measurement error by more than 1e6).
    """
    unit = np.asarray(band_edge_unit, dtype=float)
    z = np.asarray(z, dtype=float)
    if unit.shape != z.shape:
        raise ValueError("band_edge_unit must live on the z grid")
    if not np.all(np.isfinite(unit)) or np.all(unit == 0.0):
        raise ValueError("band_edge_unit must be finite and not all "
                         "zero")
    e_meas = float(e_meas_ev)
    if not (np.isfinite(e_meas) and e_meas > 0.0):
        raise ValueError("e_meas_ev must be finite and positive")
    i, j = (int(states[0]), int(states[1]))
    if i == j or i < 0 or j < 0:
        raise ValueError("states must be two distinct subband indices")
    lo, hi = float(bracket[0]), float(bracket[1])
    if not (np.isfinite(lo) and np.isfinite(hi) and lo < hi):
        raise ValueError("bracket must be a finite (lo, hi) with "
                         "lo < hi")

    def g(offset):
        return _spacing(z, params_list, unit, offset, (i, j),
                        kwargs) - e_meas

    g_lo, g_hi = g(lo), g(hi)
    if g_lo == 0.0:
        root, n_iter = lo, 0
    elif g_hi == 0.0:
        root, n_iter = hi, 0
    elif g_lo * g_hi > 0.0:
        raise ValueError(
            "the bracket does not straddle the measured spacing: the "
            f"calculated spacing is {g_lo + e_meas:.6g} eV at offset "
            f"{lo:.6g} and {g_hi + e_meas:.6g} eV at offset {hi:.6g}, "
            f"both on the same side of the measured {e_meas:.6g} eV. "
            "Widen the bracket, or check the offset profile and the "
            "states")
    else:
        n_iter = 0
        for n_iter in range(1, 200):
            mid = 0.5 * (lo + hi)
            g_mid = g(mid)
            if g_mid == 0.0 or (hi - lo) <= tol * max(1.0, abs(mid)):
                lo = hi = mid
                break
            if g_lo * g_mid < 0.0:
                hi, g_hi = mid, g_mid
            else:
                lo, g_lo = mid, g_mid
        root = 0.5 * (lo + hi)

    sens = offset_sensitivity(z, params_list, unit, root, (i, j),
                              **kwargs)
    if abs(sens) * 1e6 < 1.0:
        raise ValueError(
            f"the {i}->{j} spacing changes by only {sens:.3g} eV per "
            "eV of offset at the solution: the measurement barely "
            "feels the offset, so the calibration would amplify the "
            "measurement error enormously. Use a transition that "
            "probes the barrier (a higher subband, or a thinner well)")
    sigma_off = None
    if sigma_e_ev is not None:
        se = float(sigma_e_ev)
        if not (np.isfinite(se) and se > 0.0):
            raise ValueError("sigma_e_ev must be finite and positive")
        sigma_off = se / abs(sens)
    return {"offset_ev": float(root),
            "sigma_offset_ev": sigma_off,
            "e_model_ev": float(g(root) + e_meas),
            "sensitivity": float(sens), "n_iter": n_iter}


def offset_sensitivity(z, params_list, band_edge_unit, offset,
                       states=(0, 2), rel_step=1e-4, **kwargs):
    """dE_spacing / d(offset scale): can this measurement calibrate
    the offset at all? Central difference through the package's own
    solver; ask before measuring."""
    unit = np.asarray(band_edge_unit, dtype=float)
    off = float(offset)
    h = rel_step * max(abs(off), 1e-3)
    ep = _spacing(z, params_list, unit, off + h, states, kwargs)
    em = _spacing(z, params_list, unit, off - h, states, kwargs)
    return float((ep - em) / (2.0 * h))


def sheet_density_from_shift(z, env_i, env_j, e_i, e_j, e_meas_ev,
                             eps_r, sigma_e_ev=None):
    """Sheet density from the measured depolarization-shifted
    resonance -- an exact closed-form inversion.

    The depolarization relation E_meas = E_bare sqrt(1 + C n_s) with
    C = 2 e^2 S / (eps0 eps_r E_bare) is exactly linear in n_s, so

        n_s = ((E_meas / E_bare)^2 - 1) / C,

    with everything (S, E_bare) computed from the package's own
    envelopes -- the exact inverse of `depolarization_shift`, checked
    as a round trip in the tests.

    z, env_i, env_j, e_i, e_j, eps_r : exactly as
        `depolarization_shift` takes them.
    e_meas_ev : the measured collective resonance (eV).
    sigma_e_ev : optional measurement error; the density error bar is
        the exact derivative dn/dE = 2 E_meas / (E_bare^2 C).

    Returns dict(n_sheet, sigma_n_sheet, e_bare, alpha, S). Refuses a
    measured resonance at or below the bare spacing: the
    depolarization shift only pushes the resonance up, so such a
    measurement contradicts the model instead of calibrating it.
    """
    e_meas = float(e_meas_ev)
    if not (np.isfinite(e_meas) and e_meas > 0.0):
        raise ValueError("e_meas_ev must be finite and positive")
    base = depolarization_shift(z, env_i, env_j, e_i, e_j,
                                n_sheet=0.0, eps_r=eps_r)
    e_bare = base["E_bare"]
    S = base["S"]
    if e_meas <= e_bare:
        raise ValueError(
            f"measured resonance {e_meas:.6g} eV is not above the "
            f"bare spacing {e_bare:.6g} eV: the depolarization shift "
            "only pushes the resonance up, so no nonnegative sheet "
            "density reproduces this measurement -- check the state "
            "assignment and the bare calculation")
    # C = alpha / n_s: exactly linear, so evaluate at n_s = 1
    c_lin = depolarization_shift(z, env_i, env_j, e_i, e_j,
                                 n_sheet=1.0, eps_r=eps_r)["alpha"]
    ns = ((e_meas / e_bare) ** 2 - 1.0) / c_lin
    sigma_ns = None
    if sigma_e_ev is not None:
        se = float(sigma_e_ev)
        if not (np.isfinite(se) and se > 0.0):
            raise ValueError("sigma_e_ev must be finite and positive")
        sigma_ns = se * 2.0 * e_meas / (e_bare ** 2 * c_lin)
    return {"n_sheet": float(ns), "sigma_n_sheet": sigma_ns,
            "e_bare": float(e_bare), "alpha": float(c_lin * ns),
            "S": float(S)}


# --------------------------------------------------------------------
# Shubnikov-de Haas analysis (new in v0.13.0)
# --------------------------------------------------------------------

def _check_branches(branches):
    try:
        b = int(branches) if not isinstance(branches, bool) else 0
    except (TypeError, ValueError):
        b = 0
    if b != branches or b < 1:
        raise ValueError(
            "branches must be a positive integer: the number of "
            "spin-resolved branches that share the oscillating Fermi "
            "contour (2 for a spin-degenerate subband, 1 for one branch "
            f"of a split pair); got {branches!r}")
    return b


def sdh_frequency(n_sheet, *, branches):
    """The SdH frequency (tesla) of a Fermi contour shared by
    ``branches`` spin-resolved branches holding ``n_sheet`` holes per
    nm^2 in total: F = (h/e) n_sheet / branches (Onsager relation).

    Feed it the package's per-state occupations (one spin-resolved
    branch each, ``branches=1``) to predict the frequencies a
    measurement should show. Scalar or array input.
    """
    b = _check_branches(branches)
    n = np.asarray(n_sheet, dtype=float)
    if not (np.all(np.isfinite(n)) and np.all(n >= 0.0)):
        raise ValueError("n_sheet must be finite and >= 0 (nm^-2; "
                         "1e13 cm^-2 = 0.1 nm^-2)")
    f = H_OVER_E_T_NM2 * n / b
    return float(f) if f.ndim == 0 else f


def sheet_density_from_sdh(frequency_T, *, branches,
                           sigma_frequency_T=None) -> dict:
    """Sheet density (nm^-2) from a measured SdH frequency (tesla),
    the exact inverse of `sdh_frequency`:
    n_sheet = branches * F / (h/e).

    branches : how many spin-resolved branches share the contour (2
        for the usual spin-degenerate reading of a single frequency, 1
        when the two spin branches show separate frequencies). There is
        no default on purpose: a wrong guess is a factor of two.
    sigma_frequency_T : optional 1-sigma error of F; the density error
        follows exactly (the relation is linear).

    Returns dict(n_sheet, sigma_n_sheet, n_sheet_cm2,
    sigma_n_sheet_cm2). The frequency of oscillations periodic in 1/B
    is what a Fourier transform of the resistance against 1/B gives.
    """
    b = _check_branches(branches)
    f = float(frequency_T)
    if not (np.isfinite(f) and f > 0.0):
        raise ValueError("frequency_T must be a finite positive SdH "
                         f"frequency in tesla; got {frequency_T!r}")
    n = b * f / H_OVER_E_T_NM2
    sigma = None
    if sigma_frequency_T is not None:
        sf = float(sigma_frequency_T)
        if not (np.isfinite(sf) and sf > 0.0):
            raise ValueError("sigma_frequency_T must be finite and "
                             "positive")
        sigma = b * sf / H_OVER_E_T_NM2
    return {"n_sheet": float(n), "sigma_n_sheet": sigma,
            "n_sheet_cm2": float(n * 1e14),
            "sigma_n_sheet_cm2": None if sigma is None else sigma * 1e14}


def lk_thermal_factor(temperature_K, field_T, mass_ratio):
    """The Lifshitz-Kosevich thermal damping factor R_T = X / sinh(X),
    X = 2 pi^2 k_B T m* / (hbar e B) = LK_TESLA_PER_KELVIN * m* T / B.
    Equal to 1 at T = 0; computed without overflow at large X.
    Broadcasts over array inputs."""
    T = np.asarray(temperature_K, dtype=float)
    B = np.asarray(field_T, dtype=float)
    m = np.asarray(mass_ratio, dtype=float)
    if not (np.all(np.isfinite(T)) and np.all(T >= 0.0)):
        raise ValueError("temperature_K must be finite and >= 0")
    if not (np.all(np.isfinite(B)) and np.all(B > 0.0)):
        raise ValueError("field_T must be finite and > 0 (tesla)")
    if not (np.all(np.isfinite(m)) and np.all(m > 0.0)):
        raise ValueError("mass_ratio must be finite and > 0 (m*/m0)")
    out = _lk_R(LK_TESLA_PER_KELVIN * m * T / B)
    return float(out) if out.ndim == 0 else out


def _lk_R(X):
    """X / sinh X for X >= 0, stable at both ends."""
    X = np.asarray(X, dtype=float)
    small = X < 1e-4
    Xs = np.where(small, 1.0, X)
    big = Xs * 2.0 * np.exp(-Xs) / -np.expm1(-2.0 * Xs)
    return np.where(small, 1.0 - X * X / 6.0, big)


def _lk_dlnR_dX(X):
    """d ln(X/sinh X)/dX = 1/X - coth X (-X/3 near 0)."""
    X = np.asarray(X, dtype=float)
    small = X < 1e-4
    Xs = np.where(small, 1.0, X)
    return np.where(small, -X / 3.0, 1.0 / Xs - 1.0 / np.tanh(Xs))


def fit_sdh_mass(temperatures_K, amplitudes, field_T,
                 sigma_amplitudes=None) -> dict:
    """Cyclotron mass m*/m0 from SdH amplitudes measured at several
    temperatures and one magnetic field (Lifshitz-Kosevich fit).

    Model: A(T) = A0 * R_T, R_T = X / sinh(X),
    X = LK_TESLA_PER_KELVIN * (m*/m0) * T / B. A0 absorbs everything
    that does not depend on temperature (the Dingle factor, the field
    dependence, the contact geometry), so this assumes that the
    scattering, and with it the Dingle factor, does not change over
    the temperature range, and that each amplitude is the size of one
    oscillation (one subband: for several frequencies, take each
    frequency's peak in the Fourier transform at the same field
    window).

    temperatures_K : (M,) temperatures, M >= 3, at least 3 distinct.
    amplitudes : (M,) positive amplitudes (any unit; only ratios
        matter).
    field_T : the magnetic field of the amplitudes (tesla). For
        amplitudes from a Fourier transform over a field window, the
        usual choice is 1/B_mean = mean of 1/B over the window.
    sigma_amplitudes : optional (M,) 1-sigma errors. With them, the
        fit is weighted, the error bars come from them alone, and
        ``chi2_red`` (chi-square per degree of freedom, near 1 when the
        model and the errors are right) tests the model. Without them,
        all points weigh the same and the error bars are scaled by the
        scatter of the residuals (as `scipy.optimize.curve_fit` does
        by default); ``chi2_red`` is then not a test and is None.

    Returns dict(mass_ratio, sigma_mass_ratio, A0, sigma_A0,
    correlation, chi2_red, dof, x_max). ``x_max`` is X at the highest
    temperature; the mass is well determined only when it reaches
    about 1 or more (the amplitude must fall clearly).

    Refuses: fewer than 3 points or 3 distinct temperatures,
    non-positive amplitudes, temperatures or field, amplitudes that do
    not fall with temperature (the best fit is at the edge of the
    searched masses, X_max from 1e-3 to 60), a mass whose 1-sigma
    error is as large as the mass itself, and a fit whose error bar is
    not honest: the error bar comes from the curvature of chi-square
    at the minimum, which is right only if chi-square (with A0
    refitted) rises by one unit (one times chi2_red without
    sigma_amplitudes) at m* -/+ sigma; if either rise is below 0.7 or
    above 1.3 the data are too weakly damped for a symmetric error bar
    (measure to higher temperature or at a lower field).
    """
    T = np.asarray(temperatures_K, dtype=float)
    y = np.asarray(amplitudes, dtype=float)
    if T.ndim != 1 or y.shape != T.shape:
        raise ValueError("temperatures_K and amplitudes must be 1D "
                         "arrays of the same length")
    if T.size < 3 or np.unique(T).size < 3:
        raise ValueError("need amplitudes at 3 or more distinct "
                         "temperatures to fit a mass and an amplitude "
                         "with an error bar")
    if not (np.all(np.isfinite(T)) and np.all(T > 0.0)):
        raise ValueError("temperatures_K must be finite and > 0 (K)")
    if not (np.all(np.isfinite(y)) and np.all(y > 0.0)):
        raise ValueError("amplitudes must be finite and > 0 (use the "
                         "size of the oscillation, not a signed value)")
    B = float(field_T)
    if not (np.isfinite(B) and B > 0.0):
        raise ValueError("field_T must be finite and > 0 (tesla)")
    absolute = sigma_amplitudes is not None
    if absolute:
        s = np.asarray(sigma_amplitudes, dtype=float)
        if s.shape != y.shape or not (np.all(np.isfinite(s))
                                      and np.all(s > 0.0)):
            raise ValueError("sigma_amplitudes must be finite, > 0 and "
                             "one per amplitude")
        w = 1.0 / s ** 2
    else:
        w = np.ones_like(y)

    a = LK_TESLA_PER_KELVIN * T / B          # X = a * m

    def profile(m):
        g = _lk_R(a * m)
        A0 = (w * y * g).sum() / (w * g * g).sum()
        return float((w * (y - A0 * g) ** 2).sum()), A0

    # search ln m over X_max from 1e-3 to 60, then refine the bracket
    lo = math.log(1e-3 / a.max())
    hi = math.log(60.0 / a.max())
    grid = np.linspace(lo, hi, 601)
    chi = np.array([profile(math.exp(t))[0] for t in grid])
    k = int(np.argmin(chi))
    if k == 0 or k == grid.size - 1:
        raise ValueError(
            "the amplitudes do not determine a mass: the best fit lies "
            "at the edge of the searched range (X at the highest "
            f"temperature {'1e-3' if k == 0 else '60'}). "
            + ("They do not fall with temperature as R_T = X/sinh X "
               "requires; check the amplitudes and that the field is "
               "the same for all of them" if k == 0 else
               "They fall faster than any mass in range explains; "
               "check the units (K, T)"))
    # golden-section search on [grid[k-1], grid[k+1]]
    ga, gb = grid[k - 1], grid[k + 1]
    r = (math.sqrt(5.0) - 1.0) / 2.0
    c1, c2 = gb - r * (gb - ga), ga + r * (gb - ga)
    f1, f2 = profile(math.exp(c1))[0], profile(math.exp(c2))[0]
    for _ in range(200):
        if gb - ga < 1e-13:
            break
        if f1 < f2:
            gb, c2, f2 = c2, c1, f1
            c1 = gb - r * (gb - ga)
            f1 = profile(math.exp(c1))[0]
        else:
            ga, c1, f1 = c1, c2, f2
            c2 = ga + r * (gb - ga)
            f2 = profile(math.exp(c2))[0]
    m = math.exp(0.5 * (ga + gb))
    chi2, A0 = profile(m)

    # Gauss-Newton covariance of (A0, m) from the exact Jacobian
    X = a * m
    g = _lk_R(X)
    J = np.column_stack([g, A0 * g * _lk_dlnR_dX(X) * a])
    cov = np.linalg.inv(J.T @ (w[:, None] * J))
    dof = T.size - 2
    if absolute:
        chi2_red = chi2 / dof
    else:
        chi2_red = None
        cov = cov * (chi2 / dof)
    sA0, sm = math.sqrt(cov[0, 0]), math.sqrt(cov[1, 1])
    advice = (f"the data do not determine the mass: best fit m* = "
              f"{m:.3g} m0 with a 1-sigma error of {sm:.3g} m0 (X at the "
              f"highest temperature {a.max() * m:.3g}). Measure to "
              "higher temperature or at a lower field, so that the "
              "amplitude falls clearly")
    if not sm < m:
        raise ValueError(advice)
    # the error bar is the curvature of chi^2 at the minimum; it is
    # honest only if chi^2 (with A0 refitted) really rises by one unit
    # (s2) at m -/+ sigma. Refuse when either side is off by > 30 %.
    # (skipped for data the model fits to rounding error, sigma ~ 0)
    s2 = 1.0 if absolute else chi2 / dof
    if absolute or chi2 > 1e-20 * float((w * y * y).sum()):
        rises = [(profile(m + d)[0] - chi2) / s2 for d in (-sm, sm)]
        if not all(0.7 < r < 1.3 for r in rises):
            raise ValueError(
                advice + ". (chi-square, with A0 refitted, rises by "
                f"{rises[0]:.2f} and {rises[1]:.2f} instead of 1 at "
                "m* -/+ sigma, so the error bar would not be honest)")
    corr = cov[0, 1] / (sA0 * sm) if sA0 > 0.0 and sm > 0.0 else 0.0
    return {"mass_ratio": float(m), "sigma_mass_ratio": float(sm),
            "A0": float(A0), "sigma_A0": float(sA0),
            "correlation": float(corr),
            "chi2_red": None if chi2_red is None else float(chi2_red),
            "dof": int(dof), "x_max": float(a.max() * m)}
