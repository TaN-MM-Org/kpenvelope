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

Units as everywhere in this package: energies in eV, lengths in nm,
sheet densities in nm^-2 (1e13 cm^-2 = 0.1 nm^-2).
"""
from __future__ import annotations

import numpy as np

from .heterostructure import solve_heterostructure
from .isb import depolarization_shift

__all__ = ["fit_band_offset", "offset_sensitivity",
           "sheet_density_from_shift"]


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
