"""Fit the unknowns of one sample to its measured C-V curve and Hall
density, with error bars and a warning where the data cannot tell two
unknowns apart (new in v0.18.0).

A measured capacitance-voltage curve and a Hall density are the two
measurements almost every nitride sample gets. Together they fix the
numbers that are hardest to know beforehand: the barrier of the metal,
the barrier of the free surface, the real composition or thickness of a
layer, an extra fixed charge at an interface, the misfit-line charge, and
the parasitic capacitance of the measurement. `fit_measurements` adjusts
the ones you name until the computed curve and density match the
measured ones (weighted least squares, Levenberg-Marquardt), and reports
each with its one-sigma error and the correlation between them.

The model of the sample is yours: `build(values)` receives the current
values of the unknowns as a dict and returns the layers (and, if needed,
the line charge and extra sheet charges) for those values, so any layer,
set or composition can be fitted. The computation behind each predicted
number is the package's own:

* C-V: the quasi-static capacitance of `cv_curve` (classical electrons,
  "quasi-equilibrium") or of `cv_curve_quantum` ("quantum"), at the
  measured voltages, plus a parasitic capacitance if you fit one;
* Hall density: the electron sheet of the stack with the free-surface
  barrier, from `solve_quasi_equilibrium` (classical) or
  `solve_electrons` (quantum), over the region you name.

Reading the result: the error bars are first order (`fit_least_squares`).
A correlation close to +-1 between two unknowns means the data constrain
only a combination of them; the fit then says so in `warnings`, and the
cure is to fix one of them or to add a measurement that separates them
(the Hall density often does). A value that ends on a physical bound (an
Al fraction of 0, say) is flagged too.
"""
from __future__ import annotations

import dataclasses
import math

import numpy as np

from .driftdiffusion import (Contact, cv_curve, cv_curve_quantum,
                             solve_quasi_equilibrium)
from .fitting import FitResult, fit_least_squares

__all__ = ["fit_measurements", "MeasurementFit", "correlation_warnings"]


@dataclasses.dataclass
class MeasurementFit:
    """Result of `fit_measurements`.

    values, errors : {name: fitted value}, {name: one-sigma error}.
    correlation : (p, p) correlation matrix, in the order of `names`.
    names : the fitted unknowns.
    chi2, dof, scaled : as in `FitResult` (scaled: the error scale came
        from the scatter of the data about the fit).
    cv_voltage, cv_measured, cv_fitted : the C-V data and the fitted curve
        (F/cm^2), or None.
    sheet_measured, sheet_fitted : the Hall density and its fit (cm^-2), or
        None.
    warnings : plain sentences (correlated unknowns, values on a bound).
    fit : the underlying `FitResult`.
    """

    values: dict
    errors: dict
    correlation: np.ndarray
    names: tuple
    chi2: float
    dof: int
    scaled: bool
    cv_voltage: np.ndarray
    cv_measured: np.ndarray
    cv_fitted: np.ndarray
    sheet_measured: float
    sheet_fitted: float
    warnings: list
    fit: FitResult


def correlation_warnings(names, corr, limit=0.95):
    """Sentences for every pair of unknowns whose correlation exceeds
    `limit` in size."""
    out = []
    p = len(names)
    for i in range(p):
        for j in range(i + 1, p):
            r = float(corr[i, j])
            if abs(r) > limit:
                out.append(f"{names[i]} and {names[j]} are strongly "
                           f"{'correlated' if r > 0 else 'anti-correlated'} "
                           f"({r:+.3f}): the data mainly fix a combination "
                           "of the two, so their errors move together, and "
                           "fixing one (or adding a measurement that "
                           "separates them) would shrink the other's error.")
    return out


def _sheet(res, window):
    z, n = res.z, res.n
    lo, hi = window
    m = (z >= lo - 1e-9) & (z <= hi + 1e-9)
    zz, nn = z[m], n[m]
    return float(np.sum(0.5 * (nn[1:] + nn[:-1]) * np.diff(zz)))


def fit_measurements(build, start: dict, cv=None, sheet=None, fixed=None,
                     bounds=None, method: str = "quasi-equilibrium",
                     temperature_K: float = 300.0, dz: float = 0.5,
                     dv: float = 0.01, rel_step: float = 1e-3,
                     difference: str = "forward", max_iter: int = 30,
                     corr_limit: float = 0.95, lengths=()) -> MeasurementFit:
    """Fit the unknowns of one sample to its measured C-V curve and/or
    Hall density.

    build : callable(values dict) -> {"layers": [...], "line_charge": ...,
        "extra_sheet": {...}} (only "layers" is required). `values` holds
        every unknown and every fixed number by name.
    start : {name: starting value} of the unknowns to fit. Two names are
        used by the fitter itself and need not appear in `build`:
        "barrier" (metal barrier of the C-V gate, eV) and "surface"
        (free-surface barrier of the Hall sample, eV), and "parasitic"
        (capacitance added to the computed one, nF/cm^2).
    cv : (voltages V, capacitance F/cm^2, sigma F/cm^2 or None).
    sheet : (Hall density cm^-2, sigma cm^-2 or None, (z_min, z_max) nm or
        None for the whole stack).
    fixed : {name: value} of numbers held fixed (for example "barrier"
        when only the Hall density is fitted, or "surface").
    bounds : {name: (low, high)}; values are clipped into them, and a fit
        that ends on one is flagged. Fractions are kept in [0, 1] by
        `build` itself (the material builders refuse others).
    method : "quasi-equilibrium" (classical electrons, fast) or "quantum".
    difference, rel_step : of the finite-difference Jacobian.
    lengths : names of unknowns that are lengths in nm (layer thicknesses).
        On a grid a thickness can only change by whole grid steps, so for
        these the prediction is interpolated linearly between the two
        thicknesses on the grid around the value (at most two such
        unknowns).
    Reading errors: give sigma for both data sets (or neither, then the
    scale comes from the scatter, which needs more points than unknowns).
    """
    if cv is None and sheet is None:
        raise ValueError("give a C-V curve, a Hall density, or both")
    if method not in ("quasi-equilibrium", "quantum"):
        raise ValueError("method must be 'quasi-equilibrium' or 'quantum'")
    if not start:
        raise ValueError("name at least one unknown to fit in `start`")
    fixed = dict(fixed or {})
    bounds = dict(bounds or {})
    names = tuple(start)
    both = set(names) & set(fixed)
    if both:
        raise ValueError(f"{sorted(both)} are both fitted and fixed")
    V = Cm = Cs = None
    if cv is not None:
        V = np.asarray(cv[0], float)
        Cm = np.asarray(cv[1], float)
        Cs = None if len(cv) < 3 or cv[2] is None else np.broadcast_to(
            np.asarray(cv[2], float), V.shape).astype(float)
        if V.shape != Cm.shape or V.ndim != 1 or V.size < 1:
            raise ValueError("cv needs equal-length voltage and capacitance arrays")
        if not (np.all(np.isfinite(V)) and np.all(np.isfinite(Cm))):
            raise ValueError("the C-V data must be finite")
        if "barrier" not in names and "barrier" not in fixed:
            raise ValueError("a C-V fit needs the metal 'barrier', fitted "
                             "(in start) or fixed")
    ns = ns_sig = win = None
    if sheet is not None:
        ns = float(sheet[0])
        ns_sig = None if len(sheet) < 2 or sheet[1] is None else float(sheet[1])
        win = None if len(sheet) < 3 else sheet[2]
        if not (math.isfinite(ns) and ns > 0):
            raise ValueError("the Hall density must be positive (cm^-2)")
        if "surface" not in names and "surface" not in fixed:
            raise ValueError("a Hall fit needs the free 'surface' barrier, "
                             "fitted (in start) or fixed")
    sigmas_given = [s is not None for s in ((Cs if cv is not None else 0),
                                            (ns_sig if sheet is not None else 0))]
    n_cv = 0 if V is None else V.size
    rows = np.arange(n_cv + (1 if sheet is not None else 0), dtype=float)
    y = np.concatenate([Cm * 1e9 if V is not None else np.zeros(0),
                        [ns * 1e-13] if sheet is not None else []])
    if all(sigmas_given):
        ysig = np.concatenate([Cs * 1e9 if V is not None else np.zeros(0),
                               [ns_sig * 1e-13] if sheet is not None else []])
        if not np.all(np.isfinite(ysig) & (ysig > 0)):
            raise ValueError("measurement errors must be positive")
    elif V is not None and sheet is not None:
        raise ValueError("a joint fit of C-V and Hall data needs the errors "
                         "of both (their units differ, so the scatter "
                         "cannot set the scale)")
    else:
        ysig = None
    lengths = tuple(lengths)
    if len(lengths) > 2 or not set(lengths) <= set(names):
        raise ValueError("lengths must name at most two fitted unknowns")

    def values_of(theta):
        v = dict(fixed)
        for k, t in zip(names, theta):
            lo, hi = bounds.get(k, (-math.inf, math.inf))
            v[k] = float(min(max(t, lo), hi))
        return v

    cache = {}

    def predict(theta):
        key = tuple(np.round(theta, 15))
        if key in cache:
            return cache[key]
        v = values_of(theta)
        if not lengths:
            pred = raw(v)
        else:
            # multilinear interpolation between the grid thicknesses
            parts = []
            for k in lengths:
                lo = math.floor(v[k] / dz + 1e-9) * dz
                parts.append((k, lo, (v[k] - lo) / dz))
            pred = 0.0
            for corner in range(2 ** len(parts)):
                w = 1.0
                vv = dict(v)
                for b, (k, lo, f) in enumerate(parts):
                    up = (corner >> b) & 1
                    vv[k] = lo + up * dz
                    w *= f if up else (1.0 - f)
                if w > 0.0:
                    pred = pred + w * raw(vv)
        if len(cache) > 64:
            cache.clear()
        cache[key] = pred
        return pred

    def raw(v):
        spec = build(v)
        layers = spec["layers"]
        kw = dict(line_charge=spec.get("line_charge"),
                  extra_sheet=spec.get("extra_sheet"),
                  temperature_K=temperature_K)
        out = []
        if V is not None:
            if method == "quantum":
                _, C, _, _ = cv_curve_quantum(layers, V, barrier=v["barrier"],
                                              dz=min(dz, 0.2), dv=dv, **kw)
            else:
                _, C, _, _ = cv_curve(layers, V,
                                      top=Contact("schottky", barrier=v["barrier"]),
                                      bottom=Contact("zero-field"), dz=dz,
                                      dv=dv, **kw)
            out.append(C * 1e9 + v.get("parasitic", 0.0))
        if sheet is not None:
            tot = sum(L.thickness for L in layers)
            w = win or (0.0, tot)
            if method == "quantum":
                from .electrons import solve_electrons
                q0 = max(0.0, tot - min(tot, 120.0))
                r = solve_electrons(layers, phi_B=v["surface"], dz=min(dz, 0.2),
                                    quantum_window=(q0, tot),
                                    classical_outside=True, **kw)
                n = _sheet(r, w) * 1e14
            else:
                r = solve_quasi_equilibrium(
                    layers, Contact("schottky", barrier=v["surface"]),
                    Contact("zero-field"), dz=dz, **kw)
                n = r.sheet(*w)
            out.append(np.array([n * 1e-13]))
        return np.concatenate(out)

    def model(theta, row):
        return predict(np.asarray(theta, float))[int(row[0])]

    theta0 = np.array([float(start[k]) for k in names])
    fr = fit_least_squares(model, rows, y, theta0, y_sigma=ysig, names=names,
                           rel_step=rel_step, max_iter=max_iter, tol=1e-8,
                           difference=difference, abs_step=1e-4)
    sd = np.sqrt(np.diag(fr.cov))
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = fr.cov / np.outer(sd, sd)
    warn = correlation_warnings(names, corr, corr_limit)
    final = values_of(fr.theta)
    for k in names:
        lo, hi = bounds.get(k, (-math.inf, math.inf))
        if final[k] in (lo, hi):
            warn.append(f"{k} ended on its bound ({final[k]:g}); the best "
                        "fit may lie outside the range you allowed.")
    if not fr.converged:
        warn.append("The fit stopped before it settled; try other starting "
                    "values.")
    pred = predict(fr.theta)
    return MeasurementFit(
        values={k: final[k] for k in names},
        errors={k: float(s) for k, s in zip(names, sd)},
        correlation=corr, names=names, chi2=fr.chi2, dof=fr.dof,
        scaled=fr.scaled,
        cv_voltage=V, cv_measured=Cm,
        cv_fitted=None if V is None else pred[:n_cv] * 1e-9,
        sheet_measured=ns,
        sheet_fitted=None if sheet is None else float(pred[-1] * 1e13),
        warnings=warn, fit=fr)
