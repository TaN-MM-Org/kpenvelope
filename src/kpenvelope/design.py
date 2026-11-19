"""Find layer designs that hit a target electron density and switch-off
voltage (new in v0.18.0).

You say what you want: an electron sheet density (on the bare sample, or
under the gate at 0 V) and a switch-off (threshold) voltage, each with a
tolerance, and which design numbers may change and over what range (a
barrier's composition and thickness, say). `find_designs` maps the whole
range on a grid, refines around the best points, and returns the designs
that come closest, each with its predicted numbers. `design_error_bars`
then gives the error bars of a chosen design from the uncertainties of
its inputs (growth tolerances, barrier heights), with a budget of where
the error comes from.

Definitions (stated, because they matter):

* sheet density: electrons per cm^2 in the stack, from
  `solve_quasi_equilibrium` with a metal of the given barrier on top
  (gated) or with the free-surface barrier (bare), the bottom a thick
  base with no field (classical electrons; use the electron solver for
  subbands);
* switch-off voltage V_off: the gate voltage at which the sheet density
  has fallen to 1 percent of its value at 0 V (`threshold_voltage`), found
  by bisection to 1 mV.

The search is a grid over the design range followed by `refine` rounds
of a finer grid around the best points, so it finds the best design
within the range to the resolution of the last grid, without assuming the
targets can be met: if no design meets them, the closest ones are still
returned, with how far they miss.
"""
from __future__ import annotations

import dataclasses
import itertools
import math

import numpy as np

from .driftdiffusion import Contact, solve_quasi_equilibrium
from .uncertainty import propagate

__all__ = ["Design", "find_designs", "threshold_voltage", "sheet_density",
           "design_error_bars"]


def sheet_density(layers, barrier: float, voltage: float = 0.0,
                  temperature_K: float = 300.0, dz: float = 0.5,
                  guess=None, **kwargs):
    """Electrons per cm^2 (classical, quasi-equilibrium) with a metal of
    barrier `barrier` (eV) at `voltage` (V) on top. Returns (density,
    result)."""
    r = solve_quasi_equilibrium(layers, Contact("schottky", barrier=barrier,
                                                voltage=voltage),
                                Contact("zero-field"), temperature_K=temperature_K,
                                dz=dz, guess=guess, **kwargs)
    return r.sheet(), r


def threshold_voltage(layers, barrier: float, fraction: float = 0.01,
                      temperature_K: float = 300.0, dz: float = 0.5,
                      v_min: float = -40.0, tol: float = 1e-3, **kwargs):
    """Gate voltage (V) at which the sheet density falls to `fraction` of
    its value at 0 V (bisection to `tol`). Raises if it is not reached
    above `v_min`, or if there are no electrons at 0 V."""
    n0, r0 = sheet_density(layers, barrier, 0.0, temperature_K, dz, **kwargs)
    if not n0 > 1e8:
        raise ValueError("there are (almost) no electrons at 0 V, so there is "
                         "nothing to switch off")
    target = fraction * n0
    hi, lo = 0.0, None
    v, prev = 0.0, r0
    while v > v_min:
        v = max(v - 1.0, v_min)
        n, prev = sheet_density(layers, barrier, v, temperature_K, dz,
                                guess=prev, **kwargs)
        if n < target:
            lo = v
            break
        hi = v
    if lo is None:
        raise ValueError(f"the electrons do not switch off above {v_min} V")
    while hi - lo > tol:
        m = 0.5 * (hi + lo)
        n, _ = sheet_density(layers, barrier, m, temperature_K, dz,
                             guess=prev, **kwargs)
        if n < target:
            lo = m
        else:
            hi = m
    return 0.5 * (hi + lo)


@dataclasses.dataclass
class Design:
    """One design: its values, what it gives, and how far it misses.

    values : {name: value} of the design numbers.
    sheet_bare, sheet_gated : electrons per cm^2 (None if not asked).
    threshold : switch-off voltage (V) (None if not asked).
    miss : {target: (predicted - target) / tolerance}; a design meets a
        target when |miss| <= 1.
    score : sum of miss^2 (smaller is better).
    """

    values: dict
    sheet_bare: float
    sheet_gated: float
    threshold: float
    miss: dict
    score: float

    @property
    def meets_targets(self):
        return all(abs(m) <= 1.0 for m in self.miss.values())


def _evaluate(build, values, targets, barrier, surface, temperature_K, dz):
    layers = build(values)
    out = {"sheet_bare": None, "sheet_gated": None, "threshold": None}
    miss = {}
    try:
        if "sheet_bare" in targets:
            out["sheet_bare"], _ = sheet_density(layers, surface, 0.0,
                                                 temperature_K, dz)
        if "sheet_gated" in targets:
            out["sheet_gated"], _ = sheet_density(layers, barrier, 0.0,
                                                  temperature_K, dz)
        if "threshold" in targets:
            out["threshold"] = threshold_voltage(layers, barrier,
                                                 temperature_K=temperature_K,
                                                 dz=dz)
    except (ValueError, RuntimeError):
        return None
    for k, (val, tol) in targets.items():
        miss[k] = (out[k] - val) / tol
    return Design(values=dict(values), miss=miss,
                  score=float(sum(m * m for m in miss.values())), **out)


def find_designs(build, variables: dict, targets: dict, barrier: float = None,
                 surface: float = None, fixed: dict = None, n_grid: int = 7,
                 refine: int = 2, top: int = 5, temperature_K: float = 300.0,
                 dz: float = 0.5, return_map: bool = False):
    """Search the design range for the designs closest to the targets.

    build : callable(values dict) -> layers (bottom first).
    variables : {name: (low, high)} of the design numbers to vary (one to
        three of them).
    targets : any of "sheet_bare" (electrons per cm^2 with the free
        surface), "sheet_gated" (under the metal at 0 V) and "threshold"
        (switch-off voltage, V), each as (value, tolerance).
    barrier : metal barrier (eV), needed for "sheet_gated" and "threshold";
        surface : free-surface barrier (eV), needed for "sheet_bare".
    fixed : other numbers passed to `build` unchanged.
    n_grid : grid points per variable; refine : rounds of a finer grid
        (same size, a third of the span) around the best point.
    Returns the `top` best `Design`s (best first), and with return_map
    also the first grid: {"axes": {name: values}, "designs": [...]}.
    """
    if not (1 <= len(variables) <= 3):
        raise ValueError("vary one to three design numbers")
    allowed = {"sheet_bare", "sheet_gated", "threshold"}
    if not targets or not set(targets) <= allowed:
        raise ValueError(f"targets must be some of {sorted(allowed)}")
    for k, (val, tol) in targets.items():
        if not (math.isfinite(val) and math.isfinite(tol) and tol > 0):
            raise ValueError(f"target {k} needs a finite value and a "
                             "positive tolerance")
    if ("sheet_gated" in targets or "threshold" in targets) and barrier is None:
        raise ValueError("a gated target needs the metal barrier")
    if "sheet_bare" in targets and surface is None:
        raise ValueError("a bare-sample target needs the surface barrier")
    for k, (lo, hi) in variables.items():
        if not (math.isfinite(lo) and math.isfinite(hi) and hi > lo):
            raise ValueError(f"the range of {k} needs low < high")
    if not (isinstance(n_grid, int) and n_grid >= 2):
        raise ValueError("n_grid must be an integer >= 2")
    fixed = dict(fixed or {})
    names = list(variables)
    seen = {}

    def ev(point):
        key = tuple(round(p, 12) for p in point)
        if key not in seen:
            vals = dict(fixed, **dict(zip(names, point)))
            seen[key] = _evaluate(build, vals, targets, barrier, surface,
                                  temperature_K, dz)
        return seen[key]

    axes = {k: np.linspace(lo, hi, n_grid) for k, (lo, hi) in variables.items()}
    grid = [ev(p) for p in itertools.product(*axes.values())]
    span = {k: hi - lo for k, (lo, hi) in variables.items()}
    for _ in range(refine):
        good = sorted((d for d in seen.values() if d is not None),
                      key=lambda d: d.score)
        if not good:
            break
        for k in span:
            span[k] /= 3.0
        for best in good[:2]:
            sub = []
            for k in names:
                lo, hi = variables[k]
                c = best.values[k]
                sub.append(np.linspace(max(lo, c - span[k]),
                                       min(hi, c + span[k]), n_grid))
            for p in itertools.product(*sub):
                ev(p)
    ranked = sorted((d for d in seen.values() if d is not None),
                    key=lambda d: d.score)
    # keep designs that differ: skip near-duplicates of better ones
    picked = []
    for d in ranked:
        if all(any(abs(d.values[k] - e.values[k]) > 0.02 * (variables[k][1] - variables[k][0])
                   for k in names) for e in picked):
            picked.append(d)
        if len(picked) >= top:
            break
    if return_map:
        return picked, {"axes": {k: v.tolist() for k, v in axes.items()},
                        "designs": grid}
    return picked


def design_error_bars(build, design_values: dict, sigmas: dict,
                      barrier: float = None, surface: float = None,
                      quantities=("sheet_bare", "sheet_gated", "threshold"),
                      temperature_K: float = 300.0, dz: float = 0.5):
    """Error bars of a design's numbers from the uncertainties of its
    inputs (first order, with a budget).

    sigmas : {name: one-sigma error} of design numbers (as named in
        `build`) and of "barrier" and "surface".
    Returns {quantity: Propagation}. The slopes are secants over +-1
    sigma (a thickness on the grid moves by whole grid steps, so a smaller
    step would be noisy).
    """
    values = dict(design_values)
    if barrier is not None:
        values["barrier"] = barrier
    if surface is not None:
        values["surface"] = surface
    for k in sigmas:
        if k not in values:
            raise ValueError(f"no value for {k!r}")
    out = {}
    for q in quantities:
        if q in ("sheet_gated", "threshold") and "barrier" not in values:
            continue
        if q == "sheet_bare" and "surface" not in values:
            continue

        def f(q=q, **v):
            layers = build({k: x for k, x in v.items()
                            if k not in ("barrier", "surface")})
            if q == "sheet_bare":
                return sheet_density(layers, v["surface"], 0.0, temperature_K, dz)[0]
            if q == "sheet_gated":
                return sheet_density(layers, v["barrier"], 0.0, temperature_K, dz)[0]
            return threshold_voltage(layers, v["barrier"],
                                     temperature_K=temperature_K, dz=dz)

        out[q] = propagate(f, values, sigmas, method="linear", step_fraction=1.0)
    return out
