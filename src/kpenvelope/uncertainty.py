"""Error bars carried through any calculation of the package (new in v0.17.0).

`propagate(func, values, sigmas)` takes a function of named inputs (a line
charge, a band offset, a polarization constant, a relaxation ...) and their
1-sigma errors, and returns the error of the result, either to first order
(``method="linear"``: central differences, an error budget per input, and
optional correlations) or by sampling (``method="monte-carlo"``: normal
draws, optionally truncated to physical bounds, with percentiles). The two
agree for a linear function; where they differ, the result is not linear
over the error bars, and the Monte Carlo percentiles are the ones to quote.
"""
from __future__ import annotations

import dataclasses
import math

import numpy as np

__all__ = ["propagate", "Propagation"]


@dataclasses.dataclass
class Propagation:
    """Result of `propagate`.

    value : func at the nominal inputs.
    sigma : 1-sigma error (linear) or standard deviation of the samples.
    method : "linear" or "monte-carlo".
    budget : {input: its share of the variance} (linear, uncorrelated part;
        the shares sum to 1 when there are no correlations).
    jacobian : {input: d func / d input} (linear).
    mean, percentiles : sample mean and {2.5, 16, 50, 84, 97.5} percentiles
        (Monte Carlo); samples : the draws of func (Monte Carlo);
        rejected_fraction : share of draws redrawn because they fell outside
        `bounds`.
    """

    value: object
    sigma: object
    method: str
    budget: dict = None
    jacobian: dict = None
    mean: object = None
    percentiles: dict = None
    samples: np.ndarray = None
    rejected_fraction: float = 0.0


def _check(values, sigmas):
    if not isinstance(values, dict) or not values:
        raise ValueError("values must be a non-empty dict {name: nominal value}")
    if not isinstance(sigmas, dict):
        raise ValueError("sigmas must be a dict {name: 1-sigma error}")
    for k, s in sigmas.items():
        if k not in values:
            raise ValueError(f"sigma given for {k!r}, which is not an input")
        if not (math.isfinite(s) and s >= 0.0):
            raise ValueError(f"the sigma of {k!r} must be finite and >= 0")
    for k, v in values.items():
        if not math.isfinite(v):
            raise ValueError(f"the value of {k!r} must be finite")


def propagate(func, values: dict, sigmas: dict, method: str = "linear",
              correlation: dict = None, step_fraction: float = 0.01,
              n_samples: int = 2000, seed: int = 0, bounds: dict = None):
    """Error of func(**values) from the errors of its inputs.

    func : callable taking the inputs as keyword arguments and returning a
        number or an array.
    values : {name: nominal value}; sigmas : {name: 1-sigma error}. Inputs
        without a sigma are held fixed.
    correlation : {(name1, name2): correlation coefficient}, optional.
    step_fraction : central-difference step as a fraction of each sigma
        (linear method). Keep it well above the numerical noise of func.
    n_samples, seed : Monte Carlo draws and the seed of the generator.
    bounds : {name: (low, high)}: Monte Carlo draws outside are redrawn
        (a truncated normal); the share redrawn is reported.
    """
    _check(values, sigmas)
    free = [k for k in values if sigmas.get(k, 0.0) > 0.0]
    nominal = func(**values)
    nom = np.asarray(nominal, dtype=float)
    sig = np.array([sigmas[k] for k in free])
    C = np.eye(len(free))
    for (a, b), rho in (correlation or {}).items():
        if a not in free or b not in free or a == b:
            raise ValueError(f"correlation ({a!r}, {b!r}) needs two different "
                             "inputs that both have a sigma")
        if not (-1.0 <= rho <= 1.0):
            raise ValueError("correlation coefficients lie in [-1, 1]")
        i, j = free.index(a), free.index(b)
        C[i, j] = C[j, i] = rho
    if free and np.min(np.linalg.eigvalsh(C)) < -1e-12:
        raise ValueError("the correlation matrix is not positive semidefinite")
    cov = C * np.outer(sig, sig)
    if method == "linear":
        if not (0.0 < step_fraction <= 1.0):
            raise ValueError("step_fraction must lie in (0, 1]")
        J = []
        for k, s in zip(free, sig):
            h = step_fraction * s
            up = dict(values, **{k: values[k] + h})
            dn = dict(values, **{k: values[k] - h})
            J.append((np.asarray(func(**up), float) - np.asarray(func(**dn), float))
                     / (2.0 * h))
        if not free:
            return Propagation(nominal, 0.0 * nom, "linear", {}, {})
        J = np.array(J)                               # (p,) or (p, ...)
        var = np.einsum("i...,ij,j...->...", J, cov, J)
        indiv = {k: (J[i] * sig[i]) ** 2 for i, k in enumerate(free)}
        tot = sum(indiv.values())
        budget = {k: np.where(tot > 0, v / np.where(tot > 0, tot, 1.0), 0.0)
                  for k, v in indiv.items()}
        if not nom.ndim:
            budget = {k: float(v) for k, v in budget.items()}
        s_out = np.sqrt(np.maximum(var, 0.0))
        return Propagation(nominal, float(s_out) if not nom.ndim else s_out,
                           "linear", budget,
                           {k: (float(J[i]) if not nom.ndim else J[i])
                            for i, k in enumerate(free)})
    if method == "monte-carlo":
        if not (isinstance(n_samples, int) and n_samples >= 10):
            raise ValueError("n_samples must be an integer >= 10")
        rng = np.random.default_rng(seed)
        if free:
            w, V = np.linalg.eigh(cov)
            Lc = V * np.sqrt(np.maximum(w, 0.0))      # Lc @ Lc.T == cov
        bnd = bounds or {}
        for k in bnd:
            if k not in values:
                raise ValueError(f"bounds given for {k!r}, which is not an input")
        out, drawn, kept = [], 0, 0
        while kept < n_samples:
            z = rng.standard_normal(len(free))
            drawn += 1
            d = dict(values)
            dv = Lc @ z if free else ()
            for i, k in enumerate(free):
                d[k] = values[k] + float(dv[i])
            ok = all(lo <= d[k] <= hi for k, (lo, hi) in bnd.items())
            if not ok:
                if drawn > 100 * n_samples:
                    raise ValueError("almost every draw falls outside the bounds")
                continue
            out.append(np.asarray(func(**d), float))
            kept += 1
        S = np.array(out)
        pct = {p: np.percentile(S, p, axis=0) for p in (2.5, 16, 50, 84, 97.5)}
        if not nom.ndim:
            pct = {p: float(v) for p, v in pct.items()}
        mean = S.mean(axis=0)
        sd = S.std(axis=0, ddof=1)
        return Propagation(nominal, float(sd) if not nom.ndim else sd,
                           "monte-carlo", mean=float(mean) if not nom.ndim else mean,
                           percentiles=pct, samples=S,
                           rejected_fraction=(drawn - kept) / drawn)
    raise ValueError("method must be 'linear' or 'monte-carlo'")
