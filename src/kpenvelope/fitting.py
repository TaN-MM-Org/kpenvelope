"""Fitting model parameters to measured data, with error bars, including
errors in the settings (new in v0.17.0).

`fit_least_squares` is a general weighted least-squares fit
(Levenberg-Marquardt, central-difference Jacobian). When the settings x
also have errors (for example a relaxation read from an X-ray map), it fits
the true settings together with the parameters, minimizing

    S = sum ((y - f(theta, x_hat)) / sigma_y)^2 + sum ((x - x_hat) / sigma_x)^2

(orthogonal-distance regression), so that the uncertainty of the settings
enters the error bars of the parameters. Without reading errors the error
scale is taken from the scatter of the readings about the fit (chi^2 per
degree of freedom = 1).

`fit_interface_series` uses it for the question of the misfit paper: given
measured electron densities of samples that differ in thickness and
relaxation, what calibration charge and what misfit-line charge Lambda
reproduce them? Each prediction is a full `solve_electrons` run.
"""
from __future__ import annotations

import dataclasses

import numpy as np

__all__ = ["fit_least_squares", "FitResult", "fit_interface_series"]


@dataclasses.dataclass
class FitResult:
    """Result of `fit_least_squares`.

    theta, sigma, cov : best parameters, 1-sigma errors, covariance.
    names : parameter names (if given).
    chi2, dof : minimum of S and the degrees of freedom (n - p).
    scaled : True when the error scale was taken from the scatter.
    x_adjusted : fitted true settings (equal to x where x has no error).
    residuals : y - f(theta, x_adjusted).
    converged, iterations.
    """

    theta: np.ndarray
    sigma: np.ndarray
    cov: np.ndarray
    names: tuple
    chi2: float
    dof: int
    scaled: bool
    x_adjusted: np.ndarray
    residuals: np.ndarray
    converged: bool
    iterations: int

    @property
    def values(self):
        return dict(zip(self.names, self.theta))

    @property
    def errors(self):
        return dict(zip(self.names, self.sigma))


def fit_least_squares(model, x, y, theta0, y_sigma=None, x_sigma=None,
                      names=None, rel_step=1e-6, max_iter=100, tol=1e-10):
    """Least-squares fit of model(theta, x_row) -> prediction to readings y.

    model : callable(theta (p,), x_row (d,)) -> float. Each prediction may
        depend only on its own row of settings.
    x : (n, d) settings (or (n,)); y : (n,) readings.
    theta0 : starting parameters.
    y_sigma : 1-sigma errors of the readings (number or n numbers), or None
        to take the error scale from the scatter (needs n > p).
    x_sigma : 1-sigma errors of the settings ((n, d), n numbers, or one
        number); 0 marks an exact setting. Needs y_sigma.
    rel_step : relative central-difference step (absolute 1e-6 where the
        value is zero).
    """
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    y = np.asarray(y, dtype=float).ravel()
    n, d = x.shape
    th = np.asarray(theta0, dtype=float).ravel()
    p = th.size
    if y.size != n:
        raise ValueError("need one reading per row of settings")
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))
            and np.all(np.isfinite(th))):
        raise ValueError("settings, readings and theta0 must be finite")
    if n < p:
        raise ValueError(f"{n} readings cannot determine {p} parameters")
    names = tuple(names) if names is not None else tuple(f"p{i}" for i in range(p))
    if len(names) != p:
        raise ValueError("one name per parameter")
    if y_sigma is None:
        if x_sigma is not None and np.any(np.asarray(x_sigma) > 0):
            raise ValueError("errors in the settings need reading errors "
                             "y_sigma too: their balance cannot be taken "
                             "from the scatter")
        if n == p:
            raise ValueError("with no reading errors the scatter needs more "
                             "readings than parameters")
        sy = np.ones(n)
        scaled = True
    else:
        sy = np.broadcast_to(np.asarray(y_sigma, float), (n,)).astype(float)
        if not np.all(np.isfinite(sy) & (sy > 0)):
            raise ValueError("y_sigma must be finite and positive")
        scaled = False
    if x_sigma is None:
        sx = np.zeros((n, d))
    else:
        a = np.asarray(x_sigma, float)
        if a.ndim == 0:
            sx = np.full((n, d), float(a))
        elif a.ndim == 1 and a.size == n and d == 1:
            sx = a[:, None].copy()
        else:
            sx = np.broadcast_to(a, (n, d)).astype(float)
        if not np.all(np.isfinite(sx) & (sx >= 0)):
            raise ValueError("x_sigma must be finite and >= 0")
    free = np.argwhere(sx > 0)          # (m, 2) entries of x fitted too
    m = len(free)

    def unpack(u):
        xa = x.copy()
        for j, (r, c) in enumerate(free):
            xa[r, c] = u[p + j]
        return u[:p], xa

    def resid(u):
        t, xa = unpack(u)
        f = np.array([float(model(t, xa[i])) for i in range(n)])
        r1 = (y - f) / sy
        r2 = np.array([(x[r, c] - xa[r, c]) / sx[r, c] for r, c in free])
        return np.concatenate([r1, r2]) if m else r1

    u = np.concatenate([th, x[tuple(free.T)]]) if m else th.copy()

    def jac(u, r0):
        J = np.empty((r0.size, u.size))
        for k in range(u.size):
            h = rel_step * abs(u[k]) if u[k] != 0 else 1e-6
            up, dn = u.copy(), u.copy()
            up[k] += h
            dn[k] -= h
            J[:, k] = (resid(up) - resid(dn)) / (2 * h)
        return J

    r = resid(u)
    S = float(r @ r)
    lam = 1e-3
    converged = False
    it = 0
    for it in range(1, max_iter + 1):
        J = jac(u, r)
        A = J.T @ J
        g = J.T @ r
        accepted = False
        for _ in range(30):
            step = np.linalg.solve(A + lam * np.diag(np.diag(A) + 1e-300), -g)
            u_new = u + step
            r_new = resid(u_new)
            S_new = float(r_new @ r_new)
            if S_new <= S:
                accepted = True
                lam = max(lam / 3.0, 1e-12)
                break
            lam *= 4.0
        if not accepted:
            # no step lowers S any more: at the minimum to rounding
            converged = True
            break
        dS = S - S_new
        u, r, S = u_new, r_new, S_new
        small_step = np.all(np.abs(step) <= 1e-9 * (np.abs(u) + 1e-12))
        if dS <= tol * max(S, 1e-300) or small_step:
            converged = True
            break
    J = jac(u, r)
    covall = np.linalg.pinv(J.T @ J)
    cov = covall[:p, :p]
    dof = n - p
    if scaled:
        cov = cov * (S / dof)
    t, xa = unpack(u)
    f = np.array([float(model(t, xa[i])) for i in range(n)])
    return FitResult(theta=t, sigma=np.sqrt(np.diag(cov)), cov=cov, names=names,
                     chi2=S, dof=dof, scaled=scaled, x_adjusted=xa,
                     residuals=y - f, converged=converged, iterations=it)


def fit_interface_series(build_layers, settings, sheet_measured,
                         sheet_sigma=None, settings_sigma=None,
                         theta0=(0.0, 0.0), calibration_interface=0,
                         names=("calibration_nm2", "line_charge_e_per_nm"),
                         **solver_kwargs):
    """Fit a calibration charge and the misfit-line charge to measured
    electron densities of a series of samples.

    build_layers : callable(settings_row) -> list of `StackLayer`. Put the
        relaxation (and anything else that changes between samples) in the
        settings row so that its error can enter the fit.
    settings : (n, d) settings rows; sheet_measured : (n,) measured 2DEG
        densities (nm^-2).
    sheet_sigma : reading errors (nm^-2), or None (scale from the scatter).
    settings_sigma : errors of the settings ((n, d)), optional; needs
        sheet_sigma.
    theta0 : start for (calibration charge in e/nm^2 at
        `calibration_interface`, Lambda in e/nm). Lambda is applied to every
        relaxed interface as a fixed number (its sign is the fitted one).
    solver_kwargs : passed to `solve_electrons` (phi_B, temperature_K,
        quantum_window, ...).

    Returns a `FitResult` with the two parameters.
    """
    from .electrons import solve_electrons

    def model(theta, row):
        layers = build_layers(row)
        res = solve_electrons(layers, line_charge=float(theta[1]),
                              extra_sheet={calibration_interface: float(theta[0])},
                              **solver_kwargs)
        if not res.converged:
            raise RuntimeError("the electron solver did not converge for the "
                               f"settings {list(row)}")
        return res.sheet_quantum

    return fit_least_squares(model, settings, sheet_measured, theta0,
                             y_sigma=sheet_sigma, x_sigma=settings_sigma,
                             names=names)
