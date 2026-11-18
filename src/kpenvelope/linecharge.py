"""What is known about the misfit-line charge, how to combine it, and how to
plan the measurement that pins it down (new in v0.18.0).

The charge per length Lambda of the misfit dislocations is the least known
input of a relaxed nitride interface. Only one published series measures it
(Shen et al., Appl. Phys. Lett. 76, 2746 (2000)): a fit of its electron
densities gives Lambda = -0.84 +- 0.53 e/nm, about 60 percent. No software
can shrink that error without new data. What this module does:

* `PUBLISHED_CONSTRAINTS` lists, with their sources and assumptions, the
  published numbers that bear on Lambda: the density fit, and the upper
  limits on |Lambda| that the measured mobilities set (scattering by the
  lines cannot exceed the measured total).
* `combine_line_charge` combines Gaussian estimates with such upper limits
  into one posterior: a flat prior times the Gaussians (an inverse-variance
  Gaussian), cut off where a limit excludes the value, which is a truncated
  normal and is evaluated in closed form. The result is reported with its mean, standard deviation, median
  and central 68 and 95 percent intervals, and with the share of the
  original estimate the limits removed, so it is clear how much the limits
  did. It is only as good as the assumptions listed with each constraint.
* `plan_holography`, `plan_sum_rule` and `line_charge_error_from_density`
  predict the error of Lambda (or of the trapped part q) that a planned
  measurement would give, so the experiment that actually fixes Lambda can
  be designed before it is grown.
"""
from __future__ import annotations

import dataclasses
import math

import numpy as np

from .interfaces import misfit_line_density
from .nitride import E_PER_NM2_PER_C_M2, NitrideMaterial

__all__ = ["Constraint", "PUBLISHED_CONSTRAINTS", "combine_line_charge",
           "LineChargePosterior", "plan_holography", "plan_sum_rule",
           "line_charge_error_from_density"]

_MISFIT = ("T. M. Mahim, 'The charge of misfit dislocations decides the "
           "polarization charge at relaxed nitride interfaces' (2026, "
           "manuscript); archived results in nitride-misfit-polarization "
           "v1.3.1, doi:10.5281/zenodo.24315036")


@dataclasses.dataclass(frozen=True)
class Constraint:
    """One published piece of information about Lambda (e/nm).

    kind : "gaussian" (value +- sigma) or "upper" (|Lambda| <= value).
    value, sigma : e/nm (sigma unused for "upper").
    applies_to : the interface the number was measured on.
    assumptions : what has to hold for it to apply elsewhere.
    source : where it comes from (file and key of the archived results).
    """

    name: str
    kind: str
    value: float
    sigma: float
    applies_to: str
    assumptions: str
    source: str


#: The published constraints, copied with all digits from the archived
#: results (results/shen_uncertainty.json, results/shen_analysis.json,
#: results/elwaradi_analysis.json of the misfit analysis).
PUBLISHED_CONSTRAINTS = {
    "shen-density-pbesol": Constraint(
        "Shen series, density fit (PBEsol constants, phi_B = 0.5 eV)",
        "gaussian", -0.8448566143029012, 0.5323733875570534,
        "Al0.22Ga0.78N stretched on GaN, relaxed 78 percent",
        "the Schroedinger-Poisson model of the series, a fixed calibration "
        "charge, an error of 0.05 on the relaxation; the fit moves to "
        "-0.95, -1.09 and -1.27 e/nm for surface barriers of 1, 1.5 and 2 eV",
        _MISFIT + ": shen_uncertainty.json, phi 0.5, fit_B"),
    "shen-density-hse": Constraint(
        "Shen series, density fit (HSE constants, phi_B = 0.5 eV)",
        "gaussian", -0.5729500135068806, 0.5323732285916173,
        "Al0.22Ga0.78N stretched on GaN, relaxed 78 percent",
        "as above, with the polarization constants of Dreyer et al. (2016)",
        _MISFIT + ": shen_uncertainty_hse.json, phi 0.5, fit_B"),
    "shen-mobility": Constraint(
        "Shen series, mobility limit at 77 K", "upper", 1.277806399828856,
        math.nan, "Al0.22Ga0.78N stretched on GaN, relaxed 78 percent",
        "all scattering of the relaxed sample from the lines (so a true "
        "upper limit), the line-scattering model with Thomas-Fermi "
        "screening; over the sensitivity tests at 77 K it lies between "
        "1.03 and 1.35 e/nm",
        _MISFIT + ": shen_analysis.json, baseline 77K_mu2_0, mobility"),
    "elwaradi-mobility": Constraint(
        "Elwaradi series, mobility limit (barrier relaxed 25 percent)",
        "upper", 1.3862680293570193, math.nan,
        "Al0.3Ga0.7N compressed on a 20 nm GaN channel on AlN",
        "as for the Shen limit, for lines of the opposite sign; applies to "
        "the Shen lines only if |Lambda| is the same for both alloys and "
        "both signs of the strain. 1.55 e/nm for a relaxation of 0.20",
        _MISFIT + ": elwaradi_analysis.json, mobility_t20, R_bar_0.25"),
}


@dataclasses.dataclass
class LineChargePosterior:
    """Combined estimate of Lambda (e/nm) on a grid.

    mean, sd, median, interval68, interval95 : of the posterior.
    removed : share of the Gaussian estimate that the limits excluded.
    grid, density : the posterior density on a grid (for plotting).
    used : names (or descriptions) of the constraints combined.
    """

    mean: float
    sd: float
    median: float
    interval68: tuple
    interval95: tuple
    removed: float
    grid: np.ndarray
    density: np.ndarray
    used: list


def _as_constraint(c):
    if isinstance(c, Constraint):
        return c
    if isinstance(c, str):
        if c not in PUBLISHED_CONSTRAINTS:
            raise ValueError(f"unknown constraint {c!r}; known: "
                             f"{sorted(PUBLISHED_CONSTRAINTS)}")
        return PUBLISHED_CONSTRAINTS[c]
    raise ValueError("constraints are Constraint objects or names of "
                     "PUBLISHED_CONSTRAINTS")


def combine_line_charge(constraints, span: float = 8.0, n_grid: int = 2001):
    """Combine Gaussian estimates and upper limits on |Lambda| into one
    posterior (flat prior). At least one Gaussian is needed.

    constraints : names of `PUBLISHED_CONSTRAINTS` or `Constraint` objects.
    span, n_grid : extent (in combined sigmas) and size of the grid on which
        the posterior density is returned for plotting; the summary numbers
        are exact.
    """
    cs = [_as_constraint(c) for c in constraints]
    gs = [c for c in cs if c.kind == "gaussian"]
    us = [c for c in cs if c.kind == "upper"]
    if any(c.kind not in ("gaussian", "upper") for c in cs):
        raise ValueError("constraint kind must be 'gaussian' or 'upper'")
    if not gs:
        raise ValueError("combining needs at least one Gaussian estimate")
    for c in gs:
        if not (math.isfinite(c.value) and math.isfinite(c.sigma) and c.sigma > 0):
            raise ValueError(f"{c.name}: a Gaussian needs a finite value and sigma > 0")
    for c in us:
        if not (math.isfinite(c.value) and c.value > 0):
            raise ValueError(f"{c.name}: an upper limit must be positive")
    w = np.array([1 / c.sigma ** 2 for c in gs])
    mu = float((w * [c.value for c in gs]).sum() / w.sum())
    sg = float(1 / math.sqrt(w.sum()))
    U = min((c.value for c in us), default=math.inf)
    # The product of Gaussians is a Gaussian; the limits cut it to
    # [-U, U]: a truncated normal, evaluated in closed form.
    al = -math.inf if U == math.inf else (-U - mu) / sg
    be = math.inf if U == math.inf else (U - mu) / sg
    Phi = lambda t: 0.5 * math.erfc(-t / math.sqrt(2.0))
    phi = lambda t: 0.0 if math.isinf(t) else math.exp(-0.5 * t * t) / math.sqrt(2 * math.pi)
    Z = Phi(be) - Phi(al)
    if Z < 1e-300:
        raise ValueError("the limits exclude the whole estimate")
    ta = 0.0 if math.isinf(al) else al * phi(al)
    tb = 0.0 if math.isinf(be) else be * phi(be)
    mean = mu + sg * (phi(al) - phi(be)) / Z
    var = sg * sg * (1 + (ta - tb) / Z - ((phi(al) - phi(be)) / Z) ** 2)

    def q(f):          # quantile by bisection on the exact CDF
        lo_, hi_ = max(mu - 40 * sg, -U), min(mu + 40 * sg, U)
        target = Phi(al) + f * Z
        for _ in range(200):
            m = 0.5 * (lo_ + hi_)
            if Phi((m - mu) / sg) < target:
                lo_ = m
            else:
                hi_ = m
        return 0.5 * (lo_ + hi_)

    x = np.linspace(max(mu - span * sg, -U), min(mu + span * sg, U), n_grid)
    dens = np.exp(-0.5 * ((x - mu) / sg) ** 2) / (sg * math.sqrt(2 * math.pi) * Z)
    return LineChargePosterior(
        mean=float(mean), sd=float(math.sqrt(max(var, 0.0))), median=q(0.5),
        interval68=(q(0.15865525393145707), q(0.8413447460685429)),
        interval95=(q(0.025), q(0.975)), removed=float(1.0 - Z), grid=x,
        density=dens, used=[c.name for c in cs])


def plan_holography(film: NitrideMaterial, eps_c: float, relaxation: float,
                    charge_error_C_m2: float = 1e-3, n_layers: int = 1) -> float:
    """Error of Lambda (e/nm) from a measured interface charge change with
    error `charge_error_C_m2`, for one layer of mismatch eps_c relaxed by
    `relaxation` (the polarization constants taken as known):
    sigma_Lambda = sigma_charge / L_A, divided by sqrt(n_layers) for
    independent layers. Electron holography reaches about 1e-3 C/m^2 at
    InGaN/GaN interfaces (Lan et al., Phys. Rev. B 113, 155302 (2026))."""
    if not (math.isfinite(charge_error_C_m2) and charge_error_C_m2 > 0):
        raise ValueError("charge_error_C_m2 must be positive")
    if not (isinstance(n_layers, int) and n_layers >= 1):
        raise ValueError("n_layers must be an integer >= 1")
    if not (0.0 < relaxation <= 1.0) or eps_c == 0.0:
        raise ValueError("need 0 < relaxation <= 1 and a nonzero mismatch")
    LA = misfit_line_density(relaxation, eps_c, film.a)
    return charge_error_C_m2 * E_PER_NM2_PER_C_M2 / LA / math.sqrt(n_layers)


def plan_sum_rule(film: NitrideMaterial, eps_tension: float,
                  eps_compression: float, r_tension: float,
                  r_compression: float, charge_error_C_m2: float = 1e-3) -> float:
    """Error of the trapped charge q (e/nm) from the sum rule y+ + y- =
    -2 q e / b, with an error `charge_error_C_m2` on each of the two measured
    charge changes: sigma_q = (b/2) sqrt(e+^2 + e-^2), e = sigma/(2 r |eps|).
    Needs no polarization constant."""
    if not (eps_tension > 0 > eps_compression):
        raise ValueError("need eps_tension > 0 > eps_compression")
    if not (0 < r_tension <= 1 and 0 < r_compression <= 1):
        raise ValueError("relaxations must lie in (0, 1]")
    if not charge_error_C_m2 > 0:
        raise ValueError("charge_error_C_m2 must be positive")
    e1 = charge_error_C_m2 / (2 * r_tension * eps_tension)
    e2 = charge_error_C_m2 / (2 * r_compression * abs(eps_compression))
    return film.a / 2 * math.hypot(e1, e2) * E_PER_NM2_PER_C_M2


def line_charge_error_from_density(build_layers, settings_row, line_charge,
                                   density_error_nm2, settings_error=None,
                                   step_line_charge=0.01, step_settings=None,
                                   n_samples: int = 1, extra_sheet=None,
                                   **solver_kwargs) -> dict:
    """Error of Lambda (e/nm) that a measured 2DEG density would give.

    build_layers(row) -> layers, as for `fit_interface_series`; settings_row
    the planned sample (for example [thickness, relaxation]); line_charge
    the Lambda around which to linearize; density_error_nm2 the error of
    the measured density; settings_error the errors of the settings (same
    length as the row, 0 for exact ones), which add through the slopes.
    n_samples independent samples of the same design divide the error by
    sqrt(n_samples). Returns {"sigma_line_charge", "dn_dLambda",
    "dn_dsettings"}.
    """
    from .electrons import solve_electrons

    row = np.asarray(settings_row, dtype=float)
    if not (math.isfinite(density_error_nm2) and density_error_nm2 > 0):
        raise ValueError("density_error_nm2 must be positive")
    se = np.zeros_like(row) if settings_error is None else np.asarray(settings_error, float)
    if se.shape != row.shape or np.any(se < 0):
        raise ValueError("settings_error must match the row and be >= 0")

    def n(lam, r):
        res = solve_electrons(build_layers(r), line_charge=float(lam),
                              extra_sheet=extra_sheet, **solver_kwargs)
        if not res.converged:
            raise RuntimeError("the electron solver did not converge")
        return res.sheet_quantum

    h = step_line_charge
    dn_dl = (n(line_charge + h, row) - n(line_charge - h, row)) / (2 * h)
    if dn_dl == 0.0:
        raise ValueError("the density does not depend on Lambda for this "
                         "sample (is the layer relaxed?)")
    var = density_error_nm2 ** 2
    dns = np.zeros_like(row)
    for k in range(row.size):
        if se[k] > 0:
            hk = (step_settings[k] if step_settings is not None else 0.01 * se[k])
            up, dn = row.copy(), row.copy()
            up[k] += hk
            dn[k] -= hk
            dns[k] = (n(line_charge, up) - n(line_charge, dn)) / (2 * hk)
            var += (dns[k] * se[k]) ** 2
    return {"sigma_line_charge": math.sqrt(var) / abs(dn_dl) / math.sqrt(n_samples),
            "dn_dLambda": dn_dl, "dn_dsettings": dns}
