"""How much each subband energy depends on each material parameter
(new in v0.15.0).

The six-band matrix is linear in every one of the nine parameters
A1..A6, delta1, delta2, delta3 of every material, and in a common
scale of the band-edge profile: H = sum_t theta_t dH/dtheta_t + (the
potential and strain terms). For an eigenvalue E with normalized
eigenvector v of the assembled (discrete) Hermitian matrix, the
derivative is therefore exactly

    dE/dtheta = v^H (dH/dtheta) v

(the Hellmann-Feynman theorem; for a matrix it is first-order
perturbation theory, valid for any Hermitian matrix with a simple
eigenvalue). Each dH/dtheta is the package's own assembly with that one
parameter set to 1 and all others to 0, so no finite-difference step
is involved. Where several states share one energy (a Kramers pair at
k_t = 0, or a level of the synthetic demo set) the derivative of their
common energy is taken from the matrix of dH/dtheta inside that level:
its trace divided by the number of states, the derivative of the
level's mean energy, which does not depend on the basis the solver
picked inside the level. A Kramers pair is never split by these
parameters (they do not break time reversal), so there the value is
the derivative of each member. When a parameter does split a level
(possible only for a level that is degenerate by accident, as in the
demo set), the individual energies have a kink there and the state is
flagged in ``level_split``.

From the derivatives (a Jacobian J) and independent 1-sigma errors of
the parameters, the linear error propagation gives the covariance of
the energies, C = J diag(sigma^2) J^T, and with it the error of any
spacing E_i - E_j, sqrt(C_ii + C_jj - 2 C_ij). No parameter errors are
shipped: they depend on the source of the parameters, so you pass
yours. The propagation is linear: it holds while the energies change
linearly over +/- one sigma, which a Monte Carlo test checks for 2 %
errors on the GaN set.

Anchors in the tests: on the synthetic demo set the derivatives equal
the closed forms of the discrete well (dE/dA1 = c (2 - 2 cos(l pi /
(N + 1))) / dz^2 at k_t = 0, dE/dA2 = c k_t^2); on GaN, at zero and
finite in-plane momentum and for a two-material stack, they equal
central finite differences of the solver's energies; the band-edge
derivative equals `offset_sensitivity`; and the propagated errors
equal the spread of a Monte Carlo sample.
"""
from __future__ import annotations

import numpy as np

from .hamiltonian import (assemble_hamiltonian, check_grid, check_n_states,
                          check_potential)
from .heterostructure import assemble_heterostructure
from .params import WurtziteParameters
from .solver import top_eigenpairs

__all__ = ["PARAMETER_NAMES", "parameter_sensitivities"]

#: the parameters `parameter_sensitivities` differentiates with
#: respect to, in this order (A1..A6 dimensionless, the deltas in eV)
PARAMETER_NAMES = ("A1", "A2", "A3", "A4", "A5", "A6",
                   "delta1", "delta2", "delta3")


def _unit_set(name=None):
    """A parameter set with every parameter 0 except ``name`` (= 1)."""
    vals = {k: 0.0 for k in PARAMETER_NAMES}
    if name is not None:
        vals[name] = 1.0
    return WurtziteParameters(**vals, eps_r=float("nan"),
                              reference="unit set for a derivative "
                                        "(internal; not a material)")


def _clusters(vals, tol):
    """Runs of (descending) eigenvalues closer than tol to the
    previous one."""
    out = [[0]]
    for i in range(1, vals.size):
        if abs(vals[i - 1] - vals[i]) <= tol:
            out[-1].append(i)
        else:
            out.append([i])
    return [np.asarray(c) for c in out]


def _sigma_vector(sigmas, materials, names):
    """sigmas (a mapping, or one mapping per material) -> (n_mat, 9)."""
    n_mat = len(materials)
    if isinstance(sigmas, dict):
        if n_mat != 1:
            raise ValueError(
                f"this structure has {n_mat} materials (see the "
                "'materials' entry of the result); pass one mapping of "
                "parameter errors per material, in that order (an empty "
                "mapping for a material without errors)")
        per = [sigmas]
    else:
        per = list(sigmas)
        if len(per) != n_mat:
            raise ValueError(
                f"one mapping of parameter errors per material is "
                f"needed ({n_mat}); got {len(per)}")
    out = np.zeros((n_mat, len(names)))
    for m, d in enumerate(per):
        if d is None:
            continue
        if not isinstance(d, dict):
            raise ValueError("parameter errors must be given as a "
                             "mapping such as {'A1': 0.1, 'delta1': "
                             "0.002}")
        for key, val in d.items():
            if key not in names:
                raise ValueError(
                    f"unknown parameter {key!r}; the parameters are "
                    f"{', '.join(names)}")
            v = float(val)
            if not (np.isfinite(v) and v >= 0.0):
                raise ValueError(f"the error of {key} must be finite and "
                                 f">= 0; got {val!r}")
            out[m, names.index(key)] = v
    return out


def parameter_sensitivities(z, params, band_edge=None, kx: float = 0.0,
                            ky: float = 0.0, potential=None,
                            n_states: int = 8, strain=None, sigmas=None,
                            sigma_band_edge_scale=None) -> dict:
    """Exact derivatives of the top subband energies with respect to
    every material parameter (Hellmann-Feynman), and optionally their
    propagated errors (new in v0.15.0).

    z : uniform grid (nm).
    params : one `WurtziteParameters` (one material between hard walls,
        as `solve_subbands` takes it), or one parameter set per grid
        point (a layered stack, as `solve_heterostructure` takes it, for
        example from `layered_profile`). The materials are the distinct
        parameter-set objects, in the order they first appear.
    band_edge, kx, ky, potential, n_states, strain : as in
        `solve_heterostructure` (``strain`` may be one tensor, or one
        per grid point for a stack); without ``band_edge`` and with one
        parameter set, exactly the matrix of `solve_subbands`.
    sigmas : optional 1-sigma parameter errors, as a mapping such as
        ``{"A1": 0.12, "delta1": 0.002}`` (A's dimensionless, deltas in
        eV) for a single material, or a list with one mapping per
        material (in the order of the result's ``materials``). Missing
        parameters count as exact. The errors are taken as independent;
        for correlated errors use the returned derivatives with your
        covariance, C = J C_theta J^T.
    sigma_band_edge_scale : optional 1-sigma error of a common factor on
        the whole band-edge profile (0.1 for a 10 % uncertain offset).

    Returns a dict with
      energies : (n_states,) eV, descending, as the solvers return them.
      names : `PARAMETER_NAMES`.  materials : list of the parameter sets.
      derivatives : (n_states, n_materials, 9) dE/dtheta (eV per unit
        of the parameter: per unit of A, and eV per eV for the deltas).
      d_band_edge_scale : (n_states,) dE/ds for band_edge -> s *
        band_edge at s = 1 (eV), or None without a band edge. For a
        profile offset * shape, dE/d(offset) = d_band_edge_scale /
        offset.
      level_split : (n_states,) bool, True where a parameter splits a
        level of several states (then the value is the derivative of
        the level's mean energy; see the module docstring).
      covariance : (n_states, n_states) eV^2 and sigma_energies :
        (n_states,) eV, when errors were given; otherwise None.
    """
    z = check_grid(z)
    n = z.size
    n_states = check_n_states(n_states, n)
    potential = check_potential(potential, n)
    single = isinstance(params, WurtziteParameters)
    kx, ky = float(kx), float(ky)
    if not (np.isfinite(kx) and np.isfinite(ky)):
        raise ValueError("kx and ky must be finite (nm^-1)")
    if single and band_edge is None and (
            strain is None or np.shape(strain) == (3, 3)):
        params_list = None
        materials = [params]
        H = assemble_hamiltonian(params, z, kx, ky, potential,
                                 strain=strain)
    else:
        params_list = [params] * n if single else list(params)
        if len(params_list) != n:
            raise ValueError("one parameter set per grid point required "
                             f"({n}); got {len(params_list)}")
        if not all(isinstance(p, WurtziteParameters) for p in params_list):
            raise ValueError("params must be WurtziteParameters, one set "
                             "or one per grid point")
        materials = []
        for p in params_list:
            if not any(p is m for m in materials):
                materials.append(p)
        H = assemble_heterostructure(z, params_list, band_edge, kx, ky,
                                     potential, strain_list=strain)
    names = list(PARAMETER_NAMES)

    # eigenpairs, with enough extra states that every level touching
    # the first n_states is complete
    scale = max(1.0, float(np.abs(H).max()))
    tol = 1e-10 * scale
    m = min(6 * n, n_states + 6)
    while True:
        vals, X = top_eigenpairs(H, n, m, kx, ky)
        clusters = _clusters(vals, tol)
        last = [c for c in clusters if c[0] < n_states][-1]
        if m == 6 * n or last[-1] < m - 1:
            break
        m = min(6 * n, 2 * m)
    clusters = [c for c in clusters if c[0] < n_states]

    Xc = X[:, :clusters[-1][-1] + 1]

    def projected(Hd=None, diag=None):
        """Level-mean derivative of each state and whether dH/dtheta
        (the matrix Hd, or the diagonal matrix diag) splits its level."""
        if diag is None:
            Y = Hd @ Xc
            big = float(np.abs(Hd).max())
        else:
            Y = diag[:, None] * Xc
            big = float(np.abs(diag).max())
        G = Xc.conj().T @ Y
        d = np.empty(n_states)
        split = np.zeros(n_states, dtype=bool)
        lim = 1e-9 * max(1.0, big)
        for c in clusters:
            block = G[np.ix_(c, c)]
            mean = float(np.trace(block).real) / c.size
            if c.size > 1:
                ev = np.linalg.eigvalsh(0.5 * (block + block.conj().T))
                split_c = bool(ev[-1] - ev[0] > lim)
            else:
                split_c = False
            keep = c[c < n_states]
            d[keep] = mean
            split[keep] = split_c
        return d, split

    deriv = np.empty((n_states, len(materials), len(names)))
    level_split = np.zeros(n_states, dtype=bool)
    zero = _unit_set(None)
    for k, name in enumerate(names):
        unit = _unit_set(name)
        for j, mat in enumerate(materials):
            if params_list is None:
                Hd = assemble_hamiltonian(unit, z, kx, ky)
            else:
                plist = [unit if p is mat else zero for p in params_list]
                Hd = assemble_heterostructure(z, plist, None, kx, ky)
            deriv[:, j, k], s = projected(Hd)
            level_split |= s
    d_edge = None
    if band_edge is not None:
        be = np.asarray(band_edge, dtype=float)
        d_edge, s = projected(diag=np.repeat(be, 6))
        level_split |= s

    cov = sig = None
    if sigmas is not None or sigma_band_edge_scale is not None:
        sv = (_sigma_vector(sigmas, materials, names) if sigmas is not None
              else np.zeros((len(materials), len(names))))
        J = deriv.reshape(n_states, -1)
        var = sv.reshape(-1) ** 2
        cov = (J * var[None, :]) @ J.T
        if sigma_band_edge_scale is not None:
            if d_edge is None:
                raise ValueError("sigma_band_edge_scale needs a band_edge")
            se = float(sigma_band_edge_scale)
            if not (np.isfinite(se) and se >= 0.0):
                raise ValueError("sigma_band_edge_scale must be finite "
                                 "and >= 0")
            cov = cov + se ** 2 * np.outer(d_edge, d_edge)
        sig = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    return {"energies": vals[:n_states].copy(),
            "names": PARAMETER_NAMES, "materials": materials,
            "derivatives": deriv, "d_band_edge_scale": d_edge,
            "level_split": level_split, "covariance": cov,
            "sigma_energies": sig}
