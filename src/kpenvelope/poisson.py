"""Electrostatics of a polarization-bound hole gas.

The confining potential is not imposed: the gas is balanced by a fixed
negative sheet charge of density ps (the polarization bound charge of
the interface), and Gauss's law for the displacement field gives the
hole potential energy

    dV_h/dz = (e^2 / (eps_r(z) eps_0)) * ( ps * Theta(z - z_s)
                                           - integral_{z_0}^{z} p(z') dz' )

with zero field to the left of all charge. It is positive just past
the sheet (holes are held against it) and vanishes once the whole gas
lies below z. Units: eV, nm, densities nm^-2 (sheet) and nm^-3
(volume).

z_s is the position of the fixed sheet. By default it is the first grid
point z_0, the historical behaviour, which is right when the grid
starts at the interface (hard-wall solver). For a finite-barrier stack
whose grid starts inside a barrier, pass the interface position as
``sheet_z`` (new in v0.12.0): otherwise the whole left barrier carries
the sheet's full field. eps_r may be one number or one value per grid
point (new in v0.12.0); the displacement field D = eps0 eps_r E is what
the charges fix, so a layer of lower permittivity carries a stronger
electric field.
"""
from __future__ import annotations

import numpy as np

# e^2 / (4 pi eps0) = 1.4399645 eV nm  ->  e^2/eps0 = 4*pi*that
E2_OVER_EPS0 = 4.0 * np.pi * 1.4399645


def _check_eps(eps_r, n):
    """Validate a scalar or per-point permittivity; return it unchanged
    (a scalar stays a Python/NumPy scalar)."""
    eps = np.asarray(eps_r, dtype=float)
    if eps.ndim == 0:
        if not (np.isfinite(eps) and eps > 0.0):
            raise ValueError(
                "eps_r must be a finite positive number (a cited value); "
                f"got {float(eps)!r}")
        return eps_r
    if eps.shape != (n,):
        raise ValueError("a per-point eps_r must have one value per grid "
                         f"point ({n}); got shape {eps.shape}")
    if not (np.all(np.isfinite(eps)) and np.all(eps > 0.0)):
        raise ValueError(
            "every eps_r value must be finite and positive; a NaN usually "
            "means a barrier-only parameter set (e.g. aln_rinke2008) with "
            "no vetted permittivity -- supply a cited value for it")
    return eps


def sheet_mask(z, sheet_z):
    """1.0 at grid points at or right of the fixed sheet, else 0.0.

    None means the sheet sits at the first grid point (every point is
    right of it). A point within 1e-9 grid steps of ``sheet_z`` counts
    as right of it, so an interface given in nm lands on its grid node
    despite rounding in ``numpy.linspace``.
    """
    z = np.asarray(z, dtype=float)
    if sheet_z is None:
        return np.ones_like(z)
    sz = float(sheet_z)
    dz = z[1] - z[0]
    if not (np.isfinite(sz) and z[0] - 1e-9 * dz <= sz <= z[-1]):
        raise ValueError(
            f"sheet_z = {sz!r} nm must lie on the grid span "
            f"[{z[0]}, {z[-1]}] nm")
    return (z >= sz - 1e-9 * dz).astype(float)


def hole_potential(z, p_density, ps, eps_r, sheet_z=None):
    """Integrate the hole potential energy V_h(z) from the density profile.

    z : uniform grid (nm).
    p_density : hole volume density on the grid (nm^-3).
    ps : density of the fixed negative sheet charge (nm^-2).
    eps_r : static relative permittivity, one number or one value per
        grid point (new in v0.12.0).
    sheet_z : position of the fixed sheet (nm); None (default) puts it
        at z[0], the historical behaviour (new in v0.12.0).

    V_h(z[0]) = 0. With a scalar eps_r and sheet_z=None the arithmetic
    is exactly that of earlier versions.
    """
    z = np.asarray(z, dtype=float)
    dz = z[1] - z[0]
    eps = _check_eps(eps_r, z.size)
    cum = np.cumsum(p_density) * dz          # integral_0^z p
    if sheet_z is None:
        fixed = ps
    else:
        fixed = ps * sheet_mask(z, sheet_z)
    field = (E2_OVER_EPS0 / eps) * (fixed - cum)
    vh = np.concatenate([[0.0], np.cumsum(0.5 * (field[1:] + field[:-1]) * dz)])
    return vh
