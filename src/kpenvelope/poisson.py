"""Electrostatics of a polarization-bound hole gas.

The confining potential is not imposed: the gas is balanced by the fixed
polarization bound charge at the interface (z = 0), and Gauss's law gives
the hole potential energy

    dV_h/dz = (e^2 / (eps_r eps_0)) * ( ps - integral_0^z p(z') dz' )

which is positive at the interface (holes are held against it) and
vanishes once the whole gas lies below z. Units: eV, nm, densities nm^-2
(sheet) and nm^-3 (volume).
"""
from __future__ import annotations

import numpy as np

# e^2 / (4 pi eps0) = 1.4399645 eV nm  ->  e^2/eps0 = 4*pi*that
E2_OVER_EPS0 = 4.0 * np.pi * 1.4399645


def hole_potential(z, p_density, ps, eps_r):
    """Integrate the hole potential energy V_h(z) from the density profile.

    z : uniform grid (nm), starting at the interface z=0.
    p_density : hole volume density on the grid (nm^-3).
    ps : total sheet density (nm^-2).
    """
    z = np.asarray(z, dtype=float)
    dz = z[1] - z[0]
    cum = np.cumsum(p_density) * dz          # integral_0^z p
    field = (E2_OVER_EPS0 / eps_r) * (ps - cum)
    vh = np.concatenate([[0.0], np.cumsum(0.5 * (field[1:] + field[:-1]) * dz)])
    return vh
