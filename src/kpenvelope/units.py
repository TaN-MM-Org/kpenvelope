"""Unit conversions between laboratory and package conventions.

The package works in nm, eV and nm^-2 throughout; laboratories quote
sheet densities in cm^-2. The conversion is an exact power of ten
(1 cm = 1e7 nm, so 1 cm^-2 = 1e-14 nm^-2) -- these two helpers exist
because the factor of 1e-14 is the single most common way to be wrong
by orders of magnitude when driving the solver from measured numbers.
Each conversion is one floating-point multiplication, so a round trip
can differ from the input in the last bit (a relative error of order
1e-16). The tests assert the worked value used throughout the
documentation (4.6e13 cm^-2 = 0.46 nm^-2, both ways) and one round
trip (1.2345e13 cm^-2), each with ==.
"""
from __future__ import annotations

import numpy as np

__all__ = ["sheet_density_from_cm2", "sheet_density_to_cm2",
           "energy_from_wavenumber", "energy_to_wavenumber"]


def sheet_density_from_cm2(ps_cm2):
    """Sheet density in nm^-2 from a value in cm^-2 (x 1e-14)."""
    return ps_cm2 * 1e-14


def sheet_density_to_cm2(ps_nm2):
    """Sheet density in cm^-2 from a value in nm^-2 (x 1e14)."""
    return ps_nm2 * 1e14


# h c / e in eV m, from the exact 2019 SI values of h, c and e
_HC_EV_M = 6.62607015e-34 * 299792458.0 / 1.602176634e-19


def energy_from_wavenumber(nu_cm1):
    """Photon energy in eV from a wavenumber in cm^-1 (new in v0.13.0).

    Infrared spectrometers (FTIR) report intersubband lines in cm^-1;
    the package works in eV. E = h c nu, with h, c and e exact in the
    2019 SI, so 1000 cm^-1 = 0.12398419843320026 eV. Scalar or array.
    """
    nu = np.asarray(nu_cm1, dtype=float)
    out = nu * (_HC_EV_M * 100.0)
    return float(out) if out.ndim == 0 else out


def energy_to_wavenumber(e_ev):
    """Wavenumber in cm^-1 of a photon energy in eV (new in v0.13.0);
    the inverse of `energy_from_wavenumber`."""
    e = np.asarray(e_ev, dtype=float)
    out = e / (_HC_EV_M * 100.0)
    return float(out) if out.ndim == 0 else out
