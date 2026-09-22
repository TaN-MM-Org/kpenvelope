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

__all__ = ["sheet_density_from_cm2", "sheet_density_to_cm2"]


def sheet_density_from_cm2(ps_cm2):
    """Sheet density in nm^-2 from a value in cm^-2 (x 1e-14)."""
    return ps_cm2 * 1e-14


def sheet_density_to_cm2(ps_nm2):
    """Sheet density in cm^-2 from a value in nm^-2 (x 1e14)."""
    return ps_nm2 * 1e14
