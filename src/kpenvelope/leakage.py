"""Current over and through barriers: thermionic emission and tunneling
(new in v0.18.0).

Electrons cross a barrier in two ways: over its top (thermionic
emission) and through it (tunneling). Both follow from one formula, that
of R. Tsu and L. Esaki, Appl. Phys. Lett. 22, 562 (1973), for a barrier
between two reservoirs with Fermi levels F_1 and F_2:

    J = (A* T / k) int T(E) ln[(1 + exp((F_1 - E)/kT)) /
                               (1 + exp((F_2 - E)/kT))] dE,

with the Richardson constant A* = q m k^2 / (2 pi^2 hbar^3) of the
electron mass m (120.17 A cm^-2 K^-2 for the free electron mass) and the
transmission T(E) of the barrier at energy E. Here T(E) is the WKB
(Wentzel-Kramers-Brillouin) transmission, exp(-2 int kappa dz) with
kappa = sqrt(2 m (E_c(z) - E)) / hbar over the region where the band lies
above E, and 1 above the top of the barrier. Over a thick barrier only
the top counts and the formula becomes the thermionic-emission law
J = A* T^2 exp(-phi / kT) [exp(qV / kT) - 1] (checked in the tests); for a
triangular barrier at low temperature it gives the exponent of
Fowler-Nordheim tunneling (checked too).

`gate_leakage` applies this to the band profile under a metal gate, from
the electron layer up through the barrier into the metal, at each gate
voltage, using the quasi-equilibrium bands of `solve_quasi_equilibrium`
(the leakage is assumed small enough not to change the bands). The drift-
diffusion solver can also use thermionic emission as the boundary
condition of a metal contact (`Contact(..., thermionic=True)`).

Limits: one band with one mass along the path (the barrier's, or the one
you give), no image-force lowering of the barrier, no trap-assisted or
defect-assisted current, WKB transmission (no resonances, no reflection
above the barrier).
"""
from __future__ import annotations

import math

import numpy as np

from .electrons import HBAR, KB, M0
from .nitride import QE

__all__ = ["richardson_constant", "richardson_velocity",
           "thermionic_emission_current", "wkb_transmission",
           "tsu_esaki_current", "gate_leakage"]

_KB_SI = KB * QE           # J/K


def richardson_constant(mass_m0: float) -> float:
    """A* = q m k^2 / (2 pi^2 hbar^3), in A cm^-2 K^-2."""
    if not (math.isfinite(mass_m0) and mass_m0 > 0):
        raise ValueError("mass_m0 must be > 0")
    m = mass_m0 * M0
    return QE * m * _KB_SI ** 2 / (2 * math.pi ** 2 * HBAR ** 3) * 1e-4


def richardson_velocity(mass_m0: float, temperature_K: float) -> float:
    """v_R = A* T^2 / (q N_c) = sqrt(kT / (2 pi m)), in cm/s."""
    if not (math.isfinite(temperature_K) and temperature_K > 0):
        raise ValueError("temperature_K must be > 0")
    return math.sqrt(_KB_SI * temperature_K / (2 * math.pi * mass_m0 * M0)) * 100.0


def thermionic_emission_current(barrier_eV: float, voltage: float,
                                mass_m0: float, temperature_K: float = 300.0):
    """J = A* T^2 exp(-barrier / kT) [exp(V / kT) - 1] (A/cm^2), V positive
    forward."""
    kT = KB * temperature_K
    return richardson_constant(mass_m0) * temperature_K ** 2 * math.exp(
        -barrier_eV / kT) * math.expm1(voltage / kT)


def wkb_transmission(z_nm, ec_eV, energy_eV, mass_m0):
    """WKB transmission exp(-2 int kappa dz) of the band profile ec(z)
    (eV, on the grid z in nm) at energy E (eV); 1 where the band nowhere
    exceeds E. mass_m0 may be one number or one per grid point."""
    z = np.asarray(z_nm, float) * 1e-9
    ec = np.asarray(ec_eV, float)
    m = np.broadcast_to(np.asarray(mass_m0, float), ec.shape) * M0
    E = np.atleast_1d(np.asarray(energy_eV, float))
    dE = np.clip(ec[None, :] - E[:, None], 0.0, None) * QE
    kap = np.sqrt(2.0 * m[None, :] * dE) / HBAR
    integral = np.sum(0.5 * (kap[:, 1:] + kap[:, :-1]) * np.diff(z)[None, :], axis=1)
    t = np.exp(-2.0 * integral)
    return t if np.ndim(energy_eV) else float(t[0])


def tsu_esaki_current(z_nm, ec_eV, fermi_left: float, fermi_right: float,
                      mass_m0: float, temperature_K: float = 300.0,
                      n_energy: int = 4000, transport_mass=None):
    """Current density (A/cm^2) from the left reservoir to the right one
    through the band profile ec(z) (positive when electrons flow from left
    to right, so the electric current flows from right to left).

    mass_m0 : mass in the barrier (one number or one per point), used for
        kappa. transport_mass : the mass of the supply function (A*);
        default mass_m0 (or its mean if it varies).
    The energy runs from the higher of the two band edges at the ends up
    to 40 kT above the barrier top and the higher Fermi level.
    """
    ec = np.asarray(ec_eV, float)
    T = float(temperature_K)
    kT = KB * T
    mt = transport_mass if transport_mass is not None else float(np.mean(mass_m0))
    e_lo = max(ec[0], ec[-1])
    e_hi = max(ec.max(), fermi_left, fermi_right) + 40 * kT
    E = np.linspace(e_lo, e_hi, n_energy)
    tr = wkb_transmission(z_nm, ec, E, mass_m0)
    supply = (np.logaddexp(0.0, (fermi_left - E) / kT)
              - np.logaddexp(0.0, (fermi_right - E) / kT))
    trap = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    return richardson_constant(mt) * T / KB * float(trap(tr * supply, E))


def gate_leakage(layers, barrier: float, voltages, temperature_K: float = 300.0,
                 mass_m0=None, dz: float = 0.2, line_charge=None,
                 reference: str = "H", interfaces=None, extra_sheet=None,
                 n_energy: int = 4000):
    """Gate leakage of a metal on top of the stack, thermionic emission and
    tunneling together, at each gate voltage (A/cm^2). The sign is that of
    a diode: positive under forward bias (electrons flow from the stack
    into the metal), negative under reverse bias (electrons flow from the
    metal into the stack, the usual leakage of a negative gate).

    The bands come from `solve_quasi_equilibrium` (electrons at the Fermi
    level of the side contacts, zero field at the bottom). The path runs
    from the deepest point of the conduction band in the top layers (the
    electron layer) up to the metal; electrons there have the Fermi level
    0, the metal -V.

    mass_m0 : electron mass in the barrier (default: the mass of each
        layer's record, point by point).
    Below the switch-off voltage the electron layer is gone; in this model
    the bands then stop changing with the gate (nothing is left to
    deplete), so the leakage stops changing too. Trust it above the
    switch-off voltage.
    Returns (voltages, current density, list of quasi-equilibrium results).
    """
    from .driftdiffusion import Contact, solve_quasi_equilibrium

    res, J, prev = [], [], None
    for v in voltages:
        r = solve_quasi_equilibrium(layers, Contact("schottky", barrier=barrier,
                                                    voltage=float(v)),
                                    Contact("zero-field"), temperature_K=temperature_K,
                                    dz=dz, guess=prev, line_charge=line_charge,
                                    reference=reference, interfaces=interfaces,
                                    extra_sheet=extra_sheet)
        prev = r
        z, ec = r.z, r.ec
        # the electron layer: the lowest conduction band in the top part
        # of the stack (below the barrier the band rises again into the
        # buffer, which does not take part)
        top = z[-1]
        region = z >= top - min(top, 200.0)
        i0 = int(np.argmin(np.where(region, ec, np.inf)))
        zz, ee = z[i0:], ec[i0:]
        if mass_m0 is None:
            edges = np.cumsum([0.0] + [L.thickness for L in layers])
            idx = np.clip(np.searchsorted(edges, zz, side="right") - 1, 0,
                          len(layers) - 1)
            m = np.array([layers[i].material.me for i in idx])
        else:
            m = mass_m0
        # metal Fermi level -V; the metal side of the path is the metal's
        # conduction band, deep below; Tsu-Esaki with the metal at the end
        ee = np.append(ee, ee[-1] - 50.0)      # metal: no barrier on its side
        zz = np.append(zz, zz[-1] + 1e-6)
        mm = np.append(np.broadcast_to(m, zz[:-1].shape), np.atleast_1d(m)[-1])
        j = tsu_esaki_current(zz, ee, 0.0, -float(v), mm, temperature_K,
                              n_energy=n_energy, transport_mass=float(np.atleast_1d(m)[0]))
        J.append(j)
        res.append(r)
    return np.asarray(voltages, float), np.array(J), res
