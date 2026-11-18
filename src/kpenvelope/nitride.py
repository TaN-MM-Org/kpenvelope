"""Cited material numbers for c-plane (Al,Ga)N layers (new in v0.17.0).

Everything the interface-charge and electron modules need about a
nitride layer lives in one record, `NitrideMaterial`, and every number
in it carries its source. `algan(x)` builds the record of relaxed
Al(x)Ga(1-x)N from the cited end points; any other layer (InGaN,
ScAlN, a different parameter set) can be described by filling a
`NitrideMaterial` yourself, with its `reference` field naming the
source, exactly as for `WurtziteParameters`.

Two polarization sets are shipped, because the choice matters and the
literature has not settled it:

* ``"pbesol-mahim2026"``: Berry-phase calculations with the PBEsol
  functional (Quantum ESPRESSO 6.7), archived with every input and output
  file in T. M. Mahim, nitride-misfit-polarization, version 1.3.1,
  Zenodo, doi:10.5281/zenodo.24315036 (results/dft_polarization.json;
  full precision is kept here). P_H is the formal polarization on the
  branch of the layered-hexagonal (H) reference, e31 is the proper
  constant with relaxed ions, e33 the clamped-area constant, and
  C13/C33 comes from biaxial strains of +-1 percent with c and u relaxed.
* ``"hse-dreyer2016"``: hybrid-functional (HSE) values of C. E. Dreyer,
  A. Janotti, C. G. Van de Walle and D. Vanderbilt, Phys. Rev. X 6,
  021038 (2016), Table I (P_eff, H reference) and Table II (proper e31,
  e33), copied without change. That paper gives no C13/C33, so this set
  takes the elastic ratio from the PBEsol set above unless you pass your
  own (`algan(..., c13_over_c33=...)`), and says so in its reference.

The other numbers of `algan` are those of the analysis the misfit-line
module reproduces (nitride-misfit-polarization, src/nmp/materials.py):

* I. Vurgaftman and J. R. Meyer, J. Appl. Phys. 94, 3675 (2003), Table I:
  in-plane lattice constants at 300 K (GaN 3.189, AlN 3.112 Angstrom),
  band gaps at 0 K (GaN 3.510, AlN 6.25 eV) with Varshni alpha (0.909,
  1.799 meV/K) and beta (830, 1462 K), electron masses along c (GaN 0.20,
  AlN 0.32 m0); AlGaN gap bowing 0.7 eV (Table III).
* O. Ambacher et al., J. Appl. Phys. 87, 334 (2000): static dielectric
  constants along c, 10.4 (GaN) and 10.7 (AlN) (Table I), and the
  conduction-band offset rule Delta E_c = 0.7 [E_g(x) - E_g(0)] (Eq. (19)).

Alloys use linear interpolation (Vegard) between the end points, except
the gap, which has the bowing above. Optional knobs (`cbo_fraction`,
`psp_bowing`) exist for sensitivity tests and are off by default.
"""
from __future__ import annotations

import dataclasses
import math

__all__ = ["NitrideMaterial", "algan", "POLARIZATION_SETS",
           "polarization_quantum", "zinc_blende_formal_polarization",
           "QE", "E_PER_NM2_PER_C_M2"]

#: elementary charge (C), exact in the 2019 SI
QE = 1.602176634e-19
#: 1 C/m^2 expressed in elementary charges per nm^2
E_PER_NM2_PER_C_M2 = 1.0 / QE * 1e-18

_PBESOL = ("T. M. Mahim, nitride-misfit-polarization v1.3.1, Zenodo, "
           "doi:10.5281/zenodo.24315036 (PBEsol Berry-phase calculations, "
           "results/dft_polarization.json)")
_HSE = ("C. E. Dreyer, A. Janotti, C. G. Van de Walle and D. Vanderbilt, "
        "Phys. Rev. X 6, 021038 (2016), Tables I and II (HSE)")

#: Polarization constants (C/m^2): P_H (H reference), proper e31, e33,
#: and C13/C33 (dimensionless; None where the source gives none).
POLARIZATION_SETS = {
    "pbesol-mahim2026": {
        "reference": _PBESOL,
        "GaN": {"P_H": 1.3368546000000001, "e31": -0.4387631999999933,
                "e33": 0.7621140000000092,
                "c13_over_c33": 0.24177224254593732},
        "AlN": {"P_H": 1.3329925999999999, "e31": -0.6648395000000004,
                "e33": 1.640010000000004,
                "c13_over_c33": 0.30257225299928636},
    },
    "hse-dreyer2016": {
        "reference": _HSE,
        "GaN": {"P_H": 1.312, "e31": -0.551, "e33": 1.020,
                "c13_over_c33": None},
        "AlN": {"P_H": 1.351, "e31": -0.676, "e33": 1.569,
                "c13_over_c33": None},
        "InN": {"P_H": 1.026, "e31": -0.604, "e33": 1.238,
                "c13_over_c33": None},
    },
}

# Vurgaftman and Meyer (2003) and Ambacher et al. (2000); see the module
# docstring. Lattice constants in nm.
_A300_NM = {"GaN": 0.3189, "AlN": 0.3112}
_EG0 = {"GaN": 3.510, "AlN": 6.25}
_VARSHNI = {"GaN": (0.909e-3, 830.0), "AlN": (1.799e-3, 1462.0)}
_ME_PAR = {"GaN": 0.20, "AlN": 0.32}
_EPS33 = {"GaN": 10.4, "AlN": 10.7}
_BOW_EG = 0.7
_CBO = 0.7
_BAND_REF = ("I. Vurgaftman and J. R. Meyer, J. Appl. Phys. 94, 3675 (2003) "
             "(lattice constants, gaps, Varshni, masses, gap bowing); "
             "O. Ambacher et al., J. Appl. Phys. 87, 334 (2000) (dielectric "
             "constants, conduction-band offset rule)")


@dataclasses.dataclass(frozen=True)
class NitrideMaterial:
    """One relaxed c-plane wurtzite layer material, every number cited.

    name : label used in messages.
    a : in-plane lattice constant of the relaxed material (nm). It is also
        the in-plane edge component b of its misfit dislocations.
    P_H : formal spontaneous polarization along [0001] on the branch of the
        layered-hexagonal reference (C/m^2), metal-polar sign convention.
    e31 : PROPER piezoelectric constant (C/m^2). The improper response
        e31 - P_H is formed where it is needed; do not pass the improper
        value here.
    e33 : piezoelectric constant along c (C/m^2).
    c13_over_c33 : elastic ratio; the c-axis strain of a biaxially strained
        layer is -2 (C13/C33) eps.
    reference : provenance of the numbers. Required, on purpose.
    eg : band gap (eV) at the temperature the record was built for, or NaN.
    ec : conduction-band edge relative to that of relaxed GaN (eV), or NaN.
    me : electron mass along c (m0), or NaN.
    eps_r : static relative permittivity along c, or NaN.

    Band numbers may be NaN for a layer used only for its charge; the
    electron solver refuses a NaN where it needs a number.
    """

    name: str
    a: float
    P_H: float
    e31: float
    e33: float
    c13_over_c33: float
    reference: str
    eg: float = math.nan
    ec: float = math.nan
    me: float = math.nan
    eps_r: float = math.nan

    def __post_init__(self):
        for f in ("a", "P_H", "e31", "e33", "c13_over_c33"):
            v = getattr(self, f)
            if not (isinstance(v, (int, float)) and math.isfinite(v)):
                raise ValueError(f"NitrideMaterial.{f} must be a finite "
                                 f"number; got {v!r}")
        if self.a <= 0.0:
            raise ValueError("the lattice constant a (nm) must be positive")
        if self.c13_over_c33 < 0.0:
            raise ValueError("C13/C33 must be >= 0")
        if not (isinstance(self.reference, str) and self.reference.strip()):
            raise ValueError("NitrideMaterial needs a non-empty reference "
                             "naming the source of its numbers")
        for f in ("me", "eps_r"):
            v = getattr(self, f)
            if math.isfinite(v) and v <= 0.0:
                raise ValueError(f"NitrideMaterial.{f} must be positive "
                                 "(or NaN when not needed)")


def _gap(name, T):
    a, b = _VARSHNI[name]
    return _EG0[name] - a * T * T / (T + b)


def algan(x: float, temperature_K: float = 300.0,
          polarization: str = "pbesol-mahim2026",
          c13_over_c33: float = None, cbo_fraction: float = _CBO,
          psp_bowing: float = 0.0) -> NitrideMaterial:
    """Relaxed Al(x)Ga(1-x)N, 0 <= x <= 1, with cited end points.

    polarization : ``"pbesol-mahim2026"`` (default) or ``"hse-dreyer2016"``
        (see the module docstring).
    c13_over_c33 : override of the elastic ratio. Needed nowhere for the
        PBEsol set; for the HSE set the PBEsol ratio is used when None, as
        in the analysis the module reproduces, and the reference says so.
    cbo_fraction : share of the gap difference in the conduction band
        (0.7, Ambacher et al.); for sensitivity tests.
    psp_bowing : bowing of the spontaneous polarization (C/m^2), added as
        psp_bowing * x (1 - x); 0 by default. Vurgaftman and Meyer recommend
        a bowing that corresponds to +0.021 C/m^2 in this convention.
    """
    if not (isinstance(x, (int, float)) and 0.0 <= x <= 1.0):
        raise ValueError(f"the Al fraction x must be in [0, 1]; got {x!r}")
    if not (math.isfinite(temperature_K) and temperature_K > 0.0):
        raise ValueError("temperature_K must be finite and positive")
    if polarization not in POLARIZATION_SETS:
        raise ValueError(f"unknown polarization set {polarization!r}; "
                         f"choose from {sorted(POLARIZATION_SETS)}")
    if not (math.isfinite(cbo_fraction) and 0.0 < cbo_fraction < 1.0):
        raise ValueError("cbo_fraction must lie strictly between 0 and 1")
    ps = POLARIZATION_SETS[polarization]
    ref = ps["reference"]
    el = {m: ps[m]["c13_over_c33"] for m in ("GaN", "AlN")}
    if c13_over_c33 is not None:
        if not (math.isfinite(c13_over_c33) and c13_over_c33 >= 0.0):
            raise ValueError("c13_over_c33 must be finite and >= 0")
        ratio = float(c13_over_c33)
        ref += "; C13/C33 = %r supplied by the caller" % ratio
    elif el["GaN"] is None:
        pb = POLARIZATION_SETS["pbesol-mahim2026"]
        el = {m: pb[m]["c13_over_c33"] for m in ("GaN", "AlN")}
        ratio = x * el["AlN"] + (1 - x) * el["GaN"]
        ref += ("; C13/C33 not given by that source, taken from "
                + _PBESOL)
    else:
        ratio = x * el["AlN"] + (1 - x) * el["GaN"]

    def lin(f):
        return x * f("AlN") + (1 - x) * f("GaN")

    T = float(temperature_K)
    eg = lin(lambda n: _gap(n, T)) - _BOW_EG * x * (1 - x)
    return NitrideMaterial(
        name="Al%gGa%gN" % (x, 1 - x) if 0 < x < 1 else ("AlN" if x == 1 else "GaN"),
        a=lin(lambda n: _A300_NM[n]),
        P_H=lin(lambda n: ps[n]["P_H"]) + psp_bowing * x * (1 - x),
        e31=lin(lambda n: ps[n]["e31"]),
        e33=lin(lambda n: ps[n]["e33"]),
        c13_over_c33=ratio,
        reference=ref + "; " + _BAND_REF,
        eg=eg,
        ec=cbo_fraction * (eg - _gap("GaN", T)),
        me=lin(lambda n: _ME_PAR[n]),
        eps_r=lin(lambda n: _EPS33[n]),
    )


def polarization_quantum(a_nm: float) -> float:
    """Polarization quantum along c, e c / Omega = 2 e / (sqrt(3) a^2) (C/m^2)."""
    a = a_nm * 1e-9
    return 2.0 * QE / (math.sqrt(3.0) * a * a)


def zinc_blende_formal_polarization(a_nm: float) -> float:
    """Formal polarization of the zinc-blende reference in the hexagonal
    setting, e sqrt(3) / (2 a^2) = (3/4) of the quantum (C/m^2).

    P_ZB = P_H - this (Dreyer et al., Phys. Rev. X 6, 021038 (2016)). It is
    also exactly the charge per area removed by emptying one 3/4-filled
    cation dangling bond per surface site (electron counting).
    """
    a = a_nm * 1e-9
    return QE * math.sqrt(3.0) / (2.0 * a * a)
