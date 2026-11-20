"""Cited material numbers for c-plane nitride layers: AlGaN, InGaN, InAlN
and ScAlN (AlGaN new in v0.17.0; the others new in v0.18.0).

Everything the interface-charge and electron modules need about a
nitride layer lives in one record, `NitrideMaterial`, and every number
in it carries its source. The builders

* `algan(x)`  Al(x)Ga(1-x)N, 0 <= x <= 1,
* `ingan(x)`  In(x)Ga(1-x)N, 0 <= x <= 1,
* `inaln(x)`  In(x)Al(1-x)N, 0 <= x <= 1,
* `scaln(x)`  Sc(x)Al(1-x)N, 0 <= x <= 0.5 (the range of its source),
* `gan(...)`  the GaN partner of any of the sets below,

make the record of a relaxed alloy from cited end points. Any other layer
can be described by filling a `NitrideMaterial` yourself, and any number
of a built record can be swapped with `with_numbers`, which adds your
source to the record's reference.

Polarization sets. The interface charge is a small difference of large
polarizations, so the set matters, and layers from different sets should
not meet at an interface (`interface_charges` warns when they do). Every
record carries the name of its set in `parameter_set`.

* ``"pbesol-mahim2026"`` (GaN, AlN): Berry-phase calculations with the
  PBEsol functional (Quantum ESPRESSO 6.7), archived with every input and
  output file in T. M. Mahim, nitride-misfit-polarization, version 1.3.1,
  Zenodo, doi:10.5281/zenodo.24315036 (results/dft_polarization.json;
  full precision is kept here). P_H is the formal polarization on the
  branch of the layered-hexagonal (H) reference, e31 is the proper
  constant with relaxed ions, e33 the clamped-area constant, and
  C13/C33 comes from biaxial strains of +-1 percent with c and u relaxed.
* ``"hse-dreyer2016"`` (GaN, AlN, InN): hybrid-functional (HSE) values of
  C. E. Dreyer, A. Janotti, C. G. Van de Walle and D. Vanderbilt, Phys.
  Rev. X 6, 021038 (2016), Table I (P_eff, H reference) and Table II
  (proper e31, e33), copied without change. That paper gives no C13/C33:
  `algan` takes the ratio from the PBEsol set (as in the analysis it
  reproduces), `ingan` and `inaln` from the elastic constants of
  Vurgaftman and Meyer (below), and each record says which.
* ``"vurgaftman2003"`` (GaN, AlN, InN): I. Vurgaftman and J. R. Meyer,
  J. Appl. Phys. 94, 3675 (2003), Table I: spontaneous polarization with
  the zinc-blende (ZB) reference (GaN -0.034, AlN -0.090, InN -0.042
  C/m^2), elastic constants c11, c12, c13, c33 and piezoelectric moduli
  d13, d33, and the polarization bowings of Table III. The piezoelectric
  constants follow from the moduli by e31 = (c11 + c12) d13 + c13 d33 and
  e33 = 2 c13 d13 + c33 d33 (O. Ambacher et al., J. Appl. Phys. 130,
  045102 (2021), Eqs. (40a), (40b)); they are used as proper constants.
* ``"ambacher2021"`` (Sc(x)Al(1-x)N, and GaN for a relaxed buffer):
  O. Ambacher, B. Christian, N. Feil, D. F. Urban, C. Elsasser,
  M. Prescher and L. Kirste, J. Appl. Phys. 130, 045102 (2021):
  a(x) Eq. (3a), P_SP(x) Eq. (18c) (ZB reference, after Bernardini et al.
  and Caro et al.), C13(x), C33(x) Eqs. (23c), (23d), e31(x), e33(x)
  Eqs. (39b), (39c), eps33(x) Eq. (8); for GaN a = 3.1986 Angstrom
  (Eq. (1a) at x = 1) and P_SP = -0.034 C/m^2 (Eq. (18a) at x = 1). The
  paper does not say whether its e31 is proper or improper; it combines
  it with ZB-reference P_SP as P_SP + 2 eps (e31 - e33 C13/C33) (its
  Eq. (46)), which is the ZB recipe of this package, so e31 is used as
  the proper constant here. Its GaN partner has no piezoelectric or
  elastic numbers in that paper; they are NaN, and the GaN record can
  only be used unstrained (as the relaxed buffer).

ZB-reference sets are stored as given and converted to the H branch on
the lattice of the relaxed alloy, P_H = P_ZB + e sqrt(3) / (2 a^2) (Dreyer
et al.), so that every record has the same meaning.

Band numbers (electron solver):

* I. Vurgaftman and J. R. Meyer (2003), Table I: in-plane lattice
  constants at 300 K (GaN 3.189, AlN 3.112, InN 3.545 Angstrom), gaps at
  0 K (3.510, 6.25, 0.78 eV) with Varshni alpha (0.909, 1.799, 0.245
  meV/K) and beta (830, 1462, 624 K), electron masses along c (0.20,
  0.32, 0.07 m0); gap bowings of Table III (AlGaN 0.7, GaInN 1.4, AlInN
  2.5 eV; the last one "provisional" in their words); and the wurtzite
  InN/GaN valence-band offset of 0.5 eV, with no bowing of the offsets
  ("For lack of other information, we ignore any bowing").
* O. Ambacher et al., J. Appl. Phys. 87, 334 (2000): static dielectric
  constants along c of GaN (10.4) and AlN (10.7) (Table I), and the
  AlGaN conduction-band rule Delta E_c = 0.7 [E_g(x) - E_g(0)] (Eq. (19)),
  which also fixes the valence band of AlN at -0.3 [E_g(AlN) - E_g(GaN)].
* InN eps33 = 14.4: V. Yu. Davydov et al., Appl. Phys. Lett. 75, 3297
  (1999), as listed in the Ioffe NSM database (InN, basic parameters,
  extraordinary direction, 300 K); the abstract of that paper gives the
  ratio eps_perp / eps_par = 0.91, which matches 13.1 / 14.4.

Alloys use linear interpolation (Vegard) between the end points except
where the source gives a bowing. The valence band of InGaN and InAlN is
linear between the binaries (offsets relative to GaN: InN +0.5 eV, AlN
from the 0.7 rule); for InAlN this is transitivity, an assumption that
Vurgaftman and Meyer leave open for the wurtzite InN/AlN junction. ScAlN
has no cited band numbers here (gap, offset and mass are NaN); add your
own with `with_numbers` before using it in the electron solver.
"""
from __future__ import annotations

import dataclasses
import math

import numpy as np

__all__ = ["NitrideMaterial", "algan", "ingan", "inaln", "scaln", "gan",
           "with_numbers", "POLARIZATION_SETS", "VURGAFTMAN2003",
           "AMBACHER2021_SCALN",
           "polarization_quantum", "zinc_blende_formal_polarization",
           "QE", "E_PER_NM2_PER_C_M2", "gap_shift", "AlloyGapShift"]

#: elementary charge (C), exact in the 2019 SI
QE = 1.602176634e-19
#: 1 C/m^2 expressed in elementary charges per nm^2
E_PER_NM2_PER_C_M2 = 1.0 / QE * 1e-18

_PBESOL = ("T. M. Mahim, nitride-misfit-polarization v1.3.1, Zenodo, "
           "doi:10.5281/zenodo.24315036 (PBEsol Berry-phase calculations, "
           "results/dft_polarization.json)")
_HSE = ("C. E. Dreyer, A. Janotti, C. G. Van de Walle and D. Vanderbilt, "
        "Phys. Rev. X 6, 021038 (2016), Tables I and II (HSE)")
_VM = ("I. Vurgaftman and J. R. Meyer, J. Appl. Phys. 94, 3675 (2003)")
_VM_POL = (_VM + ", Table I (P_sp, ZB reference; c11, c12, c13, c33; d13, "
           "d33) and Table III (P_sp bowing); e31 = (c11 + c12) d13 + "
           "c13 d33, e33 = 2 c13 d13 + c33 d33 (Ambacher et al., J. Appl. "
           "Phys. 130, 045102 (2021), Eqs. (40a), (40b))")
_AMB21 = ("O. Ambacher et al., J. Appl. Phys. 130, 045102 (2021)")

#: Vurgaftman and Meyer (2003), Table I, wurtzite binaries, as printed:
#: P_sp (C/m^2, ZB reference), elastic constants (GPa), moduli (pm/V).
VURGAFTMAN2003 = {
    "GaN": {"P_sp": -0.034, "c11": 390.0, "c12": 145.0, "c13": 106.0,
            "c33": 398.0, "d13": -1.6, "d33": 3.1},
    "AlN": {"P_sp": -0.090, "c11": 396.0, "c12": 137.0, "c13": 108.0,
            "c33": 373.0, "d13": -2.1, "d33": 5.4},
    "InN": {"P_sp": -0.042, "c11": 223.0, "c12": 115.0, "c13": 92.0,
            "c33": 224.0, "d13": -3.5, "d33": 7.6},
}
#: Table III: P_sp bowing C (C/m^2) in E(x) = ... - x (1 - x) C.
_VM_PSP_BOWING_C = {"AlGaN": -0.021, "InGaN": -0.037, "InAlN": -0.070}


def _vm_piezo(m):
    """e31, e33 (C/m^2) from the Table I moduli: GPa * pm/V = 1e-3 C/m^2."""
    t = VURGAFTMAN2003[m]
    e31 = ((t["c11"] + t["c12"]) * t["d13"] + t["c13"] * t["d33"]) * 1e-3
    e33 = (2.0 * t["c13"] * t["d13"] + t["c33"] * t["d33"]) * 1e-3
    return e31, e33


#: Polarization constants (C/m^2) per set and binary: either P_H (H
#: reference) or P_ZB (ZB reference; converted on the alloy lattice),
#: proper e31, e33, and C13/C33 (None where the set gives none; then the
#: builder states where the ratio comes from).
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
    "vurgaftman2003": {
        "reference": _VM_POL,
        **{m: {"P_ZB": VURGAFTMAN2003[m]["P_sp"],
               "e31": _vm_piezo(m)[0], "e33": _vm_piezo(m)[1],
               "c13_over_c33": VURGAFTMAN2003[m]["c13"]
               / VURGAFTMAN2003[m]["c33"]}
           for m in ("GaN", "AlN", "InN")},
    },
}

# Band numbers; see the module docstring. Lattice constants in nm.
_A300_NM = {"GaN": 0.3189, "AlN": 0.3112, "InN": 0.3545}
_EG0 = {"GaN": 3.510, "AlN": 6.25, "InN": 0.78}
_VARSHNI = {"GaN": (0.909e-3, 830.0), "AlN": (1.799e-3, 1462.0),
            "InN": (0.245e-3, 624.0)}
_ME_PAR = {"GaN": 0.20, "AlN": 0.32, "InN": 0.07}
_EPS33 = {"GaN": 10.4, "AlN": 10.7, "InN": 14.4}
_BOW_EG = {"AlGaN": 0.7, "InGaN": 1.4, "InAlN": 2.5}
_VBO_INN = 0.5           # eV, InN valence band above GaN (wurtzite)
_CBO = 0.7
_BAND_REF = ("I. Vurgaftman and J. R. Meyer, J. Appl. Phys. 94, 3675 (2003) "
             "(lattice constants, gaps, Varshni, masses, gap bowing); "
             "O. Ambacher et al., J. Appl. Phys. 87, 334 (2000) (dielectric "
             "constants, conduction-band offset rule)")
_INN_EPS_REF = ("InN eps33 = 14.4: V. Yu. Davydov et al., Appl. Phys. Lett. "
                "75, 3297 (1999), as listed in the Ioffe NSM database")
_IN_BAND_REF = (_VM + " (lattice constants, gaps, Varshni, masses, gap "
                "bowing, wurtzite InN/GaN valence-band offset 0.5 eV, no "
                "offset bowing); O. Ambacher et al., J. Appl. Phys. 87, 334 "
                "(2000) (GaN and AlN dielectric constants, AlN valence band "
                "from the 0.7 rule); " + _INN_EPS_REF)

_PIEZO_FIELDS = ("e31", "e33", "c13_over_c33")


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
    parameter_set : name of the polarization set the numbers come from
        ("" for a record you filled yourself). Layers of different sets
        meeting at an interface draw a warning.

    e31, e33 and C13/C33 may be NaN for a layer that is only ever used
    unstrained (a relaxed buffer); a strained layer refuses NaN. Band
    numbers may be NaN for a layer used only for its charge; the electron
    solver refuses a NaN where it needs a number.
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
    parameter_set: str = ""

    def __post_init__(self):
        for f in ("a", "P_H"):
            v = getattr(self, f)
            if not (isinstance(v, (int, float)) and math.isfinite(v)):
                raise ValueError(f"NitrideMaterial.{f} must be a finite "
                                 f"number; got {v!r}")
        for f in _PIEZO_FIELDS + ("eg", "ec", "me", "eps_r"):
            v = getattr(self, f)
            if not (isinstance(v, (int, float)) and not math.isinf(v)):
                raise ValueError(f"NitrideMaterial.{f} must be a finite "
                                 f"number or NaN; got {v!r}")
        if self.a <= 0.0:
            raise ValueError("the lattice constant a (nm) must be positive")
        if self.c13_over_c33 < 0.0:
            raise ValueError("C13/C33 must be >= 0")
        if not (isinstance(self.reference, str) and self.reference.strip()):
            raise ValueError("NitrideMaterial needs a non-empty reference "
                             "naming the source of its numbers")
        if not isinstance(self.parameter_set, str):
            raise ValueError("NitrideMaterial.parameter_set must be a string")
        for f in ("me", "eps_r"):
            v = getattr(self, f)
            if math.isfinite(v) and v <= 0.0:
                raise ValueError(f"NitrideMaterial.{f} must be positive "
                                 "(or NaN when not needed)")

    @property
    def has_piezo(self):
        """True when e31, e33 and C13/C33 are all numbers (not NaN)."""
        return all(math.isfinite(getattr(self, f)) for f in _PIEZO_FIELDS)

    @property
    def P_ZB(self):
        """Spontaneous polarization with the zinc-blende reference (C/m^2)."""
        return self.P_H - zinc_blende_formal_polarization(self.a)


def with_numbers(material: NitrideMaterial, reference: str,
                 **numbers) -> NitrideMaterial:
    """A copy of `material` with some numbers replaced by yours.

    reference : your source for the new numbers (required); it is added to
        the record's reference together with the names of the fields you
        changed.
    numbers : any of a, P_H, e31, e33, c13_over_c33, eg, ec, me, eps_r, and
        name. Changing a lattice or polarization number removes the
        record's parameter_set (it is no longer the published set).

    Example: ``with_numbers(scaln(0.18), "my measured offsets", ec=...,
    me=..., eg=...)``.
    """
    if not isinstance(material, NitrideMaterial):
        raise ValueError("with_numbers needs a NitrideMaterial")
    if not (isinstance(reference, str) and reference.strip()):
        raise ValueError("with_numbers needs a reference for the new numbers")
    allowed = {"name", "a", "P_H", "e31", "e33", "c13_over_c33", "eg", "ec",
               "me", "eps_r"}
    bad = set(numbers) - allowed
    if bad:
        raise ValueError(f"cannot replace {sorted(bad)}; allowed are "
                         f"{sorted(allowed)}")
    if not numbers:
        raise ValueError("give at least one number to replace")
    pset = material.parameter_set
    if set(numbers) & {"a", "P_H", "e31", "e33", "c13_over_c33"}:
        pset = ""
    changed = ", ".join(sorted(k for k in numbers if k != "name"))
    ref = material.reference + (f"; {changed} from {reference}" if changed
                                else f"; renamed ({reference})")
    return dataclasses.replace(material, reference=ref, parameter_set=pset,
                               **numbers)


def _gap(name, T):
    a, b = _VARSHNI[name]
    return _EG0[name] - a * T * T / (T + b)


def _check_x(x, top=1.0, what="x"):
    if not (isinstance(x, (int, float)) and not isinstance(x, bool)
            and 0.0 <= x <= top):
        raise ValueError(f"the {what} must be in [0, {top:g}]; got {x!r}")


def _check_T(T):
    if not (isinstance(T, (int, float)) and math.isfinite(T) and T > 0.0):
        raise ValueError("temperature_K must be finite and positive")


def _check_set(polarization, binaries):
    if polarization not in POLARIZATION_SETS:
        raise ValueError(f"unknown polarization set {polarization!r}; "
                         f"choose from {sorted(POLARIZATION_SETS)}")
    ps = POLARIZATION_SETS[polarization]
    missing = [m for m in binaries if m not in ps]
    if missing:
        have = sorted(k for k, v in POLARIZATION_SETS.items()
                      if all(m in v for m in binaries))
        raise ValueError(f"the set {polarization!r} has no {missing}; sets "
                         f"with {list(binaries)}: {have}")
    return ps


def _p_h(ps, m1, m0, x, a_nm, bowing):
    """P_H of the alloy x*m1 + (1-x)*m0 on the lattice a_nm (C/m^2)."""
    if "P_H" in ps[m1]:
        return (x * ps[m1]["P_H"] + (1 - x) * ps[m0]["P_H"]
                + bowing * x * (1 - x))
    p_zb = (x * ps[m1]["P_ZB"] + (1 - x) * ps[m0]["P_ZB"]
            + bowing * x * (1 - x))
    return p_zb + zinc_blende_formal_polarization(a_nm)


def _bowing(polarization, alloy, psp_bowing):
    """P_sp bowing b (added as b x (1 - x)) and a note for the reference."""
    if psp_bowing is not None:
        if not (isinstance(psp_bowing, (int, float))
                and math.isfinite(psp_bowing)):
            raise ValueError("psp_bowing must be a finite number (C/m^2) "
                             "or None")
        return float(psp_bowing), ("" if psp_bowing == 0.0 else
                                   "; P bowing %r C/m^2 supplied by the "
                                   "caller" % float(psp_bowing))
    if polarization == "vurgaftman2003":
        return -_VM_PSP_BOWING_C[alloy], ""
    return 0.0, ""


def _vm_ratio(m1, m0, x):
    t1, t0 = VURGAFTMAN2003[m1], VURGAFTMAN2003[m0]
    c13 = x * t1["c13"] + (1 - x) * t0["c13"]
    c33 = x * t1["c33"] + (1 - x) * t0["c33"]
    return c13 / c33


def _label(m1, m0, x, s1, s0):
    if x == 0:
        return m0
    if x == 1:
        return m1
    return "%s%g%s%gN" % (s1, x, s0, 1 - x)


def algan(x: float, temperature_K: float = 300.0,
          polarization: str = "pbesol-mahim2026",
          c13_over_c33: float = None, cbo_fraction: float = _CBO,
          psp_bowing: float = None) -> NitrideMaterial:
    """Relaxed Al(x)Ga(1-x)N, 0 <= x <= 1, with cited end points.

    polarization : ``"pbesol-mahim2026"`` (default), ``"hse-dreyer2016"``
        or ``"vurgaftman2003"`` (see the module docstring).
    c13_over_c33 : override of the elastic ratio. For the HSE set the
        PBEsol ratio is used when None, as in the analysis the module
        reproduces, and the reference says so; the Vurgaftman set uses its
        own c13 and c33 (each linear in x).
    cbo_fraction : share of the gap difference in the conduction band
        (0.7, Ambacher et al.); for sensitivity tests.
    psp_bowing : bowing of the spontaneous polarization (C/m^2), added as
        psp_bowing * x (1 - x). None (default) means the set's own
        recommendation: 0 for the PBEsol and HSE sets (they give none) and
        +0.021 for the Vurgaftman set (its Table III, C = -0.021 in
        E = ... - x (1 - x) C).
    """
    _check_x(x, what="Al fraction x")
    _check_T(temperature_K)
    ps = _check_set(polarization, ("GaN", "AlN"))
    if not (math.isfinite(cbo_fraction) and 0.0 < cbo_fraction < 1.0):
        raise ValueError("cbo_fraction must lie strictly between 0 and 1")
    bow, bow_note = _bowing(polarization, "AlGaN", psp_bowing)
    ref = ps["reference"] + bow_note
    if c13_over_c33 is not None:
        if not (math.isfinite(c13_over_c33) and c13_over_c33 >= 0.0):
            raise ValueError("c13_over_c33 must be finite and >= 0")
        ratio = float(c13_over_c33)
        ref += "; C13/C33 = %r supplied by the caller" % ratio
    elif polarization == "vurgaftman2003":
        ratio = _vm_ratio("AlN", "GaN", x)
    elif ps["GaN"]["c13_over_c33"] is None:
        pb = POLARIZATION_SETS["pbesol-mahim2026"]
        ratio = (x * pb["AlN"]["c13_over_c33"]
                 + (1 - x) * pb["GaN"]["c13_over_c33"])
        ref += ("; C13/C33 not given by that source, taken from "
                + _PBESOL)
    else:
        ratio = (x * ps["AlN"]["c13_over_c33"]
                 + (1 - x) * ps["GaN"]["c13_over_c33"])

    def lin(f):
        return x * f("AlN") + (1 - x) * f("GaN")

    T = float(temperature_K)
    a = lin(lambda n: _A300_NM[n])
    eg = lin(lambda n: _gap(n, T)) - _BOW_EG["AlGaN"] * x * (1 - x)
    return NitrideMaterial(
        name=_label("AlN", "GaN", x, "Al", "Ga"),
        a=a,
        P_H=_p_h(ps, "AlN", "GaN", x, a, bow),
        e31=lin(lambda n: ps[n]["e31"]),
        e33=lin(lambda n: ps[n]["e33"]),
        c13_over_c33=ratio,
        reference=ref + "; " + _BAND_REF,
        eg=eg,
        ec=cbo_fraction * (eg - _gap("GaN", T)),
        me=lin(lambda n: _ME_PAR[n]),
        eps_r=lin(lambda n: _EPS33[n]),
        parameter_set=polarization,
    )


def _ev(name, T):
    """Valence-band maximum relative to that of GaN (eV); see docstring."""
    if name == "GaN":
        return 0.0
    if name == "InN":
        return _VBO_INN
    if name == "AlN":
        return -(1.0 - _CBO) * (_gap("AlN", T) - _gap("GaN", T))
    raise KeyError(name)


def _in_alloy(m0, x, temperature_K, polarization, psp_bowing, alloy, s0):
    """In(x) M0(1-x) N for M0 = GaN or AlN."""
    _check_x(x, what="In fraction x")
    _check_T(temperature_K)
    ps = _check_set(polarization, ("InN", m0))
    bow, bow_note = _bowing(polarization, alloy, psp_bowing)
    ref = ps["reference"] + bow_note
    ratio = _vm_ratio("InN", m0, x)
    if polarization != "vurgaftman2003":
        ref += ("; C13/C33 not given by that source, from c13 and c33 of "
                + _VM + ", Table I (each linear in x)")

    def lin(f):
        return x * f("InN") + (1 - x) * f(m0)

    T = float(temperature_K)
    a = lin(lambda n: _A300_NM[n])
    eg = lin(lambda n: _gap(n, T)) - _BOW_EG[alloy] * x * (1 - x)
    ev = lin(lambda n: _ev(n, T))
    band_ref = _IN_BAND_REF
    if m0 == "AlN":
        band_ref += ("; InAlN valence band linear between InN and AlN "
                     "(transitivity, an assumption: the wurtzite InN/AlN "
                     "offset is left open by Vurgaftman and Meyer); gap "
                     "bowing 2.5 eV called provisional by them")
    return NitrideMaterial(
        name=_label("InN", m0, x, "In", s0),
        a=a,
        P_H=_p_h(ps, "InN", m0, x, a, bow),
        e31=lin(lambda n: ps[n]["e31"]),
        e33=lin(lambda n: ps[n]["e33"]),
        c13_over_c33=ratio,
        reference=ref + "; " + band_ref,
        eg=eg,
        ec=ev + eg - _gap("GaN", T),
        me=lin(lambda n: _ME_PAR[n]),
        eps_r=lin(lambda n: _EPS33[n]),
        parameter_set=polarization,
    )


def ingan(x: float, temperature_K: float = 300.0,
          polarization: str = "hse-dreyer2016",
          psp_bowing: float = None) -> NitrideMaterial:
    """Relaxed In(x)Ga(1-x)N, 0 <= x <= 1, with cited end points.

    polarization : ``"hse-dreyer2016"`` (default) or ``"vurgaftman2003"``;
        the PBEsol set has no InN. C13/C33 comes from the c13 and c33 of
        Vurgaftman and Meyer, Table I.
    psp_bowing : None for the set's recommendation (0 for HSE, +0.037
        C/m^2 for Vurgaftman, Table III), or your number (C/m^2).

    Gap bowing 1.4 eV; valence band +0.5 x eV above that of GaN (no offset
    bowing), so the conduction band carries the rest of the gap change.
    Pair it with ``gan(polarization=...)`` of the same set.
    """
    return _in_alloy("GaN", x, temperature_K, polarization, psp_bowing,
                     "InGaN", "Ga")


def inaln(x: float, temperature_K: float = 300.0,
          polarization: str = "hse-dreyer2016",
          psp_bowing: float = None) -> NitrideMaterial:
    """Relaxed In(x)Al(1-x)N, 0 <= x <= 1, with cited end points.

    polarization : ``"hse-dreyer2016"`` (default) or ``"vurgaftman2003"``.
    psp_bowing : None for the set's recommendation (0 for HSE, +0.070
        C/m^2 for Vurgaftman, Table III), or your number (C/m^2).

    With the lattice constants used here In(x)Al(1-x)N matches GaN at
    x = (3.189 - 3.112) / (3.545 - 3.112) = 0.178. Gap bowing 2.5 eV
    (provisional in the source); valence band linear between AlN and InN
    (transitivity; see the module docstring).
    """
    return _in_alloy("AlN", x, temperature_K, polarization, psp_bowing,
                     "InAlN", "Al")


def _amb_quad(y1, y0, b, x):
    return y1 * x + y0 * (1.0 - x) + b * x * (1.0 - x)


#: Ambacher et al. (2021) composition fits for Sc(x)Al(1-x)N, 0 <= x <= 0.5,
#: as (value at x = 1, value at x = 0, bowing) of y1 x + y0 (1-x) + b x(1-x).
AMBACHER2021_SCALN = {
    "a_angstrom": (3.741, 3.110, -0.242),       # Eq. (3a)
    "P_SP": (-0.874, -0.089, 0.741),            # Eq. (18c), C/m^2, ZB ref.
    "C13_GPa": (141.70, 108.00, 51.95),         # Eq. (23c)
    "C33_GPa": (-155.17, 373.00, 95.49),        # Eq. (23d)
    "e31": (-1.353, -0.593, 0.576),             # Eq. (39b), C/m^2
    "e33": (9.125, 1.471, -6.625),              # Eq. (39c), C/m^2
    "eps33": (89.93, 10.31, -62.48),            # Eq. (8)
}
_AMB21_SCALN_REF = (_AMB21 + ": a Eq. (3a), P_SP Eq. (18c) (ZB reference, "
                    "converted to the H branch on a(x)), C13 and C33 Eqs. "
                    "(23c), (23d), e31 and e33 Eqs. (39b), (39c) (e31 used "
                    "as the proper constant), eps33 Eq. (8); valid for "
                    "0 <= x <= 0.5 (wurtzite; cubic phase near x = 0.45 +- "
                    "0.05 per the same paper)")


def scaln(x: float) -> NitrideMaterial:
    """Relaxed Sc(x)Al(1-x)N, 0 <= x <= 0.5, from Ambacher et al. (2021).

    Every polarization, elastic and dielectric number is that paper's
    composition fit (`AMBACHER2021_SCALN`). The paper gives no band gap,
    band offset or electron mass, so eg, ec and me are NaN: supply yours
    with `with_numbers` before using the layer in `solve_electrons`. Pair
    it with ``gan(polarization="ambacher2021")`` (same set). With that
    GaN (a = 3.1986 Angstrom) the lattice match is near x = 0.20.
    """
    _check_x(x, top=0.5, what="Sc fraction x")
    t = {k: _amb_quad(*v, x) for k, v in AMBACHER2021_SCALN.items()}
    a = t["a_angstrom"] / 10.0
    return NitrideMaterial(
        name=_label("ScN", "AlN", x, "Sc", "Al"),
        a=a,
        P_H=t["P_SP"] + zinc_blende_formal_polarization(a),
        e31=t["e31"],
        e33=t["e33"],
        c13_over_c33=t["C13_GPa"] / t["C33_GPa"],
        reference=_AMB21_SCALN_REF,
        eps_r=t["eps33"],
        parameter_set="ambacher2021",
    )


def gan(polarization: str = "pbesol-mahim2026",
        temperature_K: float = 300.0) -> NitrideMaterial:
    """Relaxed GaN of a given polarization set, to pair with the alloys of
    the same set.

    For ``"pbesol-mahim2026"``, ``"hse-dreyer2016"`` and
    ``"vurgaftman2003"`` this is ``algan(0, ...)``. For ``"ambacher2021"``
    it is the GaN of Ambacher et al. (2021): a = 3.1986 Angstrom (Eq. (1a)
    at x = 1) and P_SP = -0.034 C/m^2 (Eq. (18a) at x = 1); that paper
    gives no GaN piezoelectric or elastic constants, so they are NaN and
    the record can only be the unstrained (bottom) layer. Its band numbers
    are those of `algan(0)`.
    """
    if polarization != "ambacher2021":
        return algan(0.0, temperature_K=temperature_K,
                     polarization=polarization)
    _check_T(temperature_K)
    a = (3.1095 + 0.0891) / 10.0
    g = algan(0.0, temperature_K=temperature_K)
    return NitrideMaterial(
        name="GaN", a=a,
        P_H=-0.034 + zinc_blende_formal_polarization(a),
        e31=math.nan, e33=math.nan, c13_over_c33=math.nan,
        reference=(_AMB21 + ": GaN a Eq. (1a) at x = 1, P_SP Eq. (18a) at "
                   "x = 1 (ZB reference, converted to the H branch); no "
                   "piezoelectric or elastic constants (unstrained use "
                   "only); " + _BAND_REF),
        eg=g.eg, ec=0.0, me=g.me, eps_r=g.eps_r,
        parameter_set="ambacher2021",
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


@dataclasses.dataclass(frozen=True)
class AlloyGapShift:
    """How the gap of a nitride layer follows the temperature, with the
    package's cited numbers (new in 0.23.0): the change of x E_g(end 1) +
    (1 - x) E_g(end 0) - b x (1 - x) between two temperatures, each end on
    its own Varshni curve (Vurgaftman and Meyer 2003, as in `algan`,
    `ingan`, `inaln`; the bowing b does not change with temperature), so
    that it equals what those functions give at the two temperatures.
    conduction_share: the part of the change that moves the conduction
    band (yours; Varshni's form does not say)."""

    alloy: str
    x: float
    conduction_share: float
    reference: str = _BAND_REF

    def __post_init__(self):
        if self.alloy not in _GAP_ENDS:
            raise ValueError(f"no cited gap-temperature numbers for {self.alloy!r}; "
                             f"available: {sorted(_GAP_ENDS)}")
        _check_x(self.x)
        if not (0.0 <= self.conduction_share <= 1.0):
            raise ValueError("conduction_share must be between 0 and 1")

    def shift(self, T, T0):
        """E_g(T) - E_g(T0) in eV (T may be an array)."""
        m1, m0 = _GAP_ENDS[self.alloy]
        T = np.asarray(T, float)

        def g(n, t):
            a, b = _VARSHNI[n]
            return -a * t * t / (t + b)
        x = 1.0 if m1 == m0 else float(self.x)
        return x * (g(m1, T) - g(m1, T0)) + (1 - x) * (g(m0, T) - g(m0, T0))


_GAP_ENDS = {"GaN": ("GaN", "GaN"), "AlN": ("AlN", "AlN"), "InN": ("InN", "InN"),
             "AlGaN": ("AlN", "GaN"), "InGaN": ("InN", "GaN"), "InAlN": ("InN", "AlN")}


def gap_shift(alloy: str, x: float = 0.0, conduction_share: float = 0.5) -> AlloyGapShift:
    """The temperature dependence of a nitride layer's gap from the cited
    Varshni numbers (see `AlloyGapShift`), for `Device(band_gap_temperature=)`.
    alloy: "GaN", "AlN", "InN", "AlGaN", "InGaN" or "InAlN" (no cited band
    numbers exist here for ScAlN); x: the fraction of the first-named metal.
    conduction_share 0.5 splits the change evenly between the bands, an
    assumption: give your own if you have one."""
    return AlloyGapShift(alloy, float(x), float(conduction_share))
