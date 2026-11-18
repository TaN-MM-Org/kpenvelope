"""Polar optical phonons of wurtzite layers in the dielectric continuum
model: the dielectric function, the Lyddane-Sachs-Teller check, the
interface modes of one interface and of a slab, and the Froehlich coupling
constant (new in v0.18.0).

Dielectric function. A wurtzite crystal has one polar oscillator for
fields in the basal plane (E1 symmetry) and one for fields along c (A1):

    eps_perp(w) = eps_inf_perp (w^2 - w_LO_perp^2) / (w^2 - w_TO_perp^2),
    eps_par(w)  = eps_inf_par  (w^2 - w_LO_par^2)  / (w^2 - w_TO_par^2),

whose static limits obey the Lyddane-Sachs-Teller relation
eps_0 = eps_inf (w_LO / w_TO)^2.

Interface modes (dielectric continuum model; R. Fuchs and K. L. Kliewer,
Phys. Rev. 140, A2076 (1965); N. Mori and T. Ando, Phys. Rev. B 40, 6175
(1989)). A mode of in-plane wave number q has the potential
phi = f(z) exp(i q x) with eps_perp q^2 f = eps_par f'' in each layer, so f
decays as exp(-kappa |z|) with kappa = q sqrt(eps_perp / eps_par) where
eps_perp / eps_par > 0. Matching phi and D_z = -eps_par f' at the
interfaces (worked out here from these two conditions) gives, with
eta = sign(eps_par) sqrt(eps_perp eps_par):

* one interface between two thick layers:  eta_1(w) + eta_2(w) = 0, for
  any q;
* a slab 1 of thickness d inside medium 2:  eta_1 tanh(kappa_1 d / 2) +
  eta_2 = 0 (potential even in the slab) and eta_1 coth(kappa_1 d / 2) +
  eta_2 = 0 (odd).

For isotropic media eta = eps and these are the textbook conditions. A
thin slab in vacuum has its even mode at w_TO and its odd mode at w_LO
(Fuchs and Kliewer); a thick slab has both at the single-interface
frequency. Both limits are checked in the tests. Confined and
half-space modes (where eps_perp / eps_par < 0, between the A1 and E1
frequencies) are not computed here.

Shipped numbers: GaN only, because they are the ones that could be checked
here. A. S. Barker and M. Ilegems, Phys. Rev. B 7, 743 (1973): LO phonons
746 cm^-1 (E1) and 744 cm^-1 (A1), static dielectric constants 9.5 (perp)
and 10.4 (par), eps_inf_perp = 5.35; TO phonons 559 cm^-1 (E1; Lemos et al.,
1972) and 533 cm^-1 (A1; Manchon et al., 1970); all as listed by the Ioffe
NSM database (GaN, optical properties). eps_inf_par is not listed there
from the same fit, so it is taken from the Lyddane-Sachs-Teller relation
with the A1 frequencies and eps_0_par = 10.4 (5.34); the same relation with
the E1 numbers gives eps_0_perp = 9.53 against the listed 9.5, which checks
the copied numbers. For AlN, InN or alloys supply a `PolarPhonons` record
with your cited numbers.
"""
from __future__ import annotations

import dataclasses
import math

import numpy as np

__all__ = ["PolarPhonons", "GAN_BARKER1973", "CM1_TO_MEV",
           "single_interface_modes", "slab_modes", "frohlich_constant"]

#: 1 cm^-1 in meV (h c / e, exact in the 2019 SI)
CM1_TO_MEV = 6.62607015e-34 * 2.99792458e10 / 1.602176634e-19 * 1e3


@dataclasses.dataclass(frozen=True)
class PolarPhonons:
    """Polar optical phonons of one wurtzite material (energies in meV).

    w_TO_perp, w_LO_perp : E1 TO and LO phonons (fields in the basal plane).
    w_TO_par, w_LO_par : A1 TO and LO phonons (fields along c).
    eps_inf_perp, eps_inf_par : high-frequency dielectric constants.
    reference : the source of the numbers (required).
    """

    name: str
    w_TO_perp: float
    w_LO_perp: float
    w_TO_par: float
    w_LO_par: float
    eps_inf_perp: float
    eps_inf_par: float
    reference: str

    def __post_init__(self):
        for f in ("w_TO_perp", "w_LO_perp", "w_TO_par", "w_LO_par",
                  "eps_inf_perp", "eps_inf_par"):
            v = getattr(self, f)
            if not (isinstance(v, (int, float)) and math.isfinite(v) and v > 0):
                raise ValueError(f"PolarPhonons.{f} must be finite and > 0")
        if not (self.w_LO_perp > self.w_TO_perp and self.w_LO_par > self.w_TO_par):
            raise ValueError("a polar LO phonon lies above its TO phonon")
        if not (isinstance(self.reference, str) and self.reference.strip()):
            raise ValueError("PolarPhonons needs a reference")

    def eps_perp(self, w):
        """eps for fields in the basal plane at phonon energy w (meV)."""
        w = np.asarray(w, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            return self.eps_inf_perp * (w * w - self.w_LO_perp ** 2) / (
                w * w - self.w_TO_perp ** 2)

    def eps_par(self, w):
        """eps for fields along c at phonon energy w (meV)."""
        w = np.asarray(w, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            return self.eps_inf_par * (w * w - self.w_LO_par ** 2) / (
                w * w - self.w_TO_par ** 2)

    @property
    def eps0_perp(self):
        """Static constant from the Lyddane-Sachs-Teller relation."""
        return self.eps_inf_perp * (self.w_LO_perp / self.w_TO_perp) ** 2

    @property
    def eps0_par(self):
        return self.eps_inf_par * (self.w_LO_par / self.w_TO_par) ** 2

    def eta(self, w):
        """sign(eps_par) sqrt(eps_perp eps_par); NaN where eps_perp eps_par
        < 0 (no decaying interface solution there)."""
        a, b = self.eps_perp(w), self.eps_par(w)
        prod = a * b
        with np.errstate(invalid="ignore"):
            return np.where(prod > 0, np.sign(b) * np.sqrt(np.abs(prod)), np.nan)


_GAN_REF = ("A. S. Barker and M. Ilegems, Phys. Rev. B 7, 743 (1973) (LO "
            "phonons 746 and 744 cm^-1, eps_0 9.5 and 10.4, eps_inf_perp "
            "5.35); TO phonons 559 cm^-1 (E1, Lemos et al. 1972) and 533 "
            "cm^-1 (A1, Manchon et al. 1970); as listed by the Ioffe NSM "
            "database (GaN, optical properties); eps_inf_par = 10.4 (533/744)^2 "
            "by the Lyddane-Sachs-Teller relation")

#: GaN (see the module docstring), energies in meV
GAN_BARKER1973 = PolarPhonons(
    name="GaN",
    w_TO_perp=559.0 * CM1_TO_MEV, w_LO_perp=746.0 * CM1_TO_MEV,
    w_TO_par=533.0 * CM1_TO_MEV, w_LO_par=744.0 * CM1_TO_MEV,
    eps_inf_perp=5.35, eps_inf_par=10.4 * (533.0 / 744.0) ** 2,
    reference=_GAN_REF)


def _roots(f, lo, hi, n=20001):
    """All sign changes of f on (lo, hi) refined by bisection, skipping
    poles (where |f| jumps) and NaN gaps."""
    w = np.linspace(lo, hi, n)
    v = f(w)
    out = []
    for i in range(n - 1):
        a, b = v[i], v[i + 1]
        if not (np.isfinite(a) and np.isfinite(b)) or a * b > 0:
            continue
        x0, x1 = w[i], w[i + 1]
        fa = a
        for _ in range(100):
            m = 0.5 * (x0 + x1)
            fm = float(f(np.array([m]))[0])
            if not np.isfinite(fm):
                break
            if fa * fm <= 0:
                x1 = m
            else:
                x0, fa = m, fm
        m = 0.5 * (x0 + x1)
        val = float(f(np.array([m]))[0])
        scale = max(abs(a), abs(b))
        if np.isfinite(val) and abs(val) < 1e-6 * max(1.0, scale):
            out.append(m)
    return out


def _window(m1, m2):
    lo = 0.5 * min(m1.w_TO_perp, m1.w_TO_par, m2.w_TO_perp, m2.w_TO_par)
    hi = 1.5 * max(m1.w_LO_perp, m1.w_LO_par, m2.w_LO_perp, m2.w_LO_par)
    return lo, hi


def single_interface_modes(m1: PolarPhonons, m2: PolarPhonons):
    """Energies (meV) of the interface modes of one interface between two
    thick polar layers: eta_1 + eta_2 = 0 (independent of q). A medium of
    constant permittivity can be given as a number for m2."""
    f2 = _medium(m2)
    lo, hi = _window(m1, m1 if not isinstance(m2, PolarPhonons) else m2)
    return _roots(lambda w: m1.eta(w) + f2(w), lo, hi)


def _medium(m):
    if isinstance(m, PolarPhonons):
        return m.eta
    if isinstance(m, (int, float)) and math.isfinite(m) and m > 0:
        return lambda w: np.full(np.shape(w), float(m))
    raise ValueError("the medium must be PolarPhonons or a positive permittivity")


def slab_modes(slab: PolarPhonons, outside, thickness_nm: float, q_per_nm):
    """Interface modes of a slab of `slab` (thickness in nm) inside
    `outside` (PolarPhonons, or a positive number for a non-polar medium),
    for in-plane wave numbers q (nm^-1).

    Returns {"q": q array, "even": list of energy arrays (one per branch, NaN
    where absent), "odd": same}; "even" has the potential even in the slab
    (eta_1 tanh(kappa d/2) + eta_2 = 0), "odd" odd (coth).
    """
    if not (math.isfinite(thickness_nm) and thickness_nm > 0):
        raise ValueError("thickness_nm must be > 0")
    qs = np.atleast_1d(np.asarray(q_per_nm, dtype=float))
    if np.any(~np.isfinite(qs)) or np.any(qs <= 0):
        raise ValueError("q must be > 0 (nm^-1)")
    f2 = _medium(outside)
    other = outside if isinstance(outside, PolarPhonons) else slab
    lo, hi = _window(slab, other)
    res = {"q": qs, "even": [], "odd": []}
    for kind, fn in (("even", np.tanh), ("odd", lambda x: 1.0 / np.tanh(x))):
        rows = []
        for q in qs:
            def f(w, q=q):
                with np.errstate(invalid="ignore", divide="ignore"):
                    r = slab.eps_perp(w) / slab.eps_par(w)
                    kap = q * np.sqrt(np.where(r > 0, r, np.nan))
                    return slab.eta(w) * fn(kap * thickness_nm / 2) + f2(w)
            rows.append(_roots(f, lo, hi))
        nb = max((len(r) for r in rows), default=0)
        res[kind] = [np.array([r[k] if k < len(r) else np.nan for r in rows])
                     for k in range(nb)]
    return res


def frohlich_constant(mass_m0: float, w_LO_meV: float, eps_inf: float,
                      eps0: float) -> float:
    """Froehlich coupling constant of an electron to the LO phonon,
    alpha = (e^2 / (4 pi eps_0 hbar)) sqrt(m / (2 hbar w_LO)) (1/eps_inf -
    1/eps_0) (dimensionless; isotropic model)."""
    for v in (mass_m0, w_LO_meV, eps_inf, eps0):
        if not (math.isfinite(v) and v > 0):
            raise ValueError("all inputs must be finite and > 0")
    if not eps0 > eps_inf:
        raise ValueError("a polar material has eps0 > eps_inf")
    e = 1.602176634e-19
    hbar = 1.054571817e-34
    eps_vac = 8.8541878128e-12
    m = mass_m0 * 9.1093837015e-31
    hw = w_LO_meV * 1e-3 * e
    return (e * e / (4 * math.pi * eps_vac * hbar)) * math.sqrt(m / (2 * hw)) * (
        1.0 / eps_inf - 1.0 / eps0)
