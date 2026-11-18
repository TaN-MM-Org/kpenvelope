"""Fixed charge at every interface of a c-plane nitride stack, coherent or
partly relaxed, with the charge of the misfit dislocations made explicit
(new in v0.17.0).

Conventions. z runs along [0001] of a metal-polar crystal, from the bottom
layer (the relaxed substrate or buffer) up to the surface. P is the formal
polarization along z (C/m^2). The bound charge at an interface with layer
B below and film F above is sigma = P_B - P_F; a positive sigma attracts
electrons, a negative one holes.

Polarization of a strained layer. Under biaxial in-plane strain eps (the
c axis responds with -2 (C13/C33) eps), on the branch of the
layered-hexagonal (H) reference

    P = P_H + 2 eps (e31 - P_H) - 2 (C13/C33) eps e33        (improper),

and with the zinc-blende (ZB) reference and proper constants

    P = P_ZB + 2 eps e31 - 2 (C13/C33) eps e33,  P_ZB = P_H - e sqrt(3)/(2 a^2).

The -P_H in the H branch is the improper term: the same dipole spread over
a different area (Vanderbilt, J. Phys. Chem. Solids 61, 147 (2000);
Dreyer et al., Phys. Rev. X 6, 021038 (2016)).

Relaxation. A film of natural lattice constant a_F on a layer whose
in-plane lattice is a_B has the mismatch eps_c = (a_B - a_F)/a_F. Relaxed
by a fraction r it keeps the strain (1 - r) eps_c. The relaxation is made
by misfit dislocations, each the end of an extra (or missing) half plane
with in-plane edge component b = a_F, of total length per area

    L_A = 2 r |eps_c| / b                                    (Eq. 1).

To first order in the strain the interface charge is

    sigma(r) = sigma_coh + 2 r eps_c (e31 - e33 C13/C33) + Lambda L_A   (Eq. 2),

where Lambda is the charge per length of the dislocations. Neither
reference decides Lambda. The H recipe, used on its own, quietly gives
every line the charge of a bare end, Lambda = -P_H b for a film under
tension (eps_c > 0) and +P_H b under compression; the ZB recipe quietly
gives Lambda = 0. Ends whose broken bonds follow the electron-counting
rule carry Lambda = -/+ P_ZB b, close to neutral. T. M. Mahim, "The charge
of misfit dislocations decides the polarization charge at relaxed nitride
interfaces" (2026; code and data: doi:10.5281/zenodo.24315036) derives
Eq. 2 and measures Lambda in published AlGaN/GaN data.

Because the answer depends on Lambda, `interface_charges` refuses a
relaxed interface unless you say which line charge to use: "bare",
"neutral", "electron-counting", a `LineCharge(bound_fraction, trapped)`,
or a number in e/nm.

Implementation. With reference="H" the charge is the difference of the
H-branch polarizations at the actual strains (which contains bare-end
lines) plus (Lambda - Lambda_bare) L_A; with reference="ZB" it is the
difference of the ZB-referenced polarizations plus Lambda L_A. Both are
Eq. 2 exactly, with their own coherent charge; the two coherent charges
differ only at second order in the strain (Dreyer et al.).
"""
from __future__ import annotations

import dataclasses
import math

from .nitride import (E_PER_NM2_PER_C_M2, NitrideMaterial,
                      zinc_blende_formal_polarization)

__all__ = ["StackLayer", "LineCharge", "InterfaceCharge",
           "interface_charges", "layer_strains", "misfit_line_density",
           "line_charge_limits", "strained_polarization",
           "charge_change_per_relaxation", "trapped_from_sum_rule"]

_KINDS = ("bare", "neutral", "electron-counting")


@dataclasses.dataclass(frozen=True)
class StackLayer:
    """One layer of a c-plane stack, listed from the bottom up.

    material : a `NitrideMaterial`.
    thickness : nm (> 0).
    relaxation : fraction r of its mismatch to the layer below that the
        layer has relieved, 0 (coherent) to 1 (fully relaxed). The bottom
        layer is the relaxed substrate or buffer; leave its relaxation at 0.
    donors : ionizable donor density (nm^-3; 1e18 cm^-3 = 1e-3 nm^-3).
    donor_depth : donor level below the conduction band (eV), or None for
        complete ionization.
    name : optional label (defaults to the material name).
    """

    material: NitrideMaterial
    thickness: float
    relaxation: float = 0.0
    donors: float = 0.0
    donor_depth: float = None
    name: str = None

    def __post_init__(self):
        if not isinstance(self.material, NitrideMaterial):
            raise ValueError("StackLayer.material must be a NitrideMaterial")
        if not (math.isfinite(self.thickness) and self.thickness > 0.0):
            raise ValueError("StackLayer.thickness (nm) must be finite and > 0")
        if not (math.isfinite(self.relaxation) and 0.0 <= self.relaxation <= 1.0):
            raise ValueError("StackLayer.relaxation must lie in [0, 1]; got "
                             f"{self.relaxation!r}")
        if not (math.isfinite(self.donors) and self.donors >= 0.0):
            raise ValueError("StackLayer.donors (nm^-3) must be finite and >= 0")
        if self.donor_depth is not None and not (
                math.isfinite(self.donor_depth) and self.donor_depth > 0.0):
            raise ValueError("StackLayer.donor_depth must be None or a positive "
                             "energy in eV")

    @property
    def label(self):
        return self.name or self.material.name


@dataclasses.dataclass(frozen=True)
class LineCharge:
    """A misfit-line charge split into its two physical parts.

    bound_fraction : f, the bound charge as a fraction of the bare-end
        charge; it has the sign of the bare end (it flips between a film
        under tension and one under compression). f = 1 is a bare end,
        f = 0 no bound charge.
    trapped : q >= 0, electrons trapped on the line (e/nm); negative charge
        whatever the sign of the strain.

    Lambda = -sign(eps_c) f P_H b - q e.
    """

    bound_fraction: float = 0.0
    trapped: float = 0.0

    def __post_init__(self):
        if not math.isfinite(self.bound_fraction):
            raise ValueError("LineCharge.bound_fraction must be finite")
        if not (math.isfinite(self.trapped) and self.trapped >= 0.0):
            raise ValueError("LineCharge.trapped (e/nm) must be finite and >= 0")


@dataclasses.dataclass(frozen=True)
class InterfaceCharge:
    """Fixed charge at the interface between layer `index` and `index + 1`.

    z : position above the bottom of the stack (nm).
    eps_c : mismatch of the upper layer to the in-plane lattice below it.
    relaxation : r of the upper layer.
    strain_below, strain_above : actual in-plane strains of the two layers.
    line_density : L_A (nm^-1). line_charge, bare_line_charge : e/nm.
    sigma : C/m^2; sigma_nm2 : elementary charges per nm^2 (sigma / e).
    coherent, relaxation_term, lines : the three terms of Eq. 2 (C/m^2).
    reference : "H" or "ZB".
    """

    index: int
    z: float
    eps_c: float
    relaxation: float
    strain_below: float
    strain_above: float
    line_density: float
    line_charge: float
    bare_line_charge: float
    sigma: float
    sigma_nm2: float
    coherent: float
    relaxation_term: float
    lines: float
    reference: str

    @property
    def sigma_cm2(self):
        """Elementary charges per cm^2."""
        return self.sigma_nm2 * 1e14


def strained_polarization(m: NitrideMaterial, strain: float,
                          reference: str = "H") -> float:
    """Formal polarization (C/m^2) of `m` under biaxial in-plane strain."""
    e3 = -2.0 * m.c13_over_c33 * strain
    if reference == "H":
        return m.P_H + 2.0 * strain * (m.e31 - m.P_H) + e3 * m.e33
    if reference == "ZB":
        p_zb = m.P_H - zinc_blende_formal_polarization(m.a)
        return p_zb + 2.0 * strain * m.e31 + e3 * m.e33
    raise ValueError(f"reference must be 'H' or 'ZB'; got {reference!r}")


def misfit_line_density(r: float, eps_c: float, b_nm: float) -> float:
    """Length of misfit-dislocation line per area (nm^-1): 2 r |eps_c| / b."""
    if not (0.0 <= r <= 1.0):
        raise ValueError("relaxation must lie in [0, 1]")
    if not b_nm > 0.0:
        raise ValueError("b (nm) must be positive")
    return 2.0 * r * abs(eps_c) / b_nm


def _p_h_b(m):
    """P_H b expressed in e/nm."""
    return m.P_H * E_PER_NM2_PER_C_M2 * m.a


def line_charge_limits(film: NitrideMaterial, eps_c: float) -> dict:
    """Lambda (e/nm) of bare, electron-counting and neutral lines for a film
    of mismatch eps_c (sign: - for tension, + for compression)."""
    if eps_c == 0.0:
        raise ValueError("a coherent interface with eps_c = 0 has no misfit lines")
    s = -1.0 if eps_c > 0 else 1.0
    pzb = zinc_blende_formal_polarization(film.a) * E_PER_NM2_PER_C_M2 * film.a
    return {"bare": s * _p_h_b(film),
            "electron-counting": s * (_p_h_b(film) - pzb),
            "neutral": 0.0}


def _resolve(spec, film, eps_c, index):
    if spec is None:
        raise ValueError(
            f"layer {index + 1} has relaxed, so its interface charge depends on "
            "the charge of its misfit dislocations, which no reference "
            "decides. Pass line_charge= 'bare' (the H recipe), 'neutral' (the "
            "ZB recipe), 'electron-counting', LineCharge(bound_fraction, "
            "trapped), a number in e/nm, or a dict {interface index: spec}.")
    if isinstance(spec, dict):
        return _resolve(spec.get(index), film, eps_c, index)
    if isinstance(spec, str):
        if spec not in _KINDS:
            raise ValueError(f"unknown line charge {spec!r}; use one of {_KINDS}")
        return line_charge_limits(film, eps_c)[spec]
    if isinstance(spec, LineCharge):
        s = -1.0 if eps_c > 0 else 1.0
        return s * spec.bound_fraction * _p_h_b(film) - spec.trapped
    if isinstance(spec, (int, float)) and not isinstance(spec, bool) \
            and math.isfinite(spec):
        return float(spec)
    raise ValueError(f"cannot read the line charge {spec!r}")


def layer_strains(layers):
    """(eps_c, strain, in-plane lattice) of every layer; the bottom layer is
    relaxed (eps_c = strain = 0)."""
    if len(layers) < 1:
        raise ValueError("a stack needs at least one layer")
    for L in layers:
        if not isinstance(L, StackLayer):
            raise ValueError("layers must be StackLayer objects, bottom first")
    if layers[0].relaxation != 0.0:
        raise ValueError("the bottom layer is the relaxed substrate or buffer; "
                         "leave its relaxation at 0 (add the layer it relaxed "
                         "on below it if it is strained)")
    out = [(0.0, 0.0, layers[0].material.a)]
    for L in layers[1:]:
        aF = L.material.a
        a_below = out[-1][2]
        eps_c = (a_below - aF) / aF
        strain = (1.0 - L.relaxation) * eps_c
        out.append((eps_c, strain, aF * (1.0 + strain)))
    return out


def interface_charges(layers, line_charge=None, reference: str = "H"):
    """Fixed charge at each interface of the stack (list of `InterfaceCharge`).

    layers : `StackLayer` objects, bottom (relaxed substrate) first.
    line_charge : the misfit-line charge of every relaxed interface: "bare",
        "neutral", "electron-counting", `LineCharge`, a number in e/nm, or a
        dict keyed by interface index (0 = between layers 0 and 1). Needed
        only where a layer has relaxed; refused (with an explanation) when
        missing there.
    reference : "H" (default) or "ZB"; changes the result only at second
        order in the strain.
    """
    if reference not in ("H", "ZB"):
        raise ValueError(f"reference must be 'H' or 'ZB'; got {reference!r}")
    st = layer_strains(layers)
    out = []
    z = 0.0
    for i in range(len(layers) - 1):
        z += layers[i].thickness
        below, film = layers[i].material, layers[i + 1].material
        sb = st[i][1]
        eps_c, sa, _ = st[i + 1]
        r = layers[i + 1].relaxation
        p_b = strained_polarization(below, sb, reference)
        sigma_actual = p_b - strained_polarization(film, sa, reference)
        sigma_coh = p_b - strained_polarization(film, eps_c, reference)
        k = film.e31 - film.e33 * film.c13_over_c33
        relax_term = 2.0 * r * eps_c * k
        if r > 0.0 and eps_c != 0.0:
            LA = misfit_line_density(r, eps_c, film.a)
            lam = _resolve(line_charge, film, eps_c, i)
            lam_bare = line_charge_limits(film, eps_c)["bare"]
        else:
            LA, lam = 0.0, 0.0
            lam_bare = (line_charge_limits(film, eps_c)["bare"]
                        if eps_c != 0.0 else 0.0)
        lines = lam * LA / E_PER_NM2_PER_C_M2          # e/nm * nm^-1 -> C/m^2
        if reference == "H":
            sigma = sigma_actual + (lam - lam_bare) * LA / E_PER_NM2_PER_C_M2 \
                if LA else sigma_actual
        else:
            sigma = sigma_actual + lines
        out.append(InterfaceCharge(
            index=i, z=z, eps_c=eps_c, relaxation=r, strain_below=sb,
            strain_above=sa, line_density=LA, line_charge=lam,
            bare_line_charge=lam_bare, sigma=sigma,
            sigma_nm2=sigma * E_PER_NM2_PER_C_M2, coherent=sigma_coh,
            relaxation_term=relax_term, lines=lines, reference=reference))
    return out


def charge_change_per_relaxation(film: NitrideMaterial, eps_c: float,
                                 line_charge) -> float:
    """y = [sigma(r) - sigma_coh] / (2 r |eps_c|) (C/m^2), the slope of the
    interface charge against the relieved areal strain.

    With Lambda = -/+ f P_H b - q e this is y = +-(e31 - e33 C13/C33 - f P_H)
    - q e / b (upper sign: tension). For the same alloy under tension (y+)
    and under compression (y-), everything that comes from polarization
    cancels in the sum: y+ + y- = -2 q e / b (Eq. 3 of the misfit paper), so
    the sum measures the trapped electrons without any polarization constant.
    """
    if eps_c == 0.0:
        raise ValueError("eps_c must be nonzero")
    lam = _resolve(line_charge, film, eps_c, 0) / E_PER_NM2_PER_C_M2  # C/m
    k = film.e31 - film.e33 * film.c13_over_c33
    s = 1.0 if eps_c > 0 else -1.0
    return s * k + lam / film.a


def trapped_from_sum_rule(y_tension: float, y_compression: float,
                          b_nm: float) -> float:
    """Trapped electrons q (e/nm) from the two measured slopes (C/m^2):
    q = -(y+ + y-) b / (2 e). Needs no polarization constant."""
    if not b_nm > 0.0:
        raise ValueError("b (nm) must be positive")
    return -(y_tension + y_compression) * b_nm * E_PER_NM2_PER_C_M2 / 2.0
