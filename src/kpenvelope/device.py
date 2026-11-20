"""Two- and three-dimensional device simulation: current flow, traps that
fill and empty over time, self-heating (new in v0.19.0), and since
v0.20.0 contact resistance, small-signal response, pulsed heating,
impact ionization, gate tunneling, hot electrons and shaped devices.

This is the drift-diffusion model of `solve_drift_diffusion` on a
tensor-product grid in two dimensions (x along the channel, z up the
stack) or three (x, y, z), with electrodes on the top or bottom face:
ohmic source and drain (with the heavily doped region an alloyed contact
makes under them), Schottky gates (optionally with thermionic emission
into the metal), a free surface that is either pinned or carries its own
traps, deep-level traps in the layers, a mobility that falls at high
field and with temperature, and an optional heat equation coupled to the
current.

Unknowns at every node: the electrostatic potential psi, the electron
quasi-Fermi level F_n, and, when you give a hole mass, the hole
quasi-Fermi level F_p. In a transient, the electron occupancy f of every
trap is an unknown too. Equations (lengths nm, energies eV):

    div (eps0 eps_r grad psi) = -q (p - n + N_d+ - N_a- + traps) - sheets
    dn/dt + div Gamma_n = -R_n,      dp/dt + div Gamma_p = -R_p
    N_t df/dt = R_n - R_p            (each trap, transient only)

with Scharfetter-Gummel fluxes on every edge, written as in the 1D solver
(exact zero current for a flat quasi-Fermi level, Fermi-Dirac or
Boltzmann statistics). Traps follow Shockley-Read-Hall kinetics (W.
Shockley and W. T. Read, Phys. Rev. 87, 835 (1952); R. N. Hall, Phys.
Rev. 87, 387 (1952)): electrons are captured at c_n n (1 - f) and emitted
at e_n f with e_n = c_n N_c exp(-(E_c - E_t)/kT), holes likewise, so the
occupancy relaxes to the Fermi function in equilibrium (degeneracy 1;
detailed balance is exact for Boltzmann statistics). An acceptor-like
trap holds -q N_t f, a donor-like trap +q N_t (1 - f). In steady state f
is eliminated: f = (c_n n + e_p) / (c_n n + e_n + c_p p + e_p). Without
holes, traps exchange electrons with the conduction band only.

Mobility on each edge (D. M. Caughey and R. E. Thomas, Proc. IEEE 55,
2192 (1967), with a temperature power law):

    mu = mu_0 (T/300)^(-gamma) / [1 + (mu_0 (T/300)^(-gamma) E / v_sat)^beta]^(1/beta)

where E is the driving force along each edge: by default the electric
field along the edge, |d psi / dx| ("field", the parallel-field model of
most device simulators), or optionally the gradient of the quasi-Fermi
level, |dF_n/dx| ("quasi-fermi"); the two agree in a uniform region
(checked in the tests). The field is the default because it is smooth
where carriers are sparse: with the quasi-Fermi gradient the mobility of
a nearly empty region swings with a poorly defined level, which stalls
the Newton iteration at high drain voltage. All of mu_0, v_sat, beta,
gamma come from you with a reference (`Mobility`).

Self-heating: the heat put into the lattice at each node is the Joule
heat -Gamma_n . grad F_n + Gamma_p . grad F_p (on each edge) plus
R (F_n - F_p); summed over the device it equals the electrical power
sum V I exactly (checked in the tests). It drives `thermal.solve_heat`
on a grid that adds the substrate and any lateral extension; the lattice
temperature then sets the mobility, and the two are iterated to
consistency. The carrier statistics (N_c, kT in the fluxes) stay at the
ambient temperature: the model captures the mobility drop that causes
most of the current loss of a hot transistor, not the band-gap shrinkage
or thermoelectric currents.

Terminal currents are the balance of each continuity equation over the
electrode's nodes plus, in a transient, the change of the charge on the
electrode (displacement current); they sum to zero (checked). In 2D they
are per millimeter of device width (A/mm); in 3D in amperes.

Numerics: finite volumes; a fully coupled Newton method whose Jacobian
holds the exact derivatives of the fluxes and of Poisson's equation, and
central differences of the terms that live on one node (dopants, traps,
time derivatives), solved with SciPy's sparse LU (SciPy is required
here); bias ramps with adaptive
steps; backward Euler in time with adaptive steps. On a laterally uniform
problem with electrodes covering the top and bottom it reproduces the 1D
solver; a 3D device that is uniform along y reproduces the 2D one (both
checked in the tests).

Added in v0.20.0: series resistors at the terminals (contact and gate
resistance), the small-signal admittance at any frequency (with f_T and
f_max), heat that changes in time (`pulse`), impact ionization inside
the solve and the ionization integral along a field line (avalanche
breakdown), Tsu-Esaki tunneling through the gate barrier, the electron
temperature from an energy balance, and shapes: dielectric films on
top, metal boxes (recessed gates, field plates, contacts through a
film), electrodes in several pieces, and regions cut out of the
semiconductor.

Limits, stated plainly: drift-diffusion (no ballistic transport or
velocity overshoot, no quantum confinement of the channel; the electron
temperature is computed afterwards and does not feed back), the
heterojunction is a band step between two nodes (no thermionic-emission
condition at internal interfaces), tunneling only through the gate
barrier, avalanche breakdown only, linear small-signal response, and
the grid is a tensor product of boxes (no unstructured meshes).
"""
from __future__ import annotations

import dataclasses
import math

import numpy as np

from .electrons import EPS0, HBAR, KB, M0, fermi_dirac_half
from .interfaces import InterfaceCharge, StackLayer, interface_charges
from .nitride import QE
from .twod import _graded

__all__ = ["Mobility", "Electrode", "Trap", "Surface", "Device", "DeviceState",
           "TransientResult", "capture_coefficient", "with_series", "two_port",
           "rf_figures", "Ionization", "IMPACT_GAN", "Dielectric", "Region",
           "BandToBand", "TunnelBarrier"]

_CP = QE / EPS0 * 1e9          # V nm, densities in nm^-3


# ---------------------------------------------------------------- inputs
@dataclasses.dataclass(frozen=True)
class Mobility:
    """Carrier mobility of one layer (see the module docstring).

    low_field : mu_0 at 300 K, cm^2/(V s).
    reference : where the numbers come from (required).
    saturation_velocity : v_sat, cm/s (math.inf: no velocity saturation).
    beta : shape of the saturation (>= 1; 1 gradual, 2 sharper).
    temperature_exponent : gamma in mu_0 (T/300)^(-gamma). Positive when
        phonon scattering dominates (the mobility falls as the device warms);
        negative where it rises with temperature (since 0.21.0).
    """

    low_field: float
    reference: str
    saturation_velocity: float = math.inf
    beta: float = 1.0
    temperature_exponent: float = 0.0

    def __post_init__(self):
        if not (math.isfinite(self.low_field) and self.low_field > 0):
            raise ValueError("Mobility.low_field must be > 0 (cm^2/Vs)")
        if not (isinstance(self.reference, str) and self.reference.strip()):
            raise ValueError("Mobility needs a reference for its numbers")
        if not self.saturation_velocity > 0:
            raise ValueError("Mobility.saturation_velocity must be > 0 (cm/s)")
        if not (math.isfinite(self.beta) and self.beta >= 1):
            raise ValueError("Mobility.beta must be >= 1 (1 gradual, 2 sharper)")
        if not math.isfinite(self.temperature_exponent):
            raise ValueError("Mobility.temperature_exponent must be finite")

    def value(self, field_V_cm, temperature_K=300.0):
        """mu (cm^2/Vs) at a driving field (V/cm) and temperature (K), as
        the solver uses it: the field is rounded near zero,
        E -> sqrt(E^2 + E0^2) - E0 with E0 = 1e-3 v_sat / mu_0(T), which
        keeps the model smooth where the field changes sign and moves mu by
        at most 0.1 percent of its saturation term."""
        mu0 = self.low_field * (np.asarray(temperature_K, float) / 300.0) ** (
            -self.temperature_exponent)
        E = np.abs(np.asarray(field_V_cm, float))
        if math.isinf(self.saturation_velocity):
            return mu0 * np.ones_like(E)
        E0 = 1e-3 * self.saturation_velocity / mu0
        E = np.sqrt(E * E + E0 * E0) - E0
        return mu0 / (1.0 + (mu0 * E / self.saturation_velocity) ** self.beta) ** (
            1.0 / self.beta)


@dataclasses.dataclass(frozen=True)
class Electrode:
    """A contact on the top (or bottom) face.

    name : label used for its voltage and current.
    kind : "ohmic" or "schottky".
    x0, x1 : extent along x (nm); y0, y1 : along y (3D only; default the
        whole depth).
    barrier : Schottky barrier E_c - E_F(metal) (eV), schottky only.
    thermionic : schottky only; carriers cross into the metal at the
        Richardson velocity (flux v_R (n - n_0)) instead of being held at
        their equilibrium density.
    implant_depth, implant_donors : ohmic only; a region of this donor
        density (cm^-3) from the face down to this depth (nm) under the
        contact, as alloyed or implanted contacts make. Needed when the
        layer at the contact is undoped.
    face : "top" (default) or "bottom".
    z0, z1 : make the electrode a metal box from z0 to z1 (nm, measured
        from the bottom of the stack; above its top lies the cap): a
        recessed gate, or a field plate on a dielectric (new in 0.20.0).
        Every node in the box is metal. Its barrier is the one the metal
        has on the top layer of the stack; on another layer the offset of
        the bands adds to it.
    resistance : series resistance between the outside terminal and the
        contact (ohm mm in 2D, ohm in 3D): the contact resistance of an
        ohmic contact, the metal resistance of a gate, or any lumped
        resistor (new in 0.20.0). The voltages you apply are then at the
        outside terminal.
    """

    name: str
    kind: str
    x0: float
    x1: float
    barrier: float = None
    thermionic: bool = False
    implant_depth: float = None
    implant_donors: float = None
    face: str = "top"
    y0: float = None
    y1: float = None
    resistance: float = 0.0
    z0: float = None
    z1: float = None

    def __post_init__(self):
        if not (isinstance(self.name, str) and self.name):
            raise ValueError("Electrode.name must be a non-empty string")
        if self.kind not in ("ohmic", "schottky"):
            raise ValueError("Electrode.kind must be 'ohmic' or 'schottky'")
        if self.face not in ("top", "bottom"):
            raise ValueError("Electrode.face must be 'top' or 'bottom'")
        if not (math.isfinite(self.x0) and math.isfinite(self.x1) and self.x1 > self.x0):
            raise ValueError(f"electrode {self.name!r} needs x1 > x0")
        if self.kind == "schottky":
            if self.barrier is None or not math.isfinite(self.barrier):
                raise ValueError(f"Schottky electrode {self.name!r} needs a barrier (eV)")
            if self.implant_depth is not None:
                raise ValueError("an implant belongs to an ohmic contact")
        else:
            if self.barrier is not None or self.thermionic:
                raise ValueError(f"ohmic electrode {self.name!r} takes no barrier "
                                 "or thermionic option")
        if (self.implant_depth is None) != (self.implant_donors is None):
            raise ValueError("give both implant_depth and implant_donors, or neither")
        if self.implant_depth is not None and not (
                self.implant_depth > 0 and self.implant_donors > 0):
            raise ValueError("implant_depth (nm) and implant_donors (cm^-3) must be > 0")
        if not (math.isfinite(self.resistance) and self.resistance >= 0):
            raise ValueError("Electrode.resistance must be >= 0")
        if (self.z0 is None) != (self.z1 is None):
            raise ValueError("give both z0 and z1, or neither")
        if self.z0 is not None:
            if not self.z1 >= self.z0:
                raise ValueError(f"electrode {self.name!r} needs z1 >= z0")
        if (self.y0 is None) != (self.y1 is None):
            raise ValueError("give both y0 and y1, or neither")
        if self.y0 is not None and not self.y1 > self.y0:
            raise ValueError(f"electrode {self.name!r} needs y1 > y0")


@dataclasses.dataclass(frozen=True)
class Dielectric:
    """An insulating layer on top of the semiconductor stack (passivation,
    gate dielectric), new in 0.20.0. thickness nm; eps_r with its source."""

    label: str
    thickness: float
    eps_r: float
    reference: str

    def __post_init__(self):
        if not (math.isfinite(self.thickness) and self.thickness > 0):
            raise ValueError("Dielectric.thickness must be > 0 (nm)")
        if not (math.isfinite(self.eps_r) and self.eps_r >= 1.0):
            raise ValueError("Dielectric.eps_r must be >= 1")
        if not (isinstance(self.reference, str) and self.reference.strip()):
            raise ValueError("Dielectric needs a reference for its permittivity")


@dataclasses.dataclass(frozen=True)
class Region:
    """A box of the grid whose material is replaced by an insulator: an
    etched recess or mesa (eps_r = 1, empty) or a dielectric fill (new in
    0.20.0). Coordinates in nm; z measured from the bottom of the stack,
    as everywhere (the cap layers lie above the stack's top)."""

    x0: float
    x1: float
    z0: float
    z1: float
    eps_r: float = 1.0
    label: str = "etched"
    reference: str = "empty (vacuum)"
    y0: float = None
    y1: float = None

    def __post_init__(self):
        if not (self.x1 > self.x0 and self.z1 > self.z0):
            raise ValueError("Region needs x1 > x0 and z1 > z0")
        if not (math.isfinite(self.eps_r) and self.eps_r >= 1.0):
            raise ValueError("Region.eps_r must be >= 1")
        if self.eps_r != 1.0 and self.reference == "empty (vacuum)":
            raise ValueError("a filled Region needs a reference for its eps_r")
        if (self.y0 is None) != (self.y1 is None):
            raise ValueError("give both y0 and y1, or neither")


@dataclasses.dataclass(frozen=True)
class Trap:
    """A deep level.

    kind : "acceptor" (negative when it holds an electron) or "donor"
        (positive when empty).
    level : energy (eV) below E_c (from_band="ec") or above E_v ("ev").
    density : cm^-3 in layers; cm^-2 when where="surface".
    capture_n, capture_p : capture coefficients c = sigma v_th (cm^3/s);
        see `capture_coefficient`.
    reference : where the numbers come from (required).
    where : a layer label, a tuple of labels, or "surface" (the free top
        surface between the electrodes).
    z_range : optional (z0, z1) in nm to limit a bulk trap to part of the
        stack.
    """

    kind: str
    level: float
    from_band: str
    density: float
    capture_n: float
    capture_p: float
    reference: str
    where: object
    z_range: tuple = None

    def __post_init__(self):
        if self.kind not in ("acceptor", "donor"):
            raise ValueError("Trap.kind must be 'acceptor' or 'donor'")
        if self.from_band not in ("ec", "ev"):
            raise ValueError("Trap.from_band must be 'ec' or 'ev'")
        if not (math.isfinite(self.level) and self.level >= 0):
            raise ValueError("Trap.level must be >= 0 (eV)")
        if not (math.isfinite(self.density) and self.density >= 0):
            raise ValueError("Trap.density must be >= 0")
        for f in ("capture_n", "capture_p"):
            v = getattr(self, f)
            if not (math.isfinite(v) and v >= 0):
                raise ValueError(f"Trap.{f} must be >= 0 (cm^3/s)")
        if not (isinstance(self.reference, str) and self.reference.strip()):
            raise ValueError("Trap needs a reference for its numbers")
        if self.where is None:
            raise ValueError("Trap.where must name a layer, layers, or 'surface'")
        if self.z_range is not None:
            a, b = self.z_range
            if not b > a:
                raise ValueError("Trap.z_range needs z1 > z0")


@dataclasses.dataclass(frozen=True)
class Surface:
    """The free top surface between electrodes.

    kind="pinned": E_c - E_F = barrier there (surface states pin it), the
        model of the 1D solvers. kind="states": no pinning; the surface
        carries a fixed sheet charge plus any Trap(where="surface"), and
        the field outside is neglected.
    fixed_charge : for "states", e/cm^2; None takes minus the sum of the
        interface polarization charges (the stack's polarization charges
        then add up to zero, the usual convention when the buffer's
        spontaneous polarization is taken as compensated).
    """

    kind: str = "pinned"
    barrier: float = None
    fixed_charge: float = None

    def __post_init__(self):
        if self.kind not in ("pinned", "states"):
            raise ValueError("Surface.kind must be 'pinned' or 'states'")
        if self.kind == "pinned":
            if self.barrier is None or not math.isfinite(self.barrier):
                raise ValueError("a pinned surface needs its barrier (eV)")
        elif self.barrier is not None:
            raise ValueError("a surface with states takes no barrier")
        if self.fixed_charge is not None and not math.isfinite(self.fixed_charge):
            raise ValueError("Surface.fixed_charge must be finite (e/cm^2)")


@dataclasses.dataclass(frozen=True)
class Ionization:
    """Impact ionization (new in 0.20.0): alpha = a exp(-b / E) for
    electrons and holes (A. G. Chynoweth, Phys. Rev. 109, 1537 (1958)), a
    in 1/cm, b in V/cm, E the field along each edge.

    IMPACT_GAN holds the GaN coefficients of D. Ji, B. Ercan and S.
    Chowdhury, Appl. Phys. Lett. 115, 073503 (2019), doi:10.1063/1.5099245
    (measured on GaN homojunctions), as quoted again in D. Ji et al., Appl.
    Phys. Lett. 116, 211102 (2020).

    where : layer labels in which the coefficients apply (None: every
        semiconductor layer). Elsewhere nothing ionizes, so use it to keep
        GaN numbers out of AlGaN layers they were not measured for.
    """

    a_n: float
    b_n: float
    a_p: float
    b_p: float
    reference: str
    where: tuple = None

    def __post_init__(self):
        for f in ("a_n", "b_n", "a_p", "b_p"):
            v = getattr(self, f)
            if not (math.isfinite(v) and v >= 0):
                raise ValueError(f"Ionization.{f} must be >= 0")
        if not (isinstance(self.reference, str) and self.reference.strip()):
            raise ValueError("Ionization needs a reference for its numbers")
        if self.where is not None:
            w = (self.where,) if isinstance(self.where, str) else tuple(self.where)
            if not w or not all(isinstance(v, str) for v in w):
                raise ValueError("Ionization.where must be layer labels")
            object.__setattr__(self, "where", w)

    def alpha(self, field_V_cm):
        """(alpha_n, alpha_p) in 1/cm at the field (V/cm)."""
        E = np.abs(np.asarray(field_V_cm, float))
        with np.errstate(divide="ignore", over="ignore"):
            an = np.where(E > 0, self.a_n * np.exp(-self.b_n / np.where(E > 0, E, 1.0)), 0.0)
            ap = np.where(E > 0, self.a_p * np.exp(-self.b_p / np.where(E > 0, E, 1.0)), 0.0)
        return an, ap


IMPACT_GAN = Ionization(
    2.11e9, 3.689e7, 4.39e6, 1.8e7,
    "GaN: alpha_n = 2.11e9 exp(-3.689e7/E), alpha_p = 4.39e6 exp(-1.8e7/E) "
    "1/cm (E in V/cm): D. Ji, B. Ercan, S. Chowdhury, Appl. Phys. Lett. 115, "
    "073503 (2019), doi:10.1063/1.5099245, quoted in D. Ji et al., Appl. "
    "Phys. Lett. 116, 211102 (2020)")


@dataclasses.dataclass(frozen=True)
class TunnelBarrier:
    """A layer the electrons cross by tunneling and by emission over its
    top, not by drift and diffusion (new in 0.21.0). In every column of the
    grid the electron current from the node just below the layer to the
    node just above it (or to the metal of an electrode on top of it) is
    the Tsu-Esaki current with WKB transmission through the band profile
    of the solution (`leakage.tsu_esaki_current`), coupled into the
    Newton solve. Inside the layer the electron quasi-Fermi level is
    interpolated linearly between the two ends. Holes keep drift and
    diffusion.

    layer : the layer's label. mass_m0 : electron mass in the barrier
    (None: the layer's own mass). n_energy : energy points of the
    integral.
    """

    layer: str
    mass_m0: float = None
    n_energy: int = 400

    def __post_init__(self):
        if not isinstance(self.layer, str):
            raise ValueError("TunnelBarrier.layer must be a layer label")
        if self.mass_m0 is not None and not (math.isfinite(self.mass_m0) and self.mass_m0 > 0):
            raise ValueError("TunnelBarrier.mass_m0 must be > 0")
        if int(self.n_energy) < 50:
            raise ValueError("TunnelBarrier.n_energy must be at least 50")


@dataclasses.dataclass(frozen=True)
class BandToBand:
    """Band-to-band (Zener) tunneling (new in 0.21.0): electron-hole pairs
    made at the rate of Kane's two-band formula,

        G = q^2 F^2 / (18 pi hbar^2) sqrt(m_r / E_g)
            exp(-pi sqrt(m_r) E_g^(3/2) / (2 q hbar F)),

    with F the field at each node, E_g the band gap there and m_r the
    reduced mass of the two bands (as written in Eq. 13 of arXiv:2108.09328,
    after E. O. Kane). The device scales it by 1 - exp(-dV / kT), dV the
    spread of the electrode voltages, so it vanishes in equilibrium and is
    Kane's rate under any bias of more than a few kT. It is a generation
    only: tunneling recombination and forward-bias (Esaki) tunneling are
    not modeled, so use it for reverse-biased junctions and high-field
    regions. The generation follows the field of the solution (it is
    updated between solves until it settles). reduced_mass_m0: m_r in
    units of m0; None takes 1/m_r = 1/m_e + 1/m_h from the device's
    electron and hole masses. where: layer labels (None: every
    semiconductor layer). It needs holes.
    """

    reference: str
    reduced_mass_m0: float = None
    where: tuple = None

    def __post_init__(self):
        if not (isinstance(self.reference, str) and self.reference.strip()):
            raise ValueError("BandToBand needs a reference")
        if self.reduced_mass_m0 is not None and not (
                math.isfinite(self.reduced_mass_m0) and self.reduced_mass_m0 > 0):
            raise ValueError("BandToBand.reduced_mass_m0 must be > 0")
        if self.where is not None:
            w = (self.where,) if isinstance(self.where, str) else tuple(self.where)
            if not w or not all(isinstance(v, str) for v in w):
                raise ValueError("BandToBand.where must be layer labels")
            object.__setattr__(self, "where", w)

    @staticmethod
    def rate(field_V_cm, gap_eV, mr_m0):
        """(G in cm^-3 s^-1, dG/dF in cm^-3 s^-1 per V/cm)."""
        F = np.abs(np.asarray(field_V_cm, float)) * 100.0          # V/m
        Eg = np.asarray(gap_eV, float) * QE
        m = np.asarray(mr_m0, float) * M0
        pre = QE ** 2 / (18 * np.pi * HBAR ** 2) * np.sqrt(m / Eg)
        B = np.pi * np.sqrt(m) * Eg ** 1.5 / (2 * QE * HBAR)          # V/m
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            ex = np.where(F > 0, np.exp(-B / np.where(F > 0, F, 1.0)), 0.0)
            G = pre * F * F * ex
            dG = np.where(F > 0, pre * ex * (2 * F + B), 0.0)
        return G * 1e-6, dG * 1e-6 * 100.0


def capture_coefficient(cross_section_cm2: float, mass_m0: float,
                        temperature_K: float = 300.0) -> float:
    """c = sigma v_th with the mean thermal speed v_th = sqrt(8 kT / (pi m))
    (cm^3/s). Sources differ on v_th (some use sqrt(3 kT / m), 8.5 percent
    larger); use the one your cross section was quoted with."""
    if not (cross_section_cm2 >= 0 and mass_m0 > 0 and temperature_K > 0):
        raise ValueError("cross section >= 0, mass > 0, temperature > 0")
    v = math.sqrt(8 * KB * QE * temperature_K / (math.pi * mass_m0 * M0)) * 100.0
    return cross_section_cm2 * v


# --------------------------------------------------------------- results
@dataclasses.dataclass
class DeviceState:
    """A solution of `Device` (lengths nm, energies eV, densities cm^-3).

    axes : node arrays (x, z) or (x, y, z). psi, ec, efn, efp, n, p,
    temperature : arrays of shape (len(x), [len(y),] len(z)).
    occupancy : electron occupancy of each trap (list of arrays, same
    shape; NaN where the trap is absent). bias : {electrode: V}.
    current : {electrode: current into the device through it}, A/mm (2D)
    or A (3D). charge : {electrode: charge on it}, C/mm (2D) or C (3D).
    heat : power into the lattice at each node (W/mm or W, summed over
    its control volume). converged, iterations, time (transients).
    """

    axes: tuple
    psi: np.ndarray
    ec: np.ndarray
    efn: np.ndarray
    efp: np.ndarray
    n: np.ndarray
    p: np.ndarray
    temperature: np.ndarray
    occupancy: list
    bias: dict
    current: dict
    charge: dict
    heat: np.ndarray
    converged: bool
    iterations: int
    time: float = 0.0
    internal: dict = None
    _x: np.ndarray = dataclasses.field(default=None, repr=False)
    electron_temperature: np.ndarray = None

    def sheet(self, depth: float = 150.0):
        """Electrons per cm^2 below the top `depth` nm, at each lateral
        position (shape of the lateral grid)."""
        z = self.axes[-1]
        m = z >= z[-1] - depth - 1e-9
        trap = getattr(np, "trapezoid", None) or getattr(np, "trapz")
        return trap(self.n[..., m], z[m], axis=-1) * 1e-7

    @property
    def power(self):
        """Total electrical power sum V I at the outside terminals (W/mm in
        2D, W in 3D), series resistors included."""
        return float(sum(self.bias[k] * self.current[k] for k in self.current))

    @property
    def device_power(self):
        """Power delivered to the device itself, sum V_internal I (equals
        the heat put into the lattice); without series resistors it is
        `power`."""
        v = self.internal or self.bias
        return float(sum(v[k] * self.current[k] for k in self.current))


@dataclasses.dataclass
class TransientResult:
    """times (s), current {electrode: array}, mean occupancy of each trap
    species over where it sits (list of arrays), states (the full state at
    each requested time), steps (every time step taken, s)."""

    times: np.ndarray
    current: dict
    mean_occupancy: list
    states: list
    steps: np.ndarray = None


# ---------------------------------------------------------------- device
def _need(v, what, label):
    if not (isinstance(v, (int, float)) and math.isfinite(v)):
        raise ValueError(f"layer {label!r} has no {what} (NaN); the device "
                         "solver needs a cited value")
    return float(v)


def _cv(nodes):
    h = np.diff(nodes)
    v = np.zeros(nodes.size)
    v[:-1] += 0.5 * h
    v[1:] += 0.5 * h
    return v


def _axis_grid(length, marks, fine, h_max, growth):
    """Nodes from 0 to length with every mark a node, spacing `fine` at
    the marks (not at the ends unless they are marks too)."""
    pts = sorted({0.0, float(length)} | {float(m) for m in marks
                                          if 0.0 <= m <= length})
    out = []
    for a, b in zip(pts[:-1], pts[1:]):
        ha = fine if a in marks else h_max
        hb = fine if b in marks else h_max
        out.append(_graded(a, b, ha, hb, h_max, growth))
    return np.unique(np.round(np.concatenate(out), 9))


def _z_grid(layers, extra, dz_fine, fine_depth, dz_max, growth):
    edges = np.cumsum([0.0] + [L.thickness for L in layers])
    top = edges[-1]
    zf = top - min(fine_depth, top)
    zs = [np.linspace(zf, top, max(2, int(round((top - zf) / dz_fine)) + 1))]
    if zf > 0:
        zs.append(_graded(0.0, zf, dz_max, dz_fine, dz_max, growth))
    marks = list(edges) + [e for e in extra if 0 < e < top]
    z = np.unique(np.round(np.concatenate(zs + [np.array(marks)]), 9))
    keep = np.ones(z.size, bool)
    for e in marks:
        close = (np.abs(z - e) < 0.2 * dz_fine) & (np.abs(z - e) > 1e-9)
        keep &= ~close
    return z[keep]


_SMOOTH = 1e-3     # field rounding at zero, as a fraction of v_sat / mu_0


def _edge_mu(m, d, h, Te):
    """Mobility on each edge and its derivative with respect to the edge
    difference d (V, of psi or of the quasi-Fermi level). The field is
    rounded near zero, E = sqrt((d/h)^2 + E0^2) - E0 with E0 = 1e-3 v_sat /
    mu_0, so the mobility is smooth where the field changes sign; this
    changes mu by at most 0.1 percent of the saturation term."""
    mu0 = m["low_field"] * (Te / 300.0) ** (-m["temperature_exponent"])
    vs = m["saturation_velocity"]
    beta = m["beta"]
    fin = np.isfinite(vs)
    vsf = np.where(fin, vs, 1.0)
    g = d / h * 1e7                                   # V/cm, signed
    E0 = _SMOOTH * vsf / mu0
    root = np.sqrt(g * g + E0 * E0)
    E = root - E0
    dE = np.where(root > 0, g / np.where(root > 0, root, 1.0), 0.0) * 1e7 / h
    sv = np.where(fin, mu0 * E / vsf, 0.0)
    sb = sv ** beta
    mu = mu0 / (1.0 + sb) ** (1.0 / beta)
    with np.errstate(divide="ignore", invalid="ignore"):
        dmu_dE = np.where(fin, -mu * np.where(sv > 0, sv ** (beta - 1.0), 0.0)
                          * mu0 / vsf / (1.0 + sb), 0.0)
    return mu, dmu_dE * dE


class Device:
    """A 2D (x, z) or 3D (x, y, z) device. See the module docstring.

    layers : StackLayer objects, bottom first (uniform along x and y).
    width : extent along x (nm). depth : extent along y (nm); give it for
        a 3D device, leave it None for 2D.
    electrodes : Electrode objects (at least one ohmic).
    mobility : one Mobility for every layer, or a dict keyed by layer label.
    surface : a Surface.
    traps : Trap objects.
    hole_mass : hole density-of-states mass (m0), one or one per layer, to
        include holes; then hole_mobility (Mobility or dict) is needed.
    grid : optional dict with "x", ["y",] "z" node arrays; otherwise one is
        built: z spacing dz_fine over the top fine_depth nm growing to
        dz_max below; x (and y) spacing dx_fine at electrode edges growing
        to dx_max.
    density_floor_cm3 : carriers below this density do not hold up the
        convergence of their quasi-Fermi level (as in the 1D solver).
    """

    def __init__(self, layers, width: float, electrodes, mobility, surface,
                 traps=(), depth: float = None, temperature_K: float = 300.0,
                 hole_mass=None, hole_mobility=None, statistics: str = "fermi",
                 line_charge=None, reference: str = "H", interfaces=None,
                 extra_sheet: dict = None, grid: dict = None,
                 dz_fine: float = 0.5, fine_depth: float = 60.0,
                 dz_max: float = 100.0, dx_fine: float = 5.0,
                 dx_max: float = 100.0, growth: float = 1.3,
                 density_floor_cm3: float = 1.0,
                 driving_force: str = "field", generation=None,
                 ionization: "Ionization" = None, cap=(), regions=(),
                 heterojunctions: str = "drift-diffusion", band_to_band=None,
                 doping=None, tunnel_barriers=()):
        try:
            import scipy.sparse  # noqa: F401
        except ImportError as e:      # pragma: no cover - depends on install
            raise ImportError("the 2D/3D device solver needs SciPy "
                              "(pip install scipy)") from e
        layers = list(layers)
        for L in layers:
            if not isinstance(L, StackLayer):
                raise ValueError("layers must be StackLayer objects, bottom first")
        if not (math.isfinite(width) and width > 0):
            raise ValueError("width must be > 0 (nm)")
        if depth is not None and not (math.isfinite(depth) and depth > 0):
            raise ValueError("depth must be > 0 (nm) or None")
        if statistics not in ("fermi", "boltzmann"):
            raise ValueError("statistics must be 'fermi' or 'boltzmann'")
        if not isinstance(surface, Surface):
            raise ValueError("surface must be a Surface")
        electrodes = list(electrodes)
        if not electrodes or not all(isinstance(e, Electrode) for e in electrodes):
            raise ValueError("give a list of Electrode objects")
        # several Electrode entries with one name are pieces of one metal
        # (an L-shaped gate with its field plate, for example): one
        # voltage, one current. They must agree on everything but their box.
        pieces = {}
        for e in electrodes:
            pieces.setdefault(e.name, []).append(e)
        for name, ps in pieces.items():
            if len(ps) > 1:
                a = ps[0]
                for b_ in ps[1:]:
                    if (b_.kind, b_.barrier, b_.thermionic, b_.resistance) != (
                            a.kind, a.barrier, a.thermionic, a.resistance):
                        raise ValueError(f"the pieces of electrode {name!r} must have the "
                                         "same kind, barrier, thermionic and resistance")
                if a.kind == "schottky" and any(b_.z0 is None for b_ in ps):
                    raise ValueError(f"a Schottky electrode in pieces ({name!r}) is made of "
                                     "metal boxes (give each z0, z1)")
        self._pieces = pieces
        if not any(e.kind == "ohmic" for e in electrodes):
            raise ValueError("at least one ohmic electrode is needed")
        for e in electrodes:
            if not (0 <= e.x0 and e.x1 <= width):
                raise ValueError(f"electrode {e.name!r} lies outside 0..width")
            if depth is None and e.y0 is not None:
                raise ValueError("y0/y1 are for 3D devices (give depth)")
            if depth is not None and e.y0 is not None and not (0 <= e.y0 and e.y1 <= depth):
                raise ValueError(f"electrode {e.name!r} lies outside 0..depth")
        traps = list(traps)
        labels = [L.label for L in layers]
        for t in traps:
            if not isinstance(t, Trap):
                raise ValueError("traps must be Trap objects")
            if t.where == "surface":
                if surface.kind != "states":
                    raise ValueError("surface traps need Surface(kind='states')")
            else:
                ws = (t.where,) if isinstance(t.where, str) else tuple(t.where)
                for w in ws:
                    if w not in labels:
                        raise ValueError(f"trap layer {w!r} is not a layer label")
        self.holes = hole_mass is not None
        if self.holes and hole_mobility is None:
            raise ValueError("with holes, give hole_mobility")
        if not self.holes and any(L.acceptors > 0 for L in layers):
            raise ValueError("a layer has acceptors: give hole_mass and hole_mobility")
        if interfaces is None:
            interfaces = interface_charges(layers, line_charge=line_charge,
                                           reference=reference)
        if len(interfaces) != len(layers) - 1 or not all(
                isinstance(c, InterfaceCharge) for c in interfaces):
            raise ValueError("interfaces must be the interface_charges list")
        extra = dict(extra_sheet or {})

        if ionization is not None:
            if not isinstance(ionization, Ionization):
                raise ValueError("ionization must be an Ionization")
            if hole_mass is None:
                raise ValueError("impact ionization makes holes: give hole_mass "
                                 "and hole_mobility")
        self.ionization = ionization
        if heterojunctions not in ("drift-diffusion", "thermionic"):
            raise ValueError("heterojunctions must be 'drift-diffusion' or 'thermionic'")
        self.heterojunctions = heterojunctions
        if band_to_band is not None:
            if not isinstance(band_to_band, BandToBand):
                raise ValueError("band_to_band must be a BandToBand")
            if not self.holes:
                raise ValueError("band-to-band tunneling makes holes: give hole_mass "
                                 "and hole_mobility")
        self.band_to_band = band_to_band
        self._hydro = None
        if driving_force not in ("field", "quasi-fermi"):
            raise ValueError("driving_force must be 'field' or 'quasi-fermi'")
        self.force = driving_force
        self.layers, self.traps = layers, traps
        self.electrodes = [ps[0] for ps in pieces.values()]
        self.surface = surface
        self.T = float(temperature_K)
        self.kT = KB * self.T
        self.fermi = statistics == "fermi"
        self.floor = density_floor_cm3 * 1e-21
        self.dim = 2 if depth is None else 3
        self.width, self.depth = float(width), depth
        edges = np.cumsum([0.0] + [L.thickness for L in layers])
        self.top = edges[-1]
        cap = list(cap)
        regions = list(regions)
        for c_ in cap:
            if not isinstance(c_, Dielectric):
                raise ValueError("cap must be a list of Dielectric layers, bottom first")
        for r_ in regions:
            if not isinstance(r_, Region):
                raise ValueError("regions must be Region objects")
        self.cap, self.regions = cap, regions
        cap_edges = self.top + np.cumsum([0.0] + [c_.thickness for c_ in cap])
        self.ztop = cap_edges[-1]                       # top of the grid
        for e in electrodes:
            if e.z0 is not None and not (0 <= e.z0 and e.z1 <= self.ztop + 1e-9):
                raise ValueError(f"electrode {e.name!r} lies outside the grid in z")
            if e.z0 is None and e.face == "top" and cap:
                raise ValueError(f"with a cap, give electrode {e.name!r} a metal box "
                                 "(z0, z1) through the cap to the semiconductor")
        # ---- grid
        if grid is not None:
            x = np.asarray(grid["x"], float)
            z = np.asarray(grid["z"], float)
            y = np.asarray(grid["y"], float) if self.dim == 3 else None
        else:
            xm = set()
            for e in electrodes:
                xm |= {e.x0, e.x1}
            for r_ in regions:
                xm |= {r_.x0, r_.x1}
            x = _axis_grid(width, {v for v in xm if 0 <= v <= width}, dx_fine, dx_max, growth)
            zextra = [self.top - e.implant_depth for e in electrodes
                      if e.implant_depth is not None and e.face == "top"]
            zextra += [e.implant_depth for e in electrodes
                       if e.implant_depth is not None and e.face == "bottom"]
            for t in traps:
                if t.z_range is not None:
                    zextra += list(t.z_range)
            for r_ in regions:
                zextra += [r_.z0, r_.z1]
            for e in electrodes:
                if e.z0 is not None:
                    zextra += [e.z0, e.z1]
            z = _z_grid(layers, [v for v in zextra if v < self.top], dz_fine, fine_depth,
                        dz_max, growth)
            if cap:
                zc_marks = sorted({float(v) for v in list(cap_edges)
                                   + [v for v in zextra if v > self.top]})
                pts = []
                for a_, b_ in zip(zc_marks[:-1], zc_marks[1:]):
                    pts.append(_graded(a_, b_, dz_fine, dz_fine, max(dz_fine, 10.0), growth))
                z = np.unique(np.round(np.concatenate([z] + pts), 9))
            y = None
            if self.dim == 3:
                ym = set()
                for e in electrodes:
                    if e.y0 is not None:
                        ym |= {e.y0, e.y1}
                for r_ in regions:
                    if r_.y0 is not None:
                        ym |= {r_.y0, r_.y1}
                y = _axis_grid(depth, ym, dx_fine, dx_max, growth)
        if not (abs(x[0]) < 1e-9 and abs(x[-1] - width) < 1e-9):
            raise ValueError("x must run from 0 to width")
        if not (abs(z[0]) < 1e-9 and abs(z[-1] - self.ztop) < 1e-9):
            raise ValueError("z must run from 0 to the top of the stack (and cap)")
        for e_ in list(edges[1:]) + list(cap_edges[1:-1]):
            if np.min(np.abs(z - e_)) > 1e-6:
                raise ValueError("z must contain every layer edge")
        self.axes = (x, z) if self.dim == 2 else (x, y, z)
        self.shape = tuple(a.size for a in self.axes)
        N = int(np.prod(self.shape))
        self.N = N
        nz = z.size
        # ---- materials per node (by z; cap nodes take the top layer's
        # numbers, which only matter for the metal reference) and per cell
        idx = np.clip(np.searchsorted(edges, z, side="right") - 1, 0, len(layers) - 1)
        zm = 0.5 * (z[1:] + z[:-1])
        cidx = np.clip(np.searchsorted(edges, zm, side="right") - 1, 0, len(layers) - 1)
        in_cap = zm > self.top + 1e-9
        self.node_layer, self.cell_layer = idx, cidx
        self.cell_in_cap = in_cap
        eps = [_need(L.material.eps_r, "eps_r", L.label) for L in layers]
        ec = [_need(L.material.ec, "conduction-band edge ec", L.label) for L in layers]
        eg = [_need(L.material.eg, "band gap eg", L.label) for L in layers]
        me = [_need(L.material.me, "electron mass me", L.label) for L in layers]
        bshape = (1,) * (self.dim - 1) + (nz,)

        def full(v):
            return np.broadcast_to(np.asarray(v, float).reshape(bshape), self.shape).ravel().copy()

        self.Ec0 = full([ec[i] for i in idx])
        self.Eg = full([eg[i] for i in idx])
        self.me_node = full([me[i] for i in idx])
        self.Ec0_top = ec[-1]
        # a metal box's barrier is the one to the layer under its bottom
        # face (to other layers it follows the band offsets: one metal, one
        # potential)
        self._box_ec = {}
        for name, ps in pieces.items():
            z0s = [float(e.z0) for e in ps if e.z0 is not None]
            if z0s:
                zb = min(min(z0s), self.top) - 1e-6
                k = int(np.clip(np.searchsorted(edges, zb, side="right") - 1,
                                0, len(layers) - 1))
                self._box_ec[name] = ec[k]
        pref = (KB * self.T * QE / (2 * np.pi * HBAR ** 2)) ** 1.5 * 2.0 * 1e-27
        self.Nc = pref * (self.me_node * M0) ** 1.5
        if self.holes:
            hm = ([float(hole_mass)] * len(layers) if np.ndim(hole_mass) == 0
                  else [float(v) for v in hole_mass])
            self.Nv = pref * (full([hm[i] for i in idx]) * M0) ** 1.5
        else:
            self.Nv = np.zeros(N)
        self.ND = full([layers[i].donors for i in idx])
        self.Ed = full([np.nan if layers[i].donor_depth is None
                        else layers[i].donor_depth for i in idx])
        self.NA = full([layers[i].acceptors for i in idx])
        self.Ea = full([np.nan if layers[i].acceptor_depth is None
                        else layers[i].acceptor_depth for i in idx])
        # coordinates of every node
        grids = np.meshgrid(*self.axes, indexing="ij")
        self.coords = [g.ravel() for g in grids]
        # ---- a doping profile on top of the layers' (an implant, a
        # diffusion; new in 0.21.0): (N_d, N_a) in cm^-3 at each node,
        # fully ionized
        if doping is not None:
            if callable(doping):
                vals = np.array([doping(*pt) for pt in zip(*self.coords)], float)
            else:
                vals = np.asarray(doping, float).reshape(2, -1).T
            if vals.shape != (N, 2) or not (np.all(np.isfinite(vals)) and np.all(vals >= 0)):
                raise ValueError("doping must give (N_d, N_a) >= 0 in cm^-3 at every node")
            add_d, add_a = vals[:, 0] * 1e-21, vals[:, 1] * 1e-21
            if np.any(add_a > 0) and not self.holes:
                raise ValueError("a doping profile with acceptors needs hole_mass and "
                                 "hole_mobility")
            self.ND = self.ND + add_d
            self.NA = self.NA + add_a
            self.Ed = np.where(add_d > 0, np.nan, self.Ed)
            self.Ea = np.where(add_a > 0, np.nan, self.Ea)
        X, Z = self.coords[0], self.coords[-1]
        Y = self.coords[1] if self.dim == 3 else None
        zi = np.broadcast_to(np.arange(nz).reshape(bshape), self.shape).ravel()
        self.zi = zi
        # ---- cells: semiconductor or insulator, and permittivity
        cshape = tuple(sz - 1 for sz in self.shape)
        ccent = np.meshgrid(*[0.5 * (a[1:] + a[:-1]) for a in self.axes], indexing="ij")
        cz_idx = np.broadcast_to(np.arange(nz - 1).reshape((1,) * (self.dim - 1) + (-1,)),
                                 cshape)
        cell_semi = ~in_cap[cz_idx]
        cell_eps = np.array([eps[i] for i in cidx])[cz_idx].astype(float)
        for k_, c_ in enumerate(cap):
            mzc = (zm > cap_edges[k_] - 1e-9) & (zm < cap_edges[k_ + 1] + 1e-9)
            cell_eps = np.where(mzc[cz_idx], c_.eps_r, cell_eps)
        for r_ in regions:
            box = (ccent[0] > r_.x0) & (ccent[0] < r_.x1) & (ccent[-1] > r_.z0) & (ccent[-1] < r_.z1)
            if self.dim == 3 and r_.y0 is not None:
                box &= (ccent[1] > r_.y0) & (ccent[1] < r_.y1)
            cell_semi = np.where(box, False, cell_semi)
            cell_eps = np.where(box, r_.eps_r, cell_eps)
        self.cell_semi, self.cell_eps = cell_semi, cell_eps
        self.shaped = bool(cap or regions)
        # ---- optical (or any) generation, cm^-3 s^-1 -> nm^-3 s^-1
        if generation is None:
            self.Gopt = np.zeros(N)
        elif callable(generation):
            self.Gopt = np.array([float(generation(*pt)) for pt in zip(*self.coords)]) * 1e-21
        else:
            self.Gopt = np.broadcast_to(np.asarray(generation, float),
                                        self.shape).ravel() * 1e-21
        if not (np.all(np.isfinite(self.Gopt)) and np.all(self.Gopt >= 0)):
            raise ValueError("generation must be finite and >= 0 (cm^-3 s^-1)")
        # ---- control volumes, faces and edges from the cells
        hs = [np.diff(a) for a in self.axes]
        halfw = [0.5 * h_ for h_ in hs]

        def cell_prod(skip=None):
            out = np.ones(cshape)
            for b_ in range(self.dim):
                if b_ == skip:
                    continue
                sh = [1] * self.dim
                sh[b_] = cshape[b_]
                out = out * halfw[b_].reshape(sh)
            return out

        sub = cell_prod()                                   # cell volume / 2^dim
        vol_all = np.zeros(self.shape)
        vol_semi = np.zeros(self.shape)
        import itertools
        for off in itertools.product((0, 1), repeat=self.dim):
            sl = tuple(slice(o, o + cshape[k]) for k, o in enumerate(off))
            vol_all[sl] += sub
            vol_semi[sl] += np.where(cell_semi, sub, 0.0)
        self.vol = vol_semi.ravel()
        self.vol_all = vol_all.ravel()
        self.node_semi = self.vol > 0
        if not self.node_semi.any():
            raise ValueError("no semiconductor left in the grid")
        I = np.arange(N).reshape(self.shape)
        ei, ej, eh, gp, fa, ecells = [], [], [], [], [], []
        for ax in range(self.dim):
            si = [slice(None)] * self.dim
            sj = [slice(None)] * self.dim
            si[ax] = slice(0, -1)
            sj[ax] = slice(1, None)
            ii = I[tuple(si)]
            jj = I[tuple(sj)]
            eshape = ii.shape
            g_ = np.zeros(eshape)
            f_ = np.zeros(eshape)
            qa = cell_prod(skip=ax)                         # quarter face area per cell
            # the cells that hold this edge: along ax the same index; along
            # every other axis the cell on either side (if it exists)
            others = [b_ for b_ in range(self.dim) if b_ != ax]
            for off in itertools.product((-1, 0), repeat=len(others)):
                src = [slice(None)] * self.dim
                dst = [slice(None)] * self.dim
                for b_, o in zip(others, off):
                    n_e = eshape[b_]
                    if o == -1:          # cell index = node index - 1
                        dst[b_] = slice(1, n_e)
                        src[b_] = slice(0, n_e - 1)
                    else:                # cell index = node index
                        dst[b_] = slice(0, n_e - 1)
                        src[b_] = slice(0, n_e - 1)
                src, dst = tuple(src), tuple(dst)
                g_[dst] += (cell_eps * qa)[src]
                f_[dst] += np.where(cell_semi, qa, 0.0)[src]
            h = hs[ax][np.indices(eshape)[ax]]
            ei.append(ii.ravel())
            ej.append(jj.ravel())
            eh.append(h.ravel())
            gp.append((g_ / h).ravel())
            fa.append(f_.ravel())
        self.eaxis = np.concatenate([np.full(a.size, k) for k, a in enumerate(ei)])
        self.ei = np.concatenate(ei)
        self.ej = np.concatenate(ej)
        self.eh = np.concatenate(eh)
        self.gpsi = np.concatenate(gp)
        self.earea = np.concatenate(fa)                    # flux area (semiconductor)
        # every quarter face of every cell, for averaging cell properties
        # onto edges (mobility): edge index -> (cell flat index, weight)
        self._edge_cells = self._map_edge_cells(cshape, cell_semi, cell_prod)
        # lateral area of every node (for face electrodes and thermionic flux)
        cvs = [_cv(a) for a in self.axes]
        lat_area = np.ones(self.shape[:-1])
        for ax in range(self.dim - 1):
            sh = [1] * (self.dim - 1)
            sh[ax] = cvs[ax].size
            lat_area = lat_area * cvs[ax].reshape(sh)
        self.lat_area = np.broadcast_to(lat_area[..., None], self.shape).ravel().copy()
        # ---- semiconductor surface: faces between a semiconductor cell and
        # an insulator (or the open top of the grid without a cap)
        surf = np.zeros(self.shape)
        inter_sheet = np.zeros(self.shape)
        for ax in range(self.dim):
            qa = cell_prod(skip=ax) * 2 ** (self.dim - 1) / 2 ** (self.dim - 1)
            face = cell_prod(skip=ax) * 2 ** (self.dim - 1)      # full face area
            n_ax = cshape[ax]
            lo_sl = [slice(None)] * self.dim
            hi_sl = [slice(None)] * self.dim
            lo_sl[ax] = slice(0, n_ax - 1)
            hi_sl[ax] = slice(1, n_ax)
            a_semi = cell_semi[tuple(lo_sl)]
            b_semi = cell_semi[tuple(hi_sl)]
            fa_ = face[tuple(lo_sl)]
            mixed = np.where(a_semi != b_semi, fa_, 0.0)
            # the shared face sits on node index (cell index + 1) along ax
            others = [b_ for b_ in range(self.dim) if b_ != ax]
            for off in itertools.product((0, 1), repeat=len(others)):
                dst = [slice(None)] * self.dim
                dst[ax] = slice(1, n_ax)
                for b_, o in zip(others, off):
                    dst[b_] = slice(o, o + cshape[b_])
                surf[tuple(dst)] += mixed / 2 ** (self.dim - 1)
                if ax == self.dim - 1:
                    both = np.where(a_semi & b_semi, fa_, 0.0)
                    inter_sheet[tuple(dst)] += both / 2 ** (self.dim - 1)
            del qa
        if not cap:
            # the open top of the stack
            top_cells = [slice(None)] * self.dim
            top_cells[-1] = slice(cshape[-1] - 1, cshape[-1])
            face = cell_prod(skip=self.dim - 1) * 2 ** (self.dim - 1)
            open_top = np.where(cell_semi[tuple(top_cells)], face[tuple(top_cells)], 0.0)
            others = list(range(self.dim - 1))
            for off in itertools.product((0, 1), repeat=len(others)):
                dst = [slice(None)] * self.dim
                dst[-1] = slice(nz - 1, nz)
                for b_, o in zip(others, off):
                    dst[b_] = slice(o, o + cshape[b_])
                surf[tuple(dst)] += open_top / 2 ** (self.dim - 1)
        self.surf_area = surf.ravel()
        # ---- electrodes: nodes and implants
        self.enodes = {}
        for e in electrodes:
            if e.z0 is not None:
                m = (Z >= e.z0 - 1e-9) & (Z <= e.z1 + 1e-9)
            else:
                m = (zi == nz - 1) if e.face == "top" else (zi == 0)
            m = m & (X >= e.x0 - 1e-9) & (X <= e.x1 + 1e-9)
            if self.dim == 3 and e.y0 is not None:
                m &= (Y >= e.y0 - 1e-9) & (Y <= e.y1 + 1e-9)
            nodes = np.flatnonzero(m)
            if nodes.size == 0:
                raise ValueError(f"electrode {e.name!r} covers no grid node")
            for other, on in self.enodes.items():
                if other != e.name and np.intersect1d(on, nodes).size:
                    raise ValueError(f"electrodes {other!r} and {e.name!r} overlap")
            if e.name in self.enodes:
                nodes = np.union1d(self.enodes[e.name], nodes)
            self.enodes[e.name] = nodes
            if e.implant_depth is not None:
                box = (X >= e.x0 - 1e-9) & (X <= e.x1 + 1e-9)
                if e.face == "top":
                    box &= Z >= self.top - e.implant_depth - 1e-9
                else:
                    box &= Z <= e.implant_depth + 1e-9
                if self.dim == 3 and e.y0 is not None:
                    box &= (Y >= e.y0 - 1e-9) & (Y <= e.y1 + 1e-9)
                self.ND[box] = np.maximum(self.ND[box], e.implant_donors * 1e-21)
                self.Ed[box] = np.nan
        self.edict = {e.name: e for e in self.electrodes}
        on_any = np.zeros(N, bool)
        for nodes in self.enodes.values():
            on_any[nodes] = True
        self.surface_nodes = np.flatnonzero((self.surf_area > 0) & self.node_semi & ~on_any)
        self.insulator_nodes = np.flatnonzero(~self.node_semi & ~on_any)
        # ---- mobility per edge
        self._mob_n = self._edge_mobility(mobility, "mobility")
        self._mob_p = self._edge_mobility(hole_mobility, "hole_mobility") if self.holes else None
        # ---- sheet charges: interface polarization on the semiconductor
        # part of each interface plane; the fixed charge of a surface with
        # states on its semiconductor surface
        sheet = np.zeros(nz)
        for c in interfaces:
            s_ = c.sigma_nm2 + extra.get(c.index, 0.0)
            sheet[int(np.argmin(np.abs(z - edges[c.index + 1])))] += s_
        self.sheet = full(sheet) * inter_sheet.ravel()
        if surface.kind == "states":
            fc = (surface.fixed_charge * 1e-14 if surface.fixed_charge is not None
                  else -sum(c.sigma_nm2 + extra.get(c.index, 0.0) for c in interfaces))
            sn = self.surface_nodes
            self.sheet[sn] += fc * self.surf_area[sn]
        self.interfaces = list(interfaces)
        # ---- traps per node
        self._traps = []
        for t in traps:
            Nt = np.zeros(N)
            if t.where == "surface":
                sn = self.surface_nodes
                Nt[sn] = t.density * 1e-14 * self.surf_area[sn] / self.vol[sn]
            else:
                ws = (t.where,) if isinstance(t.where, str) else tuple(t.where)
                inl = np.isin(np.array(labels, dtype=object)[idx], ws)
                m = (full(inl.astype(float)) > 0.5) & self.node_semi
                if t.z_range is not None:
                    m &= (Z >= t.z_range[0] - 1e-9) & (Z <= t.z_range[1] + 1e-9)
                Nt[m] = t.density * 1e-21
            if t.from_band == "ec":
                lev_c, lev_v = t.level, self.Eg - t.level
            else:
                lev_c, lev_v = self.Eg - t.level, t.level
            if np.any(np.asarray(lev_v + 0 * self.Eg)[Nt > 0] < 0) or np.any(
                    np.asarray(lev_c + 0 * self.Eg)[Nt > 0] < 0):
                raise ValueError("a trap level lies outside the band gap")
            cn = t.capture_n * 1e21
            cp = t.capture_p * 1e21 if self.holes else 0.0
            n1 = self.Nc * np.exp(-np.asarray(lev_c, float) / self.kT)
            p1 = (self.Nv * np.exp(-np.asarray(lev_v, float) / self.kT)
                  if self.holes else np.zeros(N))
            self._traps.append(dict(
                Nt=Nt, cn=cn, cp=cp, en=cn * n1, ep=cp * p1,
                acc=t.kind == "acceptor", where=Nt > 0))
        # ---- Dirichlet data that does not change with bias
        self._neutral = self._neutral_psi()
        self.vR = np.sqrt(KB * QE * self.T / (2 * np.pi * self.me_node * M0)) * 1e9  # nm/s
        if self.holes:
            hmn = (self.Nv / pref) ** (2.0 / 3.0) / M0
            self.vRp = np.sqrt(KB * QE * self.T / (2 * np.pi * hmn * M0)) * 1e9
        self._x0 = None
        self._setup_heterojunctions()
        self._setup_band_to_band()
        self._setup_tunnel_barriers(tunnel_barriers)

    # --------------------------------------------- heterojunctions (0.21.0)
    def _setup_heterojunctions(self):
        """Edges across an abrupt band step, for thermionic emission: the
        two nodes lie in different layers and their band edges differ."""
        i, j = self.ei, self.ej
        self._te = None
        if self.heterojunctions != "thermionic":
            return
        nl = np.broadcast_to(self.node_layer.reshape((1,) * (self.dim - 1) + (-1,)),
                             self.shape).ravel()
        cross = (nl[i] != nl[j]) & (self.earea > 0) & self.node_semi[i] & self.node_semi[j]
        dEc = self.Ec0[j] - self.Ec0[i]
        Ev0 = self.Ec0 - self.Eg
        dEv = Ev0[j] - Ev0[i]
        en = cross & (np.abs(dEc) > 1e-9)
        ep = cross & (np.abs(dEv) > 1e-9) if self.holes else np.zeros_like(en)
        self._te = dict(
            n=np.flatnonzero(en), n_hi_j=dEc[en] > 0,
            p=np.flatnonzero(ep), p_hi_j=dEv[ep] < 0)

    def _te_fluxes(self, fn, fp, n, p, Gn, Gp, deriv=None):
        """Replace the drift-diffusion flux on every band-step edge by the
        thermionic-emission flux, Gamma = v_R n_b [exp(dF/kT) - 1] with the
        density n_b and Richardson velocity v_R = sqrt(kT / 2 pi m) of the
        barrier side (the side of the higher band edge) and dF the
        quasi-Fermi step that drives the electrons over it. In
        equilibrium it vanishes exactly. With `deriv` (the dicts of
        `_flux_derivatives`, plus the occupancy ratios) the derivatives are
        replaced too."""
        te = self._te
        if te is None:
            return Gn, Gp
        kT = self.kT
        i, j = self.ei, self.ej
        Gn = Gn.copy()
        e = te["n"]
        if e.size:
            hj = te["n_hi_j"]
            ii, jj = i[e], j[e]
            dfn = np.clip((fn[jj] - fn[ii]) / kT, -700, 700)
            hi = np.where(hj, jj, ii)
            vR = self.vR[hi]
            nb = n[hi]
            sgn = np.where(hj, 1.0, -1.0)
            arg = np.where(hj, -dfn, dfn)
            Gn[e] = sgn * vR * nb * np.expm1(arg)
            if deriv is not None:
                d, rn_ = deriv["n"], deriv["rn"]
                ea = np.exp(arg)
                dnb = nb * rn_[hi] / kT                    # d n_b / d F = d n_b / d psi
                # d arg / d fn_i = +1/kT if hi=j else -1/kT ; d arg / d fn_j = -that
                da_i = np.where(hj, 1.0, -1.0) / kT
                fi = sgn * vR * nb * ea * da_i + np.where(hj, 0.0, sgn * vR * dnb * np.expm1(arg))
                fj = -sgn * vR * nb * ea * da_i + np.where(hj, sgn * vR * dnb * np.expm1(arg), 0.0)
                pi = np.where(hj, 0.0, sgn * vR * dnb * np.expm1(arg))
                pj = np.where(hj, sgn * vR * dnb * np.expm1(arg), 0.0)
                for k_, v_ in (("fi", fi), ("fj", fj), ("pi", pi), ("pj", pj)):
                    d[k_] = d[k_].copy()
                    d[k_][e] = v_
        if self.holes:
            Gp = Gp.copy()
            e = te["p"]
            if e.size:
                hj = te["p_hi_j"]
                ii, jj = i[e], j[e]
                dfp = np.clip((fp[jj] - fp[ii]) / kT, -700, 700)
                hi = np.where(hj, jj, ii)
                vR = self.vRp[hi]
                pb = p[hi]
                sgn = np.where(hj, 1.0, -1.0)
                arg = np.where(hj, dfp, -dfp)
                Gp[e] = sgn * vR * pb * np.expm1(arg)
                if deriv is not None:
                    d, rp_ = deriv["p"], deriv["rp"]
                    ea = np.exp(arg)
                    dpb = -pb * rp_[hi] / kT
                    da_i = np.where(hj, -1.0, 1.0) / kT
                    em = np.expm1(arg)
                    fi = sgn * vR * pb * ea * da_i + np.where(hj, 0.0, sgn * vR * dpb * em)
                    fj = -sgn * vR * pb * ea * da_i + np.where(hj, sgn * vR * dpb * em, 0.0)
                    pi = np.where(hj, 0.0, sgn * vR * dpb * em)
                    pj = np.where(hj, sgn * vR * dpb * em, 0.0)
                    for k_, v_ in (("fi", fi), ("fj", fj), ("pi", pi), ("pj", pj)):
                        d[k_] = d[k_].copy()
                        d[k_][e] = v_
        return Gn, Gp

    # ------------------------------------------- tunnel barriers (0.21.0)
    def _setup_tunnel_barriers(self, tbs):
        self._tb = None
        tbs = list(tbs)
        if not tbs:
            self._tb_block = None
            return
        labels = [L.label for L in self.layers]
        edges = np.cumsum([0.0] + [L.thickness for L in self.layers])
        z = self.axes[-1]
        nz = z.size
        ncol = self.N // nz
        on_el = np.zeros(self.N, bool)
        for nodes in self.enodes.values():
            on_el[nodes] = True
        links, interior, blocked_nodes = [], [], np.zeros(self.N, bool)
        for tb in tbs:
            if not isinstance(tb, TunnelBarrier):
                raise ValueError("tunnel_barriers must be TunnelBarrier objects")
            if labels.count(tb.layer) != 1:
                raise ValueError(f"TunnelBarrier: {tb.layer!r} must name exactly one layer")
            k = labels.index(tb.layer)
            if k == 0:
                raise ValueError("a tunnel barrier needs a layer below it")
            zb0, zb1 = edges[k], edges[k + 1]
            below = np.flatnonzero(z < zb0 - 1e-9)
            iz_a = int(below[-1])
            iz_b = int(np.argmin(np.abs(z - zb1)))
            if abs(z[iz_b] - zb1) > 1e-6:
                raise ValueError("the grid has no node at the top of the tunnel barrier")
            top_layer = k == len(self.layers) - 1
            ins = np.arange(iz_a + 1, iz_b + (1 if top_layer else 0))
            mass = tb.mass_m0
            for c in range(ncol):
                a = c * nz + iz_a
                b = c * nz + iz_b
                inner = c * nz + ins
                if not (self.node_semi[a] and np.all(self.node_semi[inner])):
                    continue
                blocked_nodes[inner] = True
                if top_layer:
                    # the layer ends at the top: a link only into a metal on it
                    target = b if on_el[b] else None
                else:
                    target = b
                path = np.arange(a, (target if target is not None else a) + 1) if target is not None else None
                links.append(dict(a=a, b=target, path=path, mass=mass, n_energy=int(tb.n_energy),
                                  metal=bool(target is not None and on_el[target])))
                zb = z[iz_b] if target is not None else None
                for nd in inner:
                    if on_el[nd]:
                        continue
                    if target is None:
                        interior.append((nd, a, a, 1.0))
                    else:
                        w = (zb - z[nd % nz]) / (zb - z[iz_a])
                        interior.append((nd, a, target, w))
        i, j = self.ei, self.ej
        self._tb_block = ~(blocked_nodes[i] | blocked_nodes[j])
        live = [L_ for L_ in links if L_["b"] is not None]
        self._tb = dict(links=live, interior=interior,
                        int_nodes=np.array([t[0] for t in interior], int),
                        int_a=np.array([t[1] for t in interior], int),
                        int_b=np.array([t[2] for t in interior], int),
                        int_w=np.array([t[3] for t in interior], float))
        # group links of the same length so they are evaluated together
        groups = {}
        for L_ in live:
            groups.setdefault((L_["path"].size, L_["n_energy"], L_["metal"]), []).append(L_)
        self._tb["groups"] = [
            dict(a=np.array([L_["a"] for L_ in g]), b=np.array([L_["b"] for L_ in g]),
                 path=np.stack([L_["path"] for L_ in g]),
                 mass=np.stack([np.full(L_["path"].size, L_["mass"]) if L_["mass"] is not None
                                else self.me_node[L_["path"]] for L_ in g]),
                 n_energy=key[1], metal=key[2])
            for key, g in groups.items()]

    def _block_fluxes(self, G):
        if getattr(self, "_tb_block", None) is None:
            return G
        return G * self._tb_block

    def _link_current(self, grp, psi, fn):
        """Electron flux (nm^-2 s^-1) from a to b of every link of a group:
        the Tsu-Esaki current, evaluated for all links at once."""
        from .leakage import richardson_constant
        path = grp["path"]
        z = self.axes[-1][path % self.axes[-1].size] * 1e-9          # m
        ec = self.Ec0[path] - psi[path]
        m = grp["mass"] * M0
        if grp["metal"]:
            # the far end is a metal: its electrons fill every energy (a band
            # edge far below), as in `gate_tunneling`
            z = np.concatenate([z, z[:, -1:] + 1e-15], axis=1)
            ec = np.concatenate([ec, ec[:, -1:] - 50.0], axis=1)
            m = np.concatenate([m, m[:, -1:]], axis=1)
        Fa, Fb = fn[grp["a"]], fn[grp["b"]]
        kT = self.kT
        e_lo = np.maximum(ec[:, 0], ec[:, -1])
        e_hi = np.maximum(np.maximum(ec.max(axis=1), Fa), Fb) + 40 * kT
        t = np.linspace(0.0, 1.0, grp["n_energy"])
        E = e_lo[:, None] + (e_hi - e_lo)[:, None] * t[None, :]     # (L, nE)
        dE = np.clip(ec[:, None, :] - E[:, :, None], 0.0, None) * QE
        kap = np.sqrt(2.0 * m[:, None, :] * dE) / HBAR
        integ = np.sum(0.5 * (kap[:, :, 1:] + kap[:, :, :-1]) * np.diff(z, axis=1)[:, None, :],
                       axis=2)
        tr = np.exp(-2.0 * integ)
        supply = (np.logaddexp(0.0, (Fa[:, None] - E) / kT)
                  - np.logaddexp(0.0, (Fb[:, None] - E) / kT))
        f = tr * supply
        integral = np.sum(0.5 * (f[:, 1:] + f[:, :-1]) * np.diff(E, axis=1), axis=1)
        mt = self.me_node[grp["a"]]
        Ar = np.array([richardson_constant(float(v)) for v in mt])
        J = Ar * self.T / KB * integral                                 # A/cm^2
        return J / QE * 1e-14

    def _link_balance(self, psi, fn):
        out = np.zeros(self.N)
        if self._tb is None:
            return out
        for grp in self._tb["groups"]:
            G = self._link_current(grp, psi, fn)
            A = self.lat_area[grp["a"]]
            np.add.at(out, grp["a"], G * A)
            np.add.at(out, grp["b"], -G * A)
        return out

    def _link_jacobian(self, psi, fn, nv):
        """Derivatives of the link fluxes by central differences, with
        respect to the two quasi-Fermi levels and the potential along each
        link."""
        rows, cols, vals = [], [], []
        if self._tb is None:
            return rows, cols, vals
        d = 1e-6
        for grp in self._tb["groups"]:
            A = self.lat_area[grp["a"]]
            a, b = grp["a"], grp["b"]
            for which, idx in (("fa", a), ("fb", b)):
                fp_, fm_ = fn.copy(), fn.copy()
                fp_[idx] += d
                fm_[idx] -= d
                dG = (self._link_current(grp, psi, fp_) - self._link_current(grp, psi, fm_)) / (2 * d)
                rows += [a * nv + 1, b * nv + 1]
                cols += [idx * nv + 1, idx * nv + 1]
                vals += [dG * A, -dG * A]
            for k in range(grp["path"].shape[1]):
                idx = grp["path"][:, k]
                pp, pm = psi.copy(), psi.copy()
                pp[idx] += d
                pm[idx] -= d
                dG = (self._link_current(grp, pp, fn) - self._link_current(grp, pm, fn)) / (2 * d)
                rows += [a * nv + 1, b * nv + 1]
                cols += [idx * nv, idx * nv]
                vals += [dG * A, -dG * A]
        return rows, cols, vals

    # ----------------------------------------- band-to-band tunneling (0.21.0)
    def _setup_band_to_band(self):
        bt = self.band_to_band
        self._bt = None
        if bt is None:
            return
        if bt.where is None:
            mask = self.node_semi.copy()
        else:
            labels = [L.label for L in self.layers]
            bad = [w for w in bt.where if w not in labels]
            if bad:
                raise ValueError(f"BandToBand.where names no layer: {bad}")
            lab = np.array([labels[k] for k in self.node_layer])
            nl = np.broadcast_to(np.isin(lab, bt.where).reshape(
                (1,) * (self.dim - 1) + (-1,)), self.shape).ravel()
            mask = self.node_semi & nl
        if bt.reduced_mass_m0 is not None:
            mr = np.full(self.N, float(bt.reduced_mass_m0))
        else:
            pref = (KB * self.T * QE / (2 * np.pi * HBAR ** 2)) ** 1.5 * 2.0 * 1e-27
            mh = (self.Nv / pref) ** (2.0 / 3.0) / M0
            mr = 1.0 / (1.0 / self.me_node + 1.0 / mh)
        w = self.earea * self.eh
        m_ = np.concatenate([self.ei, self.ej])
        e_ = np.concatenate([np.arange(self.ei.size)] * 2)
        ax = self.eaxis[e_]
        W = np.zeros((self.N, self.dim))
        np.add.at(W, (m_, ax), w[e_])
        self._bt = dict(mask=mask, mr=mr, m=m_, e=e_, ax=ax, w=w[e_],
                        W=np.where(W > 0, W, 1.0))

    def _node_field(self, psi):
        """Field components at every node (V/cm), the semiconductor-weighted
        mean of the edge fields along each axis, and the pieces for the
        derivatives."""
        bt = self._bt
        g = -(psi[self.ej] - psi[self.ei]) / self.eh * 1e7
        Ecomp = np.zeros((self.N, self.dim))
        np.add.at(Ecomp, (bt["m"], bt["ax"]), bt["w"] * g[bt["e"]])
        Ecomp = Ecomp / bt["W"]
        return Ecomp

    def _band_to_band_nodes(self, psi, spread=None):
        """Pairs generated per second at every node (rate times volume).
        spread: the spread of the electrode Fermi levels (eV); the rate is
        scaled by 1 - exp(-spread / kT), which makes it vanish in
        equilibrium."""
        bt = self._bt
        Ecomp = self._node_field(psi)
        F = np.sqrt(np.sum(Ecomp ** 2, axis=1))
        G, _ = BandToBand.rate(F, self.Eg, bt["mr"])
        phi = 1.0 if spread is None else -math.expm1(-abs(spread) / self.kT)
        return G * self.vol * 1e-21 * bt["mask"] * phi
    def _map_edge_cells(self, cshape, cell_semi, cell_prod):
        """For every edge, the cells around it with their (semiconductor)
        quarter-face weights: arrays (edge, cell, weight)."""
        import itertools
        E_, C_, W_ = [], [], []
        start = 0
        cflat = np.arange(int(np.prod(cshape))).reshape(cshape)
        for ax in range(self.dim):
            eshape = list(self.shape)
            eshape[ax] -= 1
            eshape = tuple(eshape)
            eidx = np.arange(int(np.prod(eshape))).reshape(eshape) + start
            qa = np.where(cell_semi, cell_prod(skip=ax), 0.0)
            others = [b_ for b_ in range(self.dim) if b_ != ax]
            for off in itertools.product((-1, 0), repeat=len(others)):
                src = [slice(None)] * self.dim
                dst = [slice(None)] * self.dim
                for b_, o in zip(others, off):
                    n_e = eshape[b_]
                    if o == -1:
                        dst[b_] = slice(1, n_e)
                        src[b_] = slice(0, n_e - 1)
                    else:
                        dst[b_] = slice(0, n_e - 1)
                        src[b_] = slice(0, n_e - 1)
                E_.append(eidx[tuple(dst)].ravel())
                C_.append(cflat[tuple(src)].ravel())
                W_.append(qa[tuple(src)].ravel())
            start += int(np.prod(eshape))
        return np.concatenate(E_), np.concatenate(C_), np.concatenate(W_)

    def _edge_mobility(self, mob, what):
        labels = [L.label for L in self.layers]
        if isinstance(mob, Mobility):
            per = [mob] * len(self.layers)
        elif isinstance(mob, dict):
            per = []
            for lab in labels:
                if lab not in mob or not isinstance(mob[lab], Mobility):
                    raise ValueError(f"{what} has no Mobility for layer {lab!r}")
                per.append(mob[lab])
        else:
            raise ValueError(f"{what} must be a Mobility or a dict of them")
        arr = {k: np.array([getattr(m, k) for m in per], float)
               for k in ("low_field", "saturation_velocity", "beta",
                         "temperature_exponent")}
        E_, C_, W_ = self._edge_cells
        nE = self.ei.size
        cz = np.unravel_index(C_, tuple(sz - 1 for sz in self.shape))[-1]
        layer_of_cell = self.cell_layer[cz]
        wsum = np.bincount(E_, W_, nE)
        safe = np.where(wsum > 0, wsum, 1.0)
        out = {}
        for k, v in arr.items():
            cv = v[layer_of_cell]
            if k == "saturation_velocity":
                with np.errstate(divide="ignore"):
                    inv = np.bincount(E_, W_ / cv, nE) / safe
                with np.errstate(divide="ignore"):
                    e = np.where(inv > 0, 1.0 / np.where(inv > 0, inv, 1.0), np.inf)
            else:
                e = np.bincount(E_, W_ * cv, nE) / safe
            # edges with no semiconductor carry no current; any value works
            e = np.where(wsum > 0, e, v[-1])
            out[k] = e
        return out

    def _occ(self, eta):
        eta = np.clip(eta, -700.0, 700.0)
        if not self.fermi:
            return np.exp(eta), eta
        F, _ = fermi_dirac_half(eta)
        F = np.asarray(F, float)
        return F, np.where(eta < -40.0, eta, np.log(np.maximum(F, 1e-300)))

    def _carriers(self, psi, fn, fp):
        Ec = self.Ec0 - psi
        Fn_, lFn = self._occ((fn - Ec) / self.kT)
        n = self.Nc * Fn_
        lnn = np.log(self.Nc) + lFn
        if self.holes:
            Ev = Ec - self.Eg
            Fp_, lFp = self._occ((Ev - fp) / self.kT)
            p = self.Nv * Fp_
            lnp = np.log(self.Nv) + lFp
        else:
            p = np.zeros_like(n)
            lnp = np.full_like(n, -745.0)
        return Ec, n, p, lnn, lnp

    def _dopants(self, Ec, fn, fp):
        kT = self.kT
        ndp = self.ND.copy()
        m = np.isfinite(self.Ed) & (self.ND > 0)
        if m.any():
            ndp[m] = self.ND[m] / (1.0 + 2.0 * np.exp(np.clip(
                (fn[m] - Ec[m] + self.Ed[m]) / kT, -700, 700)))
        nam = self.NA.copy()
        m = np.isfinite(self.Ea) & (self.NA > 0)
        if m.any():
            Ev = Ec[m] - self.Eg[m]
            nam[m] = self.NA[m] / (1.0 + 4.0 * np.exp(np.clip(
                (Ev + self.Ea[m] - fp[m]) / kT, -700, 700)))
        return ndp, nam

    def _f_steady(self, tr, n, p):
        return (tr["cn"] * n + tr["ep"]) / (tr["cn"] * n + tr["en"] + tr["cp"] * p
                                            + tr["ep"] + 1e-300)

    def _rho_local(self, psi, fn, fp, focc=None):
        Ec, n, p, lnn, lnp = self._carriers(psi, fn, fp)
        ndp, nam = self._dopants(Ec, fn, fp)
        rho = p - n + ndp - nam
        for k, tr in enumerate(self._traps):
            f = self._f_steady(tr, n, p) if focc is None else focc[k]
            rho += -tr["Nt"] * f if tr["acc"] else tr["Nt"] * (1.0 - f)
        return rho

    def _neutral_psi(self):
        """Charge-neutral psi (at F = 0) of every ohmic-contact node (the
        semiconductor ones; metal inside an insulator takes their mean)."""
        out = {}
        for e in self.electrodes:
            if e.kind != "ohmic":
                continue
            all_nodes = self.enodes[e.name]
            nodes = all_nodes[self.node_semi[all_nodes]]
            if nodes.size == 0:
                raise ValueError(f"the ohmic contact {e.name!r} touches no semiconductor")
            lo = self.Ec0[nodes] - self.Eg[nodes] - 2.0
            hi = self.Ec0[nodes] + 2.0
            zero = np.zeros(nodes.size)

            def g(ps):
                Ec = self.Ec0[nodes] - ps
                Fn_, _ = self._occ(-Ec / self.kT)
                n = self.Nc[nodes] * Fn_
                if self.holes:
                    Fp_, _ = self._occ((Ec - self.Eg[nodes]) / self.kT)
                    p = self.Nv[nodes] * Fp_
                else:
                    p = 0.0 * n
                nd = self.ND[nodes].copy()
                m = np.isfinite(self.Ed[nodes]) & (nd > 0)
                nd[m] = nd[m] / (1 + 2 * np.exp(np.clip((-Ec[m] + self.Ed[nodes][m])
                                                        / self.kT, -700, 700)))
                na = self.NA[nodes].copy()
                m = np.isfinite(self.Ea[nodes]) & (na > 0)
                Ev = Ec - self.Eg[nodes]
                na[m] = na[m] / (1 + 4 * np.exp(np.clip((Ev[m] + self.Ea[nodes][m])
                                                        / self.kT, -700, 700)))
                rho = p - n + nd - na
                for tr in self._traps:
                    f = self._f_steady({k: (v[nodes] if isinstance(v, np.ndarray) else v)
                                        for k, v in tr.items() if k in
                                        ("cn", "cp", "en", "ep")}, n, p)
                    Nt = tr["Nt"][nodes]
                    rho += -Nt * f if tr["acc"] else Nt * (1 - f)
                return rho

            if not (np.all(g(lo) > 0) and np.all(g(hi) < 0)):
                raise ValueError(
                    f"the ohmic contact {e.name!r} sits on material with no "
                    "charge-neutral level (undoped, holes left out); give it "
                    "implant_depth and implant_donors, or dope the layer")
            for _ in range(200):
                m_ = 0.5 * (lo + hi)
                pos = g(m_) > 0
                lo = np.where(pos, m_, lo)
                hi = np.where(pos, hi, m_)
            del zero
            ps = 0.5 * (lo + hi)
            full_ps = np.full(all_nodes.size, float(np.mean(ps)))
            full_ps[self.node_semi[all_nodes]] = ps
            out[e.name] = full_ps
        return out

    # ------------------------------------------------------------- layout
    @property
    def nv(self):
        return 2 + int(self.holes) + (len(self._traps) if self._transient else 0)

    _transient = False
    _density_update = True
    _qf_margin = 0.0
    _damp_capped = False

    def _unpack(self, x):
        X = x.reshape(self.N, self.nv)
        psi, fn = X[:, 0], X[:, 1]
        fp = X[:, 2] if self.holes else np.zeros(self.N)
        k0 = 2 + int(self.holes)
        focc = [X[:, k0 + k] for k in range(len(self._traps))] if self._transient else None
        return psi, fn, fp, focc

    def _dirichlet(self, bias):
        """(node index array, psi value, F value or None) per electrode and
        pinned surface."""
        out = []
        for e in self.electrodes:
            V = float(bias.get(e.name, 0.0))
            nodes = self.enodes[e.name]
            if e.kind == "ohmic":
                ps = self._neutral[e.name] + V
            elif e.z0 is not None:
                # one metal: one potential, referred to the layer under it
                ps = np.full(nodes.size, self._box_ec[e.name] + V - e.barrier)
            else:
                ps = self.Ec0[nodes] + V - e.barrier
            out.append((e, nodes, ps, -V))
        return out

    # ----------------------------------------------------------- residual
    def _fluxes(self, psi, fn, fp, n, p, lnn, lnp, Tn):
        kT = self.kT
        i, j, h = self.ei, self.ej, self.eh
        Te = 0.5 * (Tn[i] + Tn[j])
        dfn = fn[j] - fn[i]
        dpsi_e = psi[j] - psi[i]
        w = fn / kT - lnn
        dw = w[j] - w[i]
        from .driftdiffusion import bernoulli
        if self._hydro is None:
            mu, _ = _edge_mu(self._mob_n, dfn if self.force == "quasi-fermi" else dpsi_e,
                             h, Te)
            Dn = mu * 1e14 * kT
            Gn = -Dn / h * n[i] * bernoulli(dw) * np.expm1(np.clip(dfn / kT, -700, 700))
        else:
            hy = self._hydro
            Tlog, rho, r = hy["Tlog"], hy["rho"], hy["r"]
            c = dw * rho + r
            s_ = np.clip(dfn / kT + dw * (rho - 1.0) + r, -700, 700)
            Gn = -hy["mu"] * 1e14 * KB * Tlog / h * n[i] * bernoulli(c) * np.expm1(s_)
        if self.holes:
            dfp = fp[j] - fp[i]
            mu, _ = _edge_mu(self._mob_p, dfp if self.force == "quasi-fermi" else dpsi_e, h, Te)
            Dp = mu * 1e14 * kT
            v = fp / kT + lnp
            dv = v[j] - v[i]
            Gp = -Dp / h * p[i] * bernoulli(-dv) * np.expm1(np.clip(-dfp / kT, -700, 700))
        else:
            Gp = np.zeros_like(Gn)
        Gn, Gp = self._te_fluxes(fn, fp, n, p, Gn, Gp)
        Gn = self._block_fluxes(Gn)
        return Gn, Gp

    def _balances(self, x, Tn, dt=None, old=None):
        """Raw balances (Poisson, electrons, holes, traps) before boundary
        conditions, and the pieces needed for currents and heat."""
        psi, fn, fp, focc = self._unpack(x)
        Ec, n, p, lnn, lnp = self._carriers(psi, fn, fp)
        ndp, nam = self._dopants(Ec, fn, fp)
        rho = p - n + ndp - nam
        Rn = np.zeros(self.N)
        Rp = np.zeros(self.N)
        rt = []
        for k, tr in enumerate(self._traps):
            if self._transient:
                f = focc[k]
                rn_ = tr["Nt"] * (tr["cn"] * n * (1 - f) - tr["en"] * f)
                rp_ = tr["Nt"] * (tr["cp"] * p * f - tr["ep"] * (1 - f))
                rt.append(rn_ - rp_)
            else:
                f = self._f_steady(tr, n, p)
                # net SRH rate: c_n c_p (n p - n1 p1) / (c_n (n + n1) + c_p (p + p1))
                den = tr["cn"] * n + tr["en"] + tr["cp"] * p + tr["ep"]
                u = tr["Nt"] * (tr["cn"] * tr["cp"] * n * p - tr["en"] * tr["ep"]) / (
                    den + 1e-300)
                rn_ = u
                rp_ = u
            Rn += rn_
            Rp += rp_
            rho += -tr["Nt"] * f if tr["acc"] else tr["Nt"] * (1.0 - f)
        i, j = self.ei, self.ej
        fl = self.gpsi * (psi[j] - psi[i])
        rP = (np.bincount(i, fl, self.N) - np.bincount(j, fl, self.N)
              + _CP * (self.vol * rho + self.sheet))
        Gn, Gp = self._fluxes(psi, fn, fp, n, p, lnn, lnp, Tn)
        A = self.earea
        rN = np.bincount(i, Gn * A, self.N) - np.bincount(j, Gn * A, self.N) + self.vol * Rn
        if self._tb is not None:
            rN = rN + self._link_balance(psi, fn)
        rQ = np.bincount(i, Gp * A, self.N) - np.bincount(j, Gp * A, self.N) + self.vol * Rp
        Gnode = self.vol * self.Gopt
        if self.ionization is not None:
            Gnode = Gnode + self._ionization_nodes(psi, Gn, Gp)
        rN -= Gnode
        rQ -= Gnode
        if dt is not None:
            _, n0, p0, _, _ = self._carriers(*self._unpack(old)[:3])
            rN += self.vol * (n - n0) / dt
            rQ += self.vol * (p - p0) / dt
        return dict(rP=rP, rN=rN, rQ=rQ, rt=rt, n=n, p=p, Ec=Ec, Gn=Gn, Gp=Gp,
                    psi=psi, fn=fn, fp=fp, focc=focc, Rn=Rn, Rp=Rp, Gnode=Gnode)

    def _ionization_edges(self, psi, Gn, Gp):
        """Pairs made per second on every edge by impact ionization,
        (alpha_n |Gamma_n| + alpha_p |Gamma_p|) A h, and pieces for the
        derivatives. |Gamma| is rounded within 1e-30 nm^-2 s^-1 of zero
        (sqrt(Gamma^2 + eps^2) - eps) so the term stays smooth where a
        current changes sign; that is far below any current that matters."""
        i, j, h = self.ei, self.ej, self.eh
        dps = psi[j] - psi[i]
        E = np.abs(dps) / h * 1e7
        an, ap = self.ionization.alpha(E)
        m = self._ion_mask(self.ionization)
        w = self.earea * h * 1e-7 * (m[i] & m[j])     # nm^3 per (1/cm)
        eps = 1e-30
        rn = np.sqrt(Gn * Gn + eps * eps)
        rp = np.sqrt(Gp * Gp + eps * eps)
        absn, absp = rn - eps, rp - eps
        self._ii_sign = (Gn / rn, Gp / rp)
        G = (an * absn + ap * absp) * w
        return G, E, an, ap, w, dps, absn, absp

    def _ion_mask(self, ion):
        """Nodes where `ion` applies: semiconductor nodes of its layers."""
        if ion.where is None:
            return self.node_semi
        labels = [L.label for L in self.layers]
        bad = [w for w in ion.where if w not in labels]
        if bad:
            raise ValueError(f"Ionization.where names no layer: {bad}")
        lab = np.array([labels[k] for k in self.node_layer])
        nl = np.broadcast_to(np.isin(lab, ion.where).reshape(
            (1,) * (self.dim - 1) + (-1,)), self.shape).ravel()
        return self.node_semi & nl

    def _ionization_nodes(self, psi, Gn, Gp):
        G = self._ionization_edges(psi, Gn, Gp)[0]
        return 0.5 * (np.bincount(self.ei, G, self.N) + np.bincount(self.ej, G, self.N))

    def _residual(self, x, bias, Tn, dt=None, old=None):
        b = self._balances(x, Tn, dt, old)
        rP, rN, rQ = b["rP"].copy(), b["rN"].copy(), b["rQ"].copy()
        psi, fn, fp, n, p = b["psi"], b["fn"], b["fp"], b["n"], b["p"]
        for e, nodes, ps, F in self._dirichlet(bias):
            rP[nodes] = psi[nodes] - ps
            if e.kind == "schottky" and e.thermionic:
                A = self.lat_area[nodes]
                n0 = self.Nc[nodes] * self._occ(np.full(nodes.size, -e.barrier / self.kT))[0]
                rN[nodes] += self.vR[nodes] * (n[nodes] - n0) * A
                if self.holes:
                    p0 = self.Nv[nodes] * self._occ(
                        (e.barrier - self.Eg[nodes]) / self.kT)[0]
                    rQ[nodes] += self.vRp[nodes] * (p[nodes] - p0) * A
            else:
                rN[nodes] = fn[nodes] - F
                rQ[nodes] = fp[nodes] - F
        if self.surface.kind == "pinned":
            sn = self.surface_nodes
            rP[sn] = psi[sn] - (self.Ec0[sn] - self.surface.barrier)
        # inside an insulator there are no carriers: hold the levels at 0
        ins = self.insulator_nodes
        if ins.size:
            rN[ins] = fn[ins]
            rQ[ins] = fp[ins]
        tb = self._tb
        if tb is not None and tb["int_nodes"].size:
            k_, a_, b_, w_ = tb["int_nodes"], tb["int_a"], tb["int_b"], tb["int_w"]
            rN[k_] = fn[k_] - (w_ * fn[a_] + (1.0 - w_) * fn[b_])
        cols = [rP, rN] + ([rQ] if self.holes else [])
        if self._transient:
            focc = b["focc"]
            fo = self._unpack(old)[3]
            for k, tr in enumerate(self._traps):
                w = tr["where"]
                r = np.where(w, tr["Nt"] * self.vol * (focc[k] - fo[k]) / dt
                             - self.vol * b["rt"][k], focc[k] - fo[k])
                cols.append(r)
        return np.stack(cols, axis=1).ravel()

    # ------------------------------------------------------------ Newton
    def _local(self, x, bias, dt, old, include_te=True):
        """The node-local part of every residual row (no fluxes, no
        Laplacian, no boundary replacement), shape (N, nv). With traps as
        unknowns and dt None, the trap rows are the steady rates (used by
        the small-signal analysis)."""
        psi, fn, fp, focc = self._unpack(x)
        Ec, n, p, lnn, lnp = self._carriers(psi, fn, fp)
        ndp, nam = self._dopants(Ec, fn, fp)
        rho = p - n + ndp - nam
        Rn = np.zeros(self.N)
        Rp = np.zeros(self.N)
        rt = []
        for k, tr in enumerate(self._traps):
            if self._transient:
                f = focc[k]
                rn_ = tr["Nt"] * (tr["cn"] * n * (1 - f) - tr["en"] * f)
                rp_ = tr["Nt"] * (tr["cp"] * p * f - tr["ep"] * (1 - f))
                rt.append(rn_ - rp_)
            else:
                f = self._f_steady(tr, n, p)
                den = tr["cn"] * n + tr["en"] + tr["cp"] * p + tr["ep"]
                rn_ = rp_ = tr["Nt"] * (tr["cn"] * tr["cp"] * n * p
                                        - tr["en"] * tr["ep"]) / (den + 1e-300)
            Rn += rn_
            Rp += rp_
            rho += -tr["Nt"] * f if tr["acc"] else tr["Nt"] * (1.0 - f)
        out = [_CP * self.vol * rho, self.vol * Rn]
        if self.holes:
            out.append(self.vol * Rp)
        if dt is not None:
            out[1] = out[1] + self.vol * n / dt
            if self.holes:
                out[2] = out[2] + self.vol * p / dt
        for e in self.electrodes:
            if e.kind == "schottky" and e.thermionic and include_te:
                nodes = self.enodes[e.name]
                A = self.lat_area[nodes]
                n0 = self.Nc[nodes] * self._occ(np.full(nodes.size, -e.barrier / self.kT))[0]
                out[1][nodes] += self.vR[nodes] * (n[nodes] - n0) * A
                if self.holes:
                    p0 = self.Nv[nodes] * self._occ((e.barrier - self.Eg[nodes]) / self.kT)[0]
                    out[2][nodes] += self.vRp[nodes] * (p[nodes] - p0) * A
        if self._transient:
            for k, tr in enumerate(self._traps):
                w = tr["where"]
                store = tr["Nt"] * self.vol * focc[k] / dt if dt is not None else 0.0
                out.append(np.where(w, store - self.vol * rt[k], focc[k]))
        return np.stack(out, axis=1)

    def _storage(self, x):
        """What the time derivative acts on, per node and row: vol n,
        vol p, N_t vol f (zero for Poisson and where a trap is absent)."""
        psi, fn, fp, focc = self._unpack(x)
        Ec, n, p, _, _ = self._carriers(psi, fn, fp)
        out = [np.zeros(self.N), self.vol * n]
        if self.holes:
            out.append(self.vol * p)
        if self._transient:
            for k, tr in enumerate(self._traps):
                out.append(np.where(tr["where"], tr["Nt"] * self.vol * focc[k], 0.0))
        return np.stack(out, axis=1)

    def _mass(self, x):
        """d(storage)/dx, block diagonal (central differences of node-local
        terms)."""
        import scipy.sparse as sp
        nv, N = self.nv, self.N
        nodes = np.arange(N)
        rows, cols, vals = [], [], []
        for v in range(nv):
            xp, xm = x.copy(), x.copy()
            xp[nodes * nv + v] += 1e-6
            xm[nodes * nv + v] -= 1e-6
            d = (self._storage(xp) - self._storage(xm)) / 2e-6
            for u in range(nv):
                rows.append(nodes * nv + u)
                cols.append(nodes * nv + v)
                vals.append(d[:, u])
        return sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows),
                                                     np.concatenate(cols))),
                             shape=(N * nv, N * nv)).tocsr()

    def _flux_derivatives(self, psi, fn, fp, Tn):
        """Fluxes and their exact derivatives with respect to the node
        values at both ends of every edge."""
        from .driftdiffusion import bernoulli
        kT = self.kT
        i, j, h = self.ei, self.ej, self.eh
        Te = 0.5 * (Tn[i] + Tn[j])
        Ec = self.Ec0 - psi

        def occ_d(eta):
            eta = np.clip(eta, -700.0, 700.0)
            if not self.fermi:
                F = np.exp(eta)
                return F, np.ones_like(F), eta
            F, dF = fermi_dirac_half(eta)
            F = np.asarray(F, float)
            r = np.where(eta < -40.0, 1.0, np.asarray(dF, float) / np.maximum(F, 1e-300))
            return F, r, np.where(eta < -40.0, eta, np.log(np.maximum(F, 1e-300)))

        def bern_d(xv):
            B = bernoulli(xv)
            small = np.abs(xv) < 1e-3
            with np.errstate(divide="ignore", invalid="ignore"):
                d = np.where(small, -0.5 + xv / 6.0 - xv ** 3 / 180.0,
                             B * (1.0 - B) / np.where(small, 1.0, xv) - B)
            return B, d

        def mob(m, dF):
            return _edge_mu(m, dF, h, Te)

        out = {}
        # electrons
        Fn_, rn_, lFn = occ_d((fn - Ec) / kT)
        n = self.Nc * Fn_
        dw_dfn = (1.0 - rn_) / kT
        dw_dpsi = -rn_ / kT
        lnn = np.log(self.Nc) + lFn
        w = fn / kT - lnn
        dw = w[j] - w[i]
        dfn = fn[j] - fn[i]
        qf = self.force == "quasi-fermi"
        dpsi_e = psi[j] - psi[i]
        a = n[i]
        da = n[i] * rn_[i] / kT
        if self._hydro is None:
            u = np.clip(dfn / kT, -700, 700)
            c = np.expm1(u)
            eu = np.exp(u)
            B, dB = bern_d(dw)
            mu, dmu = mob(self._mob_n, dfn if qf else dpsi_e)
            K = -kT * 1e14 / h
            BE, DW, BEu = B * c, dB * c, B * eu
            tm = K * dmu * a * B * c            # through the mobility's argument
            qfn = qf
        else:
            hy = self._hydro
            Tlog, rho, r = hy["Tlog"], hy["rho"], hy["r"]
            cc = dw * rho + r
            s_ = np.clip(dfn / kT + dw * (rho - 1.0) + r, -700, 700)
            E_ = np.expm1(s_)
            es = np.exp(s_)
            B, dB = bern_d(cc)
            mu = hy["mu"]
            K = -KB * Tlog * 1e14 / h
            BE = B * E_
            DW = dB * rho * E_ + B * es * (rho - 1.0)
            BEu = B * es
            tm = 0.0                             # the mobility depends on T_n only
            qfn = False
        G = K * mu * a * BE
        out["Gn"] = G
        out["n"] = dict(
            fi=K * (mu * da * BE - mu * a * DW * dw_dfn[i]
                    - mu * a * BEu / kT) - (tm if qfn else 0.0),
            fj=K * (mu * a * DW * dw_dfn[j] + mu * a * BEu / kT) + (tm if qfn else 0.0),
            pi=K * (mu * da * BE - mu * a * DW * dw_dpsi[i]) - (0.0 if qfn else tm),
            pj=K * (mu * a * DW * dw_dpsi[j]) + (0.0 if qfn else tm))
        out["rn"] = rn_
        if self.holes:
            Ev = Ec - self.Eg
            Fp_, rp_, lFp = occ_d((Ev - fp) / kT)
            p = self.Nv * Fp_
            lnp = np.log(self.Nv) + lFp
            dv_dfp = (1.0 - rp_) / kT
            dv_dpsi = -rp_ / kT
            v = fp / kT + lnp
            dv = v[j] - v[i]
            dfp = fp[j] - fp[i]
            up = np.clip(-dfp / kT, -700, 700)
            cp_ = np.expm1(up)
            eup = np.exp(up)
            Bp, dBp = bern_d(-dv)
            mup, dmup = mob(self._mob_p, dfp if qf else dpsi_e)
            ap = p[i]
            dap = -p[i] * rp_[i] / kT          # d p_i / d fp_i = d p_i / d psi_i
            Gp = K * mup * ap * Bp * cp_
            out["Gp"] = Gp
            tp = K * dmup * ap * Bp * cp_
            out["p"] = dict(
                fi=K * (mup * dap * Bp * cp_ + mup * ap * dBp * dv_dfp[i] * cp_
                        + mup * ap * Bp * eup / kT) - (tp if qf else 0.0),
                fj=K * (-mup * ap * dBp * dv_dfp[j] * cp_
                        - mup * ap * Bp * eup / kT) + (tp if qf else 0.0),
                pi=K * (mup * dap * Bp * cp_ + mup * ap * dBp * dv_dpsi[i] * cp_)
                - (0.0 if qf else tp),
                pj=K * (-mup * ap * dBp * dv_dpsi[j] * cp_) + (0.0 if qf else tp))
            out["rp"] = rp_
        if self._te is not None:
            Gn2, Gp2 = self._te_fluxes(fn, fp, n, p if self.holes else n, out["Gn"],
                                       out.get("Gp", np.zeros_like(out["Gn"])), deriv=out)
            out["Gn"] = Gn2
            if self.holes:
                out["Gp"] = Gp2
        out["Gn"] = self._block_fluxes(out["Gn"])
        for k_ in ("fi", "fj", "pi", "pj"):
            out["n"][k_] = self._block_fluxes(out["n"][k_])
        return out

    def _jacobian(self, x, F0, bias, Tn, dt, old, raw=False):
        """Exact derivatives of the fluxes and the Laplacian, central
        differences of the node-local terms (each node's local terms depend
        on that node only, so all nodes are perturbed at once), then the
        boundary rows."""
        import scipy.sparse as sp
        nv, N = self.nv, self.N
        psi, fn, fp, _ = self._unpack(x)
        i, j = self.ei, self.ej
        A = self.earea
        rows, cols, vals = [], [], []
        rows_raw, cols_raw, vals_raw = [], [], []

        def add(r, c, v):
            rows.append(r)
            cols.append(c)
            vals.append(v)

        # Laplacian: rP[i] += g (psi_j - psi_i), rP[j] -= g (psi_j - psi_i)
        g = self.gpsi
        add(i * nv, i * nv, -g)
        add(i * nv, j * nv, g)
        add(j * nv, i * nv, g)
        add(j * nv, j * nv, -g)
        fd = self._flux_derivatives(psi, fn, fp, Tn)
        # electrons: rN[i] += G A, rN[j] -= G A
        d = fd["n"]
        for (row_end, sgn) in ((i, 1.0), (j, -1.0)):
            add(row_end * nv + 1, i * nv + 1, sgn * A * d["fi"])
            add(row_end * nv + 1, j * nv + 1, sgn * A * d["fj"])
            add(row_end * nv + 1, i * nv, sgn * A * d["pi"])
            add(row_end * nv + 1, j * nv, sgn * A * d["pj"])
        if self.holes:
            d = fd["p"]
            for (row_end, sgn) in ((i, 1.0), (j, -1.0)):
                add(row_end * nv + 2, i * nv + 2, sgn * A * d["fi"])
                add(row_end * nv + 2, j * nv + 2, sgn * A * d["fj"])
                add(row_end * nv + 2, i * nv, sgn * A * d["pi"])
                add(row_end * nv + 2, j * nv, sgn * A * d["pj"])
        if self.ionization is not None:
            Gn, Gp = fd["Gn"], fd["Gp"]
            G, E, an, ap, w, dps, absn, absp = self._ionization_edges(psi, Gn, Gp)
            ion = self.ionization
            with np.errstate(divide="ignore", invalid="ignore"):
                dan = np.where(E > 0, an * ion.b_n / np.where(E > 0, E, 1.0) ** 2, 0.0)
                dap = np.where(E > 0, ap * ion.b_p / np.where(E > 0, E, 1.0) ** 2, 0.0)
            dEj = np.sign(dps) * 1e7 / self.eh
            sn, sp_ = self._ii_sign
            dn_, dp_ = fd["n"], fd["p"]
            base = (dan * absn + dap * absp)
            dG = [(i, 0, w * (-base * dEj + an * sn * dn_["pi"] + ap * sp_ * dp_["pi"])),
                  (j, 0, w * (base * dEj + an * sn * dn_["pj"] + ap * sp_ * dp_["pj"])),
                  (i, 1, w * an * sn * dn_["fi"]), (j, 1, w * an * sn * dn_["fj"]),
                  (i, 2, w * ap * sp_ * dp_["fi"]), (j, 2, w * ap * sp_ * dp_["fj"])]
            for col_node, v, val in dG:
                for row_node in (i, j):
                    add(row_node * nv + 1, col_node * nv + v, -0.5 * val)
                    add(row_node * nv + 2, col_node * nv + v, -0.5 * val)
        if self._tb is not None:
            lr, lc, lv = self._link_jacobian(psi, fn, nv)
            for r_, c_, v_ in zip(lr, lc, lv):
                add(r_, c_, v_)
        # local terms
        nodes = np.arange(N)
        for v in range(nv):
            dlt = 1e-6
            xp = x.copy()
            xm = x.copy()
            xp[nodes * nv + v] += dlt
            xm[nodes * nv + v] -= dlt
            dL = (self._local(xp, bias, dt, old) - self._local(xm, bias, dt, old)) / (2 * dlt)
            for u in range(nv):
                add(nodes * nv + u, nodes * nv + v, dL[:, u])
            if raw:
                dR = (self._local(xp, bias, dt, old, include_te=False)
                      - self._local(xm, bias, dt, old, include_te=False)) / (2 * dlt)
                for u in range(nv):
                    add_raw = dR[:, u] - dL[:, u]
                    rows_raw.append(nodes * nv + u)
                    cols_raw.append(nodes * nv + v)
                    vals_raw.append(add_raw)
        J = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows),
                                                  np.concatenate(cols))),
                          shape=(N * nv, N * nv)).tocsr()
        if raw:
            Jraw = J + sp.coo_matrix((np.concatenate(vals_raw), (np.concatenate(rows_raw),
                                                                 np.concatenate(cols_raw))),
                                     shape=(N * nv, N * nv)).tocsr()
        # boundary rows: identity on their own unknown
        brow = []
        for e, enod, ps, F in self._dirichlet(bias):
            brow.append(enod * nv)
            if not (e.kind == "schottky" and e.thermionic):
                brow.append(enod * nv + 1)
                if self.holes:
                    brow.append(enod * nv + 2)
        if self.surface.kind == "pinned":
            brow.append(self.surface_nodes * nv)
        if self.insulator_nodes.size:
            brow.append(self.insulator_nodes * nv + 1)
            if self.holes:
                brow.append(self.insulator_nodes * nv + 2)
        tb = self._tb
        if tb is not None and tb["int_nodes"].size:
            brow.append(tb["int_nodes"] * nv + 1)
        brow = np.concatenate(brow) if brow else np.zeros(0, int)
        keep = np.ones(N * nv)
        keep[brow] = 0.0
        J = sp.diags(keep) @ J
        e = np.zeros(N * nv)
        e[brow] = 1.0
        Jbc = (J + sp.diags(e)).tocsr()
        if tb is not None and tb["int_nodes"].size:
            k_, a_, b_, w_ = tb["int_nodes"], tb["int_a"], tb["int_b"], tb["int_w"]
            Jbc = Jbc + sp.coo_matrix(
                (np.concatenate([-w_, -(1.0 - w_)]),
                 (np.concatenate([k_ * nv + 1] * 2), np.concatenate([a_ * nv + 1, b_ * nv + 1]))),
                shape=(N * nv, N * nv)).tocsr()
        if raw:
            return Jbc, Jraw, brow
        return Jbc

    def _newton(self, x, bias, Tn, dt=None, old=None, tol=1e-8, max_iter=40,
                max_step=0.3):
        import scipy.sparse as sp
        import scipy.sparse.linalg as spl
        nv = self.nv
        k0 = 2 + int(self.holes)
        for it in range(1, max_iter + 1):
            F = self._residual(x, bias, Tn, dt, old)
            if not np.all(np.isfinite(F)):
                return x, False, it
            J = self._jacobian(x, F, bias, Tn, dt, old)
            # a quasi-Fermi level of a carrier that is vanishingly sparse
            # (below 1e-30 of the density floor) where nothing generates it
            # is updated from its own equation alone in this step (its row
            # keeps only its diagonal): its column is nearly empty, which
            # makes the coupled linear system ill-conditioned, yet it can
            # still move toward its value and leave that state
            frozen = (self._frozen(x, Tn) if not np.any(self.Gopt > 0)
                      else np.zeros(0, int))
            if frozen.size:
                keep = np.ones(J.shape[0])
                keep[frozen] = 0.0
                dg = J.diagonal()
                ok = dg != 0
                diag = np.where(ok, dg, 1.0) * (1.0 - keep)
                F = np.where((keep == 0) & ~ok, 0.0, F)
                J = sp.diags(keep) @ J + sp.diags(diag)
            s = np.asarray(abs(J).max(axis=1).todense()).ravel()
            s[s == 0] = 1.0
            Js = sp.diags(1.0 / s) @ J
            try:
                dx = spl.spsolve(Js.tocsc(), -F / s)
            except Exception:          # singular matrix
                return x, False, it
            if not np.all(np.isfinite(dx)):
                return x, False, it
            D = dx.reshape(self.N, nv)
            psi, fn, fp, _ = self._unpack(x)
            _, n, p, _, _ = self._carriers(psi, fn, fp)
            noise = self._noise(x, J, Tn)
            big = np.max(np.abs(D[:, 0]))
            # a quasi-Fermi level counts where its carrier matters (density
            # weight, as in 1D) and only by how much its change exceeds the
            # rounding noise of its own row (where much denser neighbors
            # dominate the flux balance, the level is set only to that
            # precision and cannot carry any current that matters)
            # (inside an insulator the levels are placeholders: no carriers)
            semi = self.node_semi
            fl_n = fl_p = self.floor
            en = np.maximum(np.abs(D[:, 1]) - noise[:, 1], 0.0) * semi
            big = max(big, np.max(en * n / (n + fl_n)))
            if self.holes:
                ep = np.maximum(np.abs(D[:, 2]) - noise[:, 2], 0.0) * semi
                big = max(big, np.max(ep * p / (p + fl_p)))
            if self._transient and nv > k0:
                big = max(big, np.max(np.abs(D[:, k0:])))
            big = float(big)
            # the common damping factor is set by the potential and by the
            # quasi-Fermi levels of carriers that matter, each change capped
            # at 1 eV: a level of a vanishing carrier (whose update can be
            # huge and meaningless) must not freeze everything else; it is
            # held back by the per-variable bound below instead
            damp = big if not self._damp_capped else max(
                np.max(np.abs(D[:, 0])),
                np.max(np.minimum(en, 1.0) * n / (n + fl_n)),
                np.max(np.minimum(ep, 1.0) * p / (p + fl_p)) if self.holes else 0.0,
                np.max(np.abs(D[:, k0:])) if (self._transient and nv > k0) else 0.0)
            scale = min(1.0, max_step / damp) if damp > 0 else 1.0
            # also bound every single change (as the 1D solver does), so a
            # quasi-Fermi level where its carrier is negligible cannot jump
            # far and underflow the density
            if self._density_update:
                x = self._density_step(x, scale * dx, max_step)
            else:
                x = x + np.clip(scale * dx, -max_step, max_step)
            # quasi-Fermi levels stay between the lowest and highest Fermi
            # level of the electrodes in a steady state (the maximum
            # principle of the continuity equations; a level outside that
            # range belongs to a carrier too sparse to matter and only stalls
            # the iteration). Not in time steps, where a level may lag.
            # (with hot electrons the quasi-Fermi level obeys no maximum
            # principle: thermal diffusion can push it past the contacts')
            if (old is None and not np.any(self.Gopt > 0) and self._hydro is None
                    and self._bt is None):
                X = x.reshape(self.N, nv)
                lev = [-float(bias.get(e.name, 0.0)) for e in self.electrodes]
                lo, hi = min(lev) - self._qf_margin, max(lev) + self._qf_margin
                X[:, 1] = np.clip(X[:, 1], lo, hi)
                if self.holes:
                    X[:, 2] = np.clip(X[:, 2], lo, hi)
            # keep every density representable: a carrier more than 600 kT
            # below its band (a density below about 1e-260 of N_c) is zero
            # for every purpose, but letting its level wander further makes
            # the density underflow and the matrix singular
            X = x.reshape(self.N, nv)
            Ec_ = self.Ec0 - X[:, 0]
            X[:, 1] = np.maximum(X[:, 1], Ec_ - 600.0 * self.kT)
            if self.holes:
                X[:, 2] = np.minimum(X[:, 2], Ec_ - self.Eg + 600.0 * self.kT)
            if self._transient and nv > k0:
                X[:, k0:] = np.clip(X[:, k0:], 0.0, 1.0)
            if self._bt is not None or np.any(self.Gopt > 0):
                # where generation feeds sparse carriers, a level change
                # counts with the density it leads to as well (else a large
                # rise from a vanishing density would look converged)
                _, n2, p2, _, _ = self._carriers(*self._unpack(x)[:3])
                big = max(big, float(np.max(en * n2 / (n2 + fl_n))))
                if self.holes:
                    big = max(big, float(np.max(ep * p2 / (p2 + fl_p))))
            ins = self.insulator_nodes
            if ins.size:
                X[ins, 1] = 0.0
                if self.holes:
                    X[ins, 2] = 0.0
            if big < tol:
                return x, True, it
        return x, False, max_iter

    def _frozen(self, x, Tn):
        """Unknown indices of quasi-Fermi levels to hold in this Newton step
        (see `_newton`)."""
        nv = self.nv
        psi, fn, fp, _ = self._unpack(x)
        _, n, p, lnn, lnp = self._carriers(psi, fn, fp)
        G = self.vol * self.Gopt
        if self.ionization is not None:
            Gn, Gp = self._fluxes(psi, fn, fp, n, p, lnn, lnp, Tn)
            G = G + self._ionization_nodes(psi, Gn, Gp)
        quiet = G < 1e-30
        out = []
        dn = np.flatnonzero((n < 1e-30 * self.floor) & quiet)
        out.append(dn * nv + 1)
        if self.holes:
            dp = np.flatnonzero((p < 1e-30 * self.floor) & quiet)
            out.append(dp * nv + 2)
        idx = np.concatenate(out)
        # never hold a level that a contact fixes
        fixed = []
        for e in self.electrodes:
            if not (e.kind == "schottky" and e.thermionic):
                nodes = self.enodes[e.name]
                fixed.append(nodes * nv + 1)
                if self.holes:
                    fixed.append(nodes * nv + 2)
        if fixed:
            idx = np.setdiff1d(idx, np.concatenate(fixed))
        return idx

    def _inv_occ(self, target):
        """eta with F(eta) = target (F = exp or F_1/2), vectorized."""
        t = np.maximum(np.asarray(target, float), 1e-300)
        eta = np.log(t)
        if not self.fermi:
            return eta
        # start from the Joyce-Dixon-like guess, then Newton in log F
        big = t > 1.0
        eta = np.where(big, (0.75 * math.sqrt(math.pi) * t) ** (2.0 / 3.0) + 0.1, eta)
        for _ in range(30):
            F, dF = fermi_dirac_half(np.clip(eta, -700, 700))
            F = np.maximum(np.asarray(F, float), 1e-300)
            step = (np.log(F) - np.log(t)) / np.maximum(np.asarray(dF, float) / F, 1e-12)
            eta = eta - np.clip(step, -5, 5)
            if np.max(np.abs(step)) < 1e-12:
                break
        return eta

    def _density_step(self, x, dx, max_step):
        """Apply a Newton update with the carrier densities updated
        linearly (n + dn, with dn from the linearized update of both the
        potential and the quasi-Fermi level) instead of the quasi-Fermi
        levels: the continuity equations are nearly linear in the
        densities, so a sparse carrier lands on its value in one step
        instead of crawling there one kT at a time. A density may fall at
        most a thousandfold per step; the potential is bounded by max_step."""
        nv = self.nv
        X = x.reshape(self.N, nv)
        D = dx.reshape(self.N, nv)
        kT = self.kT
        psi, fn, fp = X[:, 0], X[:, 1], X[:, 2] if self.holes else None
        dpsi = np.clip(D[:, 0], -max_step, max_step)
        Ec = self.Ec0 - psi
        Ec_new = Ec - dpsi
        out = X.copy()
        out[:, 0] = psi + dpsi
        # electrons
        eta = np.clip((fn - Ec) / kT, -700, 700)
        F, dF = (np.exp(eta), np.exp(eta)) if not self.fermi else fermi_dirac_half(eta)
        F = np.maximum(np.asarray(F, float), 1e-300)
        n = self.Nc * F
        dn = self.Nc * np.asarray(dF, float) / kT * (D[:, 1] + dpsi)
        n_new = np.maximum(n + dn, n * 1e-3)
        out[:, 1] = Ec_new + kT * self._inv_occ(n_new / self.Nc)
        if self.holes:
            Ev = Ec - self.Eg
            etap = np.clip((Ev - fp) / kT, -700, 700)
            Fp_, dFp = ((np.exp(etap), np.exp(etap)) if not self.fermi
                        else fermi_dirac_half(etap))
            Fp_ = np.maximum(np.asarray(Fp_, float), 1e-300)
            p = self.Nv * Fp_
            dp = -self.Nv * np.asarray(dFp, float) / kT * (D[:, 2] + dpsi)
            p_new = np.maximum(p + dp, p * 1e-3)
            out[:, 2] = (Ec_new - self.Eg) - kT * self._inv_occ(p_new / self.Nv)
        if nv > 2 + int(self.holes):
            out[:, 2 + int(self.holes):] = X[:, 2 + int(self.holes):] + D[:, 2 + int(self.holes):]
        # no quasi-Fermi level moves by more than 1 eV in one step
        for v in range(1, 2 + int(self.holes)):
            out[:, v] = X[:, v] + np.clip(out[:, v] - X[:, v], -1.0, 1.0)
        # where a carrier is extremely sparse its density cannot be resolved
        # by the inverse; keep the plain update there
        sparse_n = n < 1e-280
        out[sparse_n, 1] = X[sparse_n, 1] + np.clip(D[sparse_n, 1], -max_step, max_step)
        if self.holes:
            sparse_p = p < 1e-280
            out[sparse_p, 2] = X[sparse_p, 2] + np.clip(D[sparse_p, 2], -max_step, max_step)
        return out.ravel()

    def _gummel(self, x, bias, Tn, tol=1e-8, outer=40):
        """Gummel's decoupled iteration as a fallback when the coupled
        Newton method fails (often in reverse-biased junctions of wide-gap
        material, where one carrier is extremely sparse): Poisson for the
        potential with the quasi-Fermi levels held, then each continuity
        equation for its own level with the rest held, repeated. Each block
        is solved by Newton on its own unknowns; the coupled Newton method
        then finishes from the result."""
        import scipy.sparse as sp
        import scipy.sparse.linalg as spl
        nv = self.nv
        blocks = [0, 1] + ([2] if self.holes else [])
        for _ in range(outer):
            moved = 0.0
            for v in blocks:
                idx = np.arange(self.N) * nv + v
                for _inner in range(30):
                    F = self._residual(x, bias, Tn)
                    J = self._jacobian(x, F, bias, Tn, None, None)
                    Jb = J[idx][:, idx]
                    sc = np.asarray(abs(Jb).max(axis=1).todense()).ravel()
                    sc[sc == 0] = 1.0
                    try:
                        d = spl.spsolve((sp.diags(1.0 / sc) @ Jb).tocsc(), -F[idx] / sc)
                    except Exception:
                        return x, False
                    if not np.all(np.isfinite(d)):
                        return x, False
                    d = np.clip(d, -0.3, 0.3)
                    x = x.copy()
                    x[idx] += d
                    X = x.reshape(self.N, nv)
                    Ec_ = self.Ec0 - X[:, 0]
                    X[:, 1] = np.maximum(X[:, 1], Ec_ - 600.0 * self.kT)
                    if self.holes:
                        X[:, 2] = np.minimum(X[:, 2], Ec_ - self.Eg + 600.0 * self.kT)
                    psi, fn, fp, _ = self._unpack(x)
                    _, n, p, _, _ = self._carriers(psi, fn, fp)
                    w = 1.0 if v == 0 else (n if v == 1 else p) / (
                        (n if v == 1 else p) + self.floor)
                    step = float(np.max(np.abs(d) * w))
                    moved = max(moved, step)
                    if step < tol:
                        break
            if moved < 1e-6:
                break
        xn, conv, _ = self._newton(x, bias, Tn, tol=tol, max_iter=40)
        return (xn, True) if conv else (x, False)

    def _noise(self, x, J, Tn):
        """Rounding level of every quasi-Fermi level: 1e3 machine epsilons
        times the sum of the magnitudes in its row's flux balance, divided
        by the row's own derivative."""
        nv = self.nv
        psi, fn, fp, _ = self._unpack(x)
        _, n, p, lnn, lnp = self._carriers(psi, fn, fp)
        Gn, Gp = self._fluxes(psi, fn, fp, n, p, lnn, lnp, Tn)
        A = self.earea
        out = np.zeros((self.N, nv))
        d = np.abs(J.diagonal()).reshape(self.N, nv)
        tiny = 1e3 * np.finfo(float).eps
        Sn = np.bincount(self.ei, np.abs(Gn * A), self.N) + np.bincount(
            self.ej, np.abs(Gn * A), self.N)
        out[:, 1] = tiny * Sn / np.maximum(d[:, 1], 1e-300)
        if self.holes:
            Sp = np.bincount(self.ei, np.abs(Gp * A), self.N) + np.bincount(
                self.ej, np.abs(Gp * A), self.N)
            out[:, 2] = tiny * Sp / np.maximum(d[:, 2], 1e-300)
        return np.minimum(out, 0.1)

    # ------------------------------------------------------- equilibrium
    def _equilibrium_x(self):
        """Poisson alone with F_n = F_p = 0 (all electrodes at 0 V)."""
        import scipy.sparse as sp
        import scipy.sparse.linalg as spl
        if self._x0 is not None:
            return self._x0.copy()
        N = self.N
        # start: a 1D quasi-equilibrium column under the free surface
        psi = self._column_guess()
        dmask = np.zeros(N, bool)
        dval = np.zeros(N)
        for e, nodes, ps, F in self._dirichlet({}):
            dmask[nodes] = True
            dval[nodes] = ps
        if self.surface.kind == "pinned":
            sn = self.surface_nodes
            dmask[sn] = True
            dval[sn] = self.Ec0[sn] - self.surface.barrier
        psi[dmask] = dval[dmask]
        i, j, g = self.ei, self.ej, self.gpsi
        L = sp.csr_matrix((np.concatenate([g, g]), (np.concatenate([i, j]),
                                                     np.concatenate([j, i]))), shape=(N, N))
        Ld = -(np.bincount(i, g, N) + np.bincount(j, g, N))
        zero = np.zeros(N)
        free = ~dmask
        for _ in range(300):
            rho = self._rho_local(psi, zero, zero)
            d = 1e-6
            drho = (self._rho_local(psi + d, zero, zero) - self._rho_local(psi - d, zero, zero)) / (2 * d)
            r = L @ psi + Ld * psi + _CP * (self.vol * rho + self.sheet)
            Jm = L + sp.diags(Ld + _CP * self.vol * drho)
            step = np.zeros(N)
            step[free] = spl.spsolve(Jm[free][:, free].tocsc(), -r[free])
            step = np.clip(step, -0.2, 0.2)
            psi = psi + step
            if np.max(np.abs(step)) < 1e-11:
                break
        else:
            raise RuntimeError("the equilibrium Poisson solve did not converge")
        x = np.zeros((N, self.nv))
        x[:, 0] = psi
        self._x0 = x.ravel().copy()
        return x.ravel()

    def _column_guess(self):
        """psi from a 1D quasi-equilibrium solve of the stack, repeated
        along x (and y)."""
        from .driftdiffusion import Contact, solve_quasi_equilibrium
        bar = self.surface.barrier if self.surface.kind == "pinned" else 1.0
        z = self.axes[-1]
        try:
            r = solve_quasi_equilibrium(
                self.layers, Contact("schottky", barrier=bar), Contact("zero-field"),
                temperature_K=self.T, dz=min(0.5, self.top / 50),
                interfaces=self.interfaces,
                hole_mass=None)
            col = np.interp(z, r.z, r.psi)
        except Exception:
            col = self.Ec0.reshape(self.shape)[(0,) * (self.dim - 1)] - bar
        return np.broadcast_to(col.reshape((1,) * (self.dim - 1) + (-1,)),
                               self.shape).ravel().copy()

    # --------------------------------------------------------- outputs
    def _state(self, x, bias, Tn, conv, it, dt=None, old=None, time=0.0,
               old_charge=None, internal=None):
        b = self._balances(x, Tn, dt, old)
        psi, fn, fp = b["psi"], b["fn"], b["fp"]
        scale_I = QE * (1e6 if self.dim == 2 else 1.0)      # A/mm or A
        cur, chg = {}, {}
        for e in self.electrodes:
            nodes = self.enodes[e.name]
            Sn = float(np.sum(b["rN"][nodes]))
            Sp = float(np.sum(b["rQ"][nodes]))
            Q = -float(np.sum(b["rP"][nodes])) / _CP * QE * (1e6 if self.dim == 2 else 1.0)
            chg[e.name] = Q
            I = scale_I * (-Sn + Sp)
            if dt is not None and old_charge is not None:
                I += (Q - old_charge[e.name]) / dt
            cur[e.name] = I
        # heat: Joule on edges, half to each node, plus recombination heat
        i, j = self.ei, self.ej
        A = self.earea
        pe = (-b["Gn"] * (fn[j] - fn[i]) + b["Gp"] * (fp[j] - fp[i])) * A   # eV/s
        H = 0.5 * (np.bincount(i, pe, self.N) + np.bincount(j, pe, self.N))
        if self.holes or self._traps:
            H += self.vol * 0.5 * (b["Rn"] + b["Rp"]) * (fn - fp)
        H -= b["Gnode"] * (fn - fp)
        H = H * QE * (1e6 if self.dim == 2 else 1.0)        # W/mm or W
        occ = []
        for k, tr in enumerate(self._traps):
            f = (b["focc"][k] if self._transient else self._f_steady(tr, b["n"], b["p"]))
            occ.append(np.where(tr["where"], f, np.nan).reshape(self.shape))
        sh = self.shape
        # carriers, band edge and quasi-Fermi levels exist only in the
        # semiconductor (inside a dielectric or a metal they are reported
        # as 0 and NaN)
        semi = self.node_semi
        nan = np.where(semi, 1.0, np.nan)
        return DeviceState(
            axes=self.axes, psi=psi.reshape(sh), ec=(b["Ec"] * nan).reshape(sh),
            efn=(fn * nan).reshape(sh), efp=(fp * nan).reshape(sh),
            n=(b["n"] * 1e21 * semi).reshape(sh),
            p=(b["p"] * 1e21 * semi).reshape(sh), temperature=Tn.reshape(sh),
            occupancy=occ, bias=dict(bias), current=cur, charge=chg,
            heat=H.reshape(sh), converged=conv, iterations=it, time=time,
            internal=dict(internal) if internal is not None else dict(bias),
            _x=x.copy())

    # ------------------------------------------------------ linear response
    def _ports(self):
        names = [e.name for e in self.electrodes]
        if self.surface.kind == "pinned" and self.surface_nodes.size:
            names.append("surface")
        return names

    def _response(self, x, bias, Tn, svals, dt=None, old=None):
        """Admittance matrices Y[s] (dI_c / dV_k, A/(V mm) in 2D, A/V in
        3D) at the complex frequencies s (s = j omega), linearized about
        the state x. With dt, the time-step conductance of backward Euler
        (s = 0 then)."""
        import scipy.sparse as sp
        import scipy.sparse.linalg as spl
        nv, N = self.nv, self.N
        Jbc, Jraw, brow = self._jacobian(x, None, bias, Tn, dt, old, raw=True)
        keep = np.ones(N * nv)
        keep[brow] = 0.0
        M = self._mass(x)
        Mbc = sp.diags(keep) @ M
        ports = self._ports()
        scale = QE * (1e6 if self.dim == 2 else 1.0)
        rhs = np.zeros((N * nv, len(ports)))
        a = np.zeros((len(ports), N * nv))
        q = np.zeros((len(ports), N * nv))
        for k, name in enumerate(ports):
            if name == "surface":
                nodes = self.surface_nodes
                rhs[nodes * nv, k] = 1.0
            else:
                e = self.edict[name]
                nodes = self.enodes[name]
                rhs[nodes * nv, k] = 1.0
                if not (e.kind == "schottky" and e.thermionic):
                    rhs[nodes * nv + 1, k] = -1.0
                    if self.holes:
                        rhs[nodes * nv + 2, k] = -1.0
                a[k, nodes * nv + 1] = -scale
                if self.holes:
                    a[k, nodes * nv + 2] = scale
            q[k, nodes * nv] = -scale / _CP
        out = []
        for s_ in svals:
            A = Jbc + s_ * Mbc
            # scale every row by its largest entry, as in the Newton steps
            sc = np.asarray(abs(A).max(axis=1).todense()).ravel()
            sc[sc == 0] = 1.0
            A = (sp.diags(1.0 / sc) @ A).tocsc()
            b_ = rhs / sc[:, None]
            if isinstance(s_, complex):
                lu = spl.splu(A.astype(complex))
                X = lu.solve(b_.astype(complex))
            else:
                lu = spl.splu(A)
                X = lu.solve(b_)
            JX = Jraw @ X
            MX = M @ X
            disp = s_ + (1.0 / dt if dt is not None else 0.0)
            Y = a @ (JX + s_ * MX) + disp * (q @ JX)
            out.append(Y)
        return ports, out

    def small_signal(self, state: DeviceState, frequencies, traps: bool = True):
        """Small-signal admittance of every terminal at each frequency
        (Hz), linearized about the steady `state` (new in 0.20.0).

        Returns (ports, Y) with Y of shape (len(frequencies), n, n):
        Y[f, c, k] = dI_c / dV_k, the current into the device through
        terminal c per volt on terminal k (S/mm in 2D, S in 3D), at the
        contacts themselves (add series resistors with `with_series`). A
        pinned surface acts as an extra terminal held at its potential;
        it is listed as "surface". traps: let the traps follow the signal
        (their capture and emission set how fast they can).
        """
        f = np.atleast_1d(np.asarray(frequencies, float))
        if np.any(f < 0) or not np.all(np.isfinite(f)):
            raise ValueError("frequencies must be finite and >= 0")
        bias = dict(state.internal or state.bias)
        Tn = state.temperature.ravel().copy()
        use_traps = traps and bool(self._traps)
        if use_traps:
            X0 = state._x.reshape(self.N, 2 + int(self.holes))
            _, n, p, _, _ = self._carriers(*self._unpack_static(state._x))
            occ = [np.where(tr["where"], self._f_steady(tr, n, p), 0.0) for tr in self._traps]
            x = np.concatenate([X0, np.stack(occ, axis=1)], axis=1).ravel()
            self._transient = True
        else:
            x = state._x.copy()
        try:
            svals = [complex(0.0, 2 * math.pi * fi) if fi > 0 else 0.0 for fi in f]
            ports, Ys = self._response(x, bias, Tn, svals)
        finally:
            self._transient = False
        return ports, np.array([np.asarray(Y, complex) for Y in Ys])

    def _conductance(self, x, bias, Tn, dt=None, old=None):
        ports, Ys = self._response(x, bias, Tn, [0.0], dt=dt, old=old)
        names = [e.name for e in self.electrodes]
        idx = [ports.index(nm) for nm in names]
        return np.real(Ys[0])[np.ix_(idx, idx)]

    # ------------------------------------------------------------- public
    def equilibrium(self) -> DeviceState:
        """All electrodes at 0 V, without generation (true equilibrium)."""
        Tn = np.full(self.N, self.T)
        bias = {e.name: 0.0 for e in self.electrodes}
        if self._x0 is not None:
            return self._state(self._x0.copy(), bias, Tn, True, 0)
        G = self.Gopt
        try:
            self.Gopt = np.zeros(self.N)
            x = self._equilibrium_x()
            x, conv, it = self._newton(x, bias, Tn)
            if not conv:
                raise RuntimeError("equilibrium did not converge")
            self._x0 = x.copy()
            return self._state(x, bias, Tn, True, it)
        finally:
            self.Gopt = G

    def _solve_band_to_band(self, bias, guess, thermal, max_bias_step, tol, thermal_tol,
                            max_thermal_iter, energy_relaxation, rtol=1e-6, max_outer=80):
        """Band-to-band tunneling as a generation that follows the field:
        solve with the generation of the present field held fixed (the
        robust path of optical generation), update it from the new field,
        and repeat until it changes by less than rtol."""
        if thermal is not None or energy_relaxation is not None:
            raise ValueError("band-to-band tunneling cannot be combined with thermal or "
                             "energy_relaxation yet")
        G_user = self.Gopt.copy()
        vol = np.where(self.vol > 0, self.vol, 1.0)
        target = self._check_bias(bias)
        Tl = np.full(self.N, self.T)
        self._bt_inner = True
        try:
            st = self.solve(bias, guess=guess, max_bias_step=max_bias_step, tol=tol)
            x, vint = st._x.copy(), dict(st.internal or st.bias)
            G_now = G_user.copy()
            last = None
            for k in range(1, max_outer + 1):
                X = x.reshape(self.N, self.nv)
                spread = max(vint.values()) - min(vint.values())
                Gb = self._band_to_band_nodes(X[:, 0], spread) / vol
                tot = float(np.sum(Gb * self.vol))
                if last is None and tot == 0.0:
                    # no generation anywhere (equilibrium, or fields far too
                    # low): the solution without it stands
                    st.band_to_band_iterations = k
                    st.band_to_band_pairs = 0.0
                    return st
                if last is not None and abs(tot - last) <= rtol * max(abs(tot), 1e-300):
                    st = self._state(x, target, Tl, True, 0, internal=vint)
                    st.band_to_band_iterations = k
                    st.band_to_band_pairs = tot
                    return st
                last = tot
                G_new = G_user + Gb
                if k == 1 and guess is None:
                    # first appearance: switch it on from the dark
                    self.Gopt = G_new
                    x = self._switch_on_generation(x, vint, Tl, tol)
                else:
                    x = self._move_generation(x, vint, Tl, tol, G_now, G_new)
                G_now = G_new
            raise RuntimeError("band-to-band generation and the field did not settle")
        finally:
            self.Gopt = G_user
            self._bt_inner = False

    def _move_generation(self, x, bias, Tn, tol, G0, G1):
        """Change a fixed generation from G0 to G1 at fixed voltages, in
        steps that shrink when Newton's method needs them to; in time if
        that fails."""
        frac, step = 0.0, 1.0
        while frac < 1.0:
            f = min(1.0, frac + step)
            self.Gopt = G0 + f * (G1 - G0)
            xn, conv, it = self._newton(x, bias, Tn, tol=tol, max_iter=40)
            if conv:
                x, frac = xn, f
                step = min(1.0, step * 2.0)
                continue
            step /= 4.0
            if step < 1e-3:
                self.Gopt = G1
                xs = self._pseudo_transient(x, bias, Tn, tol)
                if xs is None:
                    raise RuntimeError("could not follow the band-to-band generation")
                return xs
        return x

    def _switch_on_generation(self, x, bias, Tn, tol):
        """Turn the generation on at fixed voltages, in steps of a factor
        (log scale), halving the step when Newton does not converge."""
        G = self.Gopt
        x0 = x.copy()
        try:
            # try it all at once first; if that fails, start low and climb
            self.Gopt = G
            xn, conv, it = self._newton(x, bias, Tn, tol=tol, max_iter=60)
            if conv:
                return xn
            lg, lstep = -6.0, 2.0
            ok = False
            for _ in range(200):
                nxt = min(0.0, lg + lstep) if ok else lg
                self.Gopt = G * 10 ** nxt
                xn, conv, it = self._newton(x, bias, Tn, tol=tol, max_iter=60)
                if conv:
                    x, lg, ok = xn, nxt, True
                    if lg >= 0.0:
                        return x
                    lstep = min(3.0, lstep * 1.5)
                elif not ok:
                    lg -= 2.0
                    if lg < -30:
                        break
                else:
                    lstep /= 2
                    if lstep < 1e-3:
                        break
            # last resort: switch it on in time (pseudo-transient
            # continuation): the time derivative keeps every step well
            # conditioned, and the steps grow until the state is steady
            self.Gopt = G
            xs = self._pseudo_transient(x0, bias, Tn, tol)
            if xs is not None:
                return xs
            raise RuntimeError("could not switch the generation on")
        finally:
            self.Gopt = G

    def _pseudo_transient(self, x, bias, Tn, tol, dt=1e-15, dt_end=1e3):
        """March dn/dt + div Gamma = -R with backward-Euler steps that grow
        from `dt` to `dt_end` seconds, then settle with the steady Newton
        method. Returns the steady state, or None."""
        if self._transient:
            return None
        for _ in range(600):
            xn, conv, it = self._newton(x, bias, Tn, dt=dt, old=x, tol=tol, max_iter=30)
            if conv:
                x = xn
                if dt >= dt_end:
                    xs, conv_s, _ = self._newton(x, bias, Tn, tol=tol, max_iter=60)
                    if conv_s:
                        return xs
                    dt_end *= 1e3
                    if dt_end > 1e15:
                        return None
                dt *= 4.0 if it <= 8 else 1.5
            else:
                dt /= 4.0
                if dt < 1e-24:
                    return None
        return None

    def _check_bias(self, bias):
        for k, v in bias.items():
            if k not in self.edict:
                raise ValueError(f"no electrode named {k!r}")
            if not math.isfinite(v):
                raise ValueError("voltages must be finite")
        return {e.name: float(bias.get(e.name, 0.0)) for e in self.electrodes}

    def _ramp(self, x, start, target, Tn, max_bias_step, tol):
        """Ramp the voltages from `start` to `target` with adaptive steps;
        each step starts from a linear extrapolation of the last two
        solutions (a secant predictor)."""
        dist = max(abs(target[k] - start[k]) for k in target)
        if dist == 0:
            x, conv, it = self._newton(x, target, Tn, tol=tol)
            if not conv:
                raise RuntimeError("the solve did not converge")
            return x, it
        cap = min(1.0, max_bias_step / dist)
        step = cap
        frac, it_total = 0.0, 0
        prev = None                       # (frac, x) of the solution before
        while frac < 1.0 - 1e-12:
            f = min(1.0, frac + step)
            b = {k: start[k] + f * (target[k] - start[k]) for k in target}
            guess = x
            if prev is not None and frac > prev[0]:
                # extrapolate the potential and the quasi-Fermi levels of
                # carriers that matter; a level of a vanishing carrier is
                # kept (extrapolating it only throws it further off)
                d = (x - prev[1]) * (f - frac) / (frac - prev[0])
                X = x.reshape(self.N, self.nv)
                _, n, p, _, _ = self._carriers(*self._unpack(x)[:3])
                D = d.reshape(self.N, self.nv).copy()
                D[:, 1] *= n > 1e3 * self.floor
                if self.holes:
                    D[:, 2] *= p > 1e3 * self.floor
                guess = x + D.ravel()
            if guess is not x:
                # the predictor gets a few iterations; if it does not pay off,
                # start again from the last solution
                xn, conv, it = self._newton(guess, b, Tn, tol=tol, max_iter=8)
                if not conv:
                    xn, conv, it2 = self._newton(x, b, Tn, tol=tol, max_iter=25)
                    it += it2
            else:
                xn, conv, it = self._newton(x, b, Tn, tol=tol, max_iter=25)
            if not conv and step * dist < 0.05:
                # small steps keep failing: try Gummel's decoupled iteration
                xg, convg = self._gummel(x, b, Tn, tol=tol)
                if convg:
                    xn, conv = xg, True
            it_total += it
            if conv:
                prev = (frac, x)
                x, frac = xn, f
                if it <= 6:
                    step = min(step * 1.6, cap)
            else:
                step *= 0.35
                if step * dist < 1e-4:
                    raise RuntimeError(
                        f"the bias ramp stalled near {b}; try a finer grid or "
                        "smaller max_bias_step")
        return x, it_total

    def solve(self, bias: dict, guess: DeviceState = None, thermal=None,
              max_bias_step: float = 0.5, tol: float = 1e-8,
              thermal_tol: float = 0.01, max_thermal_iter: int = 30,
              energy_relaxation: float = None) -> DeviceState:
        """Steady state at the given voltages {electrode: V}; others at 0 V.

        guess : an earlier state of this device to start from (its bias is
            ramped to the new one).
        thermal : a `thermal.ThermalModel` to include self-heating (the
            lattice temperature and the current are iterated until the
            temperature changes by less than thermal_tol K).
        energy_relaxation : the electrons' energy relaxation time tau_E (s,
            with your source) to solve the energy balance together with the
            currents (hydrodynamic model, new in 0.21.0): the mobility and
            the diffusion follow the electron temperature, which is returned
            as `electron_temperature`. Not with `thermal`.
        """
        target = self._check_bias(bias)
        if self._bt is not None and not getattr(self, "_bt_inner", False):
            return self._solve_band_to_band(bias, guess, thermal, max_bias_step, tol,
                                            thermal_tol, max_thermal_iter, energy_relaxation)
        if energy_relaxation is not None:
            if thermal is not None:
                raise ValueError("energy_relaxation and thermal cannot be combined yet")
            if not (math.isfinite(energy_relaxation) and energy_relaxation > 0):
                raise ValueError("energy_relaxation must be > 0 (s)")
            Tl = np.full(self.N, self.T)
            if guess is not None:
                x0, start = guess._x.copy(), dict(guess.internal or guess.bias)
                Tg = guess.electron_temperature
            else:
                st0 = self.solve(bias, max_bias_step=max_bias_step, tol=tol)
                x0, start, Tg = st0._x.copy(), dict(st0.internal or st0.bias), None
            if Tg is None:
                # reach the voltages with drift-diffusion first
                x0, start, _ = self._settle(x0, start, target, Tl, max_bias_step, tol)
            return self._hydro_solve(x0, start, target, Tl, energy_relaxation,
                                     max_bias_step, tol, start_Tn=Tg)
        if guess is not None:
            x, start = guess._x.copy(), dict(guess.internal or guess.bias)
            Tn = guess.temperature.ravel().copy()
            if x.size != self.N * self.nv:
                raise ValueError("guess is from a different device")
        else:
            st = self.equilibrium()
            x, start = st._x.copy(), dict(st.bias)
            Tn = np.full(self.N, self.T)
            if np.any(self.Gopt > 0):
                # reach the voltages in the dark, then turn the light on
                G = self.Gopt
                try:
                    self.Gopt = np.zeros(self.N)
                    x, vint0, _ = self._settle(x, start, target, Tn, max_bias_step, tol)
                finally:
                    self.Gopt = G
                x = self._switch_on_generation(x, vint0, Tn, tol)
                start = dict(vint0)
        if thermal is None:
            Tn = np.full(self.N, self.T)
            x, vint, it = self._settle(x, start, target, Tn, max_bias_step, tol)
            return self._state(x, target, Tn, True, it, internal=vint)
        # electrothermal: settle at the present temperature, then iterate
        x, vint, it = self._settle(x, start, target, Tn, max_bias_step, tol)
        th = self._thermal_setup(thermal)
        for k in range(1, max_thermal_iter + 1):
            st = self._state(x, target, Tn, True, it, internal=vint)
            Tnew = self._temperature(th, st.heat, Tn)
            change = float(np.max(np.abs(Tnew - Tn)))
            T_next = Tn + 0.8 * (Tnew - Tn) if k > 3 else Tnew
            x, vint, it = self._settle_temperature(x, vint, target, Tn, T_next,
                                                   max_bias_step, tol)
            Tn = T_next
            if change < thermal_tol:
                st = self._state(x, target, Tn, True, it, internal=vint)
                st.thermal_iterations = k
                return st
        raise RuntimeError("self-heating did not converge (the device may be "
                           "in thermal runaway, or lower the bias)")

    def _settle_temperature(self, x, vint, target, T0, T1, max_bias_step, tol, depth=0):
        """Settle at the lattice temperature T1 starting from a solution at
        T0; if one jump is too far for Newton, go there in smaller steps."""
        try:
            return self._settle(x, vint, target, T1, max_bias_step, tol)
        except RuntimeError:
            if depth >= 6:
                raise
        Tm = 0.5 * (T0 + T1)
        x, vint, _ = self._settle_temperature(x, vint, target, T0, Tm, max_bias_step,
                                              tol, depth + 1)
        return self._settle_temperature(x, vint, target, Tm, T1, max_bias_step, tol,
                                        depth + 1)

    def _settle(self, x, start, target, Tn, max_bias_step, tol):
        """Ramp to the outside voltages `target`; with series resistors,
        Newton on the contact voltages V (V + R I(V) = target), using the
        device's exact small-signal conductance."""
        R = np.array([e.resistance for e in self.electrodes])
        names = [e.name for e in self.electrodes]
        if not np.any(R > 0):
            x, it = self._ramp(x, start, target, Tn, max_bias_step, tol)
            return x, dict(target), it
        vint = dict(start)
        it_total = 0
        Vt = np.array([target[nm] for nm in names])
        for _ in range(60):
            x, it = self._ramp(x, start, vint, Tn, max_bias_step, tol)
            it_total += it
            start = dict(vint)
            st = self._state(x, vint, Tn, True, it)
            I = np.array([st.current[nm] for nm in names])
            V = np.array([vint[nm] for nm in names])
            phi = V + R * I - Vt
            if np.max(np.abs(phi)) < 1e-9:
                return x, vint, it_total
            G = self._conductance(x, vint, Tn)
            dV = np.linalg.solve(np.eye(len(names)) + R[:, None] * G, -phi)
            big = np.max(np.abs(dV))
            if big > max_bias_step:
                dV *= max_bias_step / big
            vint = {nm: float(V[k] + dV[k]) for k, nm in enumerate(names)}
        raise RuntimeError("the series-resistance iteration did not converge")

    def sweep(self, electrode: str, values, bias: dict = None, guess=None,
              thermal=None, **kw):
        """Steady states along a sweep of one electrode's voltage, the
        others held at `bias`. Returns (values, {electrode: current array},
        states)."""
        base = dict(bias or {})
        states, prev = [], guess
        for v in values:
            b = dict(base)
            b[electrode] = float(v)
            prev = self.solve(b, guess=prev, thermal=thermal, **kw)
            states.append(prev)
        cur = {e.name: np.array([s.current[e.name] for s in states])
               for e in self.electrodes}
        return np.asarray(values, float), cur, states

    def emission_times(self):
        """Mean time (s) one trapped electron (and hole) of each trap waits
        before it is emitted, 1 / e_n and 1 / e_p, where the trap sits (the
        median over its nodes; inf where there is no emission)."""
        out = []
        for tr in self._traps:
            w = tr["where"]
            en = np.median(tr["en"][w]) if np.any(w) else 0.0
            ep = np.median(np.broadcast_to(tr["ep"], (self.N,))[w]) if np.any(w) else 0.0
            out.append((1.0 / en if en > 0 else math.inf, 1.0 / ep if ep > 0 else math.inf))
        return out

    # --------------------------------------------------------- transient
    def transient(self, start: DeviceState, bias: dict, times,
                  rise_time: float = 1e-9, keep=(), tol: float = 1e-8,
                  max_bias_step: float = 0.5,
                  initial_occupancy=None) -> TransientResult:
        """Switch from the steady state `start` to the voltages `bias` (a
        linear change over `rise_time` seconds) and follow the device in
        time with backward Euler, the traps filling and emptying.

        times : increasing output times (s) after the switch; steps in
            between are chosen adaptively (log spacing works well).
        keep : indices of `times` at which to store the full state.
        initial_occupancy : optional list (one per trap) of a number or an
            array of the grid's shape to start the traps from instead of
            the steady occupancy of `start` (for example all full after a
            stress you did not simulate). The first step then also settles
            the potential to the new trap charge.
        The lattice temperature stays at that of `start`.
        """
        if not self._traps:
            raise ValueError("a transient needs traps (nothing else here has "
                             "memory on the time scales of interest)")
        times = np.asarray(times, float)
        if times.ndim != 1 or times.size == 0 or np.any(np.diff(times) <= 0) or times[0] <= 0:
            raise ValueError("times must be positive and increasing")
        target = self._check_bias(bias)
        v0 = dict(start.bias)
        vint = dict(start.internal or start.bias)
        Tn = start.temperature.ravel().copy()
        # extended unknowns: append the steady occupancies of `start`
        X0 = start._x.reshape(self.N, 2 + int(self.holes))
        occ0 = []
        _, n, p, _, _ = self._carriers(*(list(self._unpack_static(start._x))))
        for k, tr in enumerate(self._traps):
            f0 = self._f_steady(tr, n, p)
            if initial_occupancy is not None and initial_occupancy[k] is not None:
                f0 = np.broadcast_to(np.asarray(initial_occupancy[k], float),
                                     self.shape).ravel()
                if not np.all((f0 >= 0) & (f0 <= 1)):
                    raise ValueError("an occupancy lies between 0 and 1")
            occ0.append(np.where(tr["where"], f0, 0.0))
        if initial_occupancy is not None and len(initial_occupancy) != len(self._traps):
            raise ValueError("initial_occupancy needs one entry per trap")
        self._transient = True
        try:
            x = np.concatenate([X0, np.stack(occ0, axis=1)], axis=1).ravel()
            chg = dict(start.charge)
            t = 0.0
            cur = {e.name: [] for e in self.electrodes}
            mocc = [[] for _ in self._traps]
            states = []
            steps = []
            dt = min(times[0], rise_time if rise_time > 0 else times[0]) / 10.0

            def bias_at(tt):
                f = min(1.0, tt / rise_time) if rise_time > 0 else 1.0
                return {k: v0[k] + f * (target[k] - v0[k]) for k in target}

            for kk, tout in enumerate(times):
                while t < tout * (1 - 1e-12):
                    h = min(dt, tout - t)
                    if t < rise_time:
                        h = min(h, rise_time - t if rise_time - t > 0 else h)
                        # keep the voltage change per step modest
                        rate = max(abs(target[k] - v0[k]) for k in target) / rise_time
                        if rate > 0:
                            h = min(h, max_bias_step / rate)
                    b = bias_at(t + h)
                    xn, conv, it, st, vint_n = self._time_step(
                        x, b, vint, Tn, h, chg, t + h, tol)
                    if conv:
                        vint = vint_n
                        chg = dict(st.charge)
                        x = xn
                        t += h
                        steps.append(h)
                        if it <= 5:
                            dt = h * 2.0
                        elif it > 12:
                            dt = h * 0.7
                        else:
                            dt = h
                    else:
                        dt = h * 0.25
                        if dt < 1e-18:
                            raise RuntimeError(f"the transient stalled at t = {t:g} s")
                for e in self.electrodes:
                    cur[e.name].append(st.current[e.name])
                for k, tr in enumerate(self._traps):
                    o = st.occupancy[k].ravel()
                    w = tr["where"]
                    mocc[k].append(float(np.sum(o[w] * tr["Nt"][w] * self.vol[w])
                                         / np.sum(tr["Nt"][w] * self.vol[w])))
                if kk in keep:
                    states.append(st)
        finally:
            self._transient = False
        return TransientResult(times=times,
                               current={k: np.array(v) for k, v in cur.items()},
                               mean_occupancy=[np.array(m) for m in mocc],
                               states=states, steps=np.array(steps))

    def _time_step(self, x, b, vint, Tn, h, chg, time, tol):
        """One backward-Euler step to the outside voltages b; with series
        resistors, Newton on the contact voltages using the time-step
        conductance (which includes the displacement current)."""
        R = np.array([e.resistance for e in self.electrodes])
        names = [e.name for e in self.electrodes]
        if not np.any(R > 0):
            xn, conv, it = self._newton(x, b, Tn, dt=h, old=x, tol=tol)
            st = (self._state(xn, b, Tn, True, it, dt=h, old=x, time=time,
                              old_charge=chg) if conv else None)
            return xn, conv, it, st, dict(b)
        V = np.array([vint[nm] for nm in names])
        Vt = np.array([b[nm] for nm in names])
        xn = x
        it_total = 0
        for _ in range(30):
            vd = {nm: float(V[k]) for k, nm in enumerate(names)}
            xn, conv, it = self._newton(xn, vd, Tn, dt=h, old=x, tol=tol)
            it_total += it
            if not conv:
                return xn, False, it_total, None, vint
            st = self._state(xn, b, Tn, True, it, dt=h, old=x, time=time,
                             old_charge=chg, internal=vd)
            I = np.array([st.current[nm] for nm in names])
            phi = V + R * I - Vt
            if np.max(np.abs(phi)) < 1e-9:
                return xn, True, it_total, st, vd
            G = self._conductance(xn, vd, Tn, dt=h, old=x)
            V = V + np.linalg.solve(np.eye(len(names)) + R[:, None] * G, -phi)
        return xn, False, it_total, None, vint

    def _unpack_static(self, x):
        X = x.reshape(self.N, 2 + int(self.holes))
        fp = X[:, 2] if self.holes else np.zeros(self.N)
        return X[:, 0], X[:, 1], fp

    # ----------------------------------------------------------- thermal
    def _thermal_setup(self, model):
        from .thermal import ThermalModel
        if not isinstance(model, ThermalModel):
            raise ValueError("thermal must be a ThermalModel")
        lat = []
        maps = []
        for ax in range(self.dim - 1):
            a = self.axes[ax]
            ext = model.lateral_extension
            if ext > 0:
                h0 = a[1] - a[0]
                left = _graded(-ext, 0.0, model.dz_max, h0, model.dz_max, model.growth)
                right = _graded(a[-1], a[-1] + ext, a[-1] - a[-2], model.dz_max,
                                model.dz_max, model.growth)
                full = np.unique(np.round(np.concatenate([left, a, right]), 9))
            else:
                full = a.copy()
            lat.append(full)
            maps.append(np.searchsorted(full, np.round(a, 9)))
        z = self.axes[-1]
        sub = list(model.substrate)
        if sub:
            tot = sum(L.thickness for L in sub)
            edges = np.cumsum([0.0] + [L.thickness for L in sub])
            pts = []
            for a, b in zip(edges[:-1], edges[1:]):
                pts.append(_graded(a, b, min(model.dz_max, (b - a) / 4),
                                   min(z[1] - z[0], (b - a) / 4) if b == tot else
                                   min(model.dz_max, (b - a) / 4),
                                   model.dz_max, model.growth))
            zs = np.unique(np.round(np.concatenate(pts), 9))
            zfull = np.unique(np.round(np.concatenate([zs, z + tot]), 9))
            zmap = np.searchsorted(zfull, np.round(z + tot, 9))
            mats = []
            zc = 0.5 * (zfull[1:] + zfull[:-1])
            for c in zc:
                if c < tot:
                    k = min(int(np.searchsorted(edges, c, side="right") - 1), len(sub) - 1)
                    mats.append(sub[k].material)
                else:
                    k = min(int(np.searchsorted(z + tot, c) - 1), z.size - 2)
                    mats.append(self._thermal_cell(model, k))
        else:
            zfull, zmap = z.copy(), np.arange(z.size)
            mats = [self._thermal_cell(model, k) for k in range(z.size - 1)]
        return dict(model=model, lat=lat, maps=maps, z=zfull, zmap=zmap, mats=mats)

    def _thermal_cell(self, model, k):
        """Thermal material of z cell k: its layer's, or its cap layer's.
        (Etched regions keep the layer's material: heat flow through a
        recess is not modeled.)"""
        if self.cell_in_cap[k]:
            zc = 0.5 * (self.axes[-1][k] + self.axes[-1][k + 1])
            top = self.top
            for c_ in self.cap:
                if zc <= top + c_.thickness + 1e-9:
                    return model.material_for(c_.label)
                top += c_.thickness
        return model.material_for(self.layers[self.cell_layer[k]].label)

    def _temperature(self, th, heat, Tn):
        from .thermal import solve_heat
        shape = tuple(a.size for a in th["lat"]) + (th["z"].size,)
        H = np.zeros(shape)
        sl = np.ix_(*th["maps"], th["zmap"])
        # W/mm -> W/m of depth in 2D; W in 3D
        H[sl] = heat * (1e3 if self.dim == 2 else 1.0)
        T, _ = solve_heat(th["lat"], th["z"], th["mats"], H,
                          th["model"].sink_temperature_K)
        return T[sl].ravel()


    def pulse(self, bias: dict, times, thermal, guess: DeviceState = None,
              keep=(), thermal_tol: float = 0.01, max_bias_step: float = 0.5,
              tol: float = 1e-8):
        """Self-heating in time (new in 0.20.0): the device starts at the
        heat-sink temperature, is switched to `bias` at t = 0, and the
        lattice warms up while the current flows (pulsed I-V). The heat
        equation C dT/dt = div(k grad T) + H is stepped by backward Euler to
        each of `times` (s); the carriers follow at once (they settle in
        picoseconds, the lattice in nanoseconds to milliseconds), so at each
        step the electrical steady state at the present temperature and the
        heat equation are iterated to agree. Every thermal material needs
        its density and specific heat. The traps stay at their steady
        occupancy (use `transient` for trap dynamics).

        Returns (times, {electrode: current array}, hottest temperature
        array, states at the indices in `keep`).
        """
        import scipy.sparse as sp
        from .thermal import _HeatGrid
        times = np.asarray(times, float)
        if times.ndim != 1 or np.any(np.diff(times) <= 0) or times[0] <= 0:
            raise ValueError("times must be positive and increasing")
        th = self._thermal_setup(thermal)
        for m in th["mats"]:
            if not m.has_capacity:
                raise ValueError(f"{m.name} has no density and specific heat; "
                                 "heat that changes in time needs them")
        g = _HeatGrid(th["lat"], th["z"], th["mats"])
        sl = np.ix_(*th["maps"], th["zmap"])
        Ts = thermal.sink_temperature_K
        Tfull = np.full(g.N, float(Ts))
        if Ts != self.T:
            raise ValueError("the heat sink must be at the device temperature "
                             "for a pulse from cold")
        target = self._check_bias(bias)
        Tn = np.full(self.N, self.T)
        st = guess if guess is not None else self.solve(target)
        if guess is not None:
            st = self.solve(target, guess=guess)
        x, vint = st._x.copy(), dict(st.internal or st.bias)
        cur = {e.name: [] for e in self.electrodes}
        tmax, states = [], []
        t_prev = 0.0
        for kk, t in enumerate(times):
            dt = t - t_prev
            Told = Tfull.copy()
            Tk = Tfull.copy()
            Tprev_dev = Tk.reshape(g.shape)[sl].ravel()
            for _ in range(60):
                Tdev = Tk.reshape(g.shape)[sl].ravel()
                x, vint, it = self._settle_temperature(x, vint, target, Tprev_dev, Tdev,
                                                       max_bias_step, tol)
                Tprev_dev = Tdev
                stk = self._state(x, target, Tdev, True, it, internal=vint)
                H = np.zeros(g.shape)
                H[sl] = stk.heat * (1e3 if self.dim == 2 else 1.0)
                C = g.capacity(Tk)
                A = g.matrix(Tk) + sp.diags(C / dt)
                Tnew = g.solve(A, H.ravel() + C / dt * Told, Ts)
                change = float(np.max(np.abs(Tnew - Tk)))
                Tk = Tnew
                if change < thermal_tol:
                    break
            else:
                raise RuntimeError(f"the heating step to t = {t:g} s did not converge")
            Tfull = Tk
            Tdev = Tfull.reshape(g.shape)[sl].ravel()
            x, vint, it = self._settle(x, vint, target, Tdev, max_bias_step, tol)
            stk = self._state(x, target, Tdev, True, it, internal=vint, time=t)
            for e in self.electrodes:
                cur[e.name].append(stk.current[e.name])
            tmax.append(float(Tdev.max()))
            if kk in keep:
                states.append(stk)
            t_prev = t
        return times, {k: np.array(v) for k, v in cur.items()}, np.array(tmax), states


    # ------------------------------------------------------------ breakdown
    def field(self, state: DeviceState):
        """Electric field -grad psi at every node (V/cm), one array per axis."""
        psi = state.psi
        return [-np.gradient(psi, a, axis=k) * 1e7 for k, a in enumerate(self.axes)]

    def ionization_integral(self, state: DeviceState, ionization: "Ionization" = None,
                            start=None, max_steps: int = 20000):
        """The ionization integrals along the field line through the point
        of highest field (or through `start`, a coordinate tuple in nm), the
        breakdown criterion used with a computed field (new in 0.20.0):

            I_n = int_A^B alpha_n exp(-int_A^s (alpha_n - alpha_p) ds') ds,

        for electrons that enter at the low-potential end A and drift to the
        high-potential end B, and I_p likewise for holes from B (R. J.
        McIntyre, IEEE Trans. Electron Devices 13, 164 (1966)). Avalanche
        breakdown is reached when they reach 1 (the two reach 1 together).
        The field comes from `state` (which need not include impact
        ionization itself). Returns a dict with I_n, I_p, the path (nm),
        the field along it (V/cm) and the peak field. Only the semiconductor
        (the layers of `ionization.where`, if given) ionizes: the start is
        the highest field there, and alpha is 0 where the path leaves it.
        """
        from scipy.interpolate import RegularGridInterpolator
        ion = ionization or self.ionization
        if ion is None:
            raise ValueError("give an Ionization (for example IMPACT_GAN)")
        Ef = self.field(state)
        interps = [RegularGridInterpolator(self.axes, c, bounds_error=False,
                                           fill_value=np.nan) for c in Ef]
        mag = np.sqrt(sum(c ** 2 for c in Ef))
        semi = self._ion_mask(ion).reshape(self.shape)
        in_semi = RegularGridInterpolator(self.axes, semi.astype(float), bounds_error=False,
                                          fill_value=0.0)
        mag = np.where(semi, mag, 0.0)
        if start is None:
            k = np.unravel_index(int(np.argmax(mag)), mag.shape)
            start = np.array([a[i] for a, i in zip(self.axes, k)], float)
        start = np.asarray(start, float)
        hmin = min(float(np.min(np.diff(a))) for a in self.axes)
        lo = np.array([a[0] for a in self.axes])
        hi = np.array([a[-1] for a in self.axes])

        def trace(sign):
            pts = [start.copy()]
            p_ = start.copy()
            for _ in range(max_steps):
                e = np.array([f(np.clip(p_, lo, hi))[0] for f in interps])
                m = np.linalg.norm(e)
                if not np.isfinite(m) or m < 1e3:
                    break
                d = sign * e / m
                ds = 0.5 * hmin
                mid = p_ + 0.5 * ds * d
                e2 = np.array([f(np.clip(mid, lo, hi))[0] for f in interps])
                m2 = np.linalg.norm(e2)
                if not np.isfinite(m2) or m2 == 0:
                    break
                q = p_ + ds * sign * e2 / m2
                if np.any(q < lo - 1e-9) or np.any(q > hi + 1e-9):
                    break
                p_ = np.clip(q, lo, hi)
                pts.append(p_.copy())
            return np.array(pts)

        to_A = trace(+1.0)          # along E: toward low potential
        to_B = trace(-1.0)          # against E: toward high potential
        path = np.vstack([to_A[::-1], to_B[1:]])          # A ... B
        Eabs = np.linalg.norm(np.array([[f(np.clip(pt, lo, hi))[0] for f in interps]
                                        for pt in path]), axis=1)
        s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))])
        sc = s * 1e-7                                     # cm
        an, ap = ion.alpha(Eabs)
        inside = in_semi(np.clip(path, lo, hi)) > 0.5
        an, ap = an * inside, ap * inside
        d = an - ap
        cum = np.concatenate([[0.0], np.cumsum(0.5 * (d[1:] + d[:-1]) * np.diff(sc))])
        fn_ = an * np.exp(-cum)
        In = float(np.sum(0.5 * (fn_[1:] + fn_[:-1]) * np.diff(sc)))
        cumB = cum[-1] - cum                              # int_s^B (an - ap)
        fp_ = ap * np.exp(cumB)
        Ip = float(np.sum(0.5 * (fp_[1:] + fp_[:-1]) * np.diff(sc)))
        return {"I_n": In, "I_p": Ip, "path": path, "field": Eabs,
                "peak_field": float(mag.max()), "start": start}

    def breakdown_voltage(self, electrode: str, voltages, bias: dict = None,
                          ionization: "Ionization" = None, guess=None):
        """Sweep one electrode and return the voltage where the larger
        ionization integral reaches 1 (log-linear interpolation of
        1 - I), with the integrals at every step; None if it is not reached
        in the range."""
        V, I, S = [], [], []
        prev = guess
        for v in voltages:
            b = dict(bias or {})
            b[electrode] = float(v)
            prev = self.solve(b, guess=prev)
            r = self.ionization_integral(prev, ionization)
            V.append(float(v))
            I.append(max(r["I_n"], r["I_p"]))
            S.append(prev)
            if I[-1] >= 1.0:
                break
        vbr = None
        if len(I) == 1 and I[0] >= 1.0:
            raise ValueError(f"the ionization integral is already {I[0]:.3g} at the "
                             "first voltage; start lower")
        if len(I) > 1 and I[-1] >= 1.0:
            a, b_ = 1.0 - I[-2], 1.0 - I[-1]
            vbr = V[-2] + (V[-1] - V[-2]) * a / (a - b_) if a != b_ else V[-1]
        return vbr, np.array(V), np.array(I), S


    def breakdown_by_current(self, electrode: str, voltages, bias: dict = None,
                             current: float = 1e-3, guess=None, **solve_kw):
        """Breakdown voltage by a current criterion (new in 0.21.0): sweep
        one electrode until the magnitude of its current reaches `current`
        (A/mm in 2D, A in 3D; 1 mA/mm is a common choice for transistors in
        the off state), and interpolate log(I) linearly in V between the last
        two steps. Whatever the device includes counts: leakage under the
        gate or through the buffer (punch-through), tunneling through
        barriers (`TunnelBarrier`), band-to-band tunneling, impact
        ionization in the solve. Returns a dict: voltage (None if not
        reached; the first voltage if the current is already there, with
        first_step True), the swept voltages, every electrode's current at each, and
        `share`, each electrode's part of the returning current at the last
        step (where the current goes: the gate, the source, ...)."""
        V, cur, S = [], {e.name: [] for e in self.electrodes}, []
        prev = guess
        vbr = None
        for v in voltages:
            b = dict(bias or {})
            b[electrode] = float(v)
            prev = self.solve(b, guess=prev, **solve_kw)
            V.append(float(v))
            for e in self.electrodes:
                cur[e.name].append(prev.current[e.name])
            S.append(prev)
            I = abs(prev.current[electrode])
            if I >= current:
                if len(V) > 1:
                    I0 = max(abs(cur[electrode][-2]), 1e-300)
                    if I0 < current:
                        a, b_ = math.log(I0), math.log(I)
                        vbr = V[-2] + (V[-1] - V[-2]) * (math.log(current) - a) / (b_ - a)
                    else:
                        vbr = V[-1]
                else:
                    vbr = V[-1]
                break
        first = vbr is not None and len(V) == 1
        last = S[-1].current
        others = {k: v for k, v in last.items() if k != electrode}
        tot = sum(abs(v) for v in others.values()) or 1.0
        share = {k: abs(v) / tot for k, v in others.items()}
        return dict(voltage=vbr, voltages=np.array(V), first_step=first,
                    current={k: np.array(v) for k, v in cur.items()}, share=share,
                    states=S)

    def field_check(self, state: DeviceState, critical: dict):
        """Peak field in each material against a critical field you give
        (new in 0.21.0): critical = {label: (MV/cm, reference)} for layer
        labels, cap (Dielectric) labels and Region labels. The field is taken
        at cell centres from the potential at the cell's corners. Returns
        {label: dict(peak MV/cm, where (x, z) nm, critical, ratio,
        reference)}; ratio >= 1 means that material breaks down there."""
        crit = {}
        for k, v in critical.items():
            if not (isinstance(v, tuple) and len(v) == 2 and math.isfinite(v[0]) and v[0] > 0
                    and isinstance(v[1], str) and v[1].strip()):
                raise ValueError("critical fields are {label: (MV/cm, reference)}")
            crit[k] = v
        psi = state.psi
        axes = self.axes
        dim = self.dim
        comps = []
        for ax in range(dim):
            d = np.diff(psi, axis=ax) / np.diff(axes[ax]).reshape(
                [-1 if b == ax else 1 for b in range(dim)])
            for b in range(dim):
                if b != ax:
                    d = 0.5 * (d.take(np.arange(d.shape[b] - 1), axis=b)
                               + d.take(np.arange(1, d.shape[b]), axis=b))
            comps.append(d * 1e7)                         # V/cm
        F = np.sqrt(sum(c * c for c in comps)) / 1e6    # MV/cm
        labels = self._cell_labels()
        cc = np.meshgrid(*[0.5 * (a[1:] + a[:-1]) for a in axes], indexing="ij")
        out = {}
        for lab, (ec, ref) in crit.items():
            m = labels == lab
            if not m.any():
                raise ValueError(f"no cell is made of {lab!r}")
            k = np.unravel_index(int(np.argmax(np.where(m, F, -1.0))), F.shape)
            pk = float(F[k])
            out[lab] = dict(peak=pk, where=tuple(float(c[k]) for c in cc), critical=float(ec),
                            ratio=pk / float(ec), reference=ref)
        return out

    def _cell_labels(self):
        """The material label of every cell (layer, cap layer or region)."""
        cshape = tuple(sz - 1 for sz in self.shape)
        cz = np.broadcast_to(np.arange(cshape[-1]).reshape((1,) * (self.dim - 1) + (-1,)),
                             cshape)
        lab = np.array([L.label for L in self.layers], dtype=object)[self.cell_layer][cz]
        lab = lab.copy()
        zc = 0.5 * (self.axes[-1][1:] + self.axes[-1][:-1])
        top = self.top
        for c_ in self.cap:
            m = (zc > top - 1e-9) & (zc < top + c_.thickness + 1e-9)
            lab = np.where(m[cz], c_.label, lab)
            top += c_.thickness
        ccent = np.meshgrid(*[0.5 * (a[1:] + a[:-1]) for a in self.axes], indexing="ij")
        for r_ in self.regions:
            box = ((ccent[0] > r_.x0) & (ccent[0] < r_.x1) & (ccent[-1] > r_.z0)
                   & (ccent[-1] < r_.z1))
            if self.dim == 3 and r_.y0 is not None:
                box &= (ccent[1] > r_.y0) & (ccent[1] < r_.y1)
            lab = np.where(box, r_.label, lab)
        return lab

    def thermal_runaway(self, electrode: str, voltages, thermal, bias: dict = None,
                        guess=None, refine: bool = True, rtol: float = 1e-3, **solve_kw):
        """Sweep one electrode with self-heating until no self-heated steady
        state exists (thermal runaway, new in 0.21.0): returns the last
        voltage with a steady state, the first without one (None if every
        step has one), and the hottest temperature and power along the way.
        In this model the lattice temperature acts through the mobility
        only, so runaway needs a mobility that rises with temperature (a
        negative temperature_exponent); the carrier statistics and trap
        emission stay at the ambient temperature."""
        solve_kw.setdefault("max_thermal_iter", 300)
        V, Tmax, P = [], [], []
        prev = guess
        failed = None
        for v in voltages:
            b = dict(bias or {})
            b[electrode] = float(v)
            try:
                prev = self.solve(b, guess=prev, thermal=thermal, **solve_kw)
            except RuntimeError:
                failed = float(v)
                break
            V.append(float(v))
            Tmax.append(float(prev.temperature.max()))
            P.append(prev.power)
        if failed is not None and V and refine:
            # bisect between the last voltage with a steady state and the
            # first without one
            lo, hi, st_lo = V[-1], failed, prev
            while hi - lo > rtol * abs(hi):
                mid = 0.5 * (lo + hi)
                b = dict(bias or {})
                b[electrode] = mid
                try:
                    st_lo = self.solve(b, guess=st_lo, thermal=thermal, **solve_kw)
                    lo = mid
                    V.append(mid)
                    Tmax.append(float(st_lo.temperature.max()))
                    P.append(st_lo.power)
                except RuntimeError:
                    hi = mid
            failed = hi
        return dict(stable=V[-1] if V else None, runaway=failed, voltages=np.array(V),
                    hottest=np.array(Tmax), power=np.array(P))

    # ------------------------------------------------- tunneling at the gate
    def gate_tunneling(self, state: DeviceState, electrode: str = "gate",
                       depth: float = 200.0, mass_m0=None, n_energy: int = 2000):
        """Current through and over the barrier of a Schottky electrode,
        column by column under it (new in 0.20.0): along each column of
        nodes below the electrode, from the deepest point of the conduction
        band within `depth` nm (the electron layer) up to the metal, the
        Tsu-Esaki current with WKB transmission (`leakage.tsu_esaki_current`)
        between the electron quasi-Fermi level there and the metal's Fermi
        level -V. The bands are those of `state` (the current is assumed
        small enough not to change them). Use it with a gate without
        `thermionic` (over-the-barrier current is counted here). For a
        metal box (a recessed gate) the columns end at the box's lower face
        (its side walls are not counted); parts of the metal that lie on a
        dielectric (a field plate) carry none, and a gate entirely on a
        dielectric is refused, since tunneling through insulators is not
        modeled. Returns Jc for the columns on the semiconductor. The paths
        are vertical: at the gate edges, where the field turns sideways,
        this is an approximation.

        Returns (current into the device through the electrode, in A/mm
        for 2D or A for 3D, positive when electrons leave the device into
        the metal), and the current density of every column (A/cm^2).
        """
        from .leakage import tsu_esaki_current
        e = self.edict.get(electrode)
        if e is None or e.kind != "schottky" or e.face != "top":
            raise ValueError("gate_tunneling needs a Schottky electrode on the top face")
        V = float((state.internal or state.bias)[electrode])
        nz = self.axes[-1].size
        z = self.axes[-1]
        ec = state.ec.reshape(-1, nz)
        fn = state.efn.reshape(-1, nz)
        me = self.me_node.reshape(-1, nz)
        enodes = self.enodes[electrode]
        cols_all, iz_all = enodes // nz, enodes % nz
        cols = np.unique(cols_all)
        semi = self.node_semi.reshape(-1, nz)
        bottoms = np.array([int(iz_all[cols_all == c].min()) for c in cols])
        on_semi = semi[cols, bottoms]
        if not np.any(on_semi):
            raise ValueError(f"electrode {electrode!r} sits on a dielectric; "
                             "tunneling through insulators is not modeled")
        # columns where the metal lies on a dielectric (a field plate) carry none
        cols, bottoms = cols[on_semi], bottoms[on_semi]
        Jc = np.zeros(cols.size)
        for k, c in enumerate(cols):
            it = int(bottoms[k])                            # metal's lower face
            top = z[it]
            region = (z >= top - min(depth, top)) & (z <= top) & semi[c]
            i0 = int(np.argmin(np.where(region, ec[c], np.inf)))
            zz = np.append(z[i0:it + 1], top + 1e-6)
            ee = np.append(ec[c, i0:it + 1], ec[c, it] - 50.0)
            m = (me[c, i0:it + 1] if mass_m0 is None
                 else np.full(it + 1 - i0, float(mass_m0)))
            mm = np.append(m, m[-1])
            Jc[k] = tsu_esaki_current(zz, ee, float(fn[c, i0]), -V, mm, self.T,
                                      n_energy=n_energy, transport_mass=float(m[0]))
        # lateral area of each column's bottom node
        bottom = cols * nz + bottoms
        area = self.lat_area[bottom]                               # nm (2D) or nm^2 (3D)
        if self.dim == 2:
            I = float(np.sum(Jc * area * 1e-7 * 0.1))              # A per mm
        else:
            I = float(np.sum(Jc * area * 1e-14))                   # A
        return I, Jc

    # --------------------------------------------------- hot electrons
    def electron_temperature(self, state: DeviceState, tau_energy: float,
                             tol: float = 1e-6, max_iter: int = 400):
        """Electron temperature from the energy balance of the electrons
        (new in 0.20.0), with the currents and fields of `state`:

            div S = -Gamma_n . grad E_c - (3/2) k n (T_n - T_L) / tau_E,
            S = -kappa grad T_n + (5/2) k T_n Gamma_n,
            kappa = (5/2) k n D_n,  D_n = mu k T_n / q,

        where -Gamma_n . grad E_c is the work the field does on the
        electron current (J . E) and tau_E the energy relaxation time (s,
        yours, with its source; for example 0.47 ps was measured at 250 K in
        AlGaN/AlN/GaN by E. Tiras et al., Superlattices Microstruct. 51, 733
        (2012)). The energy flux is discretized with exponential fitting
        (the Scharfetter-Gummel form for convection and diffusion); T_n = T_L
        at the contacts, no energy flux through other faces. (Before 0.21.0
        the work term was -Gamma_n . grad F_n, the Joule heat of the lattice;
        the two agree where the density is uniform.)

        This is one-way coupling: the temperature follows from the
        drift-diffusion currents and does not change them. For the coupled
        model (velocity overshoot) use `solve(..., energy_relaxation=)`. In
        a uniform field it gives T_n - T_L = (2/3) tau_E mu E^2 / k
        exactly. Returns T_n (K), the shape of the grid.
        """
        if not (math.isfinite(tau_energy) and tau_energy > 0):
            raise ValueError("tau_energy must be > 0 (s)")
        x = state._x
        Tl = state.temperature.ravel()
        psi, fn, fp = self._unpack_static(x)
        _, n, p, lnn, lnp = self._carriers(psi, fn, fp)
        Gn, _ = self._fluxes(psi, fn, fp, n, p, lnn, lnp, Tl)
        i, j, h = self.ei, self.ej, self.eh
        Te = 0.5 * (Tl[i] + Tl[j])
        mu, _ = _edge_mu(self._mob_n, (fn[j] - fn[i]) if self.force == "quasi-fermi"
                         else (psi[j] - psi[i]), h, Te)
        Tn = self._energy_balance(psi, n, Gn, mu, Tl, tau_energy, tol, max_iter)
        return Tn.reshape(self.shape)

    def _energy_balance(self, psi, n, Gn, mu, Tl, tau_energy, tol=1e-6, max_iter=400,
                        start=None):
        import scipy.sparse as sp
        import scipy.sparse.linalg as spl
        from .driftdiffusion import bernoulli
        i, j, h, A = self.ei, self.ej, self.eh, self.earea
        kB = KB
        Ec = self.Ec0 - psi
        pe = -Gn * (Ec[j] - Ec[i]) * A                    # eV/s on each edge (J . E)
        H = 0.5 * (np.bincount(i, pe, self.N) + np.bincount(j, pe, self.N))
        nedge = 0.5 * (n[i] + n[j])
        loss_c = self.vol * 1.5 * kB * n / tau_energy     # eV/(s K) per node
        fixed = ~self.node_semi.copy()           # no electrons in an insulator
        for nodes in self.enodes.values():
            fixed[nodes] = True
        # where there are practically no electrons (below 1e-12 of the
        # densest node, or below 1e6 cm^-3) their temperature carries no
        # energy that matters: it is held at the lattice temperature
        fixed |= n < max(1e-12 * float(np.max(n)), 1e-15)
        Tn = Tl.copy() if start is None else np.asarray(start, float).ravel().copy()
        omega, last = 1.0, math.inf
        for _ in range(max_iter):
            Tm = 0.5 * (Tn[i] + Tn[j])
            D = mu * 1e14 * kB * Tm                       # nm^2/s
            kap = 2.5 * kB * nedge * D                    # eV/(K nm s)
            kap = np.maximum(kap, 1e-12 * float(np.max(kap)) + 1e-300)
            P = 2.5 * kB * Gn * h / kap
            a = kap / h * bernoulli(-P) * A               # coefficient of T_i
            b = kap / h * bernoulli(P) * A                # coefficient of T_j
            # S_e = a T_i - b T_j leaves node i and enters node j
            rows = np.concatenate([i, i, j, j])
            cols = np.concatenate([i, j, i, j])
            vals = np.concatenate([a, -b, -a, b])
            M = sp.csr_matrix((vals, (rows, cols)), shape=(self.N, self.N))
            M = M + sp.diags(loss_c)
            rhs = H + loss_c * Tl
            keep = (~fixed).astype(float)
            M = sp.diags(keep) @ M + sp.diags(1.0 - keep)
            rhs = np.where(fixed, Tl, rhs)
            rs = np.asarray(abs(M).max(axis=1).todense()).ravel()
            rs[rs == 0] = 1.0
            Tnew = spl.spsolve((sp.diags(1.0 / rs) @ M).tocsc(), rhs / rs)
            ch = float(np.max(np.abs(Tnew - Tn)))
            if ch > last:                      # oscillating: relax
                omega = max(0.1, omega * 0.5)
            last = ch
            Tn = Tn + omega * (Tnew - Tn)
            if ch < tol * max(1.0, float(np.max(Tn))):
                break
        else:
            raise RuntimeError("the electron energy balance did not converge")
        # keep the temperature physical (an energy balance from a poorly
        # resolved, nearly empty region can dip below the lattice)
        return np.maximum(Tn, Tl)

    # ------------------------------------------- hydrodynamic (0.21.0)
    def _hydro_mobility(self, Tn, Tl, tau):
        """Edge mobility of hot electrons: the low-field model of `Mobility`
        at the field E_eff that would heat uniform electrons to T_n,
        mu(E_eff) E_eff^2 = (3/2) (k/q) (T_n - T_L) / tau_E. In a uniform
        field the energy balance then returns E_eff = E, so the coupled
        model reproduces the drift-diffusion velocity-field curve there
        exactly; where T_n lags the field (a sudden rise) the mobility stays
        high and the velocity overshoots."""
        i, j, h = self.ei, self.ej, self.eh
        Te_n = 0.5 * (Tn[i] + Tn[j])
        Te_l = 0.5 * (Tl[i] + Tl[j])
        X = 1.5 * KB * np.maximum(Te_n - Te_l, 0.0) / tau            # V^2/s per cm^2/Vs ...
        lo = np.full(X.shape, 1e-6)
        hi = np.full(X.shape, 1e9)

        def f(E):
            mu, _ = _edge_mu(self._mob_n, E * h / 1e7, h, Te_l)
            return mu * E * E
        for _ in range(80):
            mid = np.sqrt(lo * hi)
            big = f(mid) > X
            hi = np.where(big, mid, hi)
            lo = np.where(big, lo, mid)
        E = np.where(X > 0, np.sqrt(lo * hi), 0.0)
        mu, _ = _edge_mu(self._mob_n, E * h / 1e7, h, Te_l)
        return mu

    def _set_hydro(self, Tn, Tl, tau, flux_T=None):
        """Hot-electron mobility from Tn; the temperature in the fluxes is
        flux_T (default Tn; a lower one is only a step on the way)."""
        i, j = self.ei, self.ej
        Tf = Tn if flux_T is None else flux_T
        Ti, Tj = Tf[i], Tf[j]
        r = np.log(Tj / Ti)
        small = np.abs(Tj - Ti) <= 1e-9 * Ti
        Tlog = np.where(small, 0.5 * (Ti + Tj), (Tj - Ti) / np.where(small, 1.0, r))
        self._hydro = dict(Tn=Tn.copy(), Tf=Tf.copy(), Tlog=Tlog, rho=self.T / Tlog,
                           r=np.where(small, 0.0, r), mu=self._hydro_mobility(Tn, Tl, tau))

    def _settle_hydro(self, x, vint, target, Tl, tau, Tn, Tm0, Tf0, max_bias_step, tol):
        """Solve at electron temperature Tn, starting from a solution whose
        mobility used Tm0 and whose fluxes used Tf0: both move to Tn in
        steps that shrink when Newton's method needs them to."""
        frac, step = 0.0, 1.0
        while True:
            f = min(1.0, frac + step)
            Tf = Tf0 + f * (Tn - Tf0)
            Tm = Tm0 + f * (Tn - Tm0)
            self._set_hydro(Tm, Tl, tau, flux_T=Tf)
            try:
                xn, vn, it = self._settle(x, vint, target, Tl, max_bias_step, tol)
            except RuntimeError:
                step /= 4.0
                if step < 1e-4:
                    raise RuntimeError("the hydrodynamic solve did not converge; try a "
                                       "smaller voltage step")
                continue
            x, vint, frac = xn, vn, f
            if frac >= 1.0:
                return x, vint, it
            step = min(1.0 - frac, step * 2.0)

    def _hydro_solve(self, x, vint, target, Tl, tau, max_bias_step, tol, start_Tn=None,
                     tol_T=0.05, max_outer=200):
        """Couple the energy balance to the currents: alternate the energy
        balance at fixed currents and the drift-diffusion solve at fixed
        electron temperature (with the hot-electron mobility and
        temperature in the fluxes) until T_n changes by less than tol_T K."""
        self._hydro = None
        if start_Tn is None:
            psi, fn, fp = self._unpack_static(x)
            _, n, p, lnn, lnp = self._carriers(psi, fn, fp)
            Gn, _ = self._fluxes(psi, fn, fp, n, p, lnn, lnp, Tl)
            i, j, h = self.ei, self.ej, self.eh
            mu, _ = _edge_mu(self._mob_n, psi[j] - psi[i], h, 0.5 * (Tl[i] + Tl[j]))
            Tn = self._energy_balance(psi, n, Gn, mu, Tl, tau)
        else:
            Tn = np.maximum(np.asarray(start_Tn, float).ravel(), Tl)
        omega = 0.5
        last = math.inf
        hist_T, hist_R = [], []
        # the temperatures of the mobility and of the fluxes solved so far
        Tm, Tf = (Tn.copy(), Tl.copy()) if start_Tn is None else (Tn.copy(), Tn.copy())
        try:
            for k in range(1, max_outer + 1):
                x, vint, it = self._settle_hydro(x, vint, target, Tl, tau, Tn, Tm, Tf,
                                                 max_bias_step, tol)
                Tm, Tf = Tn.copy(), Tn.copy()
                psi, fn, fp = self._unpack_static(x)
                _, n, p, lnn, lnp = self._carriers(psi, fn, fp)
                Gn, _ = self._fluxes(psi, fn, fp, n, p, lnn, lnp, Tl)
                Tnew = self._energy_balance(psi, n, Gn, self._hydro["mu"], Tl, tau, start=Tn)
                change = float(np.max(np.abs(Tnew - Tn)))
                if change > last:
                    omega = max(0.05, omega * 0.7)
                    hist_T, hist_R = [], []         # restart the acceleration
                else:
                    omega = min(1.0, omega * 1.2)
                last = change
                if change < tol_T:
                    x, vint, it = self._settle_hydro(x, vint, target, Tl, tau, Tnew, Tm, Tf,
                                                     max_bias_step, tol)
                    st = self._state(x, target, Tl, True, it, internal=vint)
                    st.electron_temperature = Tnew.reshape(self.shape)
                    st.hydro_iterations = k
                    return st
                # Anderson acceleration of the fixed point T -> EB(DD(T))
                R = Tnew - Tn
                hist_T.append(Tn.copy())
                hist_R.append(R.copy())
                if len(hist_T) > 6:
                    hist_T.pop(0)
                    hist_R.pop(0)
                if len(hist_T) >= 2:
                    dR = np.stack([hist_R[q + 1] - hist_R[q] for q in range(len(hist_R) - 1)], 1)
                    dT = np.stack([hist_T[q + 1] - hist_T[q] for q in range(len(hist_T) - 1)], 1)
                    gam, *_ = np.linalg.lstsq(dR, R, rcond=None)
                    Tacc = Tn + omega * R - (dT + omega * dR) @ gam
                    Tn = np.maximum(Tacc, Tl)
                else:
                    Tn = Tn + omega * R
            raise RuntimeError("the energy balance and the currents did not settle; "
                               "try a smaller voltage step")
        finally:
            self._hydro = None

# ------------------------------------------------------------- RF helpers
def with_series(ports, Y, resistances: dict):
    """Admittances seen through series resistors at the terminals:
    Y_ext = (1 + Y R)^-1 Y, exactly, at every frequency (new in 0.20.0).
    ports, Y : from `Device.small_signal`. resistances : {port: R} in
    ohm mm (2D) or ohm (3D); ports not named have none."""
    R = np.diag([float(resistances.get(p_, 0.0)) for p_ in ports])
    Y = np.asarray(Y, complex)
    one = np.eye(len(ports))
    return np.array([np.linalg.solve(one + Yf @ R, Yf) for Yf in Y])


def two_port(ports, Y, gate: str = "gate", drain: str = "drain"):
    """The common-source two-port: rows and columns of the gate and the
    drain, every other terminal held at signal ground. Y2[f] = [[Y11, Y12],
    [Y21, Y22]]."""
    i, j = ports.index(gate), ports.index(drain)
    return np.asarray(Y, complex)[:, [i, j]][:, :, [i, j]]


def _crossing(f, g):
    """Frequency where the gain g (falling) reaches 1: log-log
    interpolation between samples, or extrapolation from the highest
    sample at -20 dB per decade (gain proportional to 1/f) when it never
    falls to 1. Returns (frequency, how)."""
    f = np.asarray(f, float)
    g = np.asarray(g, float)
    ok = (f > 0) & np.isfinite(g) & (g > 0)
    f, g = f[ok], g[ok]
    if f.size == 0:
        return math.nan, "no data"
    below = np.flatnonzero(g <= 1.0)
    if below.size and below[0] > 0:
        k = below[0]
        a, b = math.log(f[k - 1]), math.log(f[k])
        ga, gb = math.log(g[k - 1]), math.log(g[k])
        return math.exp(a + (b - a) * ga / (ga - gb)), "interpolated"
    if below.size:
        return math.nan, "below 1 at the lowest frequency"
    return float(f[-1] * g[-1]), "extrapolated at -20 dB/decade"


def rf_figures(frequencies, Y2, reference_frequency: float = None):
    """Current gain h21 = Y21 / Y11, Mason's unilateral gain
    U = |Y21 - Y12|^2 / (4 (Re Y11 Re Y22 - Re Y12 Re Y21)) (S. J. Mason,
    IRE Trans. Circuit Theory 1, 20 (1954)), and two readings of f_T and
    f_max from a common-source two-port:

    fT, fmax : where |h21| and U actually fall to 1 in the calculation.
    fT_extrapolated, fmax_extrapolated : f |h21| and f sqrt(U) at the
        reference frequency (default: the lowest frequency of the sweep
        above zero), the -20 dB/decade extrapolation used with measured
        data. In a quasi-static device the two agree; when the drain
        current lags the gate (transit through the gate-drain region), the
        gain falls faster than 1/f and the crossing comes lower.
    With no gate resistance Re Y11 is nearly zero and f_max is not
    meaningful: add the gate resistance with `with_series`.
    """
    f = np.asarray(frequencies, float)
    Y2 = np.asarray(Y2, complex)
    y11, y12, y21, y22 = Y2[:, 0, 0], Y2[:, 0, 1], Y2[:, 1, 0], Y2[:, 1, 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        h21 = y21 / y11
        den = 4 * (y11.real * y22.real - y12.real * y21.real)
        U = np.abs(y21 - y12) ** 2 / den
    ft, how_t = _crossing(f, np.abs(h21))
    fmax, how_m = _crossing(f, np.sqrt(np.where(U > 0, U, np.nan)))
    pos = np.flatnonzero(f > 0)
    if reference_frequency is None:
        k = int(pos[0]) if pos.size else None
    else:
        k = int(np.argmin(np.abs(f - reference_frequency)))
    if k is None:
        fte = fme = math.nan
    else:
        fte = float(f[k] * abs(h21[k]))
        fme = float(f[k] * math.sqrt(U[k])) if U[k] > 0 else math.nan
    return {"h21": h21, "U": U, "fT": ft, "fT_method": how_t,
            "fmax": fmax, "fmax_method": how_m,
            "fT_extrapolated": fte, "fmax_extrapolated": fme,
            "reference_frequency": float(f[k]) if k is not None else math.nan}
