"""Two- and three-dimensional device simulation: current flow, traps that
fill and empty over time, and self-heating (new in v0.19.0).

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

Limits, stated plainly: drift-diffusion (no hot-electron energy balance,
no ballistic transport, no quantum confinement of the channel), the
heterojunction is a band step between two nodes (no thermionic-emission
condition at internal interfaces), no tunneling, no impact ionization or
breakdown, no small-signal AC analysis, the heat equation is steady
state, and the grid is a tensor product (no unstructured meshes).
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
           "TransientResult", "capture_coefficient"]

_CP = QE / EPS0 * 1e9          # V nm, densities in nm^-3


# ---------------------------------------------------------------- inputs
@dataclasses.dataclass(frozen=True)
class Mobility:
    """Carrier mobility of one layer (see the module docstring).

    low_field : mu_0 at 300 K, cm^2/(V s).
    reference : where the numbers come from (required).
    saturation_velocity : v_sat, cm/s (math.inf: no velocity saturation).
    beta : shape of the saturation (>= 1; 1 gradual, 2 sharper).
    temperature_exponent : gamma in mu_0 (T/300)^(-gamma), >= 0.
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
        if not (math.isfinite(self.temperature_exponent)
                and self.temperature_exponent >= 0):
            raise ValueError("Mobility.temperature_exponent must be >= 0")

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
        if (self.y0 is None) != (self.y1 is None):
            raise ValueError("give both y0 and y1, or neither")
        if self.y0 is not None and not self.y1 > self.y0:
            raise ValueError(f"electrode {self.name!r} needs y1 > y0")


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
    _x: np.ndarray = dataclasses.field(default=None, repr=False)

    def sheet(self, depth: float = 150.0):
        """Electrons per cm^2 below the top `depth` nm, at each lateral
        position (shape of the lateral grid)."""
        z = self.axes[-1]
        m = z >= z[-1] - depth - 1e-9
        trap = getattr(np, "trapezoid", None) or getattr(np, "trapz")
        return trap(self.n[..., m], z[m], axis=-1) * 1e-7

    @property
    def power(self):
        """Total electrical power sum V I (W/mm in 2D, W in 3D)."""
        return float(sum(self.bias[k] * self.current[k] for k in self.current))


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
                 driving_force: str = "field"):
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
        names = [e.name for e in electrodes]
        if len(set(names)) != len(names):
            raise ValueError("electrode names must be different")
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

        if driving_force not in ("field", "quasi-fermi"):
            raise ValueError("driving_force must be 'field' or 'quasi-fermi'")
        self.force = driving_force
        self.layers, self.electrodes, self.traps = layers, electrodes, traps
        self.surface = surface
        self.T = float(temperature_K)
        self.kT = KB * self.T
        self.fermi = statistics == "fermi"
        self.floor = density_floor_cm3 * 1e-21
        self.dim = 2 if depth is None else 3
        self.width, self.depth = float(width), depth
        edges = np.cumsum([0.0] + [L.thickness for L in layers])
        self.top = edges[-1]
        # ---- grid
        if grid is not None:
            x = np.asarray(grid["x"], float)
            z = np.asarray(grid["z"], float)
            y = np.asarray(grid["y"], float) if self.dim == 3 else None
        else:
            xm = set()
            for e in electrodes:
                xm |= {e.x0, e.x1}
            x = _axis_grid(width, xm, dx_fine, dx_max, growth)
            zextra = [self.top - e.implant_depth for e in electrodes
                      if e.implant_depth is not None and e.face == "top"]
            zextra += [e.implant_depth for e in electrodes
                       if e.implant_depth is not None and e.face == "bottom"]
            for t in traps:
                if t.z_range is not None:
                    zextra += list(t.z_range)
            z = _z_grid(layers, zextra, dz_fine, fine_depth, dz_max, growth)
            y = None
            if self.dim == 3:
                ym = set()
                for e in electrodes:
                    if e.y0 is not None:
                        ym |= {e.y0, e.y1}
                y = _axis_grid(depth, ym, dx_fine, dx_max, growth)
        if not (abs(x[0]) < 1e-9 and abs(x[-1] - width) < 1e-9):
            raise ValueError("x must run from 0 to width")
        if not (abs(z[0]) < 1e-9 and abs(z[-1] - self.top) < 1e-9):
            raise ValueError("z must run from 0 to the top of the stack")
        for e_ in edges[1:-1]:
            if np.min(np.abs(z - e_)) > 1e-6:
                raise ValueError("z must contain every layer edge")
        self.axes = (x, z) if self.dim == 2 else (x, y, z)
        self.shape = tuple(a.size for a in self.axes)
        N = int(np.prod(self.shape))
        self.N = N
        nz = z.size
        # ---- materials per node and per z cell
        idx = np.clip(np.searchsorted(edges, z, side="right") - 1, 0, len(layers) - 1)
        zm = 0.5 * (z[1:] + z[:-1])
        cidx = np.clip(np.searchsorted(edges, zm, side="right") - 1, 0, len(layers) - 1)
        self.node_layer, self.cell_layer = idx, cidx
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
        X, Z = self.coords[0], self.coords[-1]
        Y = self.coords[1] if self.dim == 3 else None
        zi = np.broadcast_to(np.arange(nz).reshape(bshape), self.shape).ravel()
        self.zi = zi
        # ---- electrodes: nodes and implants
        self.enodes = {}
        for e in electrodes:
            row = (zi == nz - 1) if e.face == "top" else (zi == 0)
            m = row & (X >= e.x0 - 1e-9) & (X <= e.x1 + 1e-9)
            if self.dim == 3 and e.y0 is not None:
                m &= (Y >= e.y0 - 1e-9) & (Y <= e.y1 + 1e-9)
            nodes = np.flatnonzero(m)
            if nodes.size == 0:
                raise ValueError(f"electrode {e.name!r} covers no grid node")
            for other, on in self.enodes.items():
                if np.intersect1d(on, nodes).size:
                    raise ValueError(f"electrodes {other!r} and {e.name!r} overlap")
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
        self.edict = {e.name: e for e in electrodes}
        on_any = np.zeros(N, bool)
        for nodes in self.enodes.values():
            on_any[nodes] = True
        self.surface_nodes = np.flatnonzero((zi == nz - 1) & ~on_any)
        # ---- control volumes and edges
        cvs = [_cv(a) for a in self.axes]
        vol = np.ones(self.shape)
        for ax, c in enumerate(cvs):
            sh = [1] * self.dim
            sh[ax] = c.size
            vol = vol * c.reshape(sh)
        self.vol = vol.ravel()
        lat_area = np.ones(self.shape[:-1])
        for ax in range(self.dim - 1):
            sh = [1] * (self.dim - 1)
            sh[ax] = cvs[ax].size
            lat_area = lat_area * cvs[ax].reshape(sh)
        self.lat_area = np.broadcast_to(lat_area[..., None], self.shape).ravel().copy()
        I = np.arange(N).reshape(self.shape)
        hz = np.diff(z)
        eps_cell = np.array([eps[i] for i in cidx])
        wb = np.zeros(nz)
        wa = np.zeros(nz)
        wb[1:] = 0.5 * hz
        wa[:-1] = 0.5 * hz
        ei, ej, eh, earea, eeps, eaxis, ezc, ezr = [], [], [], [], [], [], [], []
        for ax in range(self.dim):
            si = [slice(None)] * self.dim
            sj = [slice(None)] * self.dim
            si[ax] = slice(0, -1)
            sj[ax] = slice(1, None)
            ii = I[tuple(si)]
            jj = I[tuple(sj)]
            coords = np.indices(ii.shape)
            area = np.ones(ii.shape)
            for b in range(self.dim):
                if b == ax:
                    continue
                sh = [1] * self.dim
                sh[b] = cvs[b].size
                area = area * cvs[b].reshape(sh)
            h = np.diff(self.axes[ax])[coords[ax]]
            zrow = coords[-1]
            if ax == self.dim - 1:
                e_eps = eps_cell[zrow]
                zc = zrow
                zr = -np.ones_like(zrow)
            else:
                eb = np.where(zrow > 0, eps_cell[np.clip(zrow - 1, 0, nz - 2)], 0.0)
                ea = np.where(zrow < nz - 1, eps_cell[np.clip(zrow, 0, nz - 2)], 0.0)
                e_eps = (wb[zrow] * eb + wa[zrow] * ea) / (wb[zrow] + wa[zrow])
                zc = -np.ones_like(zrow)
                zr = zrow
            ei.append(ii.ravel())
            ej.append(jj.ravel())
            eh.append(h.ravel())
            earea.append(area.ravel())
            eeps.append(e_eps.ravel())
            eaxis.append(np.full(ii.size, ax))
            ezc.append(zc.ravel())
            ezr.append(zr.ravel())
        self.ei = np.concatenate(ei)
        self.ej = np.concatenate(ej)
        self.eh = np.concatenate(eh)
        self.earea = np.concatenate(earea)
        self.eps_e = np.concatenate(eeps)
        self.ezc = np.concatenate(ezc)
        self.ezr = np.concatenate(ezr)
        self.wb, self.wa = wb, wa
        self.gpsi = self.eps_e * self.earea / self.eh
        # ---- mobility per edge
        self._mob_n = self._edge_mobility(mobility, "mobility")
        self._mob_p = self._edge_mobility(hole_mobility, "hole_mobility") if self.holes else None
        # ---- sheet charges
        sheet = np.zeros(nz)
        for c in interfaces:
            s = c.sigma_nm2 + extra.get(c.index, 0.0)
            sheet[int(np.argmin(np.abs(z - edges[c.index + 1])))] += s
        self.sheet = full(sheet) * self.lat_area
        if surface.kind == "states":
            fc = (surface.fixed_charge * 1e-14 if surface.fixed_charge is not None
                  else -sum(c.sigma_nm2 + extra.get(c.index, 0.0) for c in interfaces))
            self.sheet[self.surface_nodes] += fc * self.lat_area[self.surface_nodes]
        self.interfaces = list(interfaces)
        # ---- traps per node
        self._traps = []
        for t in traps:
            Nt = np.zeros(N)
            if t.where == "surface":
                sn = self.surface_nodes
                Nt[sn] = t.density * 1e-14 * self.lat_area[sn] / self.vol[sn]
            else:
                ws = (t.where,) if isinstance(t.where, str) else tuple(t.where)
                inl = np.isin(np.array(labels, dtype=object)[idx], ws)
                m = full(inl.astype(float)) > 0.5
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

    # ------------------------------------------------------------ helpers
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
        nz = self.axes[-1].size
        zc, zr = self.ezc, self.ezr
        out = {}
        for k, v in arr.items():
            cellv = v[self.cell_layer]                    # per z cell
            e = np.empty(self.ei.size)
            m = zc >= 0
            e[m] = cellv[zc[m]]
            r = zr[~m]
            b = np.where(r > 0, cellv[np.clip(r - 1, 0, nz - 2)], 0.0)
            a = np.where(r < nz - 1, cellv[np.clip(r, 0, nz - 2)], 0.0)
            if k == "saturation_velocity":
                # average the inverse (an infinite v_sat stays infinite only
                # when both sides have it)
                with np.errstate(divide="ignore"):
                    ib = np.where(r > 0, 1.0 / b, 0.0)
                    ia = np.where(r < nz - 1, 1.0 / a, 0.0)
                inv = (self.wb[r] * ib + self.wa[r] * ia) / (self.wb[r] + self.wa[r])
                with np.errstate(divide="ignore"):
                    e[~m] = np.where(inv > 0, 1.0 / np.where(inv > 0, inv, 1.0), np.inf)
            else:
                e[~m] = (self.wb[r] * b + self.wa[r] * a) / (self.wb[r] + self.wa[r])
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
        """Charge-neutral psi (at F = 0) of every ohmic-contact node."""
        out = {}
        for e in self.electrodes:
            if e.kind != "ohmic":
                continue
            nodes = self.enodes[e.name]
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
            out[e.name] = 0.5 * (lo + hi)
        return out

    # ------------------------------------------------------------- layout
    @property
    def nv(self):
        return 2 + int(self.holes) + (len(self._traps) if self._transient else 0)

    _transient = False
    _qf_margin = 0.0

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
        mu, _ = _edge_mu(self._mob_n, dfn if self.force == "quasi-fermi" else dpsi_e, h, Te)
        Dn = mu * 1e14 * kT
        w = fn / kT - lnn
        dw = w[j] - w[i]
        from .driftdiffusion import bernoulli
        Gn = -Dn / h * n[i] * bernoulli(dw) * np.expm1(np.clip(dfn / kT, -700, 700))
        if self.holes:
            dfp = fp[j] - fp[i]
            mu, _ = _edge_mu(self._mob_p, dfp if self.force == "quasi-fermi" else dpsi_e, h, Te)
            Dp = mu * 1e14 * kT
            v = fp / kT + lnp
            dv = v[j] - v[i]
            Gp = -Dp / h * p[i] * bernoulli(-dv) * np.expm1(np.clip(-dfp / kT, -700, 700))
        else:
            Gp = np.zeros_like(Gn)
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
        rQ = np.bincount(i, Gp * A, self.N) - np.bincount(j, Gp * A, self.N) + self.vol * Rp
        if dt is not None:
            _, n0, p0, _, _ = self._carriers(*self._unpack(old)[:3])
            rN += self.vol * (n - n0) / dt
            rQ += self.vol * (p - p0) / dt
        return dict(rP=rP, rN=rN, rQ=rQ, rt=rt, n=n, p=p, Ec=Ec, Gn=Gn, Gp=Gp,
                    psi=psi, fn=fn, fp=fp, focc=focc, Rn=Rn, Rp=Rp)

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
    def _local(self, x, bias, dt, old):
        """The node-local part of every residual row (no fluxes, no
        Laplacian, no boundary replacement), shape (N, nv)."""
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
            if e.kind == "schottky" and e.thermionic:
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
                out.append(np.where(w, tr["Nt"] * self.vol * focc[k] / dt
                                    - self.vol * rt[k], focc[k]))
        return np.stack(out, axis=1)

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
        u = np.clip(dfn / kT, -700, 700)
        c = np.expm1(u)
        eu = np.exp(u)
        B, dB = bern_d(dw)
        qf = self.force == "quasi-fermi"
        dpsi_e = psi[j] - psi[i]
        mu, dmu = mob(self._mob_n, dfn if qf else dpsi_e)
        K = -kT * 1e14 / h
        a = n[i]
        da = n[i] * rn_[i] / kT
        G = K * mu * a * B * c
        out["Gn"] = G
        tm = K * dmu * a * B * c            # through the mobility's argument
        out["n"] = dict(
            fi=K * (mu * da * B * c - mu * a * dB * dw_dfn[i] * c
                    - mu * a * B * eu / kT) - (tm if qf else 0.0),
            fj=K * (mu * a * dB * dw_dfn[j] * c + mu * a * B * eu / kT) + (tm if qf else 0.0),
            pi=K * (mu * da * B * c - mu * a * dB * dw_dpsi[i] * c) - (0.0 if qf else tm),
            pj=K * (mu * a * dB * dw_dpsi[j] * c) + (0.0 if qf else tm))
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
        return out

    def _jacobian(self, x, F0, bias, Tn, dt, old):
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
        J = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows),
                                                  np.concatenate(cols))),
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
        brow = np.concatenate(brow) if brow else np.zeros(0, int)
        keep = np.ones(N * nv)
        keep[brow] = 0.0
        J = sp.diags(keep) @ J
        e = np.zeros(N * nv)
        e[brow] = 1.0
        return (J + sp.diags(e)).tocsr()

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
            en = np.maximum(np.abs(D[:, 1]) - noise[:, 1], 0.0)
            big = max(big, np.max(en * n / (n + self.floor)))
            if self.holes:
                ep = np.maximum(np.abs(D[:, 2]) - noise[:, 2], 0.0)
                big = max(big, np.max(ep * p / (p + self.floor)))
            if self._transient and nv > k0:
                big = max(big, np.max(np.abs(D[:, k0:])))
            big = float(big)
            scale = min(1.0, max_step / big) if big > 0 else 1.0
            # also bound every single change (as the 1D solver does), so a
            # quasi-Fermi level where its carrier is negligible cannot jump
            # far and underflow the density
            x = x + np.clip(scale * dx, -max_step, max_step)
            # quasi-Fermi levels stay between the lowest and highest Fermi
            # level of the electrodes in a steady state (the maximum
            # principle of the continuity equations; a level outside that
            # range belongs to a carrier too sparse to matter and only stalls
            # the iteration). Not in time steps, where a level may lag.
            if old is None:
                X = x.reshape(self.N, nv)
                lev = [-float(bias.get(e.name, 0.0)) for e in self.electrodes]
                lo, hi = min(lev) - self._qf_margin, max(lev) + self._qf_margin
                X[:, 1] = np.clip(X[:, 1], lo, hi)
                if self.holes:
                    X[:, 2] = np.clip(X[:, 2], lo, hi)
            if self._transient and nv > k0:
                X = x.reshape(self.N, nv)
                X[:, k0:] = np.clip(X[:, k0:], 0.0, 1.0)
            if big < tol:
                return x, True, it
        return x, False, max_iter

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
               old_charge=None):
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
        H = H * QE * (1e6 if self.dim == 2 else 1.0)        # W/mm or W
        occ = []
        for k, tr in enumerate(self._traps):
            f = (b["focc"][k] if self._transient else self._f_steady(tr, b["n"], b["p"]))
            occ.append(np.where(tr["where"], f, np.nan).reshape(self.shape))
        sh = self.shape
        return DeviceState(
            axes=self.axes, psi=psi.reshape(sh), ec=b["Ec"].reshape(sh),
            efn=fn.reshape(sh), efp=fp.reshape(sh), n=(b["n"] * 1e21).reshape(sh),
            p=(b["p"] * 1e21).reshape(sh), temperature=Tn.reshape(sh),
            occupancy=occ, bias=dict(bias), current=cur, charge=chg,
            heat=H.reshape(sh), converged=conv, iterations=it, time=time,
            _x=x.copy())

    # ------------------------------------------------------------- public
    def equilibrium(self) -> DeviceState:
        """All electrodes at 0 V."""
        x = self._equilibrium_x()
        Tn = np.full(self.N, self.T)
        bias = {e.name: 0.0 for e in self.electrodes}
        x, conv, it = self._newton(x, bias, Tn)
        if not conv:
            raise RuntimeError("equilibrium did not converge")
        self._x0 = x.copy()
        return self._state(x, bias, Tn, conv, it)

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
                guess = x + (x - prev[1]) * (f - frac) / (frac - prev[0])
            xn, conv, it = self._newton(guess, b, Tn, tol=tol, max_iter=25)
            if not conv and guess is not x:
                xn, conv, it2 = self._newton(x, b, Tn, tol=tol, max_iter=25)
                it += it2
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
              thermal_tol: float = 0.01, max_thermal_iter: int = 30) -> DeviceState:
        """Steady state at the given voltages {electrode: V}; others at 0 V.

        guess : an earlier state of this device to start from (its bias is
            ramped to the new one).
        thermal : a `thermal.ThermalModel` to include self-heating (the
            lattice temperature and the current are iterated until the
            temperature changes by less than thermal_tol K).
        """
        target = self._check_bias(bias)
        if guess is not None:
            x, start = guess._x.copy(), dict(guess.bias)
            Tn = guess.temperature.ravel().copy()
            if x.size != self.N * self.nv:
                raise ValueError("guess is from a different device")
        else:
            st = self.equilibrium()
            x, start = st._x.copy(), dict(st.bias)
            Tn = np.full(self.N, self.T)
        if thermal is None:
            Tn = np.full(self.N, self.T)
            x, it = self._ramp(x, start, target, Tn, max_bias_step, tol)
            return self._state(x, target, Tn, True, it)
        # electrothermal: ramp at the present temperature, then iterate
        x, it = self._ramp(x, start, target, Tn, max_bias_step, tol)
        th = self._thermal_setup(thermal)
        for k in range(1, max_thermal_iter + 1):
            st = self._state(x, target, Tn, True, it)
            Tnew = self._temperature(th, st.heat, Tn)
            change = float(np.max(np.abs(Tnew - Tn)))
            Tn = Tn + 0.8 * (Tnew - Tn) if k > 3 else Tnew
            x, it = self._ramp(x, target, target, Tn, max_bias_step, tol)
            if change < thermal_tol:
                st = self._state(x, target, Tn, True, it)
                st.thermal_iterations = k
                return st
        raise RuntimeError("self-heating did not converge (the device may be "
                           "in thermal runaway, or lower the bias)")

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
                    xn, conv, it = self._newton(x, b, Tn, dt=h, old=x, tol=tol)
                    if conv:
                        st = self._state(xn, b, Tn, True, it, dt=h, old=x, time=t + h,
                                         old_charge=chg)
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
                    li = self.cell_layer[min(int(np.searchsorted(z + tot, c) - 1), z.size - 2)]
                    mats.append(model.material_for(self.layers[li].label))
        else:
            zfull, zmap = z.copy(), np.arange(z.size)
            mats = [model.material_for(self.layers[li].label) for li in self.cell_layer]
        return dict(model=model, lat=lat, maps=maps, z=zfull, zmap=zmap, mats=mats)

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
