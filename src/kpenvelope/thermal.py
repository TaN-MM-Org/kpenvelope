"""Heat flow for self-heating: the steady heat equation with a thermal
conductivity that falls as the lattice warms up (new in v0.19.0).

    div (k(T) grad T) + H = 0,      k(T) = k_300 (T / 300 K)^(-alpha),

on a tensor-product grid (two or three dimensions, z last and growing
upward, as in the rest of the package). The bottom row is held at the
heat-sink temperature; every other face is adiabatic (no heat crosses
it). An adiabatic side wall is a mirror plane: a device whose sides are
adiabatic behaves like one finger of a row of identical fingers placed
side by side, which is how multi-finger transistors are laid out. Give
`lateral_extension` to widen the thermal domain beyond the electrical
one (half the finger pitch minus half the device width, for example).

Materials are layered (uniform along x and y), so each cell between two
z rows has one material; a face along x or y on a z row takes the
height-weighted average of the half cells below and above it (the exact
conductance of two layers side by side), and k is evaluated at the mean
temperature of the face's two nodes. The nonlinear problem is solved by
fixed-point iteration on k(T) with a sparse direct solve at each step.

Checks in the tests: a slab with k(T) of power law matches the exact
Kirchhoff solution, int_{T_sink}^{T} k dT' = q z; a strip heater on a slab
with constant k matches its Fourier-series solution; the heat leaving
through the sink equals the heat put in.

A thermal boundary resistance R (m^2 K/W, for example between a GaN
buffer and its substrate) is represented by `boundary_resistance`, a thin
layer of thickness d and conductivity d / R: in steady state it has
exactly the resistance R per area.

Material numbers shipped (each from a source that was opened and read):

* GAN_BULK: k_300 = 230 W/(m K), the highest room-temperature value of
  low-dislocation bulk GaN measured by C. Mion, J. F. Muth, E. A. Preble
  and D. Hanser (Appl. Phys. Lett. 89, 092123 (2006), doi:10.1063/
  1.2335972; the 230 W/(m K)
  figure is stated in the abstracts of that work and of their MRS paper),
  and alpha = 1.43, the dependence over 300-450 K from the same
  measurements as quoted by B. A. Danilchenko et al., Solid State Commun.
  (2007), doi:10.1016/j.ssc.2007.08.007 (arXiv:0705.4187). Thin GaN
  layers with many dislocations conduct less; give your own number if you
  have it.
* SIC_4H_ROOM: 4H-SiC, 3.7 W/(cm K) = 370 W/(m K) (Goldberg et al. 2001,
  as listed by the Ioffe Institute database). That source gives no
  temperature dependence for 4H-SiC, so alpha = 0 (held constant), which
  underestimates the heating in a hot substrate.
"""
from __future__ import annotations

import dataclasses
import math

import numpy as np

__all__ = ["ThermalMaterial", "ThermalLayer", "ThermalModel", "GAN_BULK",
           "SIC_4H_ROOM", "boundary_resistance", "solve_heat",
           "solve_heat_transient"]


@dataclasses.dataclass(frozen=True)
class ThermalMaterial:
    """k(T) = k300 (T / 300 K)^(-exponent), in W/(m K), with its source.

    For heat that changes in time (new in 0.20.0) also: density (g/cm^3)
    and the specific heat c_p = cp_a + cp_b T (J/(g K)); None where not
    known (then only steady heat flow is possible).
    """

    name: str
    k300: float
    exponent: float
    reference: str
    density: float = None
    cp_a: float = None
    cp_b: float = 0.0

    def __post_init__(self):
        if not (math.isfinite(self.k300) and self.k300 > 0):
            raise ValueError("ThermalMaterial.k300 must be > 0 (W/m/K)")
        if not math.isfinite(self.exponent):
            raise ValueError("ThermalMaterial.exponent must be finite")
        if not (isinstance(self.reference, str) and self.reference.strip()):
            raise ValueError("ThermalMaterial needs a reference for its numbers")
        if (self.density is None) != (self.cp_a is None):
            raise ValueError("give both density and cp_a, or neither")
        if self.density is not None and not (self.density > 0 and self.cp_a > 0
                                             and math.isfinite(self.cp_b)):
            raise ValueError("density and cp_a must be > 0")

    def k(self, T):
        return self.k300 * (np.asarray(T, float) / 300.0) ** (-self.exponent)

    @property
    def has_capacity(self):
        return self.density is not None

    def volumetric_heat(self, T):
        """rho c_p in J/(m^3 K)."""
        if not self.has_capacity:
            raise ValueError(f"{self.name} has no density and specific heat; "
                             "heat that changes in time needs them")
        return self.density * 1e6 * (self.cp_a + self.cp_b * np.asarray(T, float))


#: molar mass of GaN, g/mol (standard atomic weights: Ga 69.723, N 14.007)
GAN_MOLAR_MASS = 69.723 + 14.007

GAN_BULK = ThermalMaterial(
    "GaN (bulk, low dislocation density)", 230.0, 1.43,
    "k(300 K) = 230 W/(m K): C. Mion, J. F. Muth, E. A. Preble, D. Hanser, "
    "Appl. Phys. Lett. 89, 092123 (2006), doi:10.1063/1.2335972; T^-1.43 over "
    "300-450 K from the "
    "same work as quoted by B. A. Danilchenko et al., Solid State Commun. "
    "(2007), doi:10.1016/j.ssc.2007.08.007; density 6.15 g/cm^3 and "
    "c_p = 38.1 + 8.96e-3 T J/(mol K) (Barin et al. 1977, 298-1773 K), both "
    "as listed by the Ioffe Institute database",
    density=6.15, cp_a=38.1 / GAN_MOLAR_MASS, cp_b=8.96e-3 / GAN_MOLAR_MASS)

SIC_4H_ROOM = ThermalMaterial(
    "4H-SiC (room-temperature value, held constant)", 370.0, 0.0,
    "3.7 W/(cm K): Goldberg et al. (2001), as listed by the Ioffe Institute "
    "database; no temperature dependence given there for 4H-SiC. Density "
    "3.211 g/cm^3 (Gomes de Mesquita 1967, via Ioffe); specific heat "
    "0.69 J/(g K), the value Ioffe lists for 6H-SiC (Goldberg et al. 2001), "
    "used here for 4H-SiC because none is listed for it",
    density=3.211, cp_a=0.69, cp_b=0.0)


@dataclasses.dataclass(frozen=True)
class ThermalLayer:
    """A layer below the device stack that carries heat only (substrate,
    nucleation layer, boundary resistance). thickness in nm."""

    label: str
    thickness: float
    material: ThermalMaterial

    def __post_init__(self):
        if not (math.isfinite(self.thickness) and self.thickness > 0):
            raise ValueError("ThermalLayer.thickness must be > 0 (nm)")
        if not isinstance(self.material, ThermalMaterial):
            raise ValueError("ThermalLayer.material must be a ThermalMaterial")


def boundary_resistance(resistance_m2K_W: float, reference: str,
                        thickness_nm: float = 10.0,
                        label: str = "boundary resistance") -> ThermalLayer:
    """A thin layer with exactly the thermal resistance R (m^2 K / W) per
    area: thickness d and k = d / R, constant."""
    if not (math.isfinite(resistance_m2K_W) and resistance_m2K_W > 0):
        raise ValueError("the boundary resistance must be > 0 (m^2 K/W)")
    k = thickness_nm * 1e-9 / resistance_m2K_W
    return ThermalLayer(label, thickness_nm, ThermalMaterial(
        label, k, 0.0, f"R = {resistance_m2K_W:g} m^2 K/W: {reference}"))


@dataclasses.dataclass(frozen=True)
class ThermalModel:
    """What self-heating needs besides the device itself.

    device_materials : a ThermalMaterial for every device layer (one for
        all, or a dict keyed by layer label).
    substrate : ThermalLayer objects below the device stack, bottom first.
    sink_temperature_K : temperature of the bottom face.
    lateral_extension : extra width (nm) added on each side of the device
        for heat spreading (adiabatic beyond that; a mirror plane).
    dz_max, growth : grid of the substrate and extensions (nm, ratio).
    """

    device_materials: object
    substrate: tuple = ()
    sink_temperature_K: float = 300.0
    lateral_extension: float = 0.0
    dz_max: float = 5000.0
    growth: float = 1.3

    def __post_init__(self):
        if not (math.isfinite(self.sink_temperature_K) and self.sink_temperature_K > 0):
            raise ValueError("sink_temperature_K must be > 0")
        if not (math.isfinite(self.lateral_extension) and self.lateral_extension >= 0):
            raise ValueError("lateral_extension must be >= 0 (nm)")
        for L in self.substrate:
            if not isinstance(L, ThermalLayer):
                raise ValueError("substrate must be a sequence of ThermalLayer")
        if not (self.growth > 1.0 and self.dz_max > 0):
            raise ValueError("growth must be > 1 and dz_max > 0")

    def material_for(self, label):
        dm = self.device_materials
        if isinstance(dm, ThermalMaterial):
            return dm
        if isinstance(dm, dict) and label in dm and isinstance(dm[label], ThermalMaterial):
            return dm[label]
        raise ValueError(f"no thermal material for the device layer {label!r}")


def _cv(nodes):
    h = np.diff(nodes)
    v = np.zeros(nodes.size)
    v[:-1] += 0.5 * h
    v[1:] += 0.5 * h
    return v


class _HeatGrid:
    """Conductances and heat capacities of a layered tensor grid (SI)."""

    def __init__(self, lateral, z, cell_materials):
        lat = [np.asarray(a, float) * 1e-9 for a in lateral]
        zz = np.asarray(z, float) * 1e-9
        shape = tuple(a.size for a in lat) + (zz.size,)
        self.shape = shape
        self.N = int(np.prod(shape))
        if len(cell_materials) != zz.size - 1:
            raise ValueError("one material per z cell is needed")
        I = np.arange(self.N).reshape(shape)
        hz = np.diff(zz)
        cvs = [_cv(a) for a in lat] + [_cv(zz)]
        nd = len(shape)
        self.edges = []
        for ax in range(nd):
            sl_i = [slice(None)] * nd
            sl_j = [slice(None)] * nd
            sl_i[ax] = slice(0, -1)
            sl_j[ax] = slice(1, None)
            ii = I[tuple(sl_i)].ravel()
            jj = I[tuple(sl_j)].ravel()
            area = np.ones(I[tuple(sl_i)].shape)
            for b in range(nd):
                if b == ax:
                    continue
                sh = [1] * nd
                sh[b] = cvs[b].size
                area = area * cvs[b].reshape(sh)
            coords = np.indices(I[tuple(sl_i)].shape)
            if ax == nd - 1:
                hh = hz[coords[ax]].ravel()
                self.edges.append((ii, jj, area.ravel() / hh, "z", coords[ax].ravel()))
            else:
                h = np.diff(lat[ax])
                hh = h[coords[ax]].ravel()
                self.edges.append((ii, jj, area.ravel() / hh, "l", coords[nd - 1].ravel()))
        self.mats = list(cell_materials)
        self.kz = len(self.mats)
        self.k300 = np.array([m.k300 for m in self.mats])
        self.alp = np.array([m.exponent for m in self.mats])
        self.wb = np.zeros(zz.size)
        self.wa = np.zeros(zz.size)
        self.wb[1:] = 0.5 * hz
        self.wa[:-1] = 0.5 * hz
        # lateral control volume of every node (product of lateral widths)
        latv = np.ones(shape[:-1])
        for ax in range(nd - 1):
            sh = [1] * (nd - 1)
            sh[ax] = cvs[ax].size
            latv = latv * cvs[ax].reshape(sh)
        self.latv = np.broadcast_to(latv[..., None], shape).ravel()
        self.zrow = np.broadcast_to(np.arange(zz.size).reshape((1,) * (nd - 1) + (-1,)),
                                    shape).ravel()
        sink = np.zeros(shape, bool)
        sink[..., 0] = True
        self.sink = sink.ravel()

    def kcell(self, c, T):
        return self.k300[c] * (T / 300.0) ** (-self.alp[c])

    def matrix(self, T):
        import scipy.sparse as sp
        Tf = T.ravel()
        rows, cols, vals = [], [], []
        diag = np.zeros(self.N)
        for ii, jj, g, kind, idx in self.edges:
            Tm = 0.5 * (Tf[ii] + Tf[jj])
            if kind == "z":
                k = self.kcell(idx, Tm)
            else:
                below = np.clip(idx - 1, 0, self.kz - 1)
                above = np.clip(idx, 0, self.kz - 1)
                k = (self.wb[idx] * self.kcell(below, Tm) + self.wa[idx]
                     * self.kcell(above, Tm)) / (self.wb[idx] + self.wa[idx])
            G = g * k
            rows += [ii, jj]
            cols += [jj, ii]
            vals += [-G, -G]
            np.add.at(diag, ii, G)
            np.add.at(diag, jj, G)
        A = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows),
                                                  np.concatenate(cols))),
                          shape=(self.N, self.N))
        return A + sp.diags(diag)

    def capacity(self, T):
        """Heat capacity of every node's control volume, J/K (per meter of
        depth in 2D): the half cells below and above, each with its own
        rho c_p at the node temperature."""
        Tf = T.ravel()
        r = self.zrow
        below = np.clip(r - 1, 0, self.kz - 1)
        above = np.clip(r, 0, self.kz - 1)
        cb = np.zeros(self.N)
        ca = np.zeros(self.N)
        for c, m in enumerate(self.mats):
            mb = below == c
            ma = above == c
            if mb.any():
                cb[mb] = m.volumetric_heat(Tf[mb])
            if ma.any():
                ca[ma] = m.volumetric_heat(Tf[ma])
        return self.latv * (self.wb[r] * cb + self.wa[r] * ca)

    def solve(self, A, b, Ts):
        import scipy.sparse.linalg as spl
        free = ~self.sink
        Af = A[free][:, free]
        rhs = b[free] - A[free][:, self.sink] @ np.full(int(self.sink.sum()), Ts)
        Tn = np.full(self.N, float(Ts))
        Tn[free] = spl.spsolve(Af.tocsc(), rhs)
        return Tn


def solve_heat(lateral, z, cell_materials, heat, sink_temperature_K: float,
               T0=None, tol: float = 1e-7, max_iter: int = 200):
    """Steady temperature on a tensor grid.

    lateral : list of node arrays for the lateral axes (one for 2D, two
        for 3D), in nm. z : node array (nm), bottom first.
    cell_materials : ThermalMaterial of each z cell (len(z) - 1).
    heat : heat put into each node, shape (*lateral sizes, len(z)), in W
        per meter of depth for 2D and W for 3D.
    Returns T (K), same shape as heat, and the number of iterations.
    """
    g = _HeatGrid(lateral, z, cell_materials)
    H = np.asarray(heat, float).reshape(g.shape)
    if not np.all(np.isfinite(H)):
        raise ValueError("heat must be finite")
    Ts = float(sink_temperature_K)
    T = np.full(g.N, Ts) if T0 is None else np.array(T0, float).ravel()
    b = H.ravel()
    for it in range(1, max_iter + 1):
        Tn = g.solve(g.matrix(T), b, Ts)
        change = float(np.max(np.abs(Tn - T)))
        T = Tn
        if change < tol:
            return T.reshape(g.shape), it
    raise RuntimeError("the heat equation did not converge")


def solve_heat_transient(lateral, z, cell_materials, heat, sink_temperature_K: float,
                         times, T0=None, tol: float = 1e-7, max_iter: int = 100):
    """Temperature in time (new in 0.20.0): C dT/dt = div(k grad T) + H by
    backward Euler at the given times (s, increasing, the first step from
    t = 0), the heat sink at the bottom, adiabatic elsewhere. heat is an
    array (held constant) or a function heat(t, T) returning one. The
    capacities and conductivities are taken at the new temperature
    (iterated within each step). Returns (times, list of T arrays)."""
    g = _HeatGrid(lateral, z, cell_materials)
    Ts = float(sink_temperature_K)
    T = np.full(g.N, Ts) if T0 is None else np.array(T0, float).ravel()
    times = np.asarray(times, float)
    if times.ndim != 1 or np.any(np.diff(times) <= 0) or times[0] <= 0:
        raise ValueError("times must be positive and increasing")
    out = []
    t_prev = 0.0
    for t in times:
        dt = t - t_prev
        Told = T.copy()
        Tk = T.copy()
        for _ in range(max_iter):
            Hk = heat(t, Tk.reshape(g.shape)) if callable(heat) else heat
            Hk = np.asarray(Hk, float).ravel()
            C = g.capacity(Tk)
            import scipy.sparse as sp
            A = g.matrix(Tk) + sp.diags(C / dt)
            Tn = g.solve(A, Hk + C / dt * Told, Ts)
            change = float(np.max(np.abs(Tn - Tk)))
            Tk = Tn
            if change < tol:
                break
        else:
            raise RuntimeError("a heat time step did not converge")
        T = Tk
        out.append(T.reshape(g.shape).copy())
        t_prev = t
    return times, out
