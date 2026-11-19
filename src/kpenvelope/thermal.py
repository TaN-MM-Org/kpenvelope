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
           "SIC_4H_ROOM", "boundary_resistance", "solve_heat"]


@dataclasses.dataclass(frozen=True)
class ThermalMaterial:
    """k(T) = k300 (T / 300 K)^(-exponent), in W/(m K), with its source."""

    name: str
    k300: float
    exponent: float
    reference: str

    def __post_init__(self):
        if not (math.isfinite(self.k300) and self.k300 > 0):
            raise ValueError("ThermalMaterial.k300 must be > 0 (W/m/K)")
        if not math.isfinite(self.exponent):
            raise ValueError("ThermalMaterial.exponent must be finite")
        if not (isinstance(self.reference, str) and self.reference.strip()):
            raise ValueError("ThermalMaterial needs a reference for its numbers")

    def k(self, T):
        return self.k300 * (np.asarray(T, float) / 300.0) ** (-self.exponent)


GAN_BULK = ThermalMaterial(
    "GaN (bulk, low dislocation density)", 230.0, 1.43,
    "k(300 K) = 230 W/(m K): C. Mion, J. F. Muth, E. A. Preble, D. Hanser, "
    "Appl. Phys. Lett. 89, 092123 (2006), doi:10.1063/1.2335972; T^-1.43 over "
    "300-450 K from the "
    "same work as quoted by B. A. Danilchenko et al., Solid State Commun. "
    "(2007), doi:10.1016/j.ssc.2007.08.007")

SIC_4H_ROOM = ThermalMaterial(
    "4H-SiC (room-temperature value, held constant)", 370.0, 0.0,
    "3.7 W/(cm K): Goldberg et al. (2001), as listed by the Ioffe Institute "
    "database; no temperature dependence given there for 4H-SiC")


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
    import scipy.sparse as sp
    import scipy.sparse.linalg as spl

    lat = [np.asarray(a, float) * 1e-9 for a in lateral]
    zz = np.asarray(z, float) * 1e-9
    shape = tuple(a.size for a in lat) + (zz.size,)
    N = int(np.prod(shape))
    H = np.asarray(heat, float).reshape(shape)
    if not np.all(np.isfinite(H)):
        raise ValueError("heat must be finite")
    if len(cell_materials) != zz.size - 1:
        raise ValueError("one material per z cell is needed")
    I = np.arange(N).reshape(shape)
    hz = np.diff(zz)
    cvs = [_cv(a) for a in lat] + [_cv(zz)]
    nd = len(shape)
    # edges: (i, j, area / length, kind, z-cell or z-row)
    edges = []
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
            cvb = cvs[b] if b != ax else None
            sh[b] = cvb.size
            area = area * cvb.reshape(sh)
        coords = np.indices(I[tuple(sl_i)].shape)
        if ax == nd - 1:
            hh = hz[coords[ax]].ravel()
            zc = coords[ax].ravel()          # z cell index
            edges.append((ii, jj, (area.ravel() / hh), "z", zc))
        else:
            h = np.diff(lat[ax])
            hh = h[coords[ax]].ravel()
            zr = coords[nd - 1].ravel()      # z row index
            edges.append((ii, jj, (area.ravel() / hh), "l", zr))
    mats = list(cell_materials)
    kz = len(mats)
    k300 = np.array([m.k300 for m in mats])
    alp = np.array([m.exponent for m in mats])
    # half-cell weights of each z row (below, above)
    wb = np.zeros(zz.size)
    wa = np.zeros(zz.size)
    wb[1:] = 0.5 * hz
    wa[:-1] = 0.5 * hz

    def kcell(c, T):
        return k300[c] * (T / 300.0) ** (-alp[c])

    def conductances(T):
        Tf = T.ravel()
        out = []
        for ii, jj, g, kind, idx in edges:
            Tm = 0.5 * (Tf[ii] + Tf[jj])
            if kind == "z":
                k = kcell(idx, Tm)
            else:
                below = np.clip(idx - 1, 0, kz - 1)
                above = np.clip(idx, 0, kz - 1)
                k = (wb[idx] * kcell(below, Tm) + wa[idx] * kcell(above, Tm)) / (
                    wb[idx] + wa[idx])
            out.append((ii, jj, g * k))
        return out

    Ts = float(sink_temperature_K)
    T = np.full(shape, Ts) if T0 is None else np.array(T0, float).reshape(shape)
    sink = np.zeros(shape, bool)
    sink[..., 0] = True
    sinkf = sink.ravel()
    free = ~sinkf
    b = H.ravel().copy()
    it = 0
    for it in range(1, max_iter + 1):
        rows, cols, vals = [], [], []
        diag = np.zeros(N)
        for ii, jj, G in conductances(T):
            rows += [ii, jj]
            cols += [jj, ii]
            vals += [-G, -G]
            np.add.at(diag, ii, G)
            np.add.at(diag, jj, G)
        A = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows),
                                                  np.concatenate(cols))), shape=(N, N))
        A = A + sp.diags(diag)
        # Dirichlet at the sink: move known values to the right side
        Af = A[free][:, free]
        rhs = b[free] - A[free][:, sinkf] @ np.full(int(sinkf.sum()), Ts)
        Tn = np.full(N, Ts)
        Tn[free] = spl.spsolve(Af.tocsc(), rhs)
        change = float(np.max(np.abs(Tn - T.ravel())))
        T = Tn.reshape(shape)
        if change < tol:
            break
    else:
        raise RuntimeError("the heat equation did not converge")
    return T, it
