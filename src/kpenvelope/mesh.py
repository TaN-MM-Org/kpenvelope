"""Devices on unstructured triangular meshes (new in 0.21.0): curved and
slanted shapes in two dimensions.

`Mesh` holds the nodes (x, z in nm), the triangles and the region label of
every triangle. `MeshDevice` discretizes the same drift-diffusion
equations as `Device` with the box (Voronoi) finite-volume method: each
triangle gives every one of its edges the part of the dual face that runs
from the edge's midpoint to the triangle's circumcentre, and every node the
matching part of its dual cell. A triangle with an obtuse angle gives a
negative part to the edge across from it; on a Delaunay mesh the
neighbouring triangle makes it up on an interior edge, but not on the
boundary (the constructor warns where an edge's total is negative). A rectangular
grid split into right triangles gives exactly the coefficients of the
tensor-grid `Device`, which the tests use as a check.

Supported: semiconductor regions with their material, doping and mobility,
dielectric regions, electrodes on any set of nodes (ohmic, Schottky,
series resistance), deep-level traps per region, generation, impact
ionization, band-to-band tunneling, thermionic emission at band steps,
the hydrodynamic model, small-signal analysis and transients. Not on
meshes: self-heating, `TunnelBarrier`, the field-line tools
(`ionization_integral`, `breakdown_voltage`, `field_check`), and interface
polarization charges, which you give as sheets on nodes yourself
(`sheets=`).
"""
from __future__ import annotations

import dataclasses
import math
import warnings

import numpy as np

from .device import (Device, Dielectric, Electrode, Mobility, Surface, Trap, _CP,
                     _edge_mu)
from .electrons import HBAR, KB, M0
from .interfaces import StackLayer
from .nitride import QE, NitrideMaterial

__all__ = ["Mesh", "MeshRegion", "MeshDevice", "mesh_from_points", "mesh_from_grid"]


@dataclasses.dataclass
class Mesh:
    """points (N, 2) as (x, z) in nm; triangles (M, 3) node indices;
    regions (M,) a label per triangle."""

    points: np.ndarray
    triangles: np.ndarray
    regions: np.ndarray

    def __post_init__(self):
        self.points = np.asarray(self.points, float)
        self.triangles = np.asarray(self.triangles, int)
        self.regions = np.asarray(self.regions, dtype=object)
        if self.points.ndim != 2 or self.points.shape[1] != 2:
            raise ValueError("points must be (N, 2)")
        if self.triangles.ndim != 2 or self.triangles.shape[1] != 3:
            raise ValueError("triangles must be (M, 3)")
        if self.regions.shape != (self.triangles.shape[0],):
            raise ValueError("one region label per triangle")
        used = np.unique(self.triangles)
        if used.size != self.points.shape[0]:
            raise ValueError("every point must belong to a triangle")


def mesh_from_points(points, region_of):
    """Delaunay triangulation of the points; region_of(x, z) gives the label
    of the triangle with that centroid, or None to leave it out (a hole or
    the outside of a concave shape). Points no triangle uses are dropped."""
    from scipy.spatial import Delaunay
    pts = np.asarray(points, float)
    tri = Delaunay(pts).simplices
    cen = pts[tri].mean(axis=1)
    lab = [region_of(float(c[0]), float(c[1])) for c in cen]
    keep = np.array([v is not None for v in lab])
    tri = tri[keep]
    lab = np.array([v for v in lab if v is not None], dtype=object)
    used = np.unique(tri)
    remap = -np.ones(pts.shape[0], int)
    remap[used] = np.arange(used.size)
    return Mesh(pts[used], remap[tri], lab)


def mesh_from_grid(x, z, region_of):
    """A rectangular grid (node arrays x, z in nm) with every rectangle cut
    into two right triangles; region_of(x, z) at the rectangle's centre."""
    x = np.asarray(x, float)
    z = np.asarray(z, float)
    X, Z = np.meshgrid(x, z, indexing="ij")
    pts = np.column_stack([X.ravel(), Z.ravel()])
    nz = z.size
    tris, labs = [], []
    for i in range(x.size - 1):
        for k in range(nz - 1):
            a, b = i * nz + k, (i + 1) * nz + k
            c, d = (i + 1) * nz + k + 1, i * nz + k + 1
            lab = region_of(0.5 * (x[i] + x[i + 1]), 0.5 * (z[k] + z[k + 1]))
            if lab is None:
                continue
            tris += [[a, b, c], [a, c, d]]
            labs += [lab, lab]
    tris = np.array(tris, int)
    used = np.unique(tris)
    remap = -np.ones(pts.shape[0], int)
    remap[used] = np.arange(used.size)
    return Mesh(pts[used], remap[tris], np.array(labs, dtype=object))


@dataclasses.dataclass(frozen=True)
class MeshRegion:
    """What a region of the mesh is made of: a semiconductor (material, with
    donors / acceptors in nm^-3 and their depths, as in `StackLayer`, and a
    `Mobility`), or a dielectric (eps_r with its reference)."""

    material: NitrideMaterial = None
    mobility: Mobility = None
    donors: float = 0.0
    acceptors: float = 0.0
    donor_depth: float = None
    acceptor_depth: float = None
    hole_mobility: Mobility = None
    dielectric: Dielectric = None

    def __post_init__(self):
        if (self.material is None) == (self.dielectric is None):
            raise ValueError("a MeshRegion is either a semiconductor (material) or a "
                             "dielectric")
        if self.material is not None and not isinstance(self.mobility, Mobility):
            raise ValueError("a semiconductor MeshRegion needs a Mobility")

    @property
    def semiconductor(self):
        return self.material is not None


class MeshDevice(Device):
    """A two-dimensional device on a `Mesh` (see the module docstring).

    regions : {label: MeshRegion}. electrodes : `Electrode` objects (their
    x0, x1 and face are not used here) and contacts : {name: node indices}
    the metal covers. sheets : fixed sheet charges (an interface
    polarization charge, for example), each (node indices, e/cm^2, the
    length of sheet each node stands for in nm, reference). The rest as
    in `Device`.
    """

    def __init__(self, mesh: Mesh, regions: dict, electrodes, contacts: dict,
                 temperature_K: float = 300.0, hole_mass=None, statistics: str = "fermi",
                 traps=(), sheets=(), generation=None, ionization=None,
                 heterojunctions: str = "drift-diffusion", band_to_band=None,
                 doping=None, density_floor_cm3: float = 1.0,
                 driving_force: str = "field"):
        try:
            import scipy.sparse  # noqa: F401
        except ImportError as e:      # pragma: no cover
            raise ImportError("MeshDevice needs SciPy") from e
        if not isinstance(mesh, Mesh):
            raise ValueError("mesh must be a Mesh")
        for lab in np.unique(mesh.regions):
            if lab not in regions or not isinstance(regions[lab], MeshRegion):
                raise ValueError(f"no MeshRegion for region {lab!r}")
        if statistics not in ("fermi", "boltzmann"):
            raise ValueError("statistics must be 'fermi' or 'boltzmann'")
        if driving_force not in ("field", "quasi-fermi"):
            raise ValueError("driving_force must be 'field' or 'quasi-fermi'")
        self.mesh = mesh
        self.dim = 2
        P = mesh.points
        N = P.shape[0]
        self.N = N
        self.shape = (N,)
        self.coords = [P[:, 0].copy(), P[:, 1].copy()]
        self.axes = (P[:, 0].copy(), P[:, 1].copy())
        self.T = float(temperature_K)
        self.kT = KB * self.T
        self.fermi = statistics == "fermi"
        self.force = driving_force
        self.floor = density_floor_cm3 * 1e-21
        self.holes = hole_mass is not None
        self.top = float(P[:, 1].max())
        self.layers = []
        self.cap, self.regions = (), ()
        self.surface = Surface("states", fixed_charge=0.0)
        self.interfaces = []
        labels = sorted(set(mesh.regions.tolist()))
        self._region_labels = labels
        semi_t = np.array([regions[r].semiconductor for r in mesh.regions])
        eps_t = np.array([(regions[r].material.eps_r if regions[r].semiconductor
                           else regions[r].dielectric.eps_r) for r in mesh.regions], float)
        # ---- box method, triangle by triangle
        T_ = mesh.triangles
        A, B, C = P[T_[:, 0]], P[T_[:, 1]], P[T_[:, 2]]
        cc = _circumcentres(A, B, C)
        edges_t = []                   # (n1, n2, triangle, signed dual length)
        sub_t = []                     # (node, triangle, dual-cell area)
        for (u, v, w) in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
            pu, pv, pw = P[T_[:, u]], P[T_[:, v]], P[T_[:, w]]
            mid = 0.5 * (pu + pv)
            ev = pv - pu
            h = np.hypot(ev[:, 0], ev[:, 1])
            nrm = np.column_stack([-ev[:, 1], ev[:, 0]]) / h[:, None]
            # signed distance from the edge midpoint to the circumcentre,
            # positive on the side of the third vertex
            side = np.sign(np.sum((pw - mid) * nrm, axis=1))
            d = np.sum((cc - mid) * nrm, axis=1) * side
            edges_t.append((T_[:, u], T_[:, v], np.arange(T_.shape[0]), d, h))
            for nd in (T_[:, u], T_[:, v]):
                sub_t.append((nd, np.arange(T_.shape[0]), 0.25 * h * d))
        n1 = np.concatenate([e[0] for e in edges_t])
        n2 = np.concatenate([e[1] for e in edges_t])
        tt = np.concatenate([e[2] for e in edges_t])
        dd = np.concatenate([e[3] for e in edges_t])
        hh = np.concatenate([e[4] for e in edges_t])
        lo, hi = np.minimum(n1, n2), np.maximum(n1, n2)
        key = lo * N + hi
        uk, inv = np.unique(key, return_inverse=True)
        nE = uk.size
        self.ei, self.ej = (uk // N).astype(int), (uk % N).astype(int)
        h_e = np.zeros(nE)
        h_e[inv] = hh
        self.eh = h_e
        self.gpsi = np.bincount(inv, eps_t[tt] * dd, nE) / h_e
        self.earea = np.bincount(inv, np.where(semi_t[tt], dd, 0.0), nE)
        if np.any(self.earea < -1e-9 * np.abs(self.earea).max()) or np.any(self.gpsi < 0):
            warnings.warn("some edges have a negative dual face (obtuse triangles at a "
                          "boundary); refine the mesh there", RuntimeWarning, stacklevel=2)
        sn = np.concatenate([s_[0] for s_ in sub_t])
        st_ = np.concatenate([s_[1] for s_ in sub_t])
        sa = np.concatenate([s_[2] for s_ in sub_t])
        self.vol = np.bincount(sn, np.where(semi_t[st_], sa, 0.0), N)
        self.vol_all = np.bincount(sn, sa, N)
        self.node_semi = self.vol > 0
        self.eaxis = np.zeros(nE, int)
        # ---- node material: the semiconductor region with the largest share
        # of the node's dual cell (ties go to the region higher up in z)
        share = {}
        for r in labels:
            if not regions[r].semiconductor:
                continue
            m = mesh.regions[st_] == r
            share[r] = np.bincount(sn, np.where(m, sa, 0.0), N)
        cz = np.array([P[T_[mesh.regions == r]][:, :, 1].mean() if regions[r].semiconductor
                       else -np.inf for r in labels])
        semi_labels = [r for r in labels if regions[r].semiconductor]
        if not semi_labels:
            raise ValueError("the mesh has no semiconductor region")
        best = np.full(N, semi_labels[0], dtype=object)
        bestv = np.full(N, -np.inf)
        order = sorted(semi_labels, key=lambda r: cz[labels.index(r)])
        for r in order:                       # later (higher) wins ties
            v = share[r]
            better = v >= bestv - 1e-12 * np.maximum(1.0, np.abs(bestv))
            best = np.where(better & (v > 0), r, best)
            bestv = np.where(better & (v > 0), v, bestv)
        self.node_region = best
        mats = {r: regions[r] for r in semi_labels}

        def nodewise(f):
            return np.array([f(mats[r]) for r in best], float)
        self.Ec0 = nodewise(lambda R: R.material.ec)
        self.Eg = nodewise(lambda R: R.material.eg)
        self.me_node = nodewise(lambda R: R.material.me)
        pref = (KB * self.T * QE / (2 * np.pi * HBAR ** 2)) ** 1.5 * 2.0 * 1e-27
        self.Nc = pref * (self.me_node * M0) ** 1.5
        if self.holes:
            self.Nv = pref * (np.full(N, float(hole_mass)) * M0) ** 1.5
        else:
            self.Nv = np.zeros(N)
        self.ND = nodewise(lambda R: R.donors)
        self.NA = nodewise(lambda R: R.acceptors)
        self.Ed = nodewise(lambda R: np.nan if R.donor_depth is None else R.donor_depth)
        self.Ea = nodewise(lambda R: np.nan if R.acceptor_depth is None else R.acceptor_depth)
        if np.any(self.NA > 0) and not self.holes:
            raise ValueError("a region has acceptors: give hole_mass")
        if doping is not None:
            vals = np.array([doping(*pt) for pt in zip(*self.coords)], float)
            if vals.shape != (N, 2) or not (np.all(np.isfinite(vals)) and np.all(vals >= 0)):
                raise ValueError("doping must give (N_d, N_a) >= 0 in cm^-3 at every node")
            add_d, add_a = vals[:, 0] * 1e-21, vals[:, 1] * 1e-21
            if np.any(add_a > 0) and not self.holes:
                raise ValueError("a doping profile with acceptors needs hole_mass")
            self.ND = self.ND + add_d
            self.NA = self.NA + add_a
            self.Ed = np.where(add_d > 0, np.nan, self.Ed)
            self.Ea = np.where(add_a > 0, np.nan, self.Ea)
        self.ND = np.where(self.node_semi, self.ND, 0.0)
        self.NA = np.where(self.node_semi, self.NA, 0.0)
        # ---- mobility per edge, weighted by each triangle's semiconductor face
        self._mob_n = self._mesh_mobility(regions, inv, tt, dd, semi_t, "mobility")
        self._mob_p = (self._mesh_mobility(regions, inv, tt, dd, semi_t, "hole_mobility")
                       if self.holes else None)
        # ---- sheets
        self.sheet = np.zeros(N)
        for item in sheets:
            nodes, sig, length, ref = item
            if not (isinstance(ref, str) and ref.strip()):
                raise ValueError("every sheet needs a reference")
            nodes = np.asarray(nodes, int)
            self.sheet[nodes] += float(sig) * 1e-14 * np.broadcast_to(
                np.asarray(length, float), nodes.shape)
        # ---- generation
        if generation is None:
            self.Gopt = np.zeros(N)
        elif callable(generation):
            self.Gopt = np.array([float(generation(*pt)) for pt in zip(*self.coords)]) * 1e-21
        else:
            self.Gopt = np.broadcast_to(np.asarray(generation, float), (N,)).copy() * 1e-21
        if not (np.all(np.isfinite(self.Gopt)) and np.all(self.Gopt >= 0)):
            raise ValueError("generation must be finite and >= 0 (cm^-3 s^-1)")
        # ---- electrodes
        electrodes = list(electrodes)
        names = [e.name for e in electrodes]
        if len(set(names)) != len(names):
            raise ValueError("electrode names must be different on a mesh")
        if not any(e.kind == "ohmic" for e in electrodes):
            raise ValueError("at least one ohmic electrode is needed")
        self.electrodes = electrodes
        self.edict = {e.name: e for e in electrodes}
        self._pieces = {e.name: [e] for e in electrodes}
        self._box_ec = {}
        self.enodes = {}
        for e in electrodes:
            if e.name not in contacts:
                raise ValueError(f"no contact nodes for electrode {e.name!r}")
            nd = np.unique(np.asarray(contacts[e.name], int))
            if nd.size == 0:
                raise ValueError(f"electrode {e.name!r} covers no node")
            for other, on in self.enodes.items():
                if np.intersect1d(on, nd).size:
                    raise ValueError(f"electrodes {other!r} and {e.name!r} overlap")
            self.enodes[e.name] = nd
        on_any = np.zeros(N, bool)
        for nd in self.enodes.values():
            on_any[nd] = True
        self.surface_nodes = np.zeros(0, int)
        self.surf_area = np.zeros(N)
        self.insulator_nodes = np.flatnonzero(~self.node_semi & ~on_any)
        # lateral "area" of a node for thermionic contacts: its share of the
        # mesh boundary
        self.lat_area = self._boundary_length()
        self.ionization = ionization
        if ionization is not None and not self.holes:
            raise ValueError("impact ionization makes holes: give hole_mass")
        if heterojunctions not in ("drift-diffusion", "thermionic"):
            raise ValueError("heterojunctions must be 'drift-diffusion' or 'thermionic'")
        self.heterojunctions = heterojunctions
        if band_to_band is not None and not self.holes:
            raise ValueError("band-to-band tunneling makes holes: give hole_mass")
        self.band_to_band = band_to_band
        self._hydro = None
        # ---- traps per node, by region label
        self._traps = []
        for t in traps:
            if not isinstance(t, Trap):
                raise ValueError("traps must be Trap objects")
            ws = (t.where,) if isinstance(t.where, str) else tuple(t.where)
            if "surface" in ws:
                raise ValueError("surface traps are not available on meshes")
            m = np.isin(self.node_region.astype(str), ws) & self.node_semi
            Nt = np.where(m, t.density * 1e-21, 0.0)
            if t.from_band == "ec":
                lev_c, lev_v = t.level, self.Eg - t.level
            else:
                lev_c, lev_v = self.Eg - t.level, t.level
            cn = t.capture_n * 1e21
            cp = t.capture_p * 1e21 if self.holes else 0.0
            n1 = self.Nc * np.exp(-np.asarray(lev_c, float) / self.kT)
            p1 = (self.Nv * np.exp(-np.asarray(lev_v, float) / self.kT)
                  if self.holes else np.zeros(N))
            self._traps.append(dict(Nt=Nt, cn=cn, cp=cp, en=cn * n1, ep=cp * p1,
                                    acc=t.kind == "acceptor", where=Nt > 0))
        self.traps = list(traps)
        self._neutral = self._neutral_psi()
        self.vR = np.sqrt(KB * QE * self.T / (2 * np.pi * self.me_node * M0)) * 1e9
        if self.holes:
            hmn = (self.Nv / pref) ** (2.0 / 3.0) / M0
            self.vRp = np.sqrt(KB * QE * self.T / (2 * np.pi * hmn * M0)) * 1e9
        self._x0 = None
        self._setup_heterojunctions()
        self._setup_band_to_band()
        self._tb = None
        self._tb_block = None

    # ------------------------------------------------------------ geometry
    def _mesh_mobility(self, regions, inv, tt, dd, semi_t, what):
        keys = ("low_field", "saturation_velocity", "beta", "temperature_exponent")
        nE = self.ei.size
        w = np.where(semi_t[tt], np.abs(dd), 0.0)
        wsum = np.bincount(inv, w, nE)
        safe = np.where(wsum > 0, wsum, 1.0)
        out = {}
        for k in keys:
            vals = []
            for r in self.mesh.regions[tt]:
                R = regions[r]
                mob = getattr(R, what) if R.semiconductor else None
                if R.semiconductor and mob is None:
                    raise ValueError(f"region {r!r} needs a {what}")
                vals.append(getattr(mob, k) if mob is not None else 1.0)
            v = np.array(vals, float)
            if k == "saturation_velocity":
                with np.errstate(divide="ignore"):
                    inv_v = np.bincount(inv, w / v, nE) / safe
                    e = np.where(inv_v > 0, 1.0 / np.where(inv_v > 0, inv_v, 1.0), np.inf)
            else:
                e = np.bincount(inv, w * v, nE) / safe
            out[k] = np.where(wsum > 0, e, 1.0 if k != "saturation_velocity" else np.inf)
        return out

    def _boundary_length(self):
        T_ = self.mesh.triangles
        ed = np.sort(np.concatenate([T_[:, [0, 1]], T_[:, [1, 2]], T_[:, [2, 0]]]), axis=1)
        u, c = np.unique(ed, axis=0, return_counts=True)
        b = u[c == 1]
        P = self.mesh.points
        L = np.hypot(*(P[b[:, 1]] - P[b[:, 0]]).T)
        out = np.zeros(self.N)
        np.add.at(out, b[:, 0], 0.5 * L)
        np.add.at(out, b[:, 1], 0.5 * L)
        return out

    # ----------------------------------------------- overrides of grid code
    def _column_guess(self):
        """A start for the equilibrium Poisson solve: the charge-neutral level
        where the semiconductor is doped, a level 1 eV below the band edge
        elsewhere."""
        psi = self.Ec0 - 1.0
        dop = (self.ND > 0) | (self.NA > 0)
        for k in np.flatnonzero(dop & self.node_semi):
            n_net = self.ND[k] - self.NA[k]
            if n_net > 0:
                psi[k] = self.Ec0[k] + self.kT * math.log(n_net / self.Nc[k])
            elif self.holes and n_net < 0:
                psi[k] = self.Ec0[k] - self.Eg[k] - self.kT * math.log(-n_net / self.Nv[k])
        return psi

    def _setup_heterojunctions(self):
        i, j = self.ei, self.ej
        self._te = None
        if self.heterojunctions != "thermionic":
            return
        nl = self.node_region
        cross = (nl[i] != nl[j]) & (self.earea > 0) & self.node_semi[i] & self.node_semi[j]
        dEc = self.Ec0[j] - self.Ec0[i]
        Ev0 = self.Ec0 - self.Eg
        dEv = Ev0[j] - Ev0[i]
        en = cross & (np.abs(dEc) > 1e-9)
        ep = cross & (np.abs(dEv) > 1e-9) if self.holes else np.zeros_like(en)
        self._te = dict(n=np.flatnonzero(en), n_hi_j=dEc[en] > 0,
                        p=np.flatnonzero(ep), p_hi_j=dEv[ep] < 0)

    def _ion_mask(self, ion):
        if ion.where is None:
            return self.node_semi
        return self.node_semi & np.isin(self.node_region.astype(str), ion.where)

    def _setup_band_to_band(self):
        bt = self.band_to_band
        self._bt = None
        if bt is None:
            return
        mask = self.node_semi.copy()
        if bt.where is not None:
            mask &= np.isin(self.node_region.astype(str), bt.where)
        if bt.reduced_mass_m0 is not None:
            mr = np.full(self.N, float(bt.reduced_mass_m0))
        else:
            pref = (KB * self.T * QE / (2 * np.pi * HBAR ** 2)) ** 1.5 * 2.0 * 1e-27
            mh = (self.Nv / pref) ** (2.0 / 3.0) / M0
            mr = 1.0 / (1.0 / self.me_node + 1.0 / mh)
        # least-squares gradient at every node from its edges, weighted by
        # each edge's semiconductor face
        P = np.column_stack(self.coords)
        dv = P[self.ej] - P[self.ei]
        w = np.maximum(self.earea, 0.0) / self.eh ** 2
        M = np.zeros((self.N, 2, 2))
        for nd in (self.ei, self.ej):
            np.add.at(M, nd, w[:, None, None] * dv[:, :, None] * dv[:, None, :])
        M += 1e-30 * np.eye(2)[None]
        self._bt = dict(mask=mask, mr=mr, dv=dv, w=w, Minv=np.linalg.inv(M))

    def _node_field(self, psi):
        bt = self._bt
        dpsi = psi[self.ej] - psi[self.ei]
        r = np.zeros((self.N, 2))
        for nd in (self.ei, self.ej):
            np.add.at(r, nd, (bt["w"] * dpsi)[:, None] * bt["dv"])
        grad = np.einsum("nij,nj->ni", bt["Minv"], r)
        return -grad * 1e7

    def _thermal_setup(self, model):
        raise ValueError("self-heating is not available on meshes yet")

    def field(self, state):
        raise ValueError("on a mesh, take the field from state.psi with your own "
                         "interpolation (field() is for tensor grids)")

    def ionization_integral(self, *a, **k):
        raise ValueError("ionization_integral is for tensor grids")

    def breakdown_voltage(self, *a, **k):
        raise ValueError("breakdown_voltage is for tensor grids; use breakdown_by_current")

    def field_check(self, *a, **k):
        raise ValueError("field_check is for tensor grids")

    def gate_tunneling(self, *a, **k):
        raise ValueError("gate_tunneling is for tensor grids")


def _circumcentres(A, B, C):
    ax, ay = A[:, 0], A[:, 1]
    bx, by = B[:, 0], B[:, 1]
    cx, cy = C[:, 0], C[:, 1]
    d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    a2, b2, c2 = ax * ax + ay * ay, bx * bx + by * by, cx * cx + cy * cy
    ux = (a2 * (by - cy) + b2 * (cy - ay) + c2 * (ay - by)) / d
    uy = (a2 * (cx - bx) + b2 * (ax - cx) + c2 * (bx - ax)) / d
    return np.column_stack([ux, uy])
