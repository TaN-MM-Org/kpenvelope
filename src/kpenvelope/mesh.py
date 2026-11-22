"""Devices on unstructured triangular meshes (new in 0.21.0): curved and
slanted shapes in two dimensions.

`Mesh` holds the nodes (x, z in nm), the triangles and the region label of
every triangle. `MeshDevice` discretizes the same drift-diffusion
equations as `Device` with the box (Voronoi) finite-volume method: each
triangle gives every one of its edges the part of the dual face that runs
from the edge's midpoint to the triangle's circumcenter, and every node the
matching part of its dual cell. A triangle with an obtuse angle gives a
negative part to the edge across from it; on a Delaunay mesh the
neighboring triangle makes it up on an interior edge, but not on the
boundary (the constructor warns where an edge's total is negative). A rectangular
grid split into right triangles gives exactly the coefficients of the
tensor-grid `Device`, which the tests use as a check.

Supported: semiconductor regions with their material, doping and mobility,
dielectric regions, electrodes on any set of nodes (ohmic, Schottky,
series resistance), deep-level traps per region, generation, impact
ionization, local band-to-band tunneling, thermionic emission at band
steps, the hydrodynamic model, small-signal analysis and transients, and
since 0.22.0 self-heating and pulses (the heat equation on the mesh,
with its bottom as the heat sink), `TunnelBarrier` (with the barrier
meshed in vertical columns) and the full temperature model. Not on
meshes: the nonlocal band-to-band model, the field-line tools
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
    into two right triangles; region_of(x, z) at the rectangle's center."""
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


def _point_in_polygon(px, pz, poly):
    """Even-odd rule, vectorized over points (px, pz arrays)."""
    poly = np.asarray(poly, float)
    x0, z0 = poly[:, 0], poly[:, 1]
    x1, z1 = np.roll(x0, -1), np.roll(z0, -1)
    inside = np.zeros(np.shape(px), bool)
    for a, b, c, d in zip(x0, z0, x1, z1):
        cond = (b > pz) != (d > pz)
        with np.errstate(divide="ignore", invalid="ignore"):
            xc = a + (pz - b) * (c - a) / (d - b)
        inside ^= cond & (px < xc)
    return inside


def _tri_quality(P, T):
    """Smallest angle (degrees), circumcenters and areas of triangles."""
    A, B, C = P[T[:, 0]], P[T[:, 1]], P[T[:, 2]]
    a = np.hypot(*(B - C).T)
    b = np.hypot(*(C - A).T)
    c = np.hypot(*(A - B).T)
    cosA = np.clip((b * b + c * c - a * a) / (2 * b * c), -1, 1)
    cosB = np.clip((a * a + c * c - b * b) / (2 * a * c), -1, 1)
    cosC = np.clip((a * a + b * b - c * c) / (2 * a * b), -1, 1)
    ang = np.degrees(np.arccos(np.stack([cosA, cosB, cosC], 1)))
    area = 0.5 * np.abs((B[:, 0] - A[:, 0]) * (C[:, 1] - A[:, 1])
                        - (C[:, 0] - A[:, 0]) * (B[:, 1] - A[:, 1]))
    return ang.min(1), _circumcenters(A, B, C), area, np.maximum(np.maximum(a, b), c)


def quality_mesh(regions: dict, h=None, min_angle: float = 25.0, max_points: int = 200000):
    """A quality triangle mesh of regions given as polygons (new in 0.27.0).

    regions : {label: [(x, z), ...]} (nm), each a simple polygon listed
    around its boundary (or a list of such polygons); neighbors share their
    common edges, which become mesh edges. h : the largest edge length
    wanted (nm), a number or a function h(x, z); None: as fine as the
    input and the angle bound need. min_angle : the smallest angle wanted
    (degrees; Ruppert's algorithm guarantees about 20.7, and up to about
    30 usually works).

    Delaunay refinement after J. Ruppert, "A Delaunay refinement algorithm
    for quality 2-dimensional mesh generation", J. Algorithms (1995): every
    boundary and interface segment whose diametral circle holds another
    node is split at its middle, and every triangle with too small an
    angle (or too long an edge) gets a node at its circumcenter, unless
    that node would lie in a segment's diametral circle, in which case the
    segment is split instead. The result is a conforming Delaunay mesh in
    which every segment is an edge and no node lies in a segment's
    diametral circle, so the angle across from every boundary and interface
    edge is at most 90 degrees and the box method has no negative dual
    faces there. Input corners sharper than min_angle keep their angle.
    Returns a `Mesh` with each triangle labeled by the region holding its
    centroid."""
    from scipy.spatial import Delaunay, cKDTree
    if not regions:
        raise ValueError("regions must not be empty")
    if not (0 < min_angle <= 34):
        raise ValueError("min_angle must be in (0, 34] degrees")
    polys = []
    for lab, pg in regions.items():
        pl = pg if (len(pg) and np.ndim(pg[0]) == 2) else [pg]
        for q in pl:
            q = np.asarray(q, float)
            if q.ndim != 2 or q.shape[1] != 2 or q.shape[0] < 3:
                raise ValueError(f"region {lab!r}: a polygon needs at least 3 (x, z) points")
            if np.allclose(q[0], q[-1]):
                q = q[:-1]
            polys.append((lab, q))
    hfun = (h if callable(h) else (lambda x, z, v=h: np.full(np.shape(x), float(v)))
            if h is not None else None)
    # nodes and segments (shared edges once)
    key = {}
    pts = []

    def node(p):
        k_ = (round(float(p[0]), 9), round(float(p[1]), 9))
        if k_ not in key:
            key[k_] = len(pts)
            pts.append([float(p[0]), float(p[1])])
        return key[k_]
    segs = set()
    for _, q in polys:
        ids = [node(p) for p in q]
        for i in range(len(ids)):
            a_, b_ = ids[i], ids[(i + 1) % len(ids)]
            if a_ != b_:
                segs.add((min(a_, b_), max(a_, b_)))
    corner = np.zeros(len(pts), bool)
    corner[:] = True                         # input vertices
    P = np.array(pts, float)
    segs = list(segs)
    # split segments longer than h first
    if hfun is not None:
        out = []
        for a_, b_ in segs:
            pa, pb = P[a_], P[b_]
            L = float(np.hypot(*(pb - pa)))
            hm = float(np.min(hfun(np.array([pa[0], pb[0], 0.5 * (pa[0] + pb[0])]),
                                   np.array([pa[1], pb[1], 0.5 * (pa[1] + pb[1])]))))
            n = max(1, int(math.ceil(L / hm)))
            prev = a_
            for t_ in range(1, n):
                newp = pa + (pb - pa) * t_ / n
                P = np.vstack([P, newp])
                cur = P.shape[0] - 1
                out.append((min(prev, cur), max(prev, cur)))
                prev = cur
            out.append((min(prev, b_), max(prev, b_)))
        segs = out
    corner = np.concatenate([corner, np.zeros(P.shape[0] - corner.size, bool)])
    lo, hi = P.min(0), P.max(0)
    scale = float(np.max(hi - lo))

    def inside_any(x, z):
        res = np.zeros(np.shape(x), bool)
        for _, q in polys:
            res |= _point_in_polygon(x, z, q)
        return res

    def split_encroached(P, segs, extra=None):
        """Split segments with a node (or one of `extra`) inside their
        diametral circle, until none is left."""
        changed = True
        while changed:
            changed = False
            tree = cKDTree(P)
            new_segs = []
            add = []
            for a_, b_ in segs:
                m = 0.5 * (P[a_] + P[b_])
                r = 0.5 * float(np.hypot(*(P[b_] - P[a_])))
                near = tree.query_ball_point(m, r * (1 - 1e-9))
                near = [q for q in near if q != a_ and q != b_]
                if near:
                    idx = P.shape[0] + len(add)
                    add.append(m)
                    new_segs += [(min(a_, idx), max(a_, idx)), (min(idx, b_), max(idx, b_))]
                    changed = True
                else:
                    new_segs.append((a_, b_))
            if add:
                P = np.vstack([P, np.array(add)])
            segs = new_segs
            if P.shape[0] > max_points:
                raise RuntimeError("quality_mesh: too many points; loosen h or min_angle")
        return P, segs
    P, segs = split_encroached(P, segs)
    for _it in range(1000):
        tri = Delaunay(P).simplices
        cen = P[tri].mean(1)
        keep = inside_any(cen[:, 0], cen[:, 1])
        tri = tri[keep]
        amin, cc, area, lmax = _tri_quality(P, tri)
        # triangles at a sharp input corner keep their angle
        sharp = np.zeros(tri.shape[0], bool)
        if np.any(corner):
            vc = corner[np.minimum(tri, corner.size - 1)] & (tri < corner.size)
            sharp = vc.any(1) & (amin < min_angle)
            # only exempt if the small angle sits at the input vertex itself
            if sharp.any():
                A_, B_, C_ = P[tri[:, 0]], P[tri[:, 1]], P[tri[:, 2]]

                def ang(u, v, w_):
                    d1, d2 = v - u, w_ - u
                    return np.degrees(np.arccos(np.clip(np.sum(d1 * d2, 1) / (
                        np.hypot(*d1.T) * np.hypot(*d2.T)), -1, 1)))
                angs = np.stack([ang(A_, B_, C_), ang(B_, C_, A_), ang(C_, A_, B_)], 1)
                at = np.argmin(angs, 1)
                vtx = tri[np.arange(tri.shape[0]), at]
                sharp = sharp & (vtx < corner.size) & corner[np.minimum(vtx, corner.size - 1)]
        bad = (amin < min_angle) & ~sharp
        if hfun is not None:
            bad |= lmax > 1.0001 * hfun(cen[keep][:, 0], cen[keep][:, 1])
        if not bad.any():
            break
        # worst first; skip circumcenters too close to one already taken
        order = np.argsort(amin[bad])
        cand = cc[bad][order]
        crad = np.hypot(*(cand - P[tri[bad][order][:, 0]]).T)
        cand_ok = inside_any(cand[:, 0], cand[:, 1])
        cand, crad = cand[cand_ok], crad[cand_ok]
        taken = []
        tree_t = None
        for p_, r_ in zip(cand, crad):
            if taken:
                if tree_t is None or len(taken) % 64 == 0:
                    tree_t = cKDTree(np.array(taken))
                d_, _ = tree_t.query(p_)
                close = any(np.hypot(*(p_ - t2)) < 0.5 * r_ for t2 in taken[-64:])
                if d_ < 0.5 * r_ or close:
                    continue
            taken.append(p_)
        if not taken:
            break
        taken = np.array(taken)
        # a circumcenter in a segment's diametral circle splits the segment
        enc = np.zeros(taken.shape[0], bool)
        seg_split = set()
        SA = np.array([P[a_] for a_, _ in segs])
        SB = np.array([P[b_] for _, b_ in segs])
        SM = 0.5 * (SA + SB)
        SR = 0.5 * np.hypot(*(SB - SA).T)
        st = cKDTree(SM)
        rmax = float(SR.max())
        for q, p_ in enumerate(taken):
            for sidx in st.query_ball_point(p_, rmax):
                if np.hypot(*(p_ - SM[sidx])) < SR[sidx] * (1 - 1e-9):
                    enc[q] = True
                    seg_split.add(sidx)
        if seg_split:
            add, new_segs = [], []
            for sidx, (a_, b_) in enumerate(segs):
                if sidx in seg_split:
                    idx = P.shape[0] + len(add)
                    add.append(SM[sidx])
                    new_segs += [(min(a_, idx), max(a_, idx)), (min(idx, b_), max(idx, b_))]
                else:
                    new_segs.append((a_, b_))
            P = np.vstack([P, np.array(add)])
            segs = new_segs
        P = np.vstack([P, taken[~enc]])
        P, segs = split_encroached(P, segs)
        if P.shape[0] > max_points:
            raise RuntimeError("quality_mesh: too many points; loosen h or min_angle")
    else:
        warnings.warn("quality_mesh: the refinement did not finish; some triangles keep "
                      "angles below min_angle", RuntimeWarning, stacklevel=2)
    tri = Delaunay(P).simplices
    cen = P[tri].mean(1)
    labs = np.empty(tri.shape[0], dtype=object)
    labs[:] = None
    for lab, q in polys:
        m = _point_in_polygon(cen[:, 0], cen[:, 1], q) & np.array([v is None for v in labs])
        labs[m] = lab
    keep = np.array([v is not None for v in labs])
    tri, labs = tri[keep], labs[keep]
    used = np.unique(tri)
    remap = -np.ones(P.shape[0], int)
    remap[used] = np.arange(used.size)
    mesh = Mesh(P[used], remap[tri], labs)
    mesh.segments = np.array([[remap[a_], remap[b_]] for a_, b_ in segs], int)
    return mesh


def recovered_hessian(mesh: Mesh, values):
    """Second derivatives of a field given on the nodes of a mesh (new in
    0.28.0): at every node, a least-squares quadratic through the values on
    the node's neighbors and their neighbors (two rings of edges, at least
    six nodes), weighted by 1 / (1 + (d / d_mean)^2). Returns (N, 3) arrays
    (d2/dx2, d2/dxdz, d2/dz2) per node, in units of values per nm^2. Exact
    for quadratics on any mesh."""
    import scipy.sparse as sp
    P, T = mesh.points, mesh.triangles
    u = np.asarray(values, float)
    N = P.shape[0]
    if u.shape != (N,):
        raise ValueError("one value per node")
    e = np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]])
    A = sp.csr_matrix((np.ones(2 * len(e)), (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])),
                      shape=(N, N))
    A = ((A + sp.identity(N)) > 0).astype(float)
    A2 = (A @ A).tocsr()
    A2.sum_duplicates()
    cnt = np.diff(A2.indptr)
    K = int(cnt.max())
    idx = np.full((N, K), -1, int)
    for i in range(N):
        nb = A2.indices[A2.indptr[i]:A2.indptr[i + 1]]
        idx[i, :nb.size] = nb
    ok = idx >= 0
    j = np.where(ok, idx, 0)
    dx = np.where(ok, P[j, 0] - P[:, None, 0], 0.0)
    dz = np.where(ok, P[j, 1] - P[:, None, 1], 0.0)
    d2 = dx * dx + dz * dz
    dm = np.sqrt(np.maximum((d2 * ok).sum(1) / np.maximum(ok.sum(1) - 1, 1), 1e-300))
    s_ = dm[:, None]
    X = np.stack([np.ones_like(dx), dx / s_, dz / s_, 0.5 * (dx / s_) ** 2,
                  (dx / s_) * (dz / s_), 0.5 * (dz / s_) ** 2], 2)
    w = np.where(ok, 1.0 / (1.0 + d2 / s_ ** 2), 0.0)
    y = np.where(ok, u[j], 0.0)
    XtW = X * w[:, :, None]
    G = np.einsum("nki,nkj->nij", XtW, X)
    r = np.einsum("nki,nk->ni", XtW, y)
    G = G + 1e-12 * np.eye(6)[None] * np.trace(G, axis1=1, axis2=2)[:, None, None]
    c = np.linalg.solve(G, r[:, :, None])[:, :, 0]
    return np.stack([c[:, 3], c[:, 4], c[:, 5]], 1) / (dm ** 2)[:, None]


def size_from_solution(mesh: Mesh, values, tolerance, h_min, h_max, grading=0.3):
    """A size field h(x, z) for `quality_mesh` that puts small triangles
    where the field `values` (on the nodes of `mesh`) curves (new in
    0.28.0): the error of linear interpolation over a triangle of size h is
    about h^2 |lambda| / 8 with lambda the largest eigenvalue of the
    second derivatives (`recovered_hessian`), so h = sqrt(8 tolerance /
    |lambda|), kept between h_min and h_max and limited to grow by at most
    `grading` (nm per nm) from node to node. tolerance is in the units of
    values (for the potential, volts). Returns a function of (x, z)
    arrays: the nodes' sizes interpolated linearly (nearest node outside
    the mesh)."""
    from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
    if not (tolerance > 0 and 0 < h_min <= h_max and grading > 0):
        raise ValueError("tolerance > 0, 0 < h_min <= h_max and grading > 0")
    Hs = recovered_hessian(mesh, values)
    a, b, c = Hs[:, 0], Hs[:, 1], Hs[:, 2]
    lam = 0.5 * np.abs(a + c) + np.sqrt(0.25 * (a - c) ** 2 + b * b)
    with np.errstate(divide="ignore"):
        h = np.sqrt(8.0 * tolerance / np.maximum(lam, 1e-300))
    h = np.clip(h, h_min, h_max)
    # grading: h_i <= h_j + grading |x_i - x_j| along every edge
    T = mesh.triangles
    e = np.unique(np.sort(np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]]), 1),
                  axis=0)
    L = np.hypot(*(mesh.points[e[:, 0]] - mesh.points[e[:, 1]]).T)
    for _ in range(200):
        h0 = h.copy()
        np.minimum.at(h, e[:, 0], h[e[:, 1]] + grading * L)
        np.minimum.at(h, e[:, 1], h[e[:, 0]] + grading * L)
        if np.array_equal(h, h0):
            break
    lin = LinearNDInterpolator(mesh.points, h)
    near = NearestNDInterpolator(mesh.points, h)

    def size(x, z):
        x = np.asarray(x, float)
        z = np.asarray(z, float)
        v = lin(x, z)
        bad = ~np.isfinite(v)
        if np.any(bad):
            v = np.where(bad, near(x, z), v)
        return v
    return size


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
                 driving_force: str = "field", tunnel_barriers=(),
                 temperature_model: str = "mobility", band_gap_temperature=None,
                 heat_model: str = "joule", quantum=None, thermoelectric: str = "boltzmann"):
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
        self._regions_semi = {r: regions[r].semiconductor for r in regions}
        eps_t = np.array([(regions[r].material.eps_r if regions[r].semiconductor
                           else regions[r].dielectric.eps_r) for r in mesh.regions], float)
        # ---- box method, triangle by triangle
        T_ = mesh.triangles
        A, B, C = P[T_[:, 0]], P[T_[:, 1]], P[T_[:, 2]]
        cc = _circumcenters(A, B, C)
        edges_t = []                   # (n1, n2, triangle, signed dual length)
        sub_t = []                     # (node, triangle, dual-cell area)
        for (u, v, w) in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
            pu, pv, pw = P[T_[:, u]], P[T_[:, v]], P[T_[:, w]]
            mid = 0.5 * (pu + pv)
            ev = pv - pu
            h = np.hypot(ev[:, 0], ev[:, 1])
            nrm = np.column_stack([-ev[:, 1], ev[:, 0]]) / h[:, None]
            # signed distance from the edge midpoint to the circumcenter,
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
        # a dual face counts as negative when it is so against the edge's
        # own length (a right angle on a straight boundary gives a face of
        # zero, which rounding makes -1e-10 nm or so)
        if (np.any(self.earea < -1e-7 * h_e)
                or np.any(self.gpsi * h_e < -1e-7 * h_e * float(np.max(eps_t)))):
            warnings.warn("some edges have a negative dual face (obtuse triangles at a "
                          "boundary); refine the mesh there", RuntimeWarning, stacklevel=2)
        sn = np.concatenate([s_[0] for s_ in sub_t])
        st_ = np.concatenate([s_[1] for s_ in sub_t])
        sa = np.concatenate([s_[2] for s_ in sub_t])
        self._geo = dict(inv=inv, tt=tt, dd=dd, hh=hh, sn=sn, st=st_, sa=sa)
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
            # (a tie within rounding of the circumcenters counts as a tie)
            better = v >= bestv - 1e-9 * np.abs(self.vol_all)
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
        self._setup_tunnel_barriers(tunnel_barriers)
        self._set_heat_model(heat_model, thermoelectric)
        self._setup_lattice(temperature_model, band_gap_temperature, self.node_region)
        self._setup_quantum(quantum)

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
        self._bt = dict(mask=mask, mr=mr, dv=dv, w=w, Minv=np.linalg.inv(M),
                        nonlocal_=bt.model == "nonlocal")
        if bt.model == "nonlocal":
            self._bt["lines"] = self._mesh_lines(bt, mask)
            self._bt["area"] = np.array([L_["area"] for L_ in self._bt["lines"]])

    def _mesh_lines(self, bt, mask):
        """Straight, parallel tunneling paths through the mesh (new in
        0.23.0): lines along `direction` (default +z) at a spacing of
        `path_spacing` nm (default the median edge length), evenly across
        the semiconductor, each standing for the width between its
        neighbors (half a spacing at the two outer ones). Along a line the
        bands are linear inside every triangle, so the line is sampled where
        it crosses the triangle edges and the profile between is exact.
        Each sample carries the interpolation weights of its edge's two
        nodes. A stretch of the line counts if a triangle of a
        semiconductor region (and of `where`) covers it; anything else
        (a dielectric, a hole, outside) ends the paths there."""
        P = self.mesh.points
        T_ = self.mesh.triangles
        d = np.array(bt.direction if bt.direction is not None else (0.0, 1.0), float)
        d /= np.hypot(*d)
        e = np.array([d[1], -d[0]])                      # across the lines
        sP = P @ d
        tP = P @ e
        semi_tri = np.array([self._regions_semi[r] for r in self.mesh.regions])
        ok_tri = semi_tri & np.all(mask[T_], axis=1)
        if not ok_tri.any():
            return []
        tn = tP[np.unique(T_[ok_tri])]
        tmin, tmax = float(tn.min()), float(tn.max())
        h = bt.path_spacing
        if h is None:
            h = float(np.median(self.eh))
        K = max(1, int(round((tmax - tmin) / h)))
        tk = np.linspace(tmin, tmax, K + 1)
        dt = (tmax - tmin) / K
        width = np.full(K + 1, dt)
        width[0] = width[-1] = 0.5 * dt
        scale = max(1.0, float(np.ptp(sP)), float(np.ptp(tP)))
        eps = 1e-9 * scale
        tv = tP[T_]
        sv = sP[T_]
        lines = []
        for k, t0 in enumerate(tk):
            cand = np.flatnonzero((tv.min(axis=1) <= t0 + eps) & (tv.max(axis=1) >= t0 - eps))
            if cand.size == 0:
                continue
            pts_s, pts_a, pts_b, pts_w, tri_lo = [], [], [], [], []
            for c in cand:
                vs = []
                for (u, v) in ((0, 1), (1, 2), (2, 0)):
                    ta, tb = tv[c, u], tv[c, v]
                    na, nb = T_[c, u], T_[c, v]
                    if abs(ta - t0) <= eps:
                        vs.append((sv[c, u], na, na, 0.0))
                    if abs(tb - t0) <= eps:
                        vs.append((sv[c, v], nb, nb, 0.0))
                    if (ta - t0) * (tb - t0) < 0 and abs(ta - t0) > eps and abs(tb - t0) > eps:
                        lam = (t0 - ta) / (tb - ta)
                        vs.append((sv[c, u] + lam * (sv[c, v] - sv[c, u]), na, nb, lam))
                if not vs:
                    continue
                ss = [q[0] for q in vs]
                tri_lo.append((min(ss), max(ss), bool(ok_tri[c])))
                for q in vs:
                    pts_s.append(q[0])
                    pts_a.append(q[1])
                    pts_b.append(q[2])
                    pts_w.append(q[3])
            if not pts_s:
                continue
            order = np.argsort(pts_s, kind="stable")
            ps = np.asarray(pts_s)[order]
            keep = np.concatenate([[True], np.diff(ps) > eps])
            idx = order[keep]
            s_u = np.asarray(pts_s)[idx]
            a_u = np.asarray(pts_a, int)[idx]
            b_u = np.asarray(pts_b, int)[idx]
            lam = np.asarray(pts_w, float)[idx]
            n_s = s_u.size
            if n_s < 2:
                continue
            edge_ok = np.zeros(n_s - 1, bool)
            for lo, hi, good in tri_lo:
                if not good or hi - lo <= eps:
                    continue
                i0 = int(np.searchsorted(s_u, lo - eps))
                i1 = int(np.searchsorted(s_u, hi - eps))
                edge_ok[i0:i1] = True
            ok_s = np.zeros(n_s, bool)
            ok_s[:-1] |= edge_ok
            ok_s[1:] |= edge_ok
            lines.append(dict(a=a_u, b=b_u, wa=1.0 - lam, wb=lam, s=s_u - 0.0, ok_s=ok_s,
                              edge_ok=edge_ok, area=float(width[k])))
        return lines

    # ------------------------------------- quantum channel (0.26.0)
    def _setup_quantum(self, qc):
        """A QuantumChannel on a mesh: straight vertical lines through the
        mesh (as for nonlocal tunneling, `_mesh_lines`), `line_spacing` nm
        apart (default the median edge length), sampled where they cross the
        triangles' edges (the bands are linear along a line inside each
        triangle, so the samples carry the mesh's own profile). Each run of
        samples inside the window (semiconductor, no electrode node) is a
        column for the Schroedinger equation, with its walls at the samples
        just outside it; the quantum potential found on the lines is
        carried to the mesh nodes in the window, linearly along each line
        and between the two lines on either side of a node. The material on
        a sample (band offset, gap, mass) is the mean over its share of the
        line, so a material step stays where the mesh puts it. Nodes on a
        line take its value exactly: a mesh made from a tensor grid, with the
        lines on the grid's columns, gives the tensor grid's result when no
        material step lies inside the window (at a step the tensor grid
        gives a node's whole share to the layer above it, which moves the
        step by half a spacing)."""
        import types
        from .device import QuantumChannel
        self._qc, self._qL, self._qLp, self._qcols, self._qruns = None, None, None, [], []
        if qc is None:
            return
        if not isinstance(qc, QuantumChannel):
            raise ValueError("quantum must be a QuantumChannel")
        if qc.holes is not None and not self.holes:
            raise ValueError("QuantumChannel(holes=...) needs holes in the device "
                             "(hole_mass and hole mobilities)")
        P = self.mesh.points
        excl = np.zeros(self.N, bool)
        for nodes in self.enodes.values():
            excl[nodes] = True
        ok = self.node_semi & ~excl
        h = qc.line_spacing if qc.line_spacing is not None else float(np.median(self.eh))
        lines = self._mesh_lines(types.SimpleNamespace(direction=(0.0, 1.0), path_spacing=h),
                                 ok)
        w0, w1 = qc.window
        eps = 1e-9 * max(1.0, float(np.ptp(P)))
        runs = []
        for L in lines:
            a, b, wa, wb, s = L["a"], L["b"], L["wa"], L["wb"], L["s"]
            xl = float(wa[0] * P[a[0], 0] + wb[0] * P[b[0], 0])
            if qc.x_range is not None and not (qc.x_range[0] - eps <= xl
                                               <= qc.x_range[1] + eps):
                continue
            # a sample on a node is good where the node is; one between two
            # nodes where the line's triangles there are
            on_a, on_b = wb <= 1e-12, wa <= 1e-12
            good = np.where(on_a, ok[a], np.where(on_b, ok[b], L["ok_s"] & ok[a] & ok[b]))
            sel = good & (s >= w0 - eps) & (s <= w1 + eps)
            sel[0] = sel[-1] = False                 # walls need a sample outside
            k = 0
            n_s = s.size
            while k < n_s:
                if not sel[k]:
                    k += 1
                    continue
                k1 = k
                while k1 + 1 < n_s and sel[k1 + 1]:
                    k1 += 1
                if k1 - k + 1 >= 3:
                    i = np.arange(k, k1 + 1)
                    runs.append(dict(a=a[i], b=b[i], wa=wa[i], wb=wb[i], z=s[i].copy(),
                                     zl=float(s[k - 1]), zr=float(s[k1 + 1]), x=xl,
                                     full=(s, a, b, wa, wb)))
                k = k1 + 1
        if not runs:
            raise ValueError("the QuantumChannel window holds no semiconductor stretch of "
                             "at least 3 samples on any line")
        xs = np.array(sorted(set(r["x"] for r in runs)))
        # nodes in the window, within the lines' span: each takes the value
        # of the lines on its two sides at its height
        z = P[:, 1]
        qn = ok & (z >= w0 - eps) & (z <= w1 + eps) & (P[:, 0] >= xs[0] - eps) & (
            P[:, 0] <= xs[-1] + eps)
        nodes = np.flatnonzero(qn)
        k = np.clip(np.searchsorted(xs, P[nodes, 0] - eps), 1, max(xs.size - 1, 1))
        if xs.size == 1:
            klo = khi = np.zeros(nodes.size, int)
            th = np.zeros(nodes.size)
        else:
            klo, khi = k - 1, k
            th = (P[nodes, 0] - xs[klo]) / (xs[khi] - xs[klo])
            on_lo = np.abs(P[nodes, 0] - xs[klo]) <= eps
            on_hi = np.abs(P[nodes, 0] - xs[khi]) <= eps
            th = np.where(on_lo, 0.0, np.where(on_hi, 1.0, np.clip(th, 0.0, 1.0)))
        for r in runs:
            r["line"] = int(np.searchsorted(xs, r["x"] - eps))
        # material values (band offset, gap, mass) on the samples: the mean
        # over each sample's box (half way to each neighbor) of the
        # materials the line passes through there, so a material step sits
        # where the mesh puts it whichever region a node on it belongs to
        # (a node on a step belongs to one region in the device, the one
        # with the most of its control volume; along a line that would move
        # the step by up to half a spacing, differently from line to line)
        import scipy.sparse as sp
        T_ = self.mesh.triangles
        reg = self.mesh.regions
        Px = P[:, 0]
        rep = {}
        for lab in set(reg.tolist()):
            cand = np.flatnonzero(self.node_region == lab)
            if cand.size:
                rep[lab] = int(cand[0])
        txmin, txmax = Px[T_].min(axis=1), Px[T_].max(axis=1)
        seg_cache = {}

        def segments(r):
            """the line's segments between its crossings, each with the node
            whose material it takes (-1: outside the semiconductor)"""
            key = r["x"]
            if key in seg_cache:
                return seg_cache[key]
            s_, *_ = r["full"]
            zm = 0.5 * (s_[1:] + s_[:-1])
            cand = np.flatnonzero((txmin <= r["x"] + eps) & (txmax >= r["x"] - eps))
            node = -np.ones(zm.size, int)
            if cand.size:
                A_, B_, C_ = (P[T_[cand, j]] for j in range(3))
                det = ((B_[:, 1] - C_[:, 1]) * (A_[:, 0] - C_[:, 0])
                       + (C_[:, 0] - B_[:, 0]) * (A_[:, 1] - C_[:, 1]))
                for q in range(zm.size):
                    l1 = ((B_[:, 1] - C_[:, 1]) * (r["x"] - C_[:, 0])
                          + (C_[:, 0] - B_[:, 0]) * (zm[q] - C_[:, 1])) / det
                    l2 = ((C_[:, 1] - A_[:, 1]) * (r["x"] - C_[:, 0])
                          + (A_[:, 0] - C_[:, 0]) * (zm[q] - C_[:, 1])) / det
                    inside = (l1 >= -1e-9) & (l2 >= -1e-9) & (1 - l1 - l2 >= -1e-9)
                    hit = np.flatnonzero(inside)
                    if not hit.size:
                        continue
                    t = cand[hit[0]]
                    lab = reg[t]
                    if not self._regions_semi[lab]:
                        continue
                    mine = [v for v in T_[t] if self.node_region[v] == lab]
                    node[q] = int(mine[0]) if mine else rep.get(lab, -1)
            seg_cache[key] = (s_, node)
            return seg_cache[key]

        for r in runs:
            z_ = r["z"]
            n_ = z_.size
            on_a, on_b = r["wb"] <= 1e-12, r["wa"] <= 1e-12
            zb = np.concatenate([[r["zl"]], z_, [r["zr"]]])
            lo_b, hi_b = 0.5 * (zb[:-2] + zb[1:-1]), 0.5 * (zb[1:-1] + zb[2:])
            s_, node = segments(r)
            rows, cols, vals = [], [], []
            for q in range(n_):
                ov = np.minimum(s_[1:], hi_b[q]) - np.maximum(s_[:-1], lo_b[q])
                use = (ov > 0) & (node >= 0)
                if (not use.any() or ov[use].sum() <= 0) and (on_a[q] or on_b[q]):
                    rows.append(q)
                    cols.append(int(r["a"][q] if on_a[q] else r["b"][q]))
                    vals.append(1.0)
                    continue
                if not use.any() or ov[use].sum() <= 0:
                    rows += [q, q]
                    cols += [int(r["a"][q]), int(r["b"][q])]
                    vals += [float(r["wa"][q]), float(r["wb"][q])]
                    continue
                wq = ov[use] / ov[use].sum()
                rows += [q] * int(use.sum())
                cols += node[use].tolist()
                vals += wq.tolist()
            r["mat"] = sp.csr_matrix((vals, (rows, cols)), shape=(n_, self.N))
            del r["full"]
        if self._te is not None:
            e = np.concatenate([self._te["n"], self._te["p"]])
            if np.any(qn[self.ei[e]] | qn[self.ej[e]]):
                raise ValueError("heterojunctions='thermionic' at band steps inside the "
                                 "QuantumChannel window: keep the steps outside the window "
                                 "or use heterojunctions='drift-diffusion'")
        if self._tb is not None:
            tbn = np.concatenate([self._tb["int_nodes"]] +
                                 [np.ravel(L_["path"]) for L_ in self._tb["links"]])
            if np.any(qn[tbn.astype(int)]):
                raise ValueError("a tunnel barrier lies inside the QuantumChannel window")
        self._qc, self._qruns, self._qmask = qc, runs, qn
        self._qmap = dict(nodes=nodes, z=z[nodes], klo=klo, khi=khi, th=th, xs=xs)
        self._qL = np.zeros(self.N)
        self._qLp = np.zeros(self.N)
        self._qbase = dict(Ec0=self.Ec0.copy(), Eg=self.Eg.copy(),
                           en=[tr["en"] for tr in self._traps],
                           ep=[tr["ep"] for tr in self._traps])

    def _line_values(self, vals):
        """Values per run (one array per run) carried to the mapped nodes."""
        mp = self._qmap
        nl = mp["xs"].size
        out_lo = np.zeros(mp["nodes"].size)
        out_hi = np.zeros(mp["nodes"].size)
        have_lo = np.zeros(mp["nodes"].size, bool)
        have_hi = np.zeros(mp["nodes"].size, bool)
        zq = mp["z"]
        for r, v in zip(self._qruns, vals):
            k = r["line"]
            inside = (zq >= r["z"][0] - 1e-9) & (zq <= r["z"][-1] + 1e-9)
            if not inside.any():
                continue
            val = np.interp(zq, r["z"], v)
            m_lo = inside & (mp["klo"] == k)
            m_hi = inside & (mp["khi"] == k)
            out_lo[m_lo], have_lo[m_lo] = val[m_lo], True
            out_hi[m_hi], have_hi[m_hi] = val[m_hi], True
        th = mp["th"]
        res = np.where(have_lo & have_hi, (1.0 - th) * out_lo + th * out_hi,
                       np.where(have_lo, out_lo, np.where(have_hi, out_hi, 0.0)))
        if nl == 1:
            res = np.where(have_lo, out_lo, 0.0)
        return res

    def _q_columns(self, x, carrier):
        """As `Device._q_columns`, on the lines of the mesh."""
        from .device import _G2D, column_states
        qc = self._qc
        psi, fn, fp, _ = self._unpack(x)
        Ec = self._ec0_real() - psi
        Ev = Ec - self._eg_real()
        kT = np.broadcast_to(np.asarray(self.kT, float), (self.N,))
        lam_runs, wt_runs = [], []
        sub, sheets, top_share = [], [], 0.0
        for r in self._qruns:
            a, b, wa, wb = r["a"], r["b"], r["wa"], r["wb"]

            def at(v):
                return wa * v[a] + wb * v[b]
            mat = r["mat"].dot
            kt = at(kT)
            Ec0r, Egr = self._ec0_real(), self._eg_real()
            Ecs = mat(Ec0r) - at(psi)
            if carrier == "n":
                mass, mass_xy, U, F = mat(self.me_node), None, Ecs, at(fn)
            else:
                mass = np.full(a.size, qc.holes[0])
                mass_xy, U, F = qc.holes[1], -(Ecs - mat(Egr)), -at(fp)
            e_max = float(np.max(F) + 40.0 * np.max(kt))
            E, p2, w = column_states(r["z"], U, mass, qc.n_states, r["zl"], r["zr"],
                                     e_max=e_max)
            mi = ((w[:, None] * p2 * mass[:, None]).sum(0) if mass_xy is None
                  else np.full(E.size, mass_xy))
            occ = kt[:, None] * np.logaddexp(0.0, (F[:, None] - E[None, :]) / kt[:, None])
            parts = _G2D * mi[None, :] * occ * p2
            nq = parts.sum(1)
            if nq.max() > 0 and E.size == qc.n_states:
                top_share = max(top_share, float((parts[:, -1] * w).sum() / (nq * w).sum()))
            lam_runs.append(nq)
            wt_runs.append(np.full(nq.size, max(float(nq.max()), 1e-300)))
            sub.append((r["x"], E if carrier == "n" else -E))
            sheets.append(float((nq * w).sum() * 1e14))
        # the confined density, carried to the nodes; the quantum potential
        # from each node's own band edge, level and density of states (so a
        # node next to a material step keeps its own material)
        target = np.zeros(self.N)
        weight = np.zeros(self.N, bool)
        nd = self._qmap["nodes"]
        nqn = np.maximum(self._line_values(lam_runs), 0.0)
        nmax = self._line_values(wt_runs)
        ktn = kT[nd]
        if carrier == "n":
            U_, F_, dos = Ec[nd], fn[nd], self.Nc[nd]
        else:
            U_, F_, dos = -Ev[nd], -fp[nd], self.Nv[nd]
        eta = self._inv_occ(np.maximum(nqn, 1e-300) / dos)
        lam = np.clip(F_ - U_ - ktn * eta, -qc.max_shift, qc.max_shift)
        target[nd] = lam
        weight[nd] = (nqn > 1e-6 * np.maximum(nmax, 1e-300)) & (np.abs(lam) < qc.max_shift)
        return target, weight, dict(subbands=sub, sheet_cm2=sheets, top_share=top_share)

    # NEGF on the mesh's lines (new in 0.32.0): `Device.negf` with the
    # QuantumChannel's columns sampled on the lines (the band edge linear
    # along a line inside each triangle, the material as in the quantum
    # channel), and the density carried back to the nodes as the quantum
    # channel carries its own
    def _negf_columns(self, lo, hi):
        out = []
        for k, r in enumerate(self._qruns):
            if lo < r["x"] < hi:
                out.append(dict(x=r["x"], z=r["z"], zl=r["zl"], zr=r["zr"], run=k,
                                mass=r["mat"].dot(self.me_node)))
        xs = [c["x"] for c in out]
        if len(set(xs)) != len(xs):
            raise ValueError("negf needs one QuantumChannel run per line (a single "
                             "channel along z)")
        return out

    def _negf_band(self, c, psi, holes):
        r = self._qruns[c["run"]]
        at = r["wa"] * psi[r["a"]] + r["wb"] * psi[r["b"]]
        Ec = r["mat"].dot(self._ec0_real()) - at
        return -(Ec - r["mat"].dot(self._eg_real())) if holes else Ec

    def _negf_nodes(self, cols):
        return self._qmap["nodes"].copy()

    def _negf_density(self, cols, ncol):
        vals = [np.zeros(r["z"].size) for r in self._qruns]
        cover = [np.zeros(r["z"].size) for r in self._qruns]
        for c, n_ in zip(cols, ncol):
            vals[c["run"]] = np.asarray(n_, float)
            cover[c["run"]] = np.ones(len(n_))
        nd = self._qmap["nodes"]
        dens = self._line_values(vals)
        # the nodes that lie between two lines that carry NEGF columns (or on
        # one): the rest keep their drift-diffusion level
        keep = self._line_values(cover) > 1 - 1e-9
        return nd[keep], np.maximum(dens[keep], 0.0)

    def _node_field(self, psi):
        bt = self._bt
        dpsi = psi[self.ej] - psi[self.ei]
        r = np.zeros((self.N, 2))
        for nd in (self.ei, self.ej):
            np.add.at(r, nd, (bt["w"] * dpsi)[:, None] * bt["dv"])
        grad = np.einsum("nij,nj->ni", bt["Minv"], r)
        return -grad * 1e7

    # --------------------------------------- tunnel barriers (0.22.0)
    def _setup_tunnel_barriers(self, tbs):
        """`TunnelBarrier` on a mesh: the barrier region and the row of
        nodes just below it must be meshed in vertical columns (nodes with
        the same x joined by vertical edges, as `mesh_from_grid` makes them;
        the rest of the mesh may be anything). Every column through the
        barrier becomes one link, exactly as in a tensor grid: from the node
        below the barrier's bottom face to the node on its top face (or the
        metal of an electrode there), with the flux area of the column's
        vertical edge."""
        from .device import TunnelBarrier
        self._tb = None
        self._tb_block = None
        tbs = list(tbs)
        if not tbs:
            return
        P = self.mesh.points
        T_ = self.mesh.triangles
        reg = self.mesh.regions
        # triangles on each edge
        ed = np.concatenate([T_[:, [0, 1]], T_[:, [1, 2]], T_[:, [2, 0]]])
        et = np.concatenate([np.arange(T_.shape[0])] * 3)
        lo, hi = ed.min(axis=1), ed.max(axis=1)
        key = lo * self.N + hi
        edge_key = np.minimum(self.ei, self.ej) * self.N + np.maximum(self.ei, self.ej)
        e_index = {int(k): n for n, k in enumerate(edge_key)}
        tri_of = {}
        for k, t in zip(key.tolist(), et.tolist()):
            tri_of.setdefault(k, []).append(t)
        on_el = np.zeros(self.N, bool)
        for nodes in self.enodes.values():
            on_el[nodes] = True
        xr = np.round(P[:, 0], 6)
        links, interior = [], []
        blocked = np.zeros(self.N, bool)
        for tb in tbs:
            if not isinstance(tb, TunnelBarrier):
                raise ValueError("tunnel_barriers must be TunnelBarrier objects")
            if tb.layer not in set(reg.tolist()):
                raise ValueError(f"TunnelBarrier: {tb.layer!r} is not a region of the mesh")
            in_b = np.zeros(self.N, bool)
            in_b[np.unique(T_[reg == tb.layer])] = True
            covered = np.zeros(self.N, bool)
            for xv in np.unique(xr[in_b]):
                col = np.flatnonzero(xr == xv)
                col = col[np.argsort(P[col, 1])]
                inside = []
                for k in range(col.size - 1):
                    kk = int(min(col[k], col[k + 1]) * self.N + max(col[k], col[k + 1]))
                    ts = tri_of.get(kk)
                    inside.append(bool(ts) and all(reg[t] == tb.layer for t in ts))
                inside = np.array(inside, bool)
                if not inside.any():
                    continue
                ks = np.flatnonzero(inside)
                if np.any(np.diff(ks) != 1):
                    raise ValueError("a column crosses the tunnel barrier twice")
                kb0, kb1 = int(ks[0]), int(ks[-1]) + 1
                if kb0 == 0:
                    raise ValueError("a tunnel barrier needs semiconductor below it")
                a = int(col[kb0 - 1])
                b = int(col[kb1])
                kk = int(min(a, col[kb0]) * self.N + max(a, col[kb0]))
                if kk not in e_index:
                    raise ValueError("the nodes below the tunnel barrier must be joined to "
                                     "it by vertical edges (mesh it in columns)")
                area = float(self.earea[e_index[kk]])
                top = kb1 == col.size - 1 or not self.node_semi[col[kb1 + 1]]
                inner = col[kb0:kb1 + (1 if top else 0)]
                covered[col[kb0:kb1 + 1]] = True
                if not (self.node_semi[a] and np.all(self.node_semi[inner])):
                    continue
                blocked[inner] = True
                target = (b if on_el[b] else None) if top else b
                if target is not None:
                    path = col[kb0 - 1:kb1 + 1]
                    links.append(dict(a=a, b=target, path=path, z=P[path, 1].copy(),
                                      area=area, mass=tb.mass_m0, n_energy=int(tb.n_energy),
                                      metal=bool(on_el[target])))
                zb = P[b, 1]
                for nd in inner:
                    if on_el[nd]:
                        continue
                    if target is None:
                        interior.append((nd, a, a, 1.0))
                    else:
                        interior.append((nd, a, target, (zb - P[nd, 1]) / (zb - P[a, 1])))
            if np.any(in_b & ~covered):
                raise ValueError("the tunnel barrier region must be meshed in vertical "
                                 "columns (some of its nodes lie on none)")
        i, j = self.ei, self.ej
        self._tb_block = ~(blocked[i] | blocked[j])
        self._tb = dict(links=links, interior=interior,
                        int_nodes=np.array([t[0] for t in interior], int),
                        int_a=np.array([t[1] for t in interior], int),
                        int_b=np.array([t[2] for t in interior], int),
                        int_w=np.array([t[3] for t in interior], float))
        groups = {}
        for L_ in links:
            groups.setdefault((L_["path"].size, L_["n_energy"], L_["metal"]), []).append(L_)
        self._tb["groups"] = [
            dict(a=np.array([L_["a"] for L_ in g]), b=np.array([L_["b"] for L_ in g]),
                 path=np.stack([L_["path"] for L_ in g]),
                 z=np.stack([L_["z"] for L_ in g]),
                 area=np.array([L_["area"] for L_ in g]),
                 mass=np.stack([np.full(L_["path"].size, L_["mass"]) if L_["mass"] is not None
                                else self.me_node[L_["path"]] for L_ in g]),
                 n_energy=key[1], metal=key[2])
            for key, g in groups.items()]

    # ------------------------------------------------- heat (0.22.0)
    def _thermal_setup(self, model):
        """The heat equation on the mesh itself, with the same box method
        as the currents: every triangle adds k d / h to the conductance of
        each of its edges (k of its region at the mean temperature of the
        edge's two nodes) and its part of each node's dual cell to the heat
        capacity. The nodes at the bottom of the mesh (lowest z) are the
        heat sink; every other boundary is adiabatic. A ThermalModel for a
        mesh names a ThermalMaterial for every region (dielectrics too);
        the substrate goes into the mesh as regions, so `substrate` and
        `lateral_extension` must be left empty."""
        from .thermal import ThermalModel
        if not isinstance(model, ThermalModel):
            raise ValueError("thermal must be a ThermalModel")
        if model.substrate or model.lateral_extension > 0:
            raise ValueError("on a mesh the heat flows in the mesh itself: put the "
                             "substrate into the mesh as regions and leave "
                             "ThermalModel.substrate and lateral_extension empty")
        mats_t = [model.material_for(r) for r in self.mesh.regions]
        g = _MeshHeat(self, mats_t)
        return dict(model=model, mats=g.mats, grid=g)

    def _heat_grid(self, th):
        return th["grid"]

    def _heat_to_dev(self, th, g, T):
        return np.asarray(T, float).ravel().copy()

    def _heat_from_dev(self, th, g, heat):
        return np.asarray(heat, float).ravel() * 1e3          # W/mm -> W/m

    def _temperature(self, th, heat, Tn, tol=1e-7, max_iter=200):
        g = th["grid"]
        H = self._heat_from_dev(th, g, heat)
        if not np.all(np.isfinite(H)):
            raise ValueError("heat must be finite")
        Ts = float(th["model"].sink_temperature_K)
        T = np.full(g.N, Ts)
        for _ in range(max_iter):
            Tn_ = g.solve(g.matrix(T), H, Ts)
            change = float(np.max(np.abs(Tn_ - T)))
            T = Tn_
            if change < tol:
                return T
        raise RuntimeError("the heat equation did not converge")

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


class _MeshHeat:
    """Conductances and heat capacities of a mesh (SI, per meter of depth),
    with the interface of thermal._HeatGrid."""

    def __init__(self, dev, mats_t):
        geo = dev._geo
        self.N = dev.N
        self.shape = (dev.N,)
        self.ei, self.ej = dev.ei, dev.ej
        uniq, midx = [], []
        for m in mats_t:
            for k, u in enumerate(uniq):
                if u is m or u == m:
                    midx.append(k)
                    break
            else:
                uniq.append(m)
                midx.append(len(uniq) - 1)
        midx = np.array(midx, int)
        self.mats = uniq
        self.k300 = np.array([m.k300 for m in uniq])
        self.alp = np.array([m.exponent for m in uniq])
        self.c_edge = geo["inv"]
        self.c_mat = midx[geo["tt"]]
        self.c_w = geo["dd"] / geo["hh"]
        self.v_node = geo["sn"]
        self.v_mat = midx[geo["st"]]
        self.v_area = geo["sa"] * 1e-18                      # m^2
        z = dev.coords[1]
        self.sink = z <= z.min() + 1e-9 * max(1.0, float(np.ptp(z)))
        self.nE = dev.ei.size

    def matrix(self, T):
        import scipy.sparse as sp
        Tf = np.asarray(T, float).ravel()
        Tm = 0.5 * (Tf[self.ei] + Tf[self.ej])
        m = self.c_mat
        k = self.k300[m] * (Tm[self.c_edge] / 300.0) ** (-self.alp[m])
        G = np.bincount(self.c_edge, self.c_w * k, self.nE)
        N = self.N
        diag = np.bincount(self.ei, G, N) + np.bincount(self.ej, G, N)
        A = sp.csr_matrix((np.concatenate([-G, -G]),
                           (np.concatenate([self.ei, self.ej]),
                            np.concatenate([self.ej, self.ei]))), shape=(N, N))
        return A + sp.diags(diag)

    def capacity(self, T):
        Tf = np.asarray(T, float).ravel()
        c = np.zeros(self.v_node.size)
        for k, m in enumerate(self.mats):
            sel = self.v_mat == k
            if sel.any():
                c[sel] = m.volumetric_heat(Tf[self.v_node[sel]])
        return np.bincount(self.v_node, self.v_area * c, self.N)

    def solve(self, A, b, Ts):
        import scipy.sparse.linalg as spl
        free = ~self.sink
        Af = A[free][:, free]
        rhs = b[free] - A[free][:, self.sink] @ np.full(int(self.sink.sum()), Ts)
        Tn = np.full(self.N, float(Ts))
        Tn[free] = spl.spsolve(Af.tocsc(), rhs)
        return Tn


def _circumcenters(A, B, C):
    ax, ay = A[:, 0], A[:, 1]
    bx, by = B[:, 0], B[:, 1]
    cx, cy = C[:, 0], C[:, 1]
    d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    a2, b2, c2 = ax * ax + ay * ay, bx * bx + by * by, cx * cx + cy * cy
    ux = (a2 * (by - cy) + b2 * (cy - ay) + c2 * (ay - by)) / d
    uy = (a2 * (cx - bx) + b2 * (ax - cx) + c2 * (bx - ax)) / d
    return np.column_stack([ux, uy])
